FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDAARCHS=121

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        git \
        cmake \
        build-essential \
        libgoogle-perftools-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Build CTranslate2 from source in the devel image.
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2.git

WORKDIR /tmp/CTranslate2

RUN sed -i 's/cuda_select_nvcc_arch_flags(ARCH_FLAGS ${CUDA_ARCH_LIST})/set(ARCH_FLAGS "-gencode;arch=compute_121,code=sm_121;-gencode;arch=compute_121,code=compute_121")/' CMakeLists.txt && \
    grep -q 'compute_121,code=sm_121' CMakeLists.txt

RUN mkdir build

WORKDIR /tmp/CTranslate2/build

RUN cmake \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DOPENMP_RUNTIME=COMP \
        -DWITH_MKL=OFF \
        -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
        -DCMAKE_BUILD_TYPE=Release \
        ..

RUN make -j"$(nproc)" && \
    make install

WORKDIR /tmp/CTranslate2/python

RUN python3 -m pip wheel . --wheel-dir /tmp/wheels

WORKDIR /tmp

COPY requirements.txt /tmp/requirements.txt

# Build dependency wheels in the devel stage to avoid compiling in runtime.
RUN python3 -m pip wheel --wheel-dir /tmp/req-wheels -r /tmp/requirements.txt

FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        python3 \
        python3-pip \
        libgomp1 \
        libgoogle-perftools4 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

COPY --from=builder /tmp/req-wheels /tmp/req-wheels
COPY --from=builder /tmp/wheels /tmp/wheels
COPY --from=builder /usr/local/lib/libctranslate2* /usr/local/lib/

# Install the CTranslate2 Python package built in the devel stage first.
RUN pip3 install --no-cache-dir --no-deps /tmp/wheels/ctranslate2-*.whl

RUN pip3 install --no-cache-dir --no-index --find-links=/tmp/req-wheels -r requirements.txt && \
    rm -rf /tmp/wheels && \
    rm -rf /tmp/req-wheels && \
    ldconfig

COPY . .

EXPOSE 5000

ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

ENTRYPOINT ["python3", "whisper_fastapi.py"]
