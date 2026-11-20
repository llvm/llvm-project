ARG UBUNTU_VERSION=22.04
ARG SWIFT_MAJOR_VERSION=8
ARG SWIFT_MINOR_VERSION=1
ARG CUSTOM_CERT
ARG ENABLE_LEGACY_RENEGOTIATION

FROM ubuntu:$UBUNTU_VERSION

ARG SWIFT_MAJOR_VERSION
ARG SWIFT_MINOR_VERSION
ARG CUSTOM_CERT
ARG ENABLE_LEGACY_RENEGOTIATION

ENV DEBIAN_FRONTEND=noninteractive

# Install the base dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3 \
  python3-dev \
  python-is-python3 \
  wget \
  curl \
  lsb-release \
  ca-certificates \
  software-properties-common \
  build-essential \
  gnupg2 \
  python3-pip \
  git \
  pkg-config \
  libssl-dev \
  gcc \
  gfortran \
  vim \
  libarchive-dev \
  libudev-dev \
  libasound2-dev \
  libzmq3-dev \
  cmake \
  ninja-build \
  flex \
  bison \
  libelf-dev \
  bc \
  cpio \
  htop \
  jq \
  file \
  unzip \
  binutils \
  bison \
  bzip2 \
  cdbs \
  curl \
  dbus-x11 \
  devscripts \
  dpkg-dev \
  elfutils \
  fakeroot \
  flex \
  git-core \
  gperf \
  libasound2-dev \
  libatspi2.0-dev \
  libbrlapi-dev \
  libbz2-dev \
  libc6-dev \
  libcairo2-dev \
  libcap-dev \
  libcups2-dev \
  libcurl4-gnutls-dev \
  libdrm-dev \
  libelf-dev \
  libevdev-dev \
  libffi-dev \
  libfuse2 \
  libgbm-dev \
  libglib2.0-dev \
  libglu1-mesa-dev \
  libgtk-3-dev \
  libkrb5-dev \
  libnspr4-dev \
  libnss3-dev \
  libpam0g-dev \
  libpci-dev \
  libpulse-dev \
  libsctp-dev \
  libspeechd-dev \
  libsqlite3-dev \
  libssl-dev \
  libsystemd-dev \
  libudev-dev \
  libva-dev \
  libwww-perl \
  libxshmfence-dev \
  libxslt1-dev \
  libxss-dev \
  libxt-dev \
  libxtst-dev \
  lighttpd \
  locales \
  openbox \
  p7zip \
  patch \
  perl \
  pkg-config \
  rpm \
  ruby \
  subversion \
  uuid-dev \
  wdiff \
  x11-utils \
  xcompmgr \
  xz-utils \
  zip \
  libbluetooth-dev \
  libxkbcommon-dev \
  mesa-common-dev \
  zstd \
  libjpeg-dev \
  libudev1 \
  libbrlapi0.8 \
  libvulkan-dev \
  libinput-dev \
  binutils-arm-linux-gnueabihf \
  binutils-aarch64-linux-gnu \
  binutils-mipsel-linux-gnu \
  binutils-mips64el-linux-gnuabi64 \
  libc6-i386 \
  lib32stdc++6 \
  lib32gcc-s1 \
  lib32z1 \
  libasound2 \
  libatk1.0-0 \
  libatspi2.0-0 \
  libc6 \
  libcairo2 \
  libcap2 \
  libcgi-session-perl \
  libcups2 \
  libdrm2 \
  libegl1 \
  libevdev2 \
  libexpat1 \
  libfontconfig1 \
  libfreetype6 \
  libgbm1 \
  libglib2.0-0 \
  libgl1 \
  libgtk-3-0 \
  libncurses5 \
  libpam0g \
  libpango-1.0-0 \
  libpangocairo-1.0-0 \
  libpci3 \
  libpcre3 \
  libpixman-1-0 \
  libspeechd2 \
  libstdc++6 \
  libsqlite3-0 \
  libuuid1 \
  libwayland-egl1 \
  libwayland-egl1-mesa \
  libx11-6 \
  libx11-xcb1 \
  libxau6 \
  libxcb1 \
  libxcomposite1 \
  libxcursor1 \
  libxdamage1 \
  libxdmcp6 \
  libxext6 \
  libxfixes3 \
  libxi6 \
  libxinerama1 \
  libxrandr2 \
  libxrender1 \
  libxtst6 \
  x11-utils \
  xvfb \
  zlib1g \
  libpulse0 \
  libbz2-1.0 \
  libffi8 \
  libpng16-16 \
  libnspr4 \
  libnss3 \
  libvulkan1 \
  libinput10 \
  cups \
  xcb \
  libxcb-xkb-dev \
  x11-xkb-utils \
  libx11-xcb-dev \
  libxkbcommon-x11-dev \
  generate-ninja \
  cbindgen \
  nasm \
  libdbus-glib-1-dev

# Setup a custom certificate/SSL settings depending upon build arguments
# Include README.md here so that the build doesn't fail if there is no custom
# certificate specified. Then we just delete it afterwards.
COPY README.md $CUSTOM_CERT /usr/local/share/ca-certificates/
RUN rm /usr/local/share/ca-certificates/README.md \
  && update-ca-certificates
RUN if [ -n "$ENABLE_LEGACY_RENEGOTIATION" ]; then echo "Options = UnsafeLegacyRenegotiation" >> /etc/ssl/openssl.cnf ; fi

# Can this be converted into a native Ubuntu install as in the LLVM case
ENV CARGO_HOME="/cargo"
ENV RUSTUP_HOME="/rustup"
RUN curl https://sh.rustup.rs | sh -s -- -y --default-toolchain none
ENV PATH="$PATH:/cargo/bin"

# LLVM Installation
RUN git clone -b running-fixes --depth=1 https://github.com/llvm-ml/llvm-project \
    && mkdir /llvm-project/build \
    && cd /llvm-project/build \
    && cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
         -DLLVM_ENABLE_PROJECTS="clang;lld" \
         -DLLVM_ENABLE_RUNTIMES="compiler-rt;openmp" \
         -DCMAKE_INSTALL_PREFIX=/usr \
         -DLLVM_TARGETS_TO_BUILD=Native \
         -DLLVM_ENABLE_ASSERTIONS=ON \
         ../llvm \
    && ninja install \
    && cd / \
    && rm -rf /llvm-project

# Install Julia
RUN git clone -b emit-per-package-bitcode --depth=1 https://github.com/llvm-ml/julia /julia \
    && cd /julia \
    && make MARCH=x86-64 -j $(nproc) \
    && echo prefix=/usr > Make.user \
    && make MARCH=x86-64 install \
    && cd / \
    && rm -rf /julia

# Install Swift
RUN curl \
    https://download.swift.org/swift-5.8.1-release/ubuntu2204/swift-5.8.1-RELEASE/swift-5.8.1-RELEASE-ubuntu22.04.tar.gz \
    | tar -xz
RUN mv swift-5.$SWIFT_MAJOR_VERSION.$SWIFT_MINOR_VERSION-RELEASE-ubuntu22.04/usr/ /opt/swift-5.$SWIFT_MAJOR_VERSION.$SWIFT_MINOR_VERSION/
ENV PATH="${PATH}:/opt/swift-5.$SWIFT_MAJOR_VERSION.$SWIFT_MINOR_VERSION/bin/"

# Set up the Python dependencies
COPY Pipfile* ./
RUN pip3 install pipenv \
 && pipenv sync --categories "packages dev-packages" --system \
 && pipenv --clear \
 && rm Pipfile*

# Install current node
RUN curl -sL https://deb.nodesource.com/setup_20.x | bash
RUN apt-get install -y nodejs

# Clean up the Docker container to make the image smaller
RUN apt-get autoremove -y --purge \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=
