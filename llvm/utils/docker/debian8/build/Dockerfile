#===- llvm/utils/docker/debian8/build/Dockerfile -------------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===//
# Produces an image that compiles and archives clang, based on debian8.
FROM launcher.gcr.io/google/debian8:latest

LABEL maintainer "LLVM Developers"

# Install build dependencies of llvm.
# First, Update the apt's source list and include the sources of the packages.
RUN grep deb /etc/apt/sources.list | \
    sed 's/^deb/deb-src /g' >> /etc/apt/sources.list

# Install compiler, python and subversion.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates gnupg \
           build-essential python wget subversion unzip && \
    rm -rf /var/lib/apt/lists/*

# Install a newer ninja release. It seems the older version in the debian repos
# randomly crashes when compiling llvm.
RUN wget "https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip" && \
    echo "d2fea9ff33b3ef353161ed906f260d565ca55b8ca0568fa07b1d2cab90a84a07 ninja-linux.zip" \
        | sha256sum -c  && \
    unzip ninja-linux.zip -d /usr/local/bin && \
    rm ninja-linux.zip

# Import public key required for verifying signature of cmake download.
RUN gpg --keyserver hkp://pgp.mit.edu --recv 0x2D2CEF1034921684

# Download, verify and install cmake version that can compile clang into /usr/local.
# (Version in debian8 repos is is too old)
RUN mkdir /tmp/cmake-install && cd /tmp/cmake-install && \
    wget "https://cmake.org/files/v3.7/cmake-3.7.2-SHA-256.txt.asc" && \
    wget "https://cmake.org/files/v3.7/cmake-3.7.2-SHA-256.txt" && \
    gpg --verify cmake-3.7.2-SHA-256.txt.asc cmake-3.7.2-SHA-256.txt && \
    wget "https://cmake.org/files/v3.7/cmake-3.7.2-Linux-x86_64.tar.gz" && \
    ( grep "cmake-3.7.2-Linux-x86_64.tar.gz" cmake-3.7.2-SHA-256.txt | \
      sha256sum -c - ) && \
    tar xzf cmake-3.7.2-Linux-x86_64.tar.gz -C /usr/local --strip-components=1 && \
    cd / && rm -rf /tmp/cmake-install

ADD checksums /tmp/checksums
ADD scripts /tmp/scripts

# Arguments passed to build_install_clang.sh.
ARG buildscript_args

# Run the build. Results of the build will be available as /tmp/clang.tar.gz.
RUN /tmp/scripts/build_install_llvm.sh ${buildscript_args}
