#===- llvm/utils/docker/nvidia-cuda/build/Dockerfile ---------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===//
# Produces an image that compiles and archives clang, based on nvidia/cuda
# image.
FROM nvidia/cuda:8.0-devel

LABEL maintainer "LLVM Developers"

# Arguments to pass to build_install_clang.sh.
ARG buildscript_args

# Install llvm build dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake python2.7 subversion ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Run the build. Results of the build will be available as /tmp/clang.tar.gz.
ADD scripts/build_install_llvm.sh /tmp
RUN /tmp/build_install_llvm.sh ${buildscript_args}
