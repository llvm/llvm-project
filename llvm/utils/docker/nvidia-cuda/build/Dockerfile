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
    apt-get install -y --no-install-recommends ca-certificates cmake python \
		    subversion ninja-build && \
    rm -rf /var/lib/apt/lists/*

ADD checksums /tmp/checksums
ADD scripts /tmp/scripts

# Arguments passed to build_install_clang.sh.
ARG buildscript_args

# Run the build. Results of the build will be available as /tmp/clang.tar.gz.
RUN /tmp/scripts/build_install_llvm.sh ${buildscript_args}
