#===- llvm/utils/docker/debian8/release/Dockerfile -----------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===//
# A release image, containing clang installation, produced by the 'build/' image
# and adding libstdc++ and binutils.
FROM launcher.gcr.io/google/debian8:latest

LABEL maintainer "LLVM Developers"

# Install packages for minimal usefull image.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libstdc++-4.9-dev binutils && \
    rm -rf /var/lib/apt/lists/*

# Unpack clang installation into this image.
ADD clang.tar.gz /
