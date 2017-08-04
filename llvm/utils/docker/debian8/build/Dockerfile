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
    apt-get install -y --no-install-recommends build-essential python2.7 wget \
            subversion ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Install cmake version that can compile clang into /usr/local.
# (Version in debian8 repos is is too old)
RUN wget -O - "https://cmake.org/files/v3.7/cmake-3.7.2-Linux-x86_64.tar.gz" | \
    tar xzf - -C /usr/local --strip-components=1

# Arguments passed to build_install_clang.sh.
ARG buildscript_args

# Run the build. Results of the build will be available as /tmp/clang.tar.gz.
ADD scripts/build_install_llvm.sh /tmp
RUN /tmp/build_install_llvm.sh ${buildscript_args}
