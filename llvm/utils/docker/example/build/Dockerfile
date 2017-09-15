#===- llvm/utils/docker/example/build/Dockerfile -------------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===//
# This is an example Dockerfile to build an image that compiles clang.
# Replace FIXMEs to prepare your own image.

# FIXME: Replace 'ubuntu' with your base image
FROM ubuntu

# FIXME: Change maintainer name
LABEL maintainer "Maintainer <maintainer@email>"

# FIXME: Install llvm/clang build dependencies. Including compiler to
# build stage1, cmake, subversion, ninja, etc.

ADD checksums /tmp/checksums
ADD scripts /tmp/scripts

# Arguments passed to build_install_clang.sh.
ARG buildscript_args

# Run the build. Results of the build will be available as /tmp/clang.tar.gz.
RUN /tmp/scripts/build_install_llvm.sh ${buildscript_args}
