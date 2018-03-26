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

# Stage 1. Check out LLVM source code and run the build.
# FIXME: Replace 'ubuntu' with your base image
FROM ubuntu as builder
# FIXME: Change maintainer name
LABEL maintainer "Maintainer <maintainer@email>"
# FIXME: Install llvm/clang build dependencies here. Including compiler to
# build stage1, cmake, subversion, ninja, etc.

ADD checksums /tmp/checksums
ADD scripts /tmp/scripts
# Arguments passed to build_install_clang.sh.
ARG buildscript_args
# Run the build. Results of the build will be available as /tmp/clang-install.
RUN /tmp/scripts/build_install_llvm.sh ${buildscript_args}


# Stage 2. Produce a minimal release image with build results.
# FIXME: Replace 'ubuntu' with your base image.
FROM ubuntu
# FIXME: Change maintainer name.
LABEL maintainer "Maintainer <maintainer@email>"
# FIXME: Install all packages you want to have in your release container.
# A minimal useful installation should include at least libstdc++ and binutils.

# Copy build results of stage 1 to /usr/local.
COPY --from=builder /tmp/clang-install/ /usr/local/
