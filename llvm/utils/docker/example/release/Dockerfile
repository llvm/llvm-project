#===- llvm/utils/docker/example/release/Dockerfile -----------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===----------------------------------------------------------------------===//
# An image that unpacks a clang installation, compiled by the 'build/'
# container.
# Replace FIXMEs to prepare your own image.

# FIXME: Replace 'ubuntu' with your base image.
FROM ubuntu

# FIXME: Change maintainer name.
LABEL maintainer "Maintainer <maintainer@email>"

# FIXME: Install all packages you want to have in your release container.
# A minimal useful installation must include libstdc++ and binutils.

# Unpack clang installation into this container.
# It is copied to this directory by build_docker_image.sh script.
ADD clang.tar.gz /usr/local/
