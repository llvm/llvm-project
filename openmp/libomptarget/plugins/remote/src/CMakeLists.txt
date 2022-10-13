##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for remote offloading.
#
##===----------------------------------------------------------------------===##

cmake_minimum_required(VERSION 3.13.4)

# Define the suffix for the runtime messaging dumps.
add_definitions(-DTARGET_NAME=RPC)

include_directories(${LIBOMPTARGET_SRC_DIR})
include_directories(${LIBOMPTARGET_INCLUDE_DIR})
include_directories(${GRPC_INCLUDE_DIR})
include_directories(${RPC_INCLUDE_DIR})

add_library(omptarget.rtl.rpc SHARED
        ${LIBOMPTARGET_SRC_FILES}
        ${GRPC_SRC_FILES}
        ${RPC_SRC_DIR}/Utils.cpp
        Client.cpp
        rtl.cpp
)

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.rpc LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")

target_link_libraries(omptarget.rtl.rpc
  grpc++
  protobuf
  absl::synchronization
  ${OPENMP_PTHREAD_LIB}
  omp
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../../exports"
  "-Wl,--undefined-version")

# Report to the parent scope that we are building a plugin for RPC.
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS} rpc" PARENT_SCOPE)
