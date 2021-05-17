##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build server for remote offloading.
#
##===----------------------------------------------------------------------===##

include_directories(${LIBOMPTARGET_DEP_LIBELF_INCLUDE_DIRS})
include_directories(${LIBOMPTARGET_SRC_DIR})
include_directories(${LIBOMPTARGET_INCLUDE_DIR})
include_directories(${GRPC_INCLUDE_DIR})
include_directories(${RPC_INCLUDE_DIR})

add_executable(openmp-offloading-server
        ${LIBOMPTARGET_SRC_FILES}
        ${GRPC_SRC_FILES}
        ${RPC_SRC_DIR}/Utils.cpp
        Server.cpp
        OffloadingServer.cpp
)

target_link_libraries(openmp-offloading-server
        grpc++
        protobuf
        ${OPENMP_PTHREAD_LIB}
        "-ldl" "-lomp" "-fopenmp" "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../../exports" ${LIBOMPTARGET_DEP_LIBELF_LIBRARIES})
