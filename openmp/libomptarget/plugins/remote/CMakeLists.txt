##===----------------------------------------------------------------------===##
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin (client) and server for remote offloading.
#
##===----------------------------------------------------------------------===#

if (NOT(CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building remote offloading plugin: only support Linux hosts.")
  return()
endif()

if (NOT(LIBOMPTARGET_ENABLE_EXPERIMENTAL_REMOTE_PLUGIN))
  return()
endif()

find_package(Protobuf)
find_package(gRPC CONFIG)

find_program(PROTOC protoc)
find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)

if (Protobuf_FOUND AND gRPC_FOUND AND PROTOC AND GRPC_CPP_PLUGIN)
  libomptarget_say("Building remote offloading plugin.")
  set(directory "${CMAKE_BINARY_DIR}/include/openmp/libomptarget/plugins/remote/")
  file(MAKE_DIRECTORY ${directory})
  execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${directory})
  execute_process(
          COMMAND protoc --cpp_out=${directory} -I ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/openmp.proto
          COMMAND protoc --grpc_out=${directory} -I ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/openmp.proto --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
  )

  set(GRPC_SRC_FILES
      ${directory}/openmp.grpc.pb.cc
      ${directory}/openmp.pb.cc
  )

  set(GRPC_INCLUDE_DIR
      ${directory}
  )

  set(RPC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)
  set(RPC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/)
  
  add_subdirectory(src)
  add_subdirectory(server)
else()
  libomptarget_say("Not building remote offloading plugin: required libraries were not found.")
endif()

