##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Common parts which can be used by all plugins
#
##===----------------------------------------------------------------------===##

# NOTE: Don't try to build `PluginInterface` using `add_llvm_library` because we
# don't want to export `PluginInterface` while `add_llvm_library` requires that.
add_library(PluginInterface OBJECT
  PluginInterface.cpp
  GlobalHandler.cpp
  JIT.cpp
  RPC.cpp
  Utils/ELF.cpp
)

# Only enable JIT for those targets that LLVM can support.
string(TOUPPER "${LLVM_TARGETS_TO_BUILD}" TargetsSupported)
foreach(Target ${TargetsSupported})
  target_compile_definitions(PluginInterface PRIVATE "LIBOMPTARGET_JIT_${Target}")
endforeach()

# This is required when using LLVM libraries.
llvm_update_compile_flags(PluginInterface)

if (LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs
    ${LLVM_TARGETS_TO_BUILD}
    AggressiveInstCombine
    Analysis
    BinaryFormat
    BitReader
    BitWriter
    CodeGen
    Core
    Extensions
    InstCombine
    Instrumentation
    IPO
    IRReader
    Linker
    MC
    Object
    Passes
    Remarks
    ScalarOpts
    Support
    Target
    TargetParser
    TransformUtils
    Vectorize
  )
endif()

target_link_libraries(PluginInterface
  PUBLIC
    ${llvm_libs}
    OMPT
)

#if RPC_FIXME
# Include the RPC server from the `libc` project if availible.
if(TARGET llvmlibc_rpc_server AND ${LIBOMPTARGET_GPU_LIBC_SUPPORT})
  target_link_libraries(PluginInterface PRIVATE llvmlibc_rpc_server)
  target_compile_definitions(PluginInterface PRIVATE LIBOMPTARGET_RPC_SUPPORT)
elseif(${LIBOMPTARGET_GPU_LIBC_SUPPORT})
  find_library(llvmlibc_rpc_server NAMES llvmlibc_rpc_server
               PATHS ${LIBOMPTARGET_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
  if(llvmlibc_rpc_server)
    message(WARNING ${llvmlibc_rpc_server})
    target_link_libraries(PluginInterface PRIVATE llvmlibc_rpc_server)
    target_compile_definitions(PluginInterface PRIVATE LIBOMPTARGET_RPC_SUPPORT)
  endif()
endif()

if ((OMPT_TARGET_DEFAULT) AND (LIBOMPTARGET_OMPT_SUPPORT))
  target_link_libraries(PluginInterface PUBLIC OMPT)
endif()
#endif

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(PluginInterface PRIVATE
  TARGET_NAME="PluginInterface"
  DEBUG_PREFIX="PluginInterface"
)

target_include_directories(PluginInterface
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(PluginInterface PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET protected)
