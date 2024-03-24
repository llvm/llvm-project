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
add_library(PluginCommon OBJECT
  src/PluginInterface.cpp
  src/GlobalHandler.cpp
  src/JIT.cpp
  src/RPC.cpp
  src/Utils/ELF.cpp
)
add_dependencies(PluginCommon intrinsics_gen)

# Only enable JIT for those targets that LLVM can support.
string(TOUPPER "${LLVM_TARGETS_TO_BUILD}" TargetsSupported)
foreach(Target ${TargetsSupported})
	target_compile_definitions(PluginCommon PRIVATE "LIBOMPTARGET_JIT_${Target}")
endforeach()

# Include the RPC server from the `libc` project if availible.
if(TARGET llvmlibc_rpc_server AND ${LIBOMPTARGET_GPU_LIBC_SUPPORT})
	target_link_libraries(PluginCommon PRIVATE llvmlibc_rpc_server)
	target_compile_definitions(PluginCommon PRIVATE LIBOMPTARGET_RPC_SUPPORT)
elseif(${LIBOMPTARGET_GPU_LIBC_SUPPORT})
  find_library(llvmlibc_rpc_server NAMES llvmlibc_rpc_server
               PATHS ${LIBOMPTARGET_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
  if(llvmlibc_rpc_server)
		target_link_libraries(PluginCommon PRIVATE ${llvmlibc_rpc_server})
		target_compile_definitions(PluginCommon PRIVATE LIBOMPTARGET_RPC_SUPPORT)
    # We may need to get the headers directly from the 'libc' source directory.
    target_include_directories(PluginCommon PRIVATE
                               ${CMAKE_SOURCE_DIR}/../libc/utils/gpu/server
                               ${CMAKE_SOURCE_DIR}/../libc/include)
  endif()
endif()

# If we have OMPT enabled include it in the list of sources.
if (OMPT_TARGET_DEFAULT AND LIBOMPTARGET_OMPT_SUPPORT)
  target_sources(PluginCommon PRIVATE OMPT/OmptCallback.cpp)
  target_include_directories(PluginCommon PRIVATE OMPT)
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(PluginCommon PRIVATE
  TARGET_NAME="PluginInterface"
  DEBUG_PREFIX="PluginInterface"
)

target_compile_options(PluginCommon PUBLIC ${offload_compile_flags})
target_link_options(PluginCommon PUBLIC ${offload_link_flags})

target_include_directories(PluginCommon PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(PluginCommon PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET protected)
