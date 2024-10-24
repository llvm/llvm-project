# Only enable JIT for those targets that LLVM can support.
set(jit_targets_to_build "AMDGPU;NVPTX")
set(jit_targets_not_to_build ${jit_targets_to_build})
list(REMOVE_ITEM jit_targets_not_to_build ${LLVM_TARGETS_TO_BUILD})
list(REMOVE_ITEM jit_targets_to_build ${jit_targets_not_to_build})

add_llvm_library(PluginCommon STATIC
  src/PluginInterface.cpp
  src/GlobalHandler.cpp
  src/JIT.cpp
  src/RPC.cpp
  src/Utils/ELF.cpp

  LINK_COMPONENTS
  ProfileData
  ${jit_targets_to_build}
  
  BUILDTREE_ONLY
)
add_dependencies(PluginCommon intrinsics_gen LLVMProfileData)

list(TRANSFORM jit_targets_to_build
  PREPEND "LIBOMPTARGET_JIT_" OUTPUT_VARIABLE compile_defs
)
target_compile_definitions(PluginCommon PRIVATE ${compile_defs})

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
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
  ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(PluginCommon PROPERTIES POSITION_INDEPENDENT_CODE ON)
