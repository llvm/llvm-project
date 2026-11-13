# The error codes in this header are shared with liboffload, so need to be
# generated from the same source.
include(TableGen)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/../../API/OffloadAPI.td)
offload_tablegen(include/OffloadErrcodes.inc -gen-errcodes EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../API)
offload_tablegen(include/OffloadInfo.inc -gen-info EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/../../API)
add_public_tablegen_target(LLVMOffloadPluginErrcodes)

# NOTE: Don't try to build `PluginInterface` using `add_llvm_library` because we
# don't want to export `PluginInterface` while `add_llvm_library` requires that.
add_library(LLVMOffloadPluginCommon OBJECT
  src/PluginInterface.cpp
  src/GenericProfiler.cpp
  src/GlobalHandler.cpp
  src/JIT.cpp
  src/RPC.cpp
  src/OffloadError.cpp
  src/Utils/ELF.cpp
)
add_dependencies(LLVMOffloadPluginCommon intrinsics_gen LLVMOffloadPluginErrcodes OffloadAPI)

# Only enable JIT for those targets that LLVM can support.
set(supported_jit_targets AMDGPU NVPTX SPIRV)
foreach(target IN LISTS supported_jit_targets)
  if("${target}" IN_LIST LLVM_TARGETS_TO_BUILD)
    target_compile_definitions(LLVMOffloadPluginCommon PRIVATE "LIBOMPTARGET_JIT_${target}")
    if (NOT LLVM_LINK_LLVM_DYLIB)
      llvm_map_components_to_libnames(llvm_libs ${target})
      target_link_libraries(LLVMOffloadPluginCommon PRIVATE ${llvm_libs})
    endif()
  endif()
endforeach()

# Include the RPC server from the `libc` project.
include(FindLibcCommonUtils)
target_link_libraries(LLVMOffloadPluginCommon PRIVATE llvm-libc-common-utilities)

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(LLVMOffloadPluginCommon PRIVATE
  TARGET_NAME=PluginInterface
  DEBUG_PREFIX="PluginInterface"
)

target_compile_options(LLVMOffloadPluginCommon PUBLIC ${offload_compile_flags})
target_link_options(LLVMOffloadPluginCommon PUBLIC ${offload_link_flags})

target_include_directories(LLVMOffloadPluginCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/../../API
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
  ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(LLVMOffloadPluginCommon PROPERTIES POSITION_INDEPENDENT_CODE ON)
