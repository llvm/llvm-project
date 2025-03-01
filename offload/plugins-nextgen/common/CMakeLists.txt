
if(OFFLOAD_ENABLE_EMISSARY_APIS)
  set(emissary_sources
    src/Emissary.cpp
    src/EmissaryFortrt.cpp
    src/EmissaryPrint.cpp
  )
  set(OFFLOAD_EMISSARY_MPI_INCLUDE ""
	  CACHE STRING "MPI include directory for building MPI Emissary API")
  # dont enable the MPI Emissary api unless we have an MPI include dir
  # that contains mpi.h needed to build the variadic wrappers.
  if(OFFLOAD_EMISSARY_MPI_INCLUDE)
    list(APPEND emissary_sources src/EmissaryMPI.cpp)
  endif()
endif()

# NOTE: Don't try to build `PluginInterface` using `add_llvm_library` because we
# don't want to export `PluginInterface` while `add_llvm_library` requires that.
add_library(PluginCommon OBJECT
  src/PluginInterface.cpp
  src/GlobalHandler.cpp
  src/JIT.cpp
  src/RPC.cpp
  src/Utils/ELF.cpp
  ${emissary_sources}
)
add_dependencies(PluginCommon intrinsics_gen)

# Only enable JIT for those targets that LLVM can support.
set(supported_jit_targets AMDGPU NVPTX)
if (NOT LLVM_LINK_LLVM_DYLIB)
  foreach(target IN LISTS supported_jit_targets)
    if("${target}" IN_LIST LLVM_TARGETS_TO_BUILD)
      target_compile_definitions(PluginCommon PRIVATE "LIBOMPTARGET_JIT_${target}")
      llvm_map_components_to_libnames(llvm_libs ${target})
      target_link_libraries(PluginCommon PRIVATE ${llvm_libs})
    endif()
  endforeach()
endif()

# Include the RPC server from the `libc` project if available.
include(FindLibcCommonUtils)
target_link_libraries(PluginCommon PRIVATE llvm-libc-common-utilities)
if(OFFLOAD_ENABLE_EMISSARY_APIS)
  target_link_libraries(PluginCommon PRIVATE flang_rt.runtime
  -L${CMAKE_BINARY_DIR}/../../lib  -L${CMAKE_INSTALL_PREFIX}/lib)
  if(OFFLOAD_EMISSARY_MPI_INCLUDE)
    target_include_directories(PluginCommon PUBLIC ${OFFLOAD_EMISSARY_MPI_INCLUDE})
  endif()
endif()
if(TARGET llvmlibc_rpc_server AND ${LIBOMPTARGET_GPU_LIBC_SUPPORT})
	target_link_libraries(PluginCommon PRIVATE llvmlibc_rpc_server)
	target_compile_definitions(PluginCommon PRIVATE LIBOMPTARGET_RPC_SUPPORT)
elseif(${LIBOMPTARGET_GPU_LIBC_SUPPORT})
  find_library(llvmlibc_rpc_server NAMES llvmlibc_rpc_server
               PATHS ${LIBOMPTARGET_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
  if(llvmlibc_rpc_server)
    target_link_libraries(PluginCommon PRIVATE ${llvmlibc_rpc_server})
		target_compile_definitions(PluginCommon PRIVATE LIBOMPTARGET_RPC_SUPPORT)
  endif()
endif()

# If we have OMPT enabled include it in the list of sources.
if (OMPT_TARGET_DEFAULT AND LIBOMPTARGET_OMPT_SUPPORT)
  target_sources(PluginCommon PRIVATE OMPT/OmptTracing.cpp)
  target_include_directories(PluginCommon PUBLIC OMPT)
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(PluginCommon PRIVATE
  TARGET_NAME="PluginInterface"
  DEBUG_PREFIX="PluginInterface"
)

target_compile_options(PluginCommon PUBLIC ${offload_compile_flags})
target_link_options(PluginCommon PUBLIC ${offload_link_flags})
if (LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(PluginCommon PRIVATE LLVM)
else()
  target_link_libraries(PluginCommon PRIVATE LLVMProfileData)
endif()

target_include_directories(PluginCommon PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${LIBOMPTARGET_LLVM_INCLUDE_DIRS}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
  ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(PluginCommon PROPERTIES POSITION_INDEPENDENT_CODE ON)
