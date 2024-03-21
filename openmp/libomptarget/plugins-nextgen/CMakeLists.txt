##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build plugins for the user system if available.
#
##===----------------------------------------------------------------------===##

# Common interface to handle creating a plugin library.
set(common_dir ${CMAKE_CURRENT_SOURCE_DIR}/common)
function(add_target_library target_name lib_name)
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

  add_llvm_library(${target_name} SHARED
    ${common_dir}/src/PluginInterface.cpp
    ${common_dir}/src/GlobalHandler.cpp
    ${common_dir}/src/JIT.cpp
    ${common_dir}/src/RPC.cpp
    ${common_dir}/src/Utils/ELF.cpp

    NO_INSTALL_RPATH
    BUILDTREE_ONLY
  )

  target_link_libraries(${target_name} PUBLIC ${llvm_libs} ${OPENMP_PTHREAD_LIB})
  llvm_update_compile_flags(${target_name})

  # Include the RPC server from the `libc` project if availible.
  if(TARGET llvmlibc_rpc_server AND ${LIBOMPTARGET_GPU_LIBC_SUPPORT})
    target_link_libraries(${target_name} PRIVATE llvmlibc_rpc_server)
    target_compile_definitions(${target_name} PRIVATE LIBOMPTARGET_RPC_SUPPORT)
  elseif(${LIBOMPTARGET_GPU_LIBC_SUPPORT})
    find_library(llvmlibc_rpc_server NAMES llvmlibc_rpc_server
                 PATHS ${LIBOMPTARGET_LLVM_LIBRARY_DIR} NO_DEFAULT_PATH)
    if(llvmlibc_rpc_server)
      target_link_libraries(${target_name} PRIVATE ${llvmlibc_rpc_server})
      target_compile_definitions(${target_name} PRIVATE LIBOMPTARGET_RPC_SUPPORT)
      # We may need to get the headers directly from the 'libc' source directory.
      target_include_directories(${target_name} PRIVATE
                                 ${CMAKE_SOURCE_DIR}/../libc/utils/gpu/server
                                 ${CMAKE_SOURCE_DIR}/../libc/include)
    endif()
  endif()

  # Only enable JIT for those targets that LLVM can support.
  string(TOUPPER "${LLVM_TARGETS_TO_BUILD}" TargetsSupported)
  foreach(Target ${TargetsSupported})
    target_compile_definitions(${target_name} PRIVATE "LIBOMPTARGET_JIT_${Target}")
  endforeach()

  target_compile_definitions(${target_name} PRIVATE TARGET_NAME=${lib_name})
  target_compile_definitions(${target_name} PRIVATE 
                             DEBUG_PREFIX="TARGET ${lib_name} RTL")

  # If we have OMPT enabled include it in the list of sourced.
  if (OMPT_TARGET_DEFAULT AND LIBOMPTARGET_OMPT_SUPPORT)
    target_sources(${target_name} PRIVATE ${common_dir}/OMPT/OmptCallback.cpp)
  endif()

  if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    # On FreeBSD, the 'environ' symbol is undefined at link time, but resolved by
    # the dynamic linker at runtime. Therefore, allow the symbol to be undefined
    # when creating a shared library.
    target_link_libraries(${target_name} PRIVATE "-Wl,--allow-shlib-undefined")
  else()
    target_link_libraries(${target_name} PRIVATE "-Wl,-z,defs")
  endif()

  target_include_directories(${target_name} PRIVATE
    ${LIBOMPTARGET_INCLUDE_DIR}
    ${common_dir}/include
  )
  if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
    target_link_libraries(${target_name} PRIVATE
    "-Wl,--version-script=${common_dir}/../exports")
  endif()
  set_target_properties(${target_name} PROPERTIES CXX_VISIBILITY_PRESET protected)
endfunction()

add_subdirectory(amdgpu)
add_subdirectory(cuda)
add_subdirectory(host)

# Make sure the parent scope can see the plugins that will be created.
set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
