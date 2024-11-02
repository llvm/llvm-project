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
add_subdirectory(common)
function(add_target_library target_name lib_name)
  add_llvm_library(${target_name} SHARED
    LINK_COMPONENTS
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

    NO_INSTALL_RPATH
    BUILDTREE_ONLY
  )

  llvm_update_compile_flags(${target_name})
  target_link_libraries(${target_name} PRIVATE
                        PluginCommon ${OPENMP_PTHREAD_LIB})

  target_compile_definitions(${target_name} PRIVATE TARGET_NAME=${lib_name})
  target_compile_definitions(${target_name} PRIVATE 
                             DEBUG_PREFIX="TARGET ${lib_name} RTL")

  if(CMAKE_SYSTEM_NAME MATCHES "FreeBSD")
    # On FreeBSD, the 'environ' symbol is undefined at link time, but resolved by
    # the dynamic linker at runtime. Therefore, allow the symbol to be undefined
    # when creating a shared library.
    target_link_libraries(${target_name} PRIVATE "-Wl,--allow-shlib-undefined")
  else()
    target_link_libraries(${target_name} PRIVATE "-Wl,-z,defs")
  endif()

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
