##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Aggregation of parts which can be used by OpenMP tools
#
##===----------------------------------------------------------------------===##

# NOTE: Don't try to build `OMPT` using `add_llvm_library` because we
# don't want to export `OMPT` while `add_llvm_library` requires that.
add_library(OMPT OBJECT
  OmptCallback.cpp
  OmptTracing.cpp)

# This is required when using LLVM libraries.
llvm_update_compile_flags(OMPT)

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

target_link_libraries(OMPT
  PUBLIC
    ${llvm_libs}
)

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(OMPT PRIVATE
  TARGET_NAME="OMPT"
  DEBUG_PREFIX="OMPT"
)

target_include_directories(OMPT
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${LIBOMPTARGET_INCLUDE_DIR}
)

set_target_properties(OMPT PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET protected)
