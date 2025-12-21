##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
include(TableGen)

set(LLVM_LINK_COMPONENTS Support)

add_tablegen(offload-tblgen OFFLOAD
  EXPORT OFFLOAD
  APIGen.cpp
  DocGen.cpp
  EntryPointGen.cpp
  MiscGen.cpp
  GenCommon.hpp
  Generators.hpp
  offload-tblgen.cpp
  PrintGen.cpp
  RecordTypes.hpp
  )

# Make sure that C++ headers are available, if libcxx is built at the same
# time. This is important if clang is set to prefer libc++ over libstdc++
if(TARGET cxx-headers)
  add_dependencies(offload-tblgen cxx-headers)
endif()

set(OFFLOAD_TABLEGEN_EXE "${OFFLOAD_TABLEGEN_EXE}" CACHE INTERNAL "")
set(OFFLOAD_TABLEGEN_TARGET "${OFFLOAD_TABLEGEN_TARGET}" CACHE INTERNAL "")
