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
  EntryPointGen.cpp
  FuncsGen.cpp
  GenCommon.hpp
  Generators.hpp
  offload-tblgen.cpp
  PrintGen.cpp
  RecordTypes.hpp
  )

set(OFFLOAD_TABLEGEN_EXE "${OFFLOAD_TABLEGEN_EXE}" CACHE INTERNAL "")
set(OFFLOAD_TABLEGEN_TARGET "${OFFLOAD_TABLEGEN_TARGET}" CACHE INTERNAL "")

