##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
include(TableGen)

libomptarget_say("Building the offload-tblgen tool")

add_tablegen(offload-tblgen OPENMP
  EXPORT OPENMP
  APIGen.cpp
  GenCommon.hpp
  Generators.hpp
  offload-tblgen.cpp
  RecordTypes.hpp
  ValidationGen.cpp
  )
