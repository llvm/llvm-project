//===- GsymDataExtractor.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/GsymDataExtractor.h"
#include "llvm/DebugInfo/GSYM/GsymReader.h"

using namespace llvm;
using namespace gsym;

GsymDataExtractor::GsymDataExtractor(const DataExtractor &DE,
                                     const GsymReader *GR)
    : DataExtractor(DE), StrpSize(GR ? GR->getStringOffsetByteSize() : 4) {}
