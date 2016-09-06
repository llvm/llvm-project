//===- AMDGPUKernelCodeTUtils.h - helpers for amd_kernel_code_t  *- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file AMDKernelCodeTUtils.h
//===----------------------------------------------------------------------===//

#ifndef AMDKERNELCODETUTILS_H
#define AMDKERNELCODETUTILS_H

#include "AMDKernelCodeT.h"

namespace llvm {

class MCAsmLexer;
class MCAsmParser;
class raw_ostream;
class StringRef;

void printAmdKernelCodeField(const amd_kernel_code_t &C,
  int FldIndex,
  raw_ostream &OS);

void dumpAmdKernelCode(const amd_kernel_code_t *C,
  raw_ostream &OS,
  const char *tab);

bool parseAmdKernelCodeField(StringRef ID,
  MCAsmParser &Parser,
  amd_kernel_code_t &C,
  raw_ostream &Err);

}

#endif // AMDKERNELCODETUTILS_H
