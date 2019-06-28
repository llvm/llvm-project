//===- FunctionInfo.cpp -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/GSYM/FunctionInfo.h"

using namespace llvm;
using namespace gsym;

raw_ostream &llvm::gsym::operator<<(raw_ostream &OS, const FunctionInfo &FI) {
  OS << '[' << HEX64(FI.Range.Start) << '-' << HEX64(FI.Range.End) << "): "
     << "Name=" << HEX32(FI.Name) << '\n';
  for (const auto &Line : FI.Lines)
    OS << Line << '\n';
  OS << FI.Inline;
  return OS;
}
