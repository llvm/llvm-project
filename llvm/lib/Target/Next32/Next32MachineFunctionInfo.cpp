//===-- Next32MachineFunctionInfo.cpp - Next32 machine function info ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Next32MachineFunctionInfo.h"

using namespace llvm;

bool Next32MachineFunctionInfo::hasTopLevelStackFrame() const {
  return HasTopLevelStackFrame;
}

void Next32MachineFunctionInfo::setHasTopLevelStackFrame(bool s) {
  HasTopLevelStackFrame = s;
}
