//===- llvm/Support/Win32/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 Host support.
//
//===----------------------------------------------------------------------===//

#include "Windows.h"
#include <cstdio>
#include <string>

using namespace llvm;

std::string sys::getDefaultTargetTriple() {
  return LLVM_DEFAULT_TARGET_TRIPLE;
}
