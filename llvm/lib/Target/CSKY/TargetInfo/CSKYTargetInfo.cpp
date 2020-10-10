//===-- CSKYTargetInfo.cpp - CSKY Target Implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/CSKYTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheCSKYTarget() {
  static Target TheCSKYTarget;
  return TheCSKYTarget;
}

extern "C" void LLVMInitializeCSKYTargetInfo() {
  RegisterTarget<Triple::csky> X(getTheCSKYTarget(), "csky", "C-SKY", "CSKY");
}

// FIXME: Temporary stub - this function must be defined for linking
// to succeed and will be called unconditionally by llc, so must be a no-op.
// Remove once this function is properly implemented.
extern "C" void LLVMInitializeCSKYTargetMC() {}
