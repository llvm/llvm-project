//===-- Nios2TargetInfo.cpp - Nios2 Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Nios2.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheNios2Target() {
  static Target TheNios2Target;
  return TheNios2Target;
}

extern "C" void LLVMInitializeNios2TargetInfo() {
  RegisterTarget<Triple::nios2,
                 /*HasJIT=*/true>
      X(getTheNios2Target(), "nios2", "Nios2", "Nios2");
}
