//===-- Toy64TargetInfo.cpp - Toy64 Target Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Toy64TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

Target &llvm::getTheToy64Target() {
  static Target TheToy64Target;
  return TheToy64Target;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeToy64TargetInfo() {
  RegisterTarget<Triple::toy64> X(getTheToy64Target(), "toy64", "Toy64",
                                  "Toy64");
}
