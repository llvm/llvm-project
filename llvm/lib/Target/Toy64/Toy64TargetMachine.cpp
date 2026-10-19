//===-- Toy64TargetMachine.cpp - Toy64 Target Machine ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"

// using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeToy64Target() {
  // Temporary.
  //
  // The real implementation will eventually contain:
  //
  // RegisterTargetMachine<Toy64TargetMachine> X(getTheToy64Target());
}