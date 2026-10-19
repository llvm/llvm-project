//===-- Toy64MCTargetDesc.cpp - Toy64 Target Descriptions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Compiler.h"

// using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeToy64TargetMC() {
  // Temporary.
  //
  // Later this function will register Toy64's:
  //   MCInstrInfo
  //   MCRegisterInfo
  //   MCSubtargetInfo
  //   MCAsmInfo
  //   MCCodeEmitter
  //   MCAsmBackend
}