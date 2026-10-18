//===-- AGXTargetInfo.cpp - AGX target implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/AGXTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

Target &llvm::getTheAGXTarget() {
  static Target TheAGXTarget;
  return TheAGXTarget;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void LLVMInitializeAGXTargetInfo() {
  RegisterTarget<Triple::agx, /*HasJIT=*/false> X(
      getTheAGXTarget(), "agx", "Apple AGX GPUs", "AGX");
}
