//===-- M88kTargetInfo.cpp - M88k target implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/M88kTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"

using namespace llvm;

Target &llvm::getTheM88kTarget() {
  static Target TheM88kTarget;
  return TheM88kTarget;
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeM88kTargetInfo() {
  RegisterTarget<Triple::m88k, /*HasJIT=*/false> X(getTheM88kTarget(), "m88k",
                                                   "M88k", "M88k");
}
