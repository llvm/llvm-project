//===- AIRTargetInfo.cpp - AIR Target Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the AIR (Apple Intermediate Representation) target initializer.
///
//===----------------------------------------------------------------------===//

#include "TargetInfo/AIRTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
Target &getTheAIRTarget() {
  static Target TheAIRTarget;
  return TheAIRTarget;
}
} // namespace llvm

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeAIRTargetInfo() {
  RegisterTarget<Triple::air, /*HasJIT=*/false> X(
      getTheAIRTarget(), "air", "Apple Intermediate Representation (Metal)",
      "AIR");
}
