//===- MetalTargetInfo.cpp - Metal Target Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the Metal (Apple AIR) target initializer.
///
//===----------------------------------------------------------------------===//

#include "TargetInfo/MetalTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
Target &getTheMetalTarget() {
  static Target TheMetalTarget;
  return TheMetalTarget;
}
} // namespace llvm

using namespace llvm;

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeMetalTargetInfo() {
  RegisterTarget<Triple::air, /*HasJIT=*/false> X(
      getTheMetalTarget(), "air", "Apple Intermediate Representation (Metal)",
      "Metal");
}
