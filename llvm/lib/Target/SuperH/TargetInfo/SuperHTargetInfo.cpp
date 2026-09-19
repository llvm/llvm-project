//===-- SuperHTargetInfo.cpp - SuperH Target Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/SuperHTargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

Target &llvm::getTheSuperHTarget() {
  static Target TheSuperHTarget;
  return TheSuperHTarget;
}

Target &llvm::getTheSuperHLETarget() {
  static Target TheSuperHLETarget;
  return TheSuperHLETarget;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeSuperHTargetInfo() {
  RegisterTarget<Triple::sheb, /*HasJIT=*/false> X(
      getTheSuperHTarget(), "sheb", "SuperH (big endian)", "SuperH");
  RegisterTarget<Triple::shel, /*HasJIT=*/false> Y(
      getTheSuperHLETarget(), "shel", "SuperH (little endian)", "SuperH");
}
