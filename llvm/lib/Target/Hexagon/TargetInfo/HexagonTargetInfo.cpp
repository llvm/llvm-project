//===-- HexagonTargetInfo.cpp - Hexagon Target Implementation ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheHexagonTarget() {
  static Target TheHexagonTarget;
  return TheHexagonTarget;
}

extern "C" void LLVMInitializeHexagonTargetInfo() {
  RegisterTarget<Triple::hexagon, /*HasJIT=*/true> X(
      getTheHexagonTarget(), "hexagon", "Hexagon", "Hexagon");
}
