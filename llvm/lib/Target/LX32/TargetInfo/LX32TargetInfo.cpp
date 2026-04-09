//===-- LX32TargetInfo.cpp - LX32 Target Implementation -------------------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Implements TargetInfo registration for LX32.
//
// This file must remain tiny and deterministic: it only owns the global Target
// singleton and registers lx32 triple/name metadata.
//
// It is organized into the following sections:
//
//   Section 0 — Target singleton accessor
//   Section 1 — LLVMInitializeLX32TargetInfo registration entry point

#include "LX32TargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

llvm::Target &llvm::getTheLX32TargetInfo() {
  static Target TheLX32Target;
  return TheLX32Target;
}

extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeLX32TargetInfo() {
  llvm::RegisterTarget<llvm::Triple::lx32, /*HasJIT=*/false> X(
      llvm::getTheLX32TargetInfo(), "lx32", "32-bit LX32", "LX32");
}