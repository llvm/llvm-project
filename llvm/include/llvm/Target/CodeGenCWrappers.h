//===- llvm/Target/CodeGenCWrappers.h - CodeGen C Wrappers ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings wrappers for enums in llvm/Support/CodeGen.h
// that need them.  The wrappers are separated to avoid adding an indirect
// dependency on llvm/Config/Targets.def to CodeGen.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_CODEGENCWRAPPERS_H
#define LLVM_TARGET_CODEGENCWRAPPERS_H

#include "llvm-c/TargetMachine.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

inline Optional<CodeModel::Model> unwrap(LLVMCodeModel Model, bool &JIT) {
  JIT = false;
  switch (Model) {
  case LLVMCodeModelJITDefault:
    JIT = true;
    LLVM_FALLTHROUGH;
  case LLVMCodeModelDefault:
    return None;
  case LLVMCodeModelTiny:
    return CodeModel::Tiny;
  case LLVMCodeModelSmall:
    return CodeModel::Small;
  case LLVMCodeModelKernel:
    return CodeModel::Kernel;
  case LLVMCodeModelMedium:
    return CodeModel::Medium;
  case LLVMCodeModelLarge:
    return CodeModel::Large;
  }
  return CodeModel::Small;
}

inline LLVMCodeModel wrap(CodeModel::Model Model) {
  switch (Model) {
  case CodeModel::Tiny:
    return LLVMCodeModelTiny;
  case CodeModel::Small:
    return LLVMCodeModelSmall;
  case CodeModel::Kernel:
    return LLVMCodeModelKernel;
  case CodeModel::Medium:
    return LLVMCodeModelMedium;
  case CodeModel::Large:
    return LLVMCodeModelLarge;
  }
  llvm_unreachable("Bad CodeModel!");
}
} // namespace llvm

#endif
