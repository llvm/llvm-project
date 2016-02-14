//===- llvm/Support/CodeGenCWrappers.h - CodeGen C Wrappers -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines C bindings wrappers for enums in llvm/Support/CodeGen.h
// that need them.  The wrappers are separated to avoid adding an indirect
// dependency on llvm/Config/Targets.def to CodeGen.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CODEGENCWRAPPERS_H
#define LLVM_SUPPORT_CODEGENCWRAPPERS_H

#include "llvm-c/TargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/CodeGenCWrappers.h"

namespace llvm {

inline CodeModel::Model unwrap(LLVMCodeModel Model) {
  switch (Model) {
  case LLVMCodeModelDefault:
    return CodeModel::Default;
  case LLVMCodeModelJITDefault:
    return CodeModel::JITDefault;
  case LLVMCodeModelSmall:
    return CodeModel::Small;
  case LLVMCodeModelKernel:
    return CodeModel::Kernel;
  case LLVMCodeModelMedium:
    return CodeModel::Medium;
  case LLVMCodeModelLarge:
    return CodeModel::Large;
  }
  return CodeModel::Default;
}

inline LLVMCodeModel wrap(CodeModel::Model Model) {
  switch (Model) {
  case CodeModel::Default:
    return LLVMCodeModelDefault;
  case CodeModel::JITDefault:
    return LLVMCodeModelJITDefault;
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

} // end llvm namespace

#endif

