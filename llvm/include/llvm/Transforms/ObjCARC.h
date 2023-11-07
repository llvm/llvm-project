//===-- ObjCARC.h - ObjCARC Scalar Transformations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the ObjCARC Scalar Transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_OBJCARC_H
#define LLVM_TRANSFORMS_OBJCARC_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class Pass;

//===----------------------------------------------------------------------===//
//
// ObjCARCContract - Late ObjC ARC cleanups.
//
LLVM_FUNC_ABI Pass *createObjCARCContractPass();

struct LLVM_CLASS_ABI ObjCARCOptPass : public PassInfoMixin<ObjCARCOptPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct LLVM_CLASS_ABI ObjCARCContractPass : public PassInfoMixin<ObjCARCContractPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

struct LLVM_CLASS_ABI ObjCARCAPElimPass : public PassInfoMixin<ObjCARCAPElimPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

struct LLVM_CLASS_ABI ObjCARCExpandPass : public PassInfoMixin<ObjCARCExpandPass> {
  PreservedAnalyses run(Function &M, FunctionAnalysisManager &AM);
};

struct LLVM_CLASS_ABI PAEvalPass : public PassInfoMixin<PAEvalPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // End llvm namespace

#endif
