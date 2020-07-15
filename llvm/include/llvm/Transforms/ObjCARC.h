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

namespace llvm {

class Pass;

//===----------------------------------------------------------------------===//
//
// ObjCARCAPElim - ObjC ARC autorelease pool elimination.
//
Pass *createObjCARCAPElimPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCExpand - ObjC ARC preliminary simplifications.
//
Pass *createObjCARCExpandPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCContract - Late ObjC ARC cleanups.
//
Pass *createObjCARCContractPass();

//===----------------------------------------------------------------------===//
//
// ObjCARCOpt - ObjC ARC optimization.
//
Pass *createObjCARCOptPass();

class ObjCARCOptPass : public PassInfoMixin<ObjCARCOptPass> {
public:
  ObjCARCOptPass() {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // End llvm namespace

#endif
