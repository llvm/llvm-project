//===-- XCore.h - Top-level interface for XCore representation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// XCore back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XCORE_XCORE_H
#define LLVM_LIB_TARGET_XCORE_XCORE_H

#include "MCTargetDesc/XCoreMCTargetDesc.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class ModulePass;
class PassRegistry;
class TargetMachine;
class XCoreTargetMachine;

void initializeXCoreLowerThreadLocalLegacyPass(PassRegistry &p);

FunctionPass *createXCoreFrameToArgsOffsetEliminationPass();

class XCoreISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  XCoreISelDAGToDAGPass(XCoreTargetMachine &TM, CodeGenOptLevel OptLevel);
};

FunctionPass *createXCoreISelDagLegacyPass(XCoreTargetMachine &TM,
                                           CodeGenOptLevel OptLevel);

class XCoreLowerThreadLocalPass
    : public RequiredPassInfoMixin<XCoreLowerThreadLocalPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createXCoreLowerThreadLocalLegacyPass();
void initializeXCoreAsmPrinterPass(PassRegistry &);
void initializeXCoreDAGToDAGISelLegacyPass(PassRegistry &);

} // namespace llvm

#endif
