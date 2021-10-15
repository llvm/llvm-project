//===-- BPF.h - Top-level interface for BPF representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPF_H
#define LLVM_LIB_TARGET_BPF_BPF_H

#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class BPFTargetMachine;

ModulePass *createBPFAdjustOpt();
ModulePass *createBPFCheckAndAdjustIR();

FunctionPass *createBPFAbstractMemberAccess(BPFTargetMachine *TM);
FunctionPass *createBPFPreserveDIType();
FunctionPass *createBPFIRPeephole();
FunctionPass *createBPFISelDag(BPFTargetMachine &TM);
FunctionPass *createBPFMISimplifyPatchablePass();
FunctionPass *createBPFMIPeepholePass();
FunctionPass *createBPFMIPeepholeTruncElimPass();
FunctionPass *createBPFMIPreEmitPeepholePass();
FunctionPass *createBPFMIPreEmitCheckingPass();

void initializeBPFAdjustOptPass(PassRegistry&);
void initializeBPFCheckAndAdjustIRPass(PassRegistry&);

void initializeBPFAbstractMemberAccessLegacyPassPass(PassRegistry &);
void initializeBPFPreserveDITypePass(PassRegistry&);
void initializeBPFIRPeepholePass(PassRegistry&);
void initializeBPFMISimplifyPatchablePass(PassRegistry&);
void initializeBPFMIPeepholePass(PassRegistry&);
void initializeBPFMIPeepholeTruncElimPass(PassRegistry&);
void initializeBPFMIPreEmitPeepholePass(PassRegistry&);
void initializeBPFMIPreEmitCheckingPass(PassRegistry&);

class BPFAbstractMemberAccessPass
    : public PassInfoMixin<BPFAbstractMemberAccessPass> {
  BPFTargetMachine *TM;

public:
  BPFAbstractMemberAccessPass(BPFTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class BPFPreserveDITypePass : public PassInfoMixin<BPFPreserveDITypePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class BPFIRPeepholePass : public PassInfoMixin<BPFIRPeepholePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class BPFAdjustOptPass : public PassInfoMixin<BPFAdjustOptPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};
} // namespace llvm

#endif
