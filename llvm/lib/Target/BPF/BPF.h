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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class BPFRegisterBankInfo;
class BPFSubtarget;
class BPFTargetMachine;
class InstructionSelector;
class PassRegistry;

#define BPF_TRAP "__bpf_trap"

class BPFCheckAndAdjustIRPass
    : public RequiredPassInfoMixin<BPFCheckAndAdjustIRPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createBPFCheckAndAdjustIRLegacyPass();

class BPFISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  BPFISelDAGToDAGPass(BPFTargetMachine &TM);
};

FunctionPass *createBPFISelDag(BPFTargetMachine &TM);

class BPFMISimplifyPatchablePass
    : public OptionalPassInfoMixin<BPFMISimplifyPatchablePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createBPFMISimplifyPatchableLegacyPass();

class BPFMIPeepholePass : public OptionalPassInfoMixin<BPFMIPeepholePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createBPFMIPeepholeLegacyPass();

class BPFMIExpandStackArgPseudosPass
    : public RequiredPassInfoMixin<BPFMIExpandStackArgPseudosPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createBPFMIExpandStackArgPseudosLegacyPass();

class BPFMIPreEmitPeepholePass
    : public OptionalPassInfoMixin<BPFMIPreEmitPeepholePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createBPFMIPreEmitPeepholeLegacyPass();

class BPFMIPreEmitCheckingPass
    : public OptionalPassInfoMixin<BPFMIPreEmitCheckingPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createBPFMIPreEmitCheckingLegacyPass();

InstructionSelector *createBPFInstructionSelector(const BPFTargetMachine &,
                                                  const BPFSubtarget &,
                                                  const BPFRegisterBankInfo &);

void initializeBPFAsmPrinterPass(PassRegistry &);
void initializeBPFCheckAndAdjustIRLegacyPass(PassRegistry &);
void initializeBPFDAGToDAGISelLegacyPass(PassRegistry &);
void initializeBPFMIPeepholeLegacyPass(PassRegistry &);
void initializeBPFMIPreEmitCheckingLegacyPass(PassRegistry &);
void initializeBPFMIExpandStackArgPseudosLegacyPass(PassRegistry &);
void initializeBPFMIPreEmitPeepholeLegacyPass(PassRegistry &);
void initializeBPFMISimplifyPatchableLegacyPass(PassRegistry &);

class BPFAbstractMemberAccessPass
    : public RequiredPassInfoMixin<BPFAbstractMemberAccessPass> {
  BPFTargetMachine *TM;

public:
  BPFAbstractMemberAccessPass(BPFTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class BPFPreserveDITypePass
    : public RequiredPassInfoMixin<BPFPreserveDITypePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class BPFIRPeepholePass : public RequiredPassInfoMixin<BPFIRPeepholePass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class BPFASpaceCastSimplifyPass
    : public RequiredPassInfoMixin<BPFASpaceCastSimplifyPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

class BPFAdjustOptPass : public OptionalPassInfoMixin<BPFAdjustOptPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

class BPFPreserveStaticOffsetPass
    : public RequiredPassInfoMixin<BPFPreserveStaticOffsetPass> {
  bool AllowPartial;

public:
  BPFPreserveStaticOffsetPass(bool AllowPartial) : AllowPartial(AllowPartial) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static std::pair<GetElementPtrInst *, LoadInst *>
  reconstructLoad(CallInst *Call);

  static std::pair<GetElementPtrInst *, StoreInst *>
  reconstructStore(CallInst *Call);
};

} // namespace llvm

#endif
