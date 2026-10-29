//===-- RISCV.h - Top-level interface for RISC-V ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// RISC-V back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCV_H
#define LLVM_LIB_TARGET_RISCV_RISCV_H

#include "MCTargetDesc/RISCVBaseInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class FunctionPass;
class InstructionSelector;
class ModulePass;
class PassRegistry;
class RISCVRegisterBankInfo;
class RISCVSubtarget;
class RISCVTargetMachine;

class RISCVCodeGenPreparePass
    : public OptionalPassInfoMixin<RISCVCodeGenPreparePass> {
private:
  const RISCVTargetMachine *TM;

public:
  RISCVCodeGenPreparePass(const RISCVTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
FunctionPass *createRISCVCodeGenPrepareLegacyPass();
void initializeRISCVCodeGenPrepareLegacyPass(PassRegistry &);

class RISCVDeadRegisterDefinitionsPass
    : public OptionalPassInfoMixin<RISCVDeadRegisterDefinitionsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
FunctionPass *createRISCVDeadRegisterDefinitionsLegacyPass();
void initializeRISCVDeadRegisterDefinitionsLegacyPass(PassRegistry &);

class RISCVIndirectBranchTrackingPass
    : public RequiredPassInfoMixin<RISCVIndirectBranchTrackingPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
FunctionPass *createRISCVIndirectBranchTrackingLegacyPass();
void initializeRISCVIndirectBranchTrackingLegacyPass(PassRegistry &);

class RISCVLandingPadSetupPass
    : public RequiredPassInfoMixin<RISCVLandingPadSetupPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
FunctionPass *createRISCVLandingPadSetupLegacyPass();
void initializeRISCVLandingPadSetupLegacyPass(PassRegistry &);

FunctionPass *createRISCVISelDagLegacyPass(RISCVTargetMachine &TM,
                                           CodeGenOptLevel OptLevel);

class RISCVISelDAGToDAGPass : public SelectionDAGISelPass {
public:
  RISCVISelDAGToDAGPass(RISCVTargetMachine &TM, CodeGenOptLevel OptLevel);
};

class RISCVLateBranchOptPass
    : public OptionalPassInfoMixin<RISCVLateBranchOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
FunctionPass *createRISCVLateBranchOptLegacyPass();
void initializeRISCVLateBranchOptLegacyPass(PassRegistry &);

class RISCVMakeCompressiblePass
    : public OptionalPassInfoMixin<RISCVMakeCompressiblePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
FunctionPass *createRISCVMakeCompressibleLegacyPass();
void initializeRISCVMakeCompressibleLegacyPass(PassRegistry &);

class RISCVGatherScatterLoweringPass
    : public OptionalPassInfoMixin<RISCVGatherScatterLoweringPass> {
private:
  const RISCVTargetMachine *TM;

public:
  RISCVGatherScatterLoweringPass(const RISCVTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

FunctionPass *createRISCVGatherScatterLoweringLegacyPass();
void initializeRISCVGatherScatterLoweringLegacyPass(PassRegistry &);

class RISCVVectorPeepholePass
    : public OptionalPassInfoMixin<RISCVVectorPeepholePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

FunctionPass *createRISCVVectorPeepholeLegacyPass();
void initializeRISCVVectorPeepholeLegacyPass(PassRegistry &);

class RISCVOptWInstrsPass : public OptionalPassInfoMixin<RISCVOptWInstrsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVOptWInstrsLegacyPass();
void initializeRISCVOptWInstrsLegacyPass(PassRegistry &);

class RISCVFoldMemOffsetPass
    : public OptionalPassInfoMixin<RISCVFoldMemOffsetPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVFoldMemOffsetLegacyPass();
void initializeRISCVFoldMemOffsetLegacyPass(PassRegistry &);

class RISCVMergeBaseOffsetPass
    : public OptionalPassInfoMixin<RISCVMergeBaseOffsetPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVMergeBaseOffsetLegacyPass();
void initializeRISCVMergeBaseOffsetLegacyPass(PassRegistry &);

class RISCVExpandPseudoPreRAPass
    : public RequiredPassInfoMixin<RISCVExpandPseudoPreRAPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

FunctionPass *createRISCVExpandPseudoPreRALegacyPass();
void initializeRISCVExpandPseudoPreRALegacyPass(PassRegistry &);

class RISCVExpandPseudoPostRAPass
    : public RequiredPassInfoMixin<RISCVExpandPseudoPostRAPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVExpandPseudoPostRALegacyPass();
void initializeRISCVExpandPseudoPostRALegacyPass(PassRegistry &);

class RISCVExpandPseudoPreEmitPass
    : public RequiredPassInfoMixin<RISCVExpandPseudoPreEmitPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVExpandPseudoPreEmitLegacyPass();
void initializeRISCVExpandPseudoPreEmitLegacyPass(PassRegistry &);

class RISCVExpandPseudoAtomicsPass
    : public RequiredPassInfoMixin<RISCVExpandPseudoAtomicsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVExpandPseudoAtomicsLegacyPass();
void initializeRISCVExpandPseudoAtomicsLegacyPass(PassRegistry &);

class RISCVInsertVSETVLIPass
    : public RequiredPassInfoMixin<RISCVInsertVSETVLIPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVInsertVSETVLILegacyPass();
void initializeRISCVInsertVSETVLILegacyPass(PassRegistry &);
extern char &RISCVInsertVSETVLILegacyID;

class RISCVInsertReadWriteCSRPass
    : public RequiredPassInfoMixin<RISCVInsertReadWriteCSRPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVInsertReadWriteCSRLegacyPass();
void initializeRISCVInsertReadWriteCSRLegacyPass(PassRegistry &);

class RISCVInsertWriteVXRMPass
    : public RequiredPassInfoMixin<RISCVInsertWriteVXRMPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVInsertWriteVXRMLegacyPass();
void initializeRISCVInsertWriteVXRMLegacyPass(PassRegistry &);

class RISCVRedundantCopyEliminationPass
    : public OptionalPassInfoMixin<RISCVRedundantCopyEliminationPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setNoVRegs();
  }
};

FunctionPass *createRISCVRedundantCopyEliminationLegacyPass();
void initializeRISCVRedundantCopyEliminationLegacyPass(PassRegistry &);

class RISCVMoveMergePass : public OptionalPassInfoMixin<RISCVMoveMergePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVMoveMergeLegacyPass();
void initializeRISCVMoveMergeLegacyPass(PassRegistry &);

class RISCVPushPopOptPass : public OptionalPassInfoMixin<RISCVPushPopOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVPushPopOptLegacyPass();
void initializeRISCVPushPopOptLegacyPass(PassRegistry &);

class RISCVLoadStoreOptPass
    : public OptionalPassInfoMixin<RISCVLoadStoreOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVLoadStoreOptLegacyPass();
void initializeRISCVLoadStoreOptLegacyPass(PassRegistry &);

class RISCVPreAllocZilsdOptPass
    : public OptionalPassInfoMixin<RISCVPreAllocZilsdOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVPreAllocZilsdOptLegacyPass();
void initializeRISCVPreAllocZilsdOptLegacyPass(PassRegistry &);

class RISCVZacasABIFixPass
    : public RequiredPassInfoMixin<RISCVZacasABIFixPass> {
private:
  const RISCVTargetMachine *TM;

public:
  RISCVZacasABIFixPass(const RISCVTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};
FunctionPass *createRISCVZacasABIFixLegacyPass();
void initializeRISCVZacasABIFixLegacyPass(PassRegistry &);

InstructionSelector *
createRISCVInstructionSelector(const RISCVTargetMachine &,
                               const RISCVSubtarget &,
                               const RISCVRegisterBankInfo &);
void initializeRISCVDAGToDAGISelLegacyPass(PassRegistry &);

FunctionPass *createRISCVPostLegalizerCombiner();
void initializeRISCVPostLegalizerCombinerPass(PassRegistry &);

FunctionPass *createRISCVO0PreLegalizerCombiner();
void initializeRISCVO0PreLegalizerCombinerPass(PassRegistry &);

FunctionPass *createRISCVPreLegalizerCombiner();
void initializeRISCVPreLegalizerCombinerPass(PassRegistry &);

class RISCVPromoteConstantPass
    : public OptionalPassInfoMixin<RISCVPromoteConstantPass> {
private:
  const RISCVTargetMachine *TM;

public:
  RISCVPromoteConstantPass(const RISCVTargetMachine *TM) : TM(TM) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

ModulePass *createRISCVPromoteConstantLegacyPass();
void initializeRISCVPromoteConstantLegacyPass(PassRegistry &);

class RISCVVLOptimizerPass
    : public OptionalPassInfoMixin<RISCVVLOptimizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVVLOptimizerLegacyPass();
void initializeRISCVVLOptimizerLegacyPass(PassRegistry &);

class RISCVVMV0EliminationPass
    : public RequiredPassInfoMixin<RISCVVMV0EliminationPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().setIsSSA();
  }
};

FunctionPass *createRISCVVMV0EliminationLegacyPass();
void initializeRISCVVMV0EliminationLegacyPass(PassRegistry &);

class RISCVQCRelaxMarkingPass
    : public OptionalPassInfoMixin<RISCVQCRelaxMarkingPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

FunctionPass *createRISCVQCRelaxMarkingLegacyPass();
void initializeRISCVQCRelaxMarkingLegacyPass(PassRegistry &);

void initializeRISCVAsmPrinterPass(PassRegistry &);
} // namespace llvm

#endif
