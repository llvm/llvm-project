//==-- AArch64.h - Top-level interface for AArch64  --------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// AArch64 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AARCH64_AARCH64_H
#define LLVM_LIB_TARGET_AARCH64_AARCH64_H

#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AArch64RegisterBankInfo;
class AArch64Subtarget;
class AArch64TargetMachine;
enum class CodeGenOptLevel;
class FunctionPass;
class InstructionSelector;
class ModulePass;

FunctionPass *createAArch64DeadRegisterDefinitions();
FunctionPass *createAArch64RedundantCopyEliminationPass();
FunctionPass *createAArch64RedundantCondBranchPass();
FunctionPass *createAArch64CondBrTuning();
FunctionPass *createAArch64CompressJumpTablesPass();
FunctionPass *createAArch64ConditionalCompares();
FunctionPass *createAArch64AdvSIMDScalar();
FunctionPass *createAArch64ISelDag(AArch64TargetMachine &TM,
                                   CodeGenOptLevel OptLevel);
FunctionPass *createAArch64StorePairSuppressPass();
FunctionPass *createAArch64ExpandPseudoPass();
FunctionPass *createAArch64SLSHardeningPass();
FunctionPass *createAArch64SpeculationHardeningPass();
FunctionPass *createAArch64LoadStoreOptLegacyPass();
ModulePass *createAArch64LowerHomogeneousPrologEpilogPass();
FunctionPass *createAArch64SIMDInstrOptPass();
ModulePass *createAArch64PromoteConstantPass();
FunctionPass *createAArch64ConditionOptimizerLegacyPass();
FunctionPass *createAArch64A57FPLoadBalancing();
FunctionPass *createAArch64A53Fix835769LegacyPass();
FunctionPass *createFalkorHWPFFixPass();
FunctionPass *createFalkorMarkStridedAccessesPass();
FunctionPass *createAArch64PointerAuthPass();
FunctionPass *createAArch64BranchTargetsPass();
FunctionPass *createAArch64MIPeepholeOptPass();
FunctionPass *createAArch64PostCoalescerPass();

FunctionPass *createAArch64CleanupLocalDynamicTLSPass();

FunctionPass *createAArch64CollectLOHPass();
FunctionPass *createSMEABIPass();
FunctionPass *createSMEPeepholeOptPass();
FunctionPass *createMachineSMEABIPass(CodeGenOptLevel);
FunctionPass *createAArch64SRLTDefineSuperRegsPass();
ModulePass *createSVEIntrinsicOptsPass();
InstructionSelector *
createAArch64InstructionSelector(const AArch64TargetMachine &,
                                 const AArch64Subtarget &,
                                 const AArch64RegisterBankInfo &);
FunctionPass *createAArch64O0PreLegalizerCombiner();
FunctionPass *createAArch64PreLegalizerCombiner();
FunctionPass *createAArch64PostLegalizerCombiner(bool IsOptNone);
FunctionPass *createAArch64PostLegalizerLowering();
FunctionPass *createAArch64PostSelectOptimize();
FunctionPass *createAArch64StackTaggingPass(bool IsOptNone);
FunctionPass *createAArch64StackTaggingPreRAPass();
ModulePass *createAArch64Arm64ECCallLoweringPass();

void initializeAArch64A53Fix835769LegacyPass(PassRegistry &);
void initializeAArch64A57FPLoadBalancingPass(PassRegistry&);
void initializeAArch64AdvSIMDScalarLegacyPass(PassRegistry &);
void initializeAArch64AsmPrinterPass(PassRegistry &);
void initializeAArch64PointerAuthPass(PassRegistry&);
void initializeAArch64BranchTargetsLegacyPass(PassRegistry &);
void initializeAArch64CFIFixupPass(PassRegistry&);
void initializeAArch64CollectLOHLegacyPass(PassRegistry &);
void initializeAArch64CompressJumpTablesLegacyPass(PassRegistry &);
void initializeAArch64CondBrTuningPass(PassRegistry &);
void initializeAArch64ConditionOptimizerLegacyPass(PassRegistry &);
void initializeAArch64ConditionalComparesPass(PassRegistry &);
void initializeAArch64DAGToDAGISelLegacyPass(PassRegistry &);
void initializeAArch64DeadRegisterDefinitionsPass(PassRegistry&);
void initializeAArch64ExpandPseudoPass(PassRegistry &);
void initializeAArch64LoadStoreOptLegacyPass(PassRegistry &);
void initializeAArch64LowerHomogeneousPrologEpilogPass(PassRegistry &);
void initializeAArch64MIPeepholeOptPass(PassRegistry &);
void initializeAArch64O0PreLegalizerCombinerPass(PassRegistry &);
void initializeAArch64PostCoalescerPass(PassRegistry &);
void initializeAArch64PostLegalizerCombinerPass(PassRegistry &);
void initializeAArch64PostLegalizerLoweringPass(PassRegistry &);
void initializeAArch64PostSelectOptimizePass(PassRegistry &);
void initializeAArch64PreLegalizerCombinerPass(PassRegistry &);
void initializeAArch64PromoteConstantPass(PassRegistry&);
void initializeAArch64RedundantCopyEliminationPass(PassRegistry&);
void initializeAArch64RedundantCondBranchPass(PassRegistry &);
void initializeAArch64SIMDInstrOptPass(PassRegistry &);
void initializeAArch64SLSHardeningPass(PassRegistry &);
void initializeAArch64SpeculationHardeningPass(PassRegistry &);
void initializeAArch64StackTaggingPass(PassRegistry &);
void initializeAArch64StackTaggingPreRAPass(PassRegistry &);
void initializeAArch64StorePairSuppressPass(PassRegistry&);
void initializeFalkorHWPFFixPass(PassRegistry&);
void initializeFalkorMarkStridedAccessesLegacyPass(PassRegistry&);
void initializeLDTLSCleanupPass(PassRegistry&);
void initializeSMEABIPass(PassRegistry &);
void initializeSMEPeepholeOptPass(PassRegistry &);
void initializeMachineSMEABIPass(PassRegistry &);
void initializeAArch64SRLTDefineSuperRegsPass(PassRegistry &);
void initializeSVEIntrinsicOptsPass(PassRegistry &);
void initializeAArch64Arm64ECCallLoweringPass(PassRegistry &);

class AArch64LoadStoreOptPass : public PassInfoMixin<AArch64LoadStoreOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64A53Fix835769Pass : public PassInfoMixin<AArch64A53Fix835769Pass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64BranchTargetsPass
    : public PassInfoMixin<AArch64BranchTargetsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64AdvSIMDScalarPass
    : public PassInfoMixin<AArch64AdvSIMDScalarPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64CollectLOHPass : public PassInfoMixin<AArch64CollectLOHPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64CompressJumpTablesPass
    : public PassInfoMixin<AArch64CompressJumpTablesPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64ConditionOptimizerPass
    : public PassInfoMixin<AArch64ConditionOptimizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif
