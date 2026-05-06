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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Target/TargetMachine.h"
#include <map>
#include <memory>
#include <unordered_map>

struct AArch64O0PreLegalizerCombinerImplRuleConfig;
struct AArch64PreLegalizerCombinerImplRuleConfig;
struct AArch64PostLegalizerLoweringImplRuleConfig;

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
FunctionPass *createAArch64ExpandPseudoLegacyPass();
FunctionPass *createAArch64SLSHardeningPass();
FunctionPass *createAArch64SpeculationHardeningPass();
FunctionPass *createAArch64LoadStoreOptLegacyPass();
ModulePass *createAArch64LowerHomogeneousPrologEpilogPass();
FunctionPass *createAArch64SIMDInstrOptPass();
ModulePass *createAArch64PromoteConstantPass();
FunctionPass *createAArch64ConditionOptimizerLegacyPass();
FunctionPass *createAArch64A57FPLoadBalancingLegacyPass();
FunctionPass *createAArch64A53Fix835769LegacyPass();
FunctionPass *createFalkorHWPFFixPass();
FunctionPass *createFalkorMarkStridedAccessesPass();
FunctionPass *createAArch64PointerAuthPass();
FunctionPass *createAArch64BranchTargetsPass();
FunctionPass *createAArch64MIPeepholeOptLegacyPass();
FunctionPass *createAArch64PostCoalescerPass();

FunctionPass *createAArch64CleanupLocalDynamicTLSPass();

FunctionPass *createAArch64CollectLOHPass();
FunctionPass *createSMEPeepholeOptPass();
FunctionPass *createMachineSMEABIPass(CodeGenOptLevel);
FunctionPass *createAArch64SRLTDefineSuperRegsPass();
ModulePass *createSVEIntrinsicOptsPass();
InstructionSelector *
createAArch64InstructionSelector(const AArch64TargetMachine &,
                                 const AArch64Subtarget &,
                                 const AArch64RegisterBankInfo &);
class AArch64O0PreLegalizerCombinerPass
    : public OptionalPassInfoMixin<AArch64O0PreLegalizerCombinerPass> {
  std::unique_ptr<AArch64O0PreLegalizerCombinerImplRuleConfig> RuleConfig;

public:
  AArch64O0PreLegalizerCombinerPass();
  AArch64O0PreLegalizerCombinerPass(AArch64O0PreLegalizerCombinerPass &&);
  ~AArch64O0PreLegalizerCombinerPass();

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64PreLegalizerCombinerPass
    : public OptionalPassInfoMixin<AArch64PreLegalizerCombinerPass> {
  std::unique_ptr<AArch64PreLegalizerCombinerImplRuleConfig> RuleConfig;

public:
  AArch64PreLegalizerCombinerPass();
  AArch64PreLegalizerCombinerPass(AArch64PreLegalizerCombinerPass &&);
  ~AArch64PreLegalizerCombinerPass();

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64PostSelectOptimizePass
    : public OptionalPassInfoMixin<AArch64PostSelectOptimizePass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64PostLegalizerLoweringPass
    : public OptionalPassInfoMixin<AArch64PostLegalizerLoweringPass> {
  std::unique_ptr<AArch64PostLegalizerLoweringImplRuleConfig> RuleConfig;

public:
  AArch64PostLegalizerLoweringPass();
  AArch64PostLegalizerLoweringPass(AArch64PostLegalizerLoweringPass &&);
  ~AArch64PostLegalizerLoweringPass();

  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);

  MachineFunctionProperties getRequiredProperties() const {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::Legalized);
  }
};

FunctionPass *createAArch64O0PreLegalizerCombiner();
FunctionPass *createAArch64PreLegalizerCombiner();
FunctionPass *createAArch64PostLegalizerCombiner(bool IsOptNone);
FunctionPass *createAArch64PostLegalizerLowering();
FunctionPass *createAArch64PostSelectOptimize();
FunctionPass *createAArch64StackTaggingPass(bool IsOptNone);
FunctionPass *createAArch64StackTaggingPreRALegacyPass();
ModulePass *createAArch64Arm64ECCallLoweringPass();

void initializeAArch64A53Fix835769LegacyPass(PassRegistry &);
void initializeAArch64A57FPLoadBalancingLegacyPass(PassRegistry &);
void initializeAArch64AdvSIMDScalarLegacyPass(PassRegistry &);
void initializeAArch64AsmPrinterPass(PassRegistry &);
void initializeAArch64PointerAuthLegacyPass(PassRegistry &);
void initializeAArch64BranchTargetsLegacyPass(PassRegistry &);
void initializeAArch64CFIFixupPass(PassRegistry&);
void initializeAArch64CollectLOHLegacyPass(PassRegistry &);
void initializeAArch64CompressJumpTablesLegacyPass(PassRegistry &);
void initializeAArch64CondBrTuningPass(PassRegistry &);
void initializeAArch64ConditionOptimizerLegacyPass(PassRegistry &);
void initializeAArch64ConditionalComparesLegacyPass(PassRegistry &);
void initializeAArch64DAGToDAGISelLegacyPass(PassRegistry &);
void initializeAArch64DeadRegisterDefinitionsLegacyPass(PassRegistry &);
void initializeAArch64ExpandPseudoLegacyPass(PassRegistry &);
void initializeAArch64LoadStoreOptLegacyPass(PassRegistry &);
void initializeAArch64LowerHomogeneousPrologEpilogPass(PassRegistry &);
void initializeAArch64MIPeepholeOptLegacyPass(PassRegistry &);
void initializeAArch64O0PreLegalizerCombinerLegacyPass(PassRegistry &);
void initializeAArch64PostCoalescerLegacyPass(PassRegistry &);
void initializeAArch64PostLegalizerCombinerPass(PassRegistry &);
void initializeAArch64PostSelectOptimizeLegacyPass(PassRegistry &);
void initializeAArch64PostLegalizerLoweringLegacyPass(PassRegistry &);
void initializeAArch64PreLegalizerCombinerLegacyPass(PassRegistry &);
void initializeAArch64PromoteConstantPass(PassRegistry&);
void initializeAArch64RedundantCopyEliminationLegacyPass(PassRegistry &);
void initializeAArch64RedundantCondBranchLegacyPass(PassRegistry &);
void initializeAArch64SIMDInstrOptLegacyPass(PassRegistry &);
void initializeAArch64SLSHardeningPass(PassRegistry &);
void initializeAArch64SpeculationHardeningPass(PassRegistry &);
void initializeAArch64StackTaggingPass(PassRegistry &);
void initializeAArch64StackTaggingPreRALegacyPass(PassRegistry &);
void initializeAArch64StorePairSuppressPass(PassRegistry&);
void initializeFalkorHWPFFixPass(PassRegistry&);
void initializeFalkorMarkStridedAccessesLegacyPass(PassRegistry&);
void initializeLDTLSCleanupPass(PassRegistry &);
void initializeSMEPeepholeOptPass(PassRegistry &);
void initializeMachineSMEABIPass(PassRegistry &);
void initializeAArch64SRLTDefineSuperRegsPass(PassRegistry &);
void initializeSVEIntrinsicOptsPass(PassRegistry &);
void initializeAArch64Arm64ECCallLoweringPass(PassRegistry &);

class AArch64StackTaggingPreRAPass
    : public OptionalPassInfoMixin<AArch64StackTaggingPreRAPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64A57FPLoadBalancingPass
    : public OptionalPassInfoMixin<AArch64A57FPLoadBalancingPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64LoadStoreOptPass
    : public OptionalPassInfoMixin<AArch64LoadStoreOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64A53Fix835769Pass
    : public OptionalPassInfoMixin<AArch64A53Fix835769Pass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64BranchTargetsPass
    : public OptionalPassInfoMixin<AArch64BranchTargetsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64RedundantCondBranchPass
    : public OptionalPassInfoMixin<AArch64RedundantCondBranchPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64AdvSIMDScalarPass
    : public OptionalPassInfoMixin<AArch64AdvSIMDScalarPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64CollectLOHPass
    : public OptionalPassInfoMixin<AArch64CollectLOHPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64CompressJumpTablesPass
    : public OptionalPassInfoMixin<AArch64CompressJumpTablesPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64DeadRegisterDefinitionsPass
    : public OptionalPassInfoMixin<AArch64DeadRegisterDefinitionsPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64ExpandPseudoPass
    : public OptionalPassInfoMixin<AArch64ExpandPseudoPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64MIPeepholeOptPass
    : public OptionalPassInfoMixin<AArch64MIPeepholeOptPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64ConditionOptimizerPass
    : public OptionalPassInfoMixin<AArch64ConditionOptimizerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64SIMDInstrOptPass
    : public OptionalPassInfoMixin<AArch64SIMDInstrOptPass> {
  std::map<std::pair<unsigned, std::string>, bool> SIMDInstrTable;
  std::unordered_map<std::string, bool> InterlEarlyExit;

public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64PointerAuthPass
    : public OptionalPassInfoMixin<AArch64PointerAuthPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64PostCoalescerPass
    : public OptionalPassInfoMixin<AArch64PostCoalescerPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64RedundantCopyEliminationPass
    : public OptionalPassInfoMixin<AArch64RedundantCopyEliminationPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

class AArch64ConditionalComparesPass
    : public OptionalPassInfoMixin<AArch64ConditionalComparesPass> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};

} // end namespace llvm

#endif
