//===--------------- RISCVPostLegalizerLowering.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Post-legalization lowering for instructions.
///
/// This is used to offload pattern matching from the selector.
///
/// General optimization combines should be handled by either the
/// RISCVPostLegalizerCombiner or the RISCVPreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "RISCVSubtarget.h"

#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define GET_GICOMBINER_DEPS
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "riscv-postlegalizer-lowering"

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_TYPES

class RISCVPostLegalizerLoweringImpl : public Combiner {
protected:
  // TODO: Make CombinerHelper methods const.
  mutable CombinerHelper Helper;
  const RISCVPostLegalizerLoweringImplRuleConfig &RuleConfig;
  const RISCVSubtarget &STI;

public:
  RISCVPostLegalizerLoweringImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelCSEInfo *CSEInfo,
      const RISCVPostLegalizerLoweringImplRuleConfig &RuleConfig,
      const RISCVSubtarget &STI);

  static const char *getName() { return "RISCVPreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_IMPL

RISCVPostLegalizerLoweringImpl::RISCVPostLegalizerLoweringImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelCSEInfo *CSEInfo,
    const RISCVPostLegalizerLoweringImplRuleConfig &RuleConfig,
    const RISCVSubtarget &STI)
    : Combiner(MF, CInfo, TPC, /*KB*/ nullptr, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true), RuleConfig(RuleConfig),
      STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

class RISCVPostLegalizerLowering : public MachineFunctionPass {
public:
  static char ID;

  RISCVPostLegalizerLowering();

  StringRef getPassName() const override {
    return "RISCVPostLegalizerLowering";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  RISCVPostLegalizerLoweringImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void RISCVPostLegalizerLowering::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

RISCVPostLegalizerLowering::RISCVPostLegalizerLowering()
    : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool RISCVPostLegalizerLowering::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Legalized) &&
         "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*OptEnabled=*/true,
                     F.hasOptSize(), F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // PostLegalizerCombiner performs DCE, so a full DCE pass is unnecessary.
  CInfo.EnableFullDCE = false;
  RISCVPostLegalizerLoweringImpl Impl(MF, CInfo, TPC, /*CSEInfo*/ nullptr,
                                      RuleConfig, ST);
  return Impl.combineMachineInstrs();
}

char RISCVPostLegalizerLowering::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVPostLegalizerLowering, DEBUG_TYPE,
                      "Lower RISC-V MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVPostLegalizerLowering, DEBUG_TYPE,
                    "Lower RISC-V MachineInstrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createRISCVPostLegalizerLowering() {
  return new RISCVPostLegalizerLowering();
}
} // end namespace llvm
