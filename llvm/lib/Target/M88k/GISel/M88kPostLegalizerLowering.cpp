//=== M88kPostLegalizerLowering.cpp -----------------------------*- C++ -*-===//
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
/// For example, this combiner will notice that a G_OR with a shifted mask as
/// argumnet is actually MAKrwo.
///
/// General optimization combines should be handled by either the
/// M88kPostLegalizerCombiner or the M88kPreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "GISel/M88kGlobalISelUtils.h"
#include "GISel/M88kLegalizerInfo.h"
#include "M88kTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/CombinerInfo.h"
#include "llvm/CodeGen/GlobalISel/MIPatternMatch.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "M88K-postlegalizer-lowering"

using namespace llvm;
using namespace M88kGISelUtils;
using namespace MIPatternMatch;

// Replace the generic instruction with a m88k instruction, which has a
// width/offset field.
static void replaceMI(unsigned Opc, MachineInstr &MI, MachineRegisterInfo &MRI,
                      std::tuple<Register, uint32_t, uint32_t> &MatchInfo) {
  uint64_t Offset, Width;
  Register SrcReg;
  std::tie(SrcReg, Width, Offset) = MatchInfo;
  MachineIRBuilder MIB(MI);
  MachineFunction &MF = MIB.getMF();
  const M88kSubtarget &Subtarget = MF.getSubtarget<M88kSubtarget>();
  const auto *TRI = Subtarget.getRegisterInfo();
  const auto *TII = Subtarget.getInstrInfo();
  const auto *RBI = Subtarget.getRegBankInfo();
  auto Inst = MIB.buildInstr(Opc, {MI.getOperand(0).getReg()}, {SrcReg})
                  .addImm((Width << 5) | Offset);
  constrainSelectedInstRegOperands(*Inst, *TII, *TRI, *RBI);
  MI.eraseFromParent();
}

// Match G_SHL $dst, (G_AND $src, (2**width - 1)), offset
static bool
matchShiftAndToMak(MachineInstr &MI, MachineRegisterInfo &MRI,
                   std::tuple<Register, uint32_t, uint32_t> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHL);

  Register DstReg = MI.getOperand(0).getReg();
  if (!MRI.getType(DstReg).isScalar())
    return false;

  Register AndReg = MI.getOperand(1).getReg();
  Register OfsReg = MI.getOperand(2).getReg();
  int64_t Offset;
  if (!mi_match(OfsReg, MRI, m_ICst(Offset)))
    return false;

  Register SrcReg;
  int64_t Mask;
  if (!mi_match(AndReg, MRI, m_GAnd(m_Reg(SrcReg), m_ICst(Mask))))
    return false;

  // Check that the mask is a shifted mask with offset 0.
  uint64_t MaskWidth, MaskOffset;
  if (!isShiftedMask(Mask, MaskWidth, MaskOffset) || MaskOffset != 0)
    return false;

  assert(MaskWidth >= 0 && MaskWidth < 32 && "Width out of range");
  assert(Offset >= 0 && Offset < 32 && "Offset out of range");

  MatchInfo = std::make_tuple(SrcReg, static_cast<uint32_t>(MaskWidth),
                              static_cast<uint32_t>(Offset));

  return true;
}

// Lower to MAKrwo $dst, $src, width<offset>
bool applyShiftAndToMak(MachineInstr &MI, MachineRegisterInfo &MRI,
                        std::tuple<Register, uint32_t, uint32_t> &MatchInfo) {
  assert(MI.getOpcode() == TargetOpcode::G_SHL);
  replaceMI(M88k::MAKrwo, MI, MRI, MatchInfo);
  return true;
}

#define M88KPOSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_DEPS
#include "M88kGenPostLegalizeGILowering.inc"
#undef M88KPOSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_DEPS

namespace {
#define M88KPOSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_H
#include "M88kGenPostLegalizeGILowering.inc"
#undef M88KPOSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_H

class M88kPostLegalizerLoweringInfo : public CombinerInfo {
public:
  M88kGenPostLegalizerLoweringHelperRuleConfig GeneratedRuleCfg;

  M88kPostLegalizerLoweringInfo(bool OptSize, bool MinSize)
      : CombinerInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*OptEnabled = */ true, OptSize,
                     MinSize) {
    if (!GeneratedRuleCfg.parseCommandLineOption())
      report_fatal_error("Invalid rule identifier");
  }

  virtual bool combine(GISelChangeObserver &Observer, MachineInstr &MI,
                       MachineIRBuilder &B) const override;
};

bool M88kPostLegalizerLoweringInfo::combine(GISelChangeObserver &Observer,
                                            MachineInstr &MI,
                                            MachineIRBuilder &B) const {
  CombinerHelper Helper(Observer, B);
  M88kGenPostLegalizerLoweringHelper Generated(GeneratedRuleCfg);
  return Generated.tryCombineAll(Observer, MI, B, Helper);
}

#define M88KPOSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_CPP
#include "M88kGenPostLegalizeGILowering.inc"
#undef M88KPOSTLEGALIZERLOWERINGHELPER_GENCOMBINERHELPER_CPP

class M88kPostLegalizerLowering : public MachineFunctionPass {
public:
  static char ID;

  M88kPostLegalizerLowering();

  StringRef getPassName() const override { return "M88kPostLegalizerLowering"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};
} // end anonymous namespace

void M88kPostLegalizerLowering::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

// TODO This should not be needed here.
namespace llvm {
void initializeM88kPostLegalizerLoweringPass(PassRegistry &Registry);
}

M88kPostLegalizerLowering::M88kPostLegalizerLowering()
    : MachineFunctionPass(ID) {
  initializeM88kPostLegalizerLoweringPass(*PassRegistry::getPassRegistry());
}

bool M88kPostLegalizerLowering::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Legalized) &&
         "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();
  M88kPostLegalizerLoweringInfo PCInfo(F.hasOptSize(), F.hasMinSize());
  Combiner C(PCInfo, TPC);
  return C.combineMachineInstrs(MF, /*CSEInfo*/ nullptr);
}

char M88kPostLegalizerLowering::ID = 0;
INITIALIZE_PASS_BEGIN(M88kPostLegalizerLowering, DEBUG_TYPE,
                      "Lower M88k MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(M88kPostLegalizerLowering, DEBUG_TYPE,
                    "Lower M88k MachineInstrs after legalization", false, false)

namespace llvm {
FunctionPass *createM88kPostLegalizerLowering() {
  return new M88kPostLegalizerLowering();
}
} // end namespace llvm
