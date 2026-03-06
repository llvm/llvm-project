//===- MachineUniformityAnalysis.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

template <>
bool llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::hasDivergentDefs(
    const MachineInstr &I) const {
  for (auto &op : I.all_defs()) {
    if (isDivergent(op.getReg()))
      return true;
  }
  return false;
}

template <>
bool llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::markDefsDivergent(
    const MachineInstr &Instr) {
  bool insertedDivergent = false;
  const auto &MRI = F.getRegInfo();
  const auto &RBI = *F.getSubtarget().getRegBankInfo();
  const auto &TRI = *MRI.getTargetRegisterInfo();
  for (auto &op : Instr.all_defs()) {
    if (!op.getReg().isVirtual())
      continue;
    assert(!op.getSubReg());
    if (TRI.isUniformReg(MRI, RBI, op.getReg()))
      continue;
    insertedDivergent |= markDivergent(op.getReg());
  }
  return insertedDivergent;
}

template <>
bool llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::isAlwaysUniform(
    const MachineInstr &Instr) const {
  // For MIR, an instruction is "always uniform" only if it has at least one
  // virtual register def AND all those defs are in UniformOverrides.
  // Instructions with no virtual register defs (e.g., terminators like
  // G_BRCOND, G_BR) return false to ensure they can be properly processed
  // for divergence during propagation.
  bool HasVirtualDef = false;
  for (const MachineOperand &Op : Instr.all_defs()) {
    if (!Op.getReg().isVirtual())
      continue;
    HasVirtualDef = true;
    if (!UniformOverrides.contains(Op.getReg()))
      return false;
  }
  // Only return true if we found at least one virtual def and all were uniform
  return HasVirtualDef;
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::initialize() {
  const TargetInstrInfo &InstrInfo = *F.getSubtarget().getInstrInfo();
  const MachineRegisterInfo &MRI = F.getRegInfo();
  const RegisterBankInfo &RBI = *F.getSubtarget().getRegBankInfo();
  const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();

  for (const MachineBasicBlock &Block : F) {
    for (const MachineInstr &Instr : Block) {
      // Terminators are handled separately because:
      // 1. Many terminators (G_BRCOND, G_BR) have no def operands, so the
      // per-def loop below would skip them entirely.
      // 2. Divergent terminators mark the BLOCK as
      // divergent(DivergentTermBlocks), not individual values, which is
      // different from regular instructions.
      if (Instr.isTerminator()) {
        if (InstrInfo.isTerminatorDivergent(Instr)) {
          if (DivergentTermBlocks.insert(Instr.getParent()).second) {
            Worklist.push_back(&Instr);
          }
        }
        continue;
      }

      // Query uniformity for each def operand separately.
      unsigned DefIdx = 0;
      bool HasDivergentDef = false;
      for (const MachineOperand &Op : Instr.all_defs()) {
        if (!Op.getReg().isVirtual()) {
          DefIdx++;
          continue;
        }

        ValueUniformity Uniformity =
            InstrInfo.getValueUniformity(Instr, DefIdx);

        switch (Uniformity) {
        case ValueUniformity::AlwaysUniform:
          addUniformOverride(Op.getReg());
          break;
        case ValueUniformity::NeverUniform:
          // Skip registers that are inherently uniform (e.g., SGPRs on AMDGPU)
          // even if the instruction is marked as NeverUniform.
          if (!TRI.isUniformReg(MRI, RBI, Op.getReg())) {
            if (markDivergent(Op.getReg()))
              HasDivergentDef = true;
          }
          break;
        case ValueUniformity::Default:
          break;
        }
        DefIdx++;
      }
      // If any def was marked divergent, add the instruction to worklist
      // for divergence propagation to users.
      if (HasDivergentDef)
        Worklist.push_back(&Instr);
    }
  }
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::pushUsers(
    Register Reg) {
  assert(isDivergent(Reg));
  const auto &RegInfo = F.getRegInfo();
  for (MachineInstr &UserInstr : RegInfo.use_instructions(Reg)) {
    markDivergent(UserInstr);
  }
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::pushUsers(
    const MachineInstr &Instr) {
  assert(!isAlwaysUniform(Instr));
  if (Instr.isTerminator())
    return;
  for (const MachineOperand &op : Instr.all_defs()) {
    auto Reg = op.getReg();
    if (isDivergent(Reg))
      pushUsers(Reg);
  }
}

template <>
bool llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::usesValueFromCycle(
    const MachineInstr &I, const MachineCycle &DefCycle) const {
  assert(!isAlwaysUniform(I));
  for (auto &Op : I.operands()) {
    if (!Op.isReg() || !Op.readsReg())
      continue;
    auto Reg = Op.getReg();

    // FIXME: Physical registers need to be properly checked instead of always
    // returning true
    if (Reg.isPhysical())
      return true;

    auto *Def = F.getRegInfo().getVRegDef(Reg);
    if (DefCycle.contains(Def->getParent()))
      return true;
  }
  return false;
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::
    propagateTemporalDivergence(const MachineInstr &I,
                                const MachineCycle &DefCycle) {
  const auto &RegInfo = F.getRegInfo();
  for (auto &Op : I.all_defs()) {
    if (!Op.getReg().isVirtual())
      continue;
    auto Reg = Op.getReg();
    for (MachineInstr &UserInstr : RegInfo.use_instructions(Reg)) {
      if (DefCycle.contains(UserInstr.getParent()))
        continue;
      markDivergent(UserInstr);

      recordTemporalDivergence(Reg, &UserInstr, &DefCycle);
    }
  }
}

template <>
bool llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::isDivergentUse(
    const MachineOperand &U) const {
  if (!U.isReg())
    return false;

  auto Reg = U.getReg();
  if (isDivergent(Reg))
    return true;

  const auto &RegInfo = F.getRegInfo();
  auto *Def = RegInfo.getOneDef(Reg);
  if (!Def)
    return true;

  auto *DefInstr = Def->getParent();
  auto *UseInstr = U.getParent();
  return isTemporalDivergent(*UseInstr->getParent(), *DefInstr);
}

// This ensures explicit instantiation of
// GenericUniformityAnalysisImpl::ImplDeleter::operator()
template class llvm::GenericUniformityInfo<MachineSSAContext>;
template struct llvm::GenericUniformityAnalysisImplDeleter<
    llvm::GenericUniformityAnalysisImpl<MachineSSAContext>>;

MachineUniformityInfo llvm::computeMachineUniformityInfo(
    MachineFunction &F, const MachineCycleInfo &cycleInfo,
    const MachineDominatorTree &domTree, bool HasBranchDivergence) {
  assert(F.getRegInfo().isSSA() && "Expected to be run on SSA form!");
  MachineUniformityInfo UI(domTree, cycleInfo);
  if (HasBranchDivergence)
    UI.compute();
  return UI;
}

namespace {

class MachineUniformityInfoPrinterPass : public MachineFunctionPass {
public:
  static char ID;

  MachineUniformityInfoPrinterPass();

  bool runOnMachineFunction(MachineFunction &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace

AnalysisKey MachineUniformityAnalysis::Key;

MachineUniformityAnalysis::Result
MachineUniformityAnalysis::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  auto &DomTree = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  auto &CI = MFAM.getResult<MachineCycleAnalysis>(MF);
  auto &FAM = MFAM.getResult<FunctionAnalysisManagerMachineFunctionProxy>(MF)
                  .getManager();
  auto &F = MF.getFunction();
  auto &TTI = FAM.getResult<TargetIRAnalysis>(F);
  return computeMachineUniformityInfo(MF, CI, DomTree,
                                      TTI.hasBranchDivergence(&F));
}

PreservedAnalyses
MachineUniformityPrinterPass::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  auto &MUI = MFAM.getResult<MachineUniformityAnalysis>(MF);
  OS << "MachineUniformityInfo for function: ";
  MF.getFunction().printAsOperand(OS, /*PrintType=*/false);
  OS << '\n';
  MUI.print(OS);
  return PreservedAnalyses::all();
}

char MachineUniformityAnalysisPass::ID = 0;

MachineUniformityAnalysisPass::MachineUniformityAnalysisPass()
    : MachineFunctionPass(ID) {}

INITIALIZE_PASS_BEGIN(MachineUniformityAnalysisPass, "machine-uniformity",
                      "Machine Uniformity Info Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(MachineUniformityAnalysisPass, "machine-uniformity",
                    "Machine Uniformity Info Analysis", false, true)

void MachineUniformityAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<MachineCycleInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineUniformityAnalysisPass::runOnMachineFunction(MachineFunction &MF) {
  auto &DomTree = getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  auto &CI = getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  // FIXME: Query TTI::hasBranchDivergence. -run-pass seems to end up with a
  // default NoTTI
  UI = computeMachineUniformityInfo(MF, CI, DomTree, true);
  return false;
}

void MachineUniformityAnalysisPass::print(raw_ostream &OS,
                                          const Module *) const {
  OS << "MachineUniformityInfo for function: ";
  UI.getFunction().getFunction().printAsOperand(OS, /*PrintType=*/false);
  OS << '\n';
  UI.print(OS);
}

char MachineUniformityInfoPrinterPass::ID = 0;

MachineUniformityInfoPrinterPass::MachineUniformityInfoPrinterPass()
    : MachineFunctionPass(ID) {}

INITIALIZE_PASS_BEGIN(MachineUniformityInfoPrinterPass,
                      "print-machine-uniformity",
                      "Print Machine Uniformity Info Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineUniformityAnalysisPass)
INITIALIZE_PASS_END(MachineUniformityInfoPrinterPass,
                    "print-machine-uniformity",
                    "Print Machine Uniformity Info Analysis", true, true)

void MachineUniformityInfoPrinterPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineUniformityAnalysisPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineUniformityInfoPrinterPass::runOnMachineFunction(
    MachineFunction &F) {
  auto &UI = getAnalysis<MachineUniformityAnalysisPass>();
  UI.print(errs());
  return false;
}
