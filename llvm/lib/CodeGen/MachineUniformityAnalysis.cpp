//===- MachineUniformityAnalysis.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineUniformityAnalysis.h"
#include "llvm/ADT/GenericUniformityImpl.h"
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
  for (auto &op : I.operands()) {
    if (!op.isReg() || !op.isDef())
      continue;
    if (isDivergent(op.getReg()))
      return true;
  }
  return false;
}

template <>
bool llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::markDefsDivergent(
    const MachineInstr &Instr, bool AllDefsDivergent) {
  bool insertedDivergent = false;
  const auto &MRI = F.getRegInfo();
  const auto &TRI = *MRI.getTargetRegisterInfo();
  for (auto &op : Instr.operands()) {
    if (!op.isReg() || !op.isDef())
      continue;
    if (!op.getReg().isVirtual())
      continue;
    assert(!op.getSubReg());
    if (!AllDefsDivergent) {
      auto *RC = MRI.getRegClassOrNull(op.getReg());
      if (RC && !TRI.isDivergentRegClass(RC))
        continue;
    }
    insertedDivergent |= markDivergent(op.getReg());
  }
  return insertedDivergent;
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::initialize() {
  const auto &InstrInfo = *F.getSubtarget().getInstrInfo();

  for (const MachineBasicBlock &block : F) {
    for (const MachineInstr &instr : block) {
      auto uniformity = InstrInfo.getInstructionUniformity(instr);
      if (uniformity == InstructionUniformity::AlwaysUniform) {
        addUniformOverride(instr);
        continue;
      }

      if (uniformity == InstructionUniformity::NeverUniform) {
        markDefsDivergent(instr, /* AllDefsDivergent = */ false);
      }
    }
  }
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::pushUsers(
    Register Reg) {
  const auto &RegInfo = F.getRegInfo();
  for (MachineInstr &UserInstr : RegInfo.use_instructions(Reg)) {
    if (isAlwaysUniform(UserInstr))
      continue;
    if (markDivergent(UserInstr))
      Worklist.push_back(&UserInstr);
  }
}

template <>
void llvm::GenericUniformityAnalysisImpl<MachineSSAContext>::pushUsers(
    const MachineInstr &Instr) {
  assert(!isAlwaysUniform(Instr));
  if (Instr.isTerminator())
    return;
  for (const MachineOperand &op : Instr.operands()) {
    if (op.isReg() && op.isDef() && op.getReg().isVirtual())
      pushUsers(op.getReg());
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
    assert(Reg.isVirtual());
    auto *Def = F.getRegInfo().getVRegDef(Reg);
    if (DefCycle.contains(Def->getParent()))
      return true;
  }
  return false;
}

// This ensures explicit instantiation of
// GenericUniformityAnalysisImpl::ImplDeleter::operator()
template class llvm::GenericUniformityInfo<MachineSSAContext>;
template struct llvm::GenericUniformityAnalysisImplDeleter<
    llvm::GenericUniformityAnalysisImpl<MachineSSAContext>>;

MachineUniformityInfo
llvm::computeMachineUniformityInfo(MachineFunction &F,
                                   const MachineCycleInfo &cycleInfo,
                                   const MachineDomTree &domTree) {
  assert(F.getRegInfo().isSSA() && "Expected to be run on SSA form!");
  return MachineUniformityInfo(F, domTree, cycleInfo);
}

namespace {

/// Legacy analysis pass which computes a \ref MachineUniformityInfo.
class MachineUniformityAnalysisPass : public MachineFunctionPass {
  MachineUniformityInfo UI;

public:
  static char ID;

  MachineUniformityAnalysisPass();

  MachineUniformityInfo &getUniformityInfo() { return UI; }
  const MachineUniformityInfo &getUniformityInfo() const { return UI; }

  bool runOnMachineFunction(MachineFunction &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  // TODO: verify analysis
};

class MachineUniformityInfoPrinterPass : public MachineFunctionPass {
public:
  static char ID;

  MachineUniformityInfoPrinterPass();

  bool runOnMachineFunction(MachineFunction &F) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

} // namespace

char MachineUniformityAnalysisPass::ID = 0;

MachineUniformityAnalysisPass::MachineUniformityAnalysisPass()
    : MachineFunctionPass(ID) {
  initializeMachineUniformityAnalysisPassPass(*PassRegistry::getPassRegistry());
}

INITIALIZE_PASS_BEGIN(MachineUniformityAnalysisPass, "machine-uniformity",
                      "Machine Uniformity Info Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineCycleInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(MachineUniformityAnalysisPass, "machine-uniformity",
                    "Machine Uniformity Info Analysis", true, true)

void MachineUniformityAnalysisPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineCycleInfoWrapperPass>();
  AU.addRequired<MachineDominatorTree>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineUniformityAnalysisPass::runOnMachineFunction(MachineFunction &MF) {
  auto &DomTree = getAnalysis<MachineDominatorTree>().getBase();
  auto &CI = getAnalysis<MachineCycleInfoWrapperPass>().getCycleInfo();
  UI = computeMachineUniformityInfo(MF, CI, DomTree);
  return false;
}

void MachineUniformityAnalysisPass::print(raw_ostream &OS,
                                          const Module *) const {
  OS << "MachineUniformityInfo for function: " << UI.getFunction().getName()
     << "\n";
  UI.print(OS);
}

char MachineUniformityInfoPrinterPass::ID = 0;

MachineUniformityInfoPrinterPass::MachineUniformityInfoPrinterPass()
    : MachineFunctionPass(ID) {
  initializeMachineUniformityInfoPrinterPassPass(
      *PassRegistry::getPassRegistry());
}

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
