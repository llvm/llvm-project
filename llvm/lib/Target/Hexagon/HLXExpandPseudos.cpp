//===- HLXExpandPseudos.cpp - Expand HLX load/store pseudos ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass expands the HLX load/store pseudo instructions into their final
// two-instruction bundled form. It runs after packetization so the resulting
// bundles are not seen by the post-RA scheduler.
//
//   LX_L16b_pseudo -> { Ld.tmp = mem(Rs); Ld = Ld }   (LX_L16b_tmp; LX_assign)
//   LX_S16b_pseudo -> { Lx = Lx; mem(Rs) = Lx.new }   (LX_assign; LX_S16b_new)
//
//===----------------------------------------------------------------------===//

#include "Hexagon.h"
#include "HexagonInstrInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hlx-expand-pseudos"

using namespace llvm;

static cl::opt<bool> DisableHLXExpandPseudos(
    "disable-hlx-expand-pseudos", cl::init(false), cl::Hidden,
    cl::desc("Disable expansion of HLX pseudo instructions"));

namespace llvm {
FunctionPass *createHLXExpandPseudos();
void initializeHLXExpandPseudosPass(PassRegistry &);
} // namespace llvm

static bool isHLXLoadStorePseudo(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return Opc == Hexagon::LX_L16b_pseudo || Opc == Hexagon::LX_S16b_pseudo;
}

namespace {

struct HLXExpandPseudos : public MachineFunctionPass {
  static char ID;

  HLXExpandPseudos() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "HLX Expand Pseudos"; }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  const HexagonInstrInfo *HII = nullptr;

  bool expandPseudos(MachineInstr &MI);
  bool expandPseudosInBundle(MachineInstr &MI);
};

} // end anonymous namespace

char HLXExpandPseudos::ID = 0;

INITIALIZE_PASS(HLXExpandPseudos, "hlx-expand-pseudos", "HLX Expand Pseudos",
                false, false)

bool HLXExpandPseudos::expandPseudos(MachineInstr &MI) {
  bool IsBundled = MI.isBundled();
  MachineBasicBlock *MBB = MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  auto ExpandLoad = [&](unsigned AssignOpc, unsigned LoadOpc) {
    MachineOperand Op0 = MI.getOperand(0);
    MachineOperand Op1 = MI.getOperand(1);
    Register Reg0 = Op0.getReg();
    MachineInstr *Start = BuildMI(*MBB, MI, DL, HII->get(LoadOpc), Reg0)
                              .add(Op1)
                              .cloneMemRefs(MI)
                              .copyImplicitOps(MI)
                              .getInstr();
    MachineInstr *End =
        BuildMI(*MBB, MI, DL, HII->get(AssignOpc), Reg0)
            .addReg(Reg0, RegState::Kill | RegState::InternalRead)
            .getInstr();
    if (IsBundled)
      return true;
    End->bundleWithPred();
    finalizeBundle(*MBB, Start->getIterator(), MI.getIterator());
    MBB->erase(MI);
    return true;
  };

  auto ExpandStore = [&](unsigned AssignOpc, unsigned StoreOpc) {
    MachineOperand Op0 = MI.getOperand(0);
    MachineOperand Op1 = MI.getOperand(1);
    Register Reg1 = Op1.getReg();
    MachineInstr *Start = BuildMI(*MBB, MI, DL, HII->get(AssignOpc), Reg1)
                              .addReg(Reg1, RegState::Kill)
                              .getInstr();
    Op1.setIsInternalRead();
    MachineInstr *End = BuildMI(*MBB, MI, DL, HII->get(StoreOpc))
                            .add(Op0)
                            .add(Op1)
                            .cloneMemRefs(MI)
                            .copyImplicitOps(MI)
                            .getInstr();
    if (IsBundled)
      return true;
    End->bundleWithPred();
    finalizeBundle(*MBB, Start->getIterator(), MI.getIterator());
    MBB->erase(MI);
    return true;
  };

  switch (MI.getOpcode()) {
  default:
    return false;
  case Hexagon::LX_L16b_pseudo:
    return ExpandLoad(Hexagon::LX_assign, Hexagon::LX_L16b_tmp);
  case Hexagon::LX_S16b_pseudo:
    return ExpandStore(Hexagon::LX_assign, Hexagon::LX_S16b_new);
  }
}

bool HLXExpandPseudos::expandPseudosInBundle(MachineInstr &MI) {
  MachineBasicBlock::instr_iterator MII = MI.getIterator();
  MachineBasicBlock::instr_iterator MIE = MI.getParent()->instr_end();
  SmallVector<MachineBasicBlock::instr_iterator, 4> EraseList;
  bool MadeChange = false;
  for (++MII; MII != MIE && MII->isInsideBundle() && !MII->isBundle(); ++MII) {
    if (isHLXLoadStorePseudo(*MII) && expandPseudos(*MII)) {
      EraseList.push_back(MII);
      MadeChange = true;
    }
  }
  for (auto It : EraseList)
    It->eraseFromBundle();
  return MadeChange;
}

bool HLXExpandPseudos::runOnMachineFunction(MachineFunction &MF) {
  auto &HST = MF.getSubtarget<HexagonSubtarget>();
  if (!HST.useHLXOps() || DisableHLXExpandPseudos)
    return false;

  HII = static_cast<const HexagonInstrInfo *>(HST.getInstrInfo());

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.isBundle())
        MadeChange |= expandPseudosInBundle(MI);
      else if (isHLXLoadStorePseudo(MI))
        MadeChange |= expandPseudos(MI);
    }
  }
  return MadeChange;
}

FunctionPass *llvm::createHLXExpandPseudos() { return new HLXExpandPseudos(); }
