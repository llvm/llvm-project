//===-- WebAssemblyRegStackify.cpp - Register Stackification --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a register stacking pass.
///
/// This pass reorders instructions to put register uses and defs in an order
/// such that they form single-use expression trees. Registers fitting this form
/// are then marked as "stackified", meaning references to them are replaced by
/// "push" and "pop" from the stack.
///
/// This is primarily a code size optimiation, since temporary values on the
/// expression don't need to be named.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-reg-stackify"

namespace {
class WebAssemblyRegStackify final : public MachineFunctionPass {
  const char *getPassName() const override {
    return "WebAssembly Register Stackify";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyRegStackify() : MachineFunctionPass(ID) {}
};
} // end anonymous namespace

char WebAssemblyRegStackify::ID = 0;
FunctionPass *llvm::createWebAssemblyRegStackify() {
  return new WebAssemblyRegStackify();
}

bool WebAssemblyRegStackify::runOnMachineFunction(MachineFunction &MF) {
  DEBUG(dbgs() << "********** Register Stackifying **********\n"
                  "********** Function: "
               << MF.getName() << '\n');

  bool Changed = false;
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo &MFI = *MF.getInfo<WebAssemblyFunctionInfo>();

  // Walk the instructions from the bottom up. Currently we don't look past
  // block boundaries, and the blocks aren't ordered so the block visitation
  // order isn't significant, but we may want to change this in the future.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : reverse(MBB)) {
      MachineInstr *Insert = &MI;

      // Don't nest anything inside a phi.
      if (Insert->getOpcode() == TargetOpcode::PHI)
        break;

      // Iterate through the inputs in reverse order, since we'll be pulling
      // operands off the stack in FIFO order.
      for (MachineOperand &Op : reverse(Insert->uses())) {
        // We're only interested in explicit virtual register operands.
        if (!Op.isReg() || Op.isImplicit())
          continue;

        unsigned Reg = Op.getReg();
        if (!TargetRegisterInfo::isVirtualRegister(Reg))
          continue;

        // Only consider registers with a single definition.
        // TODO: Eventually we may relax this, to stackify phi transfers.
        MachineInstr *Def = MRI.getVRegDef(Reg);
        if (!Def)
          continue;

        // There's no use in nesting implicit defs inside anything.
        if (Def->getOpcode() == TargetOpcode::IMPLICIT_DEF)
          continue;

        // Single-use expression trees require defs that have one use, or that
        // they be trivially clonable.
        // TODO: Eventually we'll relax this, to take advantage of set_local
        // returning its result.
        bool OneUse = MRI.hasOneUse(Reg);
        if (!OneUse && !Def->isMoveImmediate())
          continue;

        // For now, be conservative and don't look across block boundaries,
        // unless we have something trivially clonable.
        // TODO: Be more aggressive.
        if (Def->getParent() != &MBB && !Def->isMoveImmediate())
          continue;

        // For now, be simple and don't reorder loads, stores, or side effects.
        // TODO: Be more aggressive.
        if ((Def->mayLoad() || Def->mayStore() ||
             Def->hasUnmodeledSideEffects()))
          continue;

        Changed = true;
        if (OneUse) {
          // Move the def down and nest it in the current instruction.
          MBB.insert(MachineBasicBlock::instr_iterator(Insert),
                     Def->removeFromParent());
          MFI.stackifyVReg(Reg);
          Insert = Def;
        } else {
          // Clone the def down and nest it in the current instruction.
          MachineInstr *Clone = MF.CloneMachineInstr(Def);
          unsigned OldReg = Def->getOperand(0).getReg();
          unsigned NewReg = MRI.createVirtualRegister(MRI.getRegClass(OldReg));
          assert(Op.getReg() == OldReg);
          assert(Clone->getOperand(0).getReg() == OldReg);
          Op.setReg(NewReg);
          Clone->getOperand(0).setReg(NewReg);
          MBB.insert(MachineBasicBlock::instr_iterator(Insert), Clone);
          MFI.stackifyVReg(Reg);
          Insert = Clone;
        }
      }
    }
  }

  return Changed;
}
