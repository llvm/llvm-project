//===-- MipsHazardSchedule.cpp - Workaround pipeline hazards --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass is used to workaround certain pipeline hazards. For now, this
/// covers compact branch hazards. In future this pass can be extended to other
/// pipeline hazards, such as various MIPS1 hazards, processor errata that
/// require instruction reorganization, etc.
///
/// This pass has to run after the delay slot filler as that pass can introduce
/// pipeline hazards, hence the existing hazard recognizer is not suitable.
///
/// Hazards handled: forbidden slots for MIPSR6.
///
/// A forbidden slot hazard occurs when a compact branch instruction is executed
/// and the adjacent instruction in memory is a control transfer instruction
/// such as a branch or jump, ERET, ERETNC, DERET, WAIT and PAUSE.
///
/// For example:
///
/// 0x8004      bnec    a1,v0,<P+0x18>
/// 0x8008      beqc    a1,a2,<P+0x54>
///
/// In such cases, the processor is required to signal a Reserved Instruction
/// exception.
///
/// Here, if the instruction at 0x8004 is executed, the processor will raise an
/// exception as there is a control transfer instruction at 0x8008.
///
/// There are two sources of forbidden slot hazards:
///
/// A) A previous pass has created a compact branch directly.
/// B) Transforming a delay slot branch into compact branch. This case can be
///    difficult to process as lookahead for hazards is insufficient, as
///    backwards delay slot fillling can also produce hazards in previously
///    processed instuctions.
///
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsInstrInfo.h"
#include "MipsSEInstrInfo.h"
#include "MipsTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "mips-hazard-schedule"

STATISTIC(NumInsertedNops, "Number of nops inserted");

namespace {

typedef MachineBasicBlock::iterator Iter;
typedef MachineBasicBlock::reverse_iterator ReverseIter;

class MipsHazardSchedule : public MachineFunctionPass {

public:
  MipsHazardSchedule() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "Mips Hazard Schedule"; }

  bool runOnMachineFunction(MachineFunction &F) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

private:
  static char ID;
};

char MipsHazardSchedule::ID = 0;
} // end of anonymous namespace

/// Returns a pass that clears pipeline hazards.
FunctionPass *llvm::createMipsHazardSchedule() {
  return new MipsHazardSchedule();
}

// Find the next real instruction from the current position in current basic
// block.
static Iter getNextMachineInstrInBB(Iter Position) {
  Iter I = Position, E = Position->getParent()->end();
  I = std::find_if_not(I, E,
                       [](const Iter &Insn) { return Insn->isTransient(); });

  return I;
}

// Find the next real instruction from the current position, looking through
// basic block boundaries.
static Iter getNextMachineInstr(Iter Position, MachineBasicBlock *Parent) {
  if (Position == Parent->end()) {
    MachineBasicBlock *Succ = Parent->getNextNode();
    if (Succ != nullptr && Parent->isSuccessor(Succ)) {
      Position = Succ->begin();
      Parent = Succ;
    } else {
      llvm_unreachable(
          "Should have identified the end of the function earlier!");
    }
  }

  Iter Instr = getNextMachineInstrInBB(Position);
  if (Instr == Parent->end()) {
    return getNextMachineInstr(Instr, Parent);
  }
  return Instr;
}

bool MipsHazardSchedule::runOnMachineFunction(MachineFunction &MF) {

  const MipsSubtarget *STI =
      &static_cast<const MipsSubtarget &>(MF.getSubtarget());

  // Forbidden slot hazards are only defined for MIPSR6 but not microMIPSR6.
  if (!STI->hasMips32r6() || STI->inMicroMipsMode())
    return false;

  bool Changed = false;
  const MipsInstrInfo *TII = STI->getInstrInfo();

  for (MachineFunction::iterator FI = MF.begin(); FI != MF.end(); ++FI) {
    for (Iter I = FI->begin(); I != FI->end(); ++I) {

      // Forbidden slot hazard handling. Use lookahead over state.
      if (!TII->HasForbiddenSlot(*I))
        continue;

      Iter Inst;
      bool LastInstInFunction =
          std::next(I) == FI->end() && std::next(FI) == MF.end();
      if (!LastInstInFunction) {
        Inst = getNextMachineInstr(std::next(I), &*FI);
      }

      if (LastInstInFunction || !TII->SafeInForbiddenSlot(*Inst)) {
        Changed = true;
        MIBundleBuilder(&*I)
            .append(BuildMI(MF, I->getDebugLoc(), TII->get(Mips::NOP)));
        NumInsertedNops++;
      }
    }
  }
  return Changed;
}
