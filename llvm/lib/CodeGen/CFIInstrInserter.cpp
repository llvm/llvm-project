//===------ CFIInstrInserter.cpp - Insert additional CFI instructions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Insert CFI instructions at the beginnings of basic blocks if needed. CFI
// instructions are inserted if basic blocks have incorrect offset or register
// set by prevoius blocks.
//
//===----------------------------------------------------------------------===//
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

namespace {
class CFIInstrInserter : public MachineFunctionPass {
 public:
  CFIInstrInserter() : MachineFunctionPass(ID) {
    initializeCFIInstrInserterPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  static char ID;

 private:
  StringRef getPassName() const override { return "CFI Instruction Inserter"; }

  // Check if incoming CFI information of a basic block matches outgoing CFI
  // information of the previous block. If it doesn't, insert CFI instruction at
  // the beginning of the block that corrects the CFA calculation rule for that
  // block.
  void CorrectCFA(MachineFunction &MF);

  // Return the cfa offset value that should be set at the beginning of MBB if
  // needed. The negated value is needed when creating CFI instructions that set
  // absolute offset.
  int getCorrectCFAOffset(MachineBasicBlock &MBB) {
    return -MBB.getIncomingCFAOffset();
  }

  // Were any CFI instructions inserted
  bool InsertedCFIInstr = false;
};
}

char CFIInstrInserter::ID = 0;
INITIALIZE_PASS(CFIInstrInserter, "cfiinstrinserter",
                "Check CFI info and insert CFI instructions if needed", false,
                false)

FunctionPass *llvm::createCFIInstrInserter() { return new CFIInstrInserter(); }

bool CFIInstrInserter::runOnMachineFunction(MachineFunction &MF) {
  bool NeedsDwarfCFI = (MF.getMMI().hasDebugInfo() ||
                        MF.getFunction()->needsUnwindTableEntry()) &&
                       (!MF.getTarget().getTargetTriple().isOSDarwin() &&
                        !MF.getTarget().getTargetTriple().isOSWindows());

  if (!NeedsDwarfCFI) return false;

  // Insert appropriate CFI instructions for each MBB if CFA calculation rule
  // needs to be corrected for that MBB.
  CorrectCFA(MF);

  return InsertedCFIInstr;
}

void CFIInstrInserter::CorrectCFA(MachineFunction &MF) {

  MachineBasicBlock &FirstMBB = MF.front();
  MachineBasicBlock *PrevMBB = &FirstMBB;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  InsertedCFIInstr = false;

  for (auto &MBB : MF) {
    // Skip the first MBB in a function
    if (MBB.getNumber() == FirstMBB.getNumber()) continue;

    auto MBBI = MBB.begin();
    DebugLoc DL = MBB.findDebugLoc(MBBI);

    if (PrevMBB->getOutgoingCFAOffset() != MBB.getIncomingCFAOffset()) {
      // If both outgoing offset and register of a previous block don't match
      // incoming offset and register of this block, add a def_cfa instruction
      // with the correct offset and register for this block.
      if (PrevMBB->getOutgoingCFARegister() != MBB.getIncomingCFARegister()) {
        unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createDefCfa(
            nullptr, MBB.getIncomingCFARegister(), getCorrectCFAOffset(MBB)));
        BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
        // If outgoing offset of a previous block doesn't match incoming offset
        // of this block, add a def_cfa_offset instruction with the correct
        // offset for this block.
      } else {
        unsigned CFIIndex =
            MF.addFrameInst(MCCFIInstruction::createDefCfaOffset(
                nullptr, getCorrectCFAOffset(MBB)));
        BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      }
      InsertedCFIInstr = true;
      // If outgoing register of a previous block doesn't match incoming
      // register of this block, add a def_cfa_register instruction with the
      // correct register for this block.
    } else if (PrevMBB->getOutgoingCFARegister() !=
               MBB.getIncomingCFARegister()) {
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::createDefCfaRegister(
              nullptr, MBB.getIncomingCFARegister()));
      BuildMI(MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }
    PrevMBB = &MBB;
  }
}
