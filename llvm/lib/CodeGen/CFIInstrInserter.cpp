//===------ CFIInstrInserter.cpp - Insert additional CFI instructions -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This pass verifies incoming and outgoing CFA information of basic
/// blocks. CFA information is information about offset and register set by CFI
/// directives, valid at the start and end of a basic block. This pass checks
/// that outgoing information of predecessors matches incoming information of
/// their successors. Then it checks if blocks have correct CFA calculation rule
/// set and inserts additional CFI instruction at their beginnings if they
/// don't. CFI instructions are inserted if basic blocks have incorrect offset
/// or register set by previous blocks, as a result of a non-linear layout of
/// blocks in a function.
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetSubtargetInfo.h"
using namespace llvm;

namespace {
class CFIInstrInserter : public MachineFunctionPass {
 public:
  static char ID;

  CFIInstrInserter() : MachineFunctionPass(ID) {
    initializeCFIInstrInserterPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {

    if (!MF.getMMI().hasDebugInfo() &&
        !MF.getFunction()->needsUnwindTableEntry())
      return false;

    MBBVector.resize(MF.getNumBlockIDs());
    calculateCFAInfo(MF);
#ifndef NDEBUG
    unsigned ErrorNum = verify(MF);
    if (ErrorNum)
      report_fatal_error("Found " + Twine(ErrorNum) +
                         " in/out CFI information errors.");
#endif
    bool insertedCFI = insertCFIInstrs(MF);
    MBBVector.clear();
    return insertedCFI;
  }

 private:
  struct MBBCFAInfo {
    MachineBasicBlock *MBB;
    /// Value of cfa offset valid at basic block entry.
    int IncomingCFAOffset = -1;
    /// Value of cfa offset valid at basic block exit.
    int OutgoingCFAOffset = -1;
    /// Value of cfa register valid at basic block entry.
    unsigned IncomingCFARegister = 0;
    /// Value of cfa register valid at basic block exit.
    unsigned OutgoingCFARegister = 0;
    /// If in/out cfa offset and register values for this block have already
    /// been set or not.
    bool Processed = false;
  };

  /// Contains cfa offset and register values valid at entry and exit of basic
  /// blocks.
  SmallVector<struct MBBCFAInfo, 4> MBBVector;

  /// Calculate cfa offset and register values valid at entry and exit for all
  /// basic blocks in a function.
  void calculateCFAInfo(MachineFunction &MF);
  /// Calculate cfa offset and register values valid at basic block exit by
  /// checking the block for CFI instructions. Block's incoming CFA info remains
  /// the same.
  void calculateOutgoingCFAInfo(struct MBBCFAInfo &MBBInfo);
  /// Update in/out cfa offset and register values for successors of the basic
  /// block.
  void updateSuccCFAInfo(struct MBBCFAInfo &MBBInfo);

  /// Check if incoming CFA information of a basic block matches outgoing CFA
  /// information of the previous block. If it doesn't, insert CFI instruction
  /// at the beginning of the block that corrects the CFA calculation rule for
  /// that block.
  bool insertCFIInstrs(MachineFunction &MF);
  /// Return the cfa offset value that should be set at the beginning of a MBB
  /// if needed. The negated value is needed when creating CFI instructions that
  /// set absolute offset.
  int getCorrectCFAOffset(MachineBasicBlock *MBB) {
    return -MBBVector[MBB->getNumber()].IncomingCFAOffset;
  }

  void report(const char *msg, MachineBasicBlock &MBB);
  /// Go through each MBB in a function and check that outgoing offset and
  /// register of its predecessors match incoming offset and register of that
  /// MBB, as well as that incoming offset and register of its successors match
  /// outgoing offset and register of the MBB.
  unsigned verify(MachineFunction &MF);
};
}

char CFIInstrInserter::ID = 0;
INITIALIZE_PASS(CFIInstrInserter, "cfi-instr-inserter",
                "Check CFA info and insert CFI instructions if needed", false,
                false)
FunctionPass *llvm::createCFIInstrInserter() { return new CFIInstrInserter(); }

void CFIInstrInserter::calculateCFAInfo(MachineFunction &MF) {
  // Initial CFA offset value i.e. the one valid at the beginning of the
  // function.
  int InitialOffset =
      MF.getSubtarget().getFrameLowering()->getInitialCFAOffset(MF);
  // Initial CFA register value i.e. the one valid at the beginning of the
  // function.
  unsigned InitialRegister =
      MF.getSubtarget().getFrameLowering()->getInitialCFARegister(MF);

  // Initialize MBBMap.
  for (MachineBasicBlock &MBB : MF) {
    struct MBBCFAInfo MBBInfo;
    MBBInfo.MBB = &MBB;
    MBBInfo.IncomingCFAOffset = InitialOffset;
    MBBInfo.OutgoingCFAOffset = InitialOffset;
    MBBInfo.IncomingCFARegister = InitialRegister;
    MBBInfo.OutgoingCFARegister = InitialRegister;
    MBBVector[MBB.getNumber()] = MBBInfo;
  }

  // Set in/out cfa info for all blocks in the function. This traversal is based
  // on the assumption that the first block in the function is the entry block
  // i.e. that it has initial cfa offset and register values as incoming CFA
  // information.
  for (MachineBasicBlock &MBB : MF) {
    if (MBBVector[MBB.getNumber()].Processed) continue;
    calculateOutgoingCFAInfo(MBBVector[MBB.getNumber()]);
    updateSuccCFAInfo(MBBVector[MBB.getNumber()]);
  }
}

void CFIInstrInserter::calculateOutgoingCFAInfo(struct MBBCFAInfo &MBBInfo) {
  // Outgoing cfa offset set by the block.
  int SetOffset = MBBInfo.IncomingCFAOffset;
  // Outgoing cfa register set by the block.
  unsigned SetRegister = MBBInfo.IncomingCFARegister;
  const std::vector<MCCFIInstruction> &Instrs =
      MBBInfo.MBB->getParent()->getFrameInstructions();

  // Determine cfa offset and register set by the block.
  for (MachineInstr &MI :
       make_range(MBBInfo.MBB->instr_begin(), MBBInfo.MBB->instr_end())) {
    if (MI.isCFIInstruction()) {
      unsigned CFIIndex = MI.getOperand(0).getCFIIndex();
      const MCCFIInstruction &CFI = Instrs[CFIIndex];
      if (CFI.getOperation() == MCCFIInstruction::OpDefCfaRegister) {
        SetRegister = CFI.getRegister();
      } else if (CFI.getOperation() == MCCFIInstruction::OpDefCfaOffset) {
        SetOffset = CFI.getOffset();
      } else if (CFI.getOperation() == MCCFIInstruction::OpAdjustCfaOffset) {
        SetOffset += CFI.getOffset();
      } else if (CFI.getOperation() == MCCFIInstruction::OpDefCfa) {
        SetRegister = CFI.getRegister();
        SetOffset = CFI.getOffset();
      }
    }
  }

  MBBInfo.Processed = true;

  // Update outgoing CFA info.
  MBBInfo.OutgoingCFAOffset = SetOffset;
  MBBInfo.OutgoingCFARegister = SetRegister;
}

void CFIInstrInserter::updateSuccCFAInfo(struct MBBCFAInfo &MBBInfo) {

  for (MachineBasicBlock *Succ : MBBInfo.MBB->successors()) {
    struct MBBCFAInfo &SuccInfo = MBBVector[Succ->getNumber()];
    if (SuccInfo.Processed) continue;
    SuccInfo.IncomingCFAOffset = MBBInfo.OutgoingCFAOffset;
    SuccInfo.IncomingCFARegister = MBBInfo.OutgoingCFARegister;
    calculateOutgoingCFAInfo(SuccInfo);
    updateSuccCFAInfo(SuccInfo);
  }
}

bool CFIInstrInserter::insertCFIInstrs(MachineFunction &MF) {

  const struct MBBCFAInfo *PrevMBBInfo = &MBBVector[MF.front().getNumber()];
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool InsertedCFIInstr = false;

  for (MachineBasicBlock &MBB : MF) {
    // Skip the first MBB in a function
    if (MBB.getNumber() == MF.front().getNumber()) continue;

    const struct MBBCFAInfo& MBBInfo = MBBVector[MBB.getNumber()];
    auto MBBI = MBBInfo.MBB->begin();
    DebugLoc DL = MBBInfo.MBB->findDebugLoc(MBBI);

    if (PrevMBBInfo->OutgoingCFAOffset != MBBInfo.IncomingCFAOffset) {
      // If both outgoing offset and register of a previous block don't match
      // incoming offset and register of this block, add a def_cfa instruction
      // with the correct offset and register for this block.
      if (PrevMBBInfo->OutgoingCFARegister != MBBInfo.IncomingCFARegister) {
        unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createDefCfa(
            nullptr, MBBInfo.IncomingCFARegister, getCorrectCFAOffset(&MBB)));
        BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
        // If outgoing offset of a previous block doesn't match incoming offset
        // of this block, add a def_cfa_offset instruction with the correct
        // offset for this block.
      } else {
        unsigned CFIIndex =
            MF.addFrameInst(MCCFIInstruction::createDefCfaOffset(
                nullptr, getCorrectCFAOffset(&MBB)));
        BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
            .addCFIIndex(CFIIndex);
      }
      InsertedCFIInstr = true;
      // If outgoing register of a previous block doesn't match incoming
      // register of this block, add a def_cfa_register instruction with the
      // correct register for this block.
    } else if (PrevMBBInfo->OutgoingCFARegister != MBBInfo.IncomingCFARegister) {
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::createDefCfaRegister(
              nullptr, MBBInfo.IncomingCFARegister));
      BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }
    PrevMBBInfo = &MBBInfo;
  }
  return InsertedCFIInstr;
}

void CFIInstrInserter::report(const char *msg, MachineBasicBlock &MBB) {
  errs() << '\n';
  errs() << "*** " << msg << " ***\n"
         << "- function:    " << MBB.getParent()->getName() << "\n";
  errs() << "- basic block: BB#" << MBB.getNumber() << ' ' << MBB.getName()
         << " (" << (const void *)&MBB << ')';
  errs() << '\n';
}

unsigned CFIInstrInserter::verify(MachineFunction &MF) {
  unsigned ErrorNum = 0;
  for (MachineBasicBlock &CurrMBB : MF) {
    const struct MBBCFAInfo& CurrMBBInfo = MBBVector[CurrMBB.getNumber()];
    for (MachineBasicBlock *Pred : CurrMBB.predecessors()) {
      const struct MBBCFAInfo& PredMBBInfo = MBBVector[Pred->getNumber()];
      // Check that outgoing offset values of predecessors match the incoming
      // offset value of CurrMBB
      if (PredMBBInfo.OutgoingCFAOffset != CurrMBBInfo.IncomingCFAOffset) {
        report("The outgoing offset of a predecessor is inconsistent.",
               CurrMBB);
        errs() << "Predecessor BB#" << Pred->getNumber()
               << " has outgoing offset (" << PredMBBInfo.OutgoingCFAOffset
               << "), while BB#" << CurrMBB.getNumber()
               << " has incoming offset (" << CurrMBBInfo.IncomingCFAOffset
               << ").\n";
        ErrorNum++;
      }
      // Check that outgoing register values of predecessors match the incoming
      // register value of CurrMBB
      if (PredMBBInfo.OutgoingCFARegister != CurrMBBInfo.IncomingCFARegister) {
        report("The outgoing register of a predecessor is inconsistent.",
               CurrMBB);
        errs() << "Predecessor BB#" << Pred->getNumber()
               << " has outgoing register (" << PredMBBInfo.OutgoingCFARegister
               << "), while BB#" << CurrMBB.getNumber()
               << " has incoming register (" << CurrMBBInfo.IncomingCFARegister
               << ").\n";
        ErrorNum++;
      }
    }

    for (MachineBasicBlock *Succ : CurrMBB.successors()) {
      const struct MBBCFAInfo& SuccMBBInfo = MBBVector[Succ->getNumber()];
      // Check that incoming offset values of successors match the outgoing
      // offset value of CurrMBB
      if (SuccMBBInfo.IncomingCFAOffset != CurrMBBInfo.OutgoingCFAOffset) {
        report("The incoming offset of a successor is inconsistent.", CurrMBB);
        errs() << "Successor BB#" << Succ->getNumber()
               << " has incoming offset (" << SuccMBBInfo.IncomingCFAOffset
               << "), while BB#" << CurrMBB.getNumber()
               << " has outgoing offset (" << CurrMBBInfo.OutgoingCFAOffset
               << ").\n";
        ErrorNum++;
      }
      // Check that incoming register values of successors match the outgoing
      // register value of CurrMBB
      if (SuccMBBInfo.IncomingCFARegister != CurrMBBInfo.OutgoingCFARegister) {
        report("The incoming register of a successor is inconsistent.",
               CurrMBB);
        errs() << "Successor BB#" << Succ->getNumber()
               << " has incoming register (" << SuccMBBInfo.IncomingCFARegister
               << "), while BB#" << CurrMBB.getNumber()
               << " has outgoing register (" << CurrMBBInfo.OutgoingCFARegister
               << ").\n";
        ErrorNum++;
      }
    }
  }
  return ErrorNum;
}
