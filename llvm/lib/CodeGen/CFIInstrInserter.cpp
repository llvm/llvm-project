//===------ CFIInstrInserter.cpp - Insert additional CFI instructions -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
using namespace llvm;

static cl::opt<bool> VerifyCFI("verify-cfiinstrs",
    cl::desc("Verify Call Frame Information instructions"),
    cl::init(false),
    cl::Hidden);

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
    if (!MF.needsFrameMoves())
      return false;

    MBBVector.resize(MF.getNumBlockIDs());
    calculateCFAInfo(MF);

    if (VerifyCFI) {
      if (unsigned ErrorNum = verify(MF))
        report_fatal_error("Found " + Twine(ErrorNum) +
                           " in/out CFI information errors.");
    }
    bool insertedCFI = insertCFIInstrs(MF);
    MBBVector.clear();
    return insertedCFI;
  }

 private:
#define INVALID_REG UINT_MAX
#define INVALID_OFFSET INT_MAX
   /// contains the location where CSR register is saved.
   struct CSRSavedLocation {
     enum Kind { INVALID, REGISTER, CFA_OFFSET };
     CSRSavedLocation() {
       K = Kind::INVALID;
       Reg = 0;
       Offset = 0;
     }
     Kind K;
     // Dwarf register number
     unsigned Reg;
     // CFA offset
     int64_t Offset;
     bool isValid() const { return K != Kind::INVALID; }
     bool operator==(const CSRSavedLocation &RHS) const {
       switch (K) {
       case Kind::INVALID:
         return !RHS.isValid();
       case Kind::REGISTER:
         return Reg == RHS.Reg;
       case Kind::CFA_OFFSET:
         return Offset == RHS.Offset;
       }
       llvm_unreachable("Unknown CSRSavedLocation Kind!");
     }
     void dump(raw_ostream &OS) const {
       switch (K) {
       case Kind::INVALID:
         OS << "INVALID";
         break;
       case Kind::REGISTER:
         OS << "In Dwarf register: " << Reg;
         break;
       case Kind::CFA_OFFSET:
         OS << "At CFA offset: " << Offset;
         break;
       }
     }
   };

   struct MBBCFAInfo {
     MachineBasicBlock *MBB;
     /// Value of cfa offset valid at basic block entry.
     int64_t IncomingCFAOffset = -1;
     /// Value of cfa offset valid at basic block exit.
     int64_t OutgoingCFAOffset = -1;
     /// Value of cfa register valid at basic block entry.
     unsigned IncomingCFARegister = 0;
     /// Value of cfa register valid at basic block exit.
     unsigned OutgoingCFARegister = 0;
     /// Set of locations where the callee saved registers are at basic block
     /// entry.
     SmallVector<CSRSavedLocation> IncomingCSRLocations;
     /// Set of locations where the callee saved registers are at basic block
     /// exit.
     SmallVector<CSRSavedLocation> OutgoingCSRLocations;
     /// If in/out cfa offset and register values for this block have already
     /// been set or not.
     bool Processed = false;
   };

   /// Contains cfa offset and register values valid at entry and exit of basic
   /// blocks.
   std::vector<MBBCFAInfo> MBBVector;

   /// Calculate cfa offset and register values valid at entry and exit for all
   /// basic blocks in a function.
   void calculateCFAInfo(MachineFunction &MF);
   /// Calculate cfa offset and register values valid at basic block exit by
   /// checking the block for CFI instructions. Block's incoming CFA info
   /// remains the same.
   void calculateOutgoingCFAInfo(MBBCFAInfo &MBBInfo);
   /// Update in/out cfa offset and register values for successors of the basic
   /// block.
   void updateSuccCFAInfo(MBBCFAInfo &MBBInfo);

   /// Check if incoming CFA information of a basic block matches outgoing CFA
   /// information of the previous block. If it doesn't, insert CFI instruction
   /// at the beginning of the block that corrects the CFA calculation rule for
   /// that block.
   bool insertCFIInstrs(MachineFunction &MF);
   /// Return the cfa offset value that should be set at the beginning of a MBB
   /// if needed. The negated value is needed when creating CFI instructions
   /// that set absolute offset.
   int64_t getCorrectCFAOffset(MachineBasicBlock *MBB) {
     return MBBVector[MBB->getNumber()].IncomingCFAOffset;
   }

   void reportCFAError(const MBBCFAInfo &Pred, const MBBCFAInfo &Succ);
   void reportCSRError(const MBBCFAInfo &Pred, const MBBCFAInfo &Succ);
   /// Go through each MBB in a function and check that outgoing offset and
   /// register of its predecessors match incoming offset and register of that
   /// MBB, as well as that incoming offset and register of its successors match
   /// outgoing offset and register of the MBB.
   unsigned verify(MachineFunction &MF);
};
}  // namespace

char CFIInstrInserter::ID = 0;
INITIALIZE_PASS(CFIInstrInserter, "cfi-instr-inserter",
                "Check CFA info and insert CFI instructions if needed", false,
                false)
FunctionPass *llvm::createCFIInstrInserter() { return new CFIInstrInserter(); }

void CFIInstrInserter::calculateCFAInfo(MachineFunction &MF) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  // Initial CFA offset value i.e. the one valid at the beginning of the
  // function.
  int InitialOffset =
      MF.getSubtarget().getFrameLowering()->getInitialCFAOffset(MF);
  // Initial CFA register value i.e. the one valid at the beginning of the
  // function.
  Register InitialRegister =
      MF.getSubtarget().getFrameLowering()->getInitialCFARegister(MF);
  unsigned DwarfInitialRegister = TRI.getDwarfRegNum(InitialRegister, true);
  unsigned NumRegs = TRI.getNumSupportedRegs(MF);

  // Initialize MBBMap.
  for (MachineBasicBlock &MBB : MF) {
    MBBCFAInfo &MBBInfo = MBBVector[MBB.getNumber()];
    MBBInfo.MBB = &MBB;
    MBBInfo.IncomingCFAOffset = InitialOffset;
    MBBInfo.OutgoingCFAOffset = InitialOffset;
    MBBInfo.IncomingCFARegister = DwarfInitialRegister;
    MBBInfo.OutgoingCFARegister = DwarfInitialRegister;
    MBBInfo.IncomingCSRLocations.resize(NumRegs);
    MBBInfo.OutgoingCSRLocations.resize(NumRegs);
  }

  // Record the initial location of all registers.
  MBBCFAInfo &EntryMBBInfo = MBBVector[MF.front().getNumber()];
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();
  for (int i = 0; CSRegs[i]; ++i) {
    unsigned Reg = TRI.getDwarfRegNum(CSRegs[i], true);
    CSRSavedLocation &CSRLoc = EntryMBBInfo.IncomingCSRLocations[Reg];
    CSRLoc.Reg = Reg;
  }

  // Set in/out cfa info for all blocks in the function. This traversal is based
  // on the assumption that the first block in the function is the entry block
  // i.e. that it has initial cfa offset and register values as incoming CFA
  // information.
  updateSuccCFAInfo(MBBVector[MF.front().getNumber()]);
}

void CFIInstrInserter::calculateOutgoingCFAInfo(MBBCFAInfo &MBBInfo) {
  // Outgoing cfa offset set by the block.
  int64_t &OutgoingCFAOffset = MBBInfo.OutgoingCFAOffset;
  OutgoingCFAOffset = MBBInfo.IncomingCFAOffset;
  // Outgoing cfa register set by the block.
  unsigned &OutgoingCFARegister = MBBInfo.OutgoingCFARegister;
  OutgoingCFARegister = MBBInfo.IncomingCFARegister;
  // Outgoing locations for each callee-saved register set by the block.
  SmallVector<CSRSavedLocation> &OutgoingCSRLocations =
      MBBInfo.OutgoingCSRLocations;
  OutgoingCSRLocations = MBBInfo.IncomingCSRLocations;

  MachineFunction *MF = MBBInfo.MBB->getParent();
  const std::vector<MCCFIInstruction> &Instrs = MF->getFrameInstructions();

#ifndef NDEBUG
  int RememberState = 0;
#endif

  // Determine cfa offset and register set by the block.
  for (MachineInstr &MI : *MBBInfo.MBB) {
    if (MI.isCFIInstruction()) {
      unsigned CFIIndex = MI.getOperand(0).getCFIIndex();
      const MCCFIInstruction &CFI = Instrs[CFIIndex];
      switch (CFI.getOperation()) {
      case MCCFIInstruction::OpDefCfaRegister: {
        OutgoingCFARegister = CFI.getRegister();
        break;
      }
      case MCCFIInstruction::OpDefCfaOffset: {
        OutgoingCFAOffset = CFI.getOffset();
        break;
      }
      case MCCFIInstruction::OpAdjustCfaOffset: {
        OutgoingCFAOffset += CFI.getOffset();
        break;
      }
      case MCCFIInstruction::OpDefCfa: {
        OutgoingCFARegister = CFI.getRegister();
        OutgoingCFAOffset = CFI.getOffset();
        break;
      }
      case MCCFIInstruction::OpOffset: {
        CSRSavedLocation &CSRLocation = OutgoingCSRLocations[CFI.getRegister()];
        CSRLocation.K = CSRSavedLocation::Kind::CFA_OFFSET;
        CSRLocation.Offset = CFI.getOffset();
        break;
      }
      case MCCFIInstruction::OpRegister: {
        CSRSavedLocation &CSRLocation = OutgoingCSRLocations[CFI.getRegister()];
        CSRLocation.K = CSRSavedLocation::Kind::REGISTER;
        CSRLocation.Reg = CFI.getRegister2();
        break;
      }
      case MCCFIInstruction::OpRelOffset: {
        CSRSavedLocation &CSRLocation = OutgoingCSRLocations[CFI.getRegister()];
        CSRLocation.K = CSRSavedLocation::Kind::CFA_OFFSET;
        CSRLocation.Offset = CFI.getOffset() - OutgoingCFAOffset;
        break;
      }
      case MCCFIInstruction::OpRestore: {
        unsigned Reg = CFI.getRegister();
        CSRSavedLocation &CSRLocation = OutgoingCSRLocations[Reg];
        CSRLocation.K = CSRSavedLocation::Kind::REGISTER;
        CSRLocation.Reg = Reg;
        break;
      }
      case MCCFIInstruction::OpLLVMDefAspaceCfa:
        // TODO: Add support for handling cfi_def_aspace_cfa.
#ifndef NDEBUG
        report_fatal_error(
            "Support for cfi_llvm_def_aspace_cfa not implemented! Value of CFA "
            "may be incorrect!\n");
#endif
        break;
      case MCCFIInstruction::OpRememberState:
        // TODO: Add support for handling cfi_remember_state.
#ifndef NDEBUG
        // Currently we need cfi_remember_state and cfi_restore_state to be in
        // the same BB, so it will not impact outgoing CFA.
        ++RememberState;
        if (RememberState != 1)
          MF->getContext().reportError(
              SMLoc(),
              "Support for cfi_remember_state not implemented! Value of CFA "
              "may be incorrect!\n");
#endif
        break;
      case MCCFIInstruction::OpRestoreState:
        // TODO: Add support for handling cfi_restore_state.
#ifndef NDEBUG
        --RememberState;
        if (RememberState != 0)
          MF->getContext().reportError(
              SMLoc(),
              "Support for cfi_restore_state not implemented! Value of CFA may "
              "be incorrect!\n");
#endif
        break;
      // Other CFI directives do not affect CFA value.
      case MCCFIInstruction::OpUndefined:
      case MCCFIInstruction::OpSameValue:
      case MCCFIInstruction::OpEscape:
      case MCCFIInstruction::OpWindowSave:
      case MCCFIInstruction::OpNegateRAState:
      case MCCFIInstruction::OpNegateRAStateWithPC:
      case MCCFIInstruction::OpGnuArgsSize:
      case MCCFIInstruction::OpLabel:
      case MCCFIInstruction::OpValOffset:
        break;
      }
    }
  }

#ifndef NDEBUG
  if (RememberState != 0)
    MF->getContext().reportError(
        SMLoc(),
        "Support for cfi_remember_state not implemented! Value of CFA may be "
        "incorrect!\n");
#endif

  MBBInfo.Processed = true;
}

void CFIInstrInserter::updateSuccCFAInfo(MBBCFAInfo &MBBInfo) {
  SmallVector<MachineBasicBlock *, 4> Stack;
  Stack.push_back(MBBInfo.MBB);

  do {
    MachineBasicBlock *Current = Stack.pop_back_val();
    MBBCFAInfo &CurrentInfo = MBBVector[Current->getNumber()];
    calculateOutgoingCFAInfo(CurrentInfo);
    for (auto *Succ : CurrentInfo.MBB->successors()) {
      MBBCFAInfo &SuccInfo = MBBVector[Succ->getNumber()];
      if (!SuccInfo.Processed) {
        SuccInfo.IncomingCFAOffset = CurrentInfo.OutgoingCFAOffset;
        SuccInfo.IncomingCFARegister = CurrentInfo.OutgoingCFARegister;
        SuccInfo.IncomingCSRLocations = CurrentInfo.OutgoingCSRLocations;
        Stack.push_back(Succ);
      }
    }
  } while (!Stack.empty());
}

bool CFIInstrInserter::insertCFIInstrs(MachineFunction &MF) {
  const MBBCFAInfo *PrevMBBInfo = &MBBVector[MF.front().getNumber()];
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool InsertedCFIInstr = false;

  for (MachineBasicBlock &MBB : MF) {
    // Skip the first MBB in a function
    if (MBB.getNumber() == MF.front().getNumber()) continue;

    const MBBCFAInfo &MBBInfo = MBBVector[MBB.getNumber()];
    auto MBBI = MBBInfo.MBB->begin();
    DebugLoc DL = MBBInfo.MBB->findDebugLoc(MBBI);

    // If the current MBB will be placed in a unique section, a full DefCfa
    // must be emitted.
    const bool ForceFullCFA = MBB.isBeginSection();

    if ((PrevMBBInfo->OutgoingCFAOffset != MBBInfo.IncomingCFAOffset &&
         PrevMBBInfo->OutgoingCFARegister != MBBInfo.IncomingCFARegister) ||
        ForceFullCFA) {
      // If both outgoing offset and register of a previous block don't match
      // incoming offset and register of this block, or if this block begins a
      // section, add a def_cfa instruction with the correct offset and
      // register for this block.
      unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::cfiDefCfa(
          nullptr, MBBInfo.IncomingCFARegister, getCorrectCFAOffset(&MBB)));
      BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    } else if (PrevMBBInfo->OutgoingCFAOffset != MBBInfo.IncomingCFAOffset) {
      // If outgoing offset of a previous block doesn't match incoming offset
      // of this block, add a def_cfa_offset instruction with the correct
      // offset for this block.
      unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::cfiDefCfaOffset(
          nullptr, getCorrectCFAOffset(&MBB)));
      BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    } else if (PrevMBBInfo->OutgoingCFARegister !=
               MBBInfo.IncomingCFARegister) {
      unsigned CFIIndex =
          MF.addFrameInst(MCCFIInstruction::createDefCfaRegister(
              nullptr, MBBInfo.IncomingCFARegister));
      BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }

    if (ForceFullCFA) {
      MF.getSubtarget().getFrameLowering()->emitCalleeSavedFrameMovesFullCFA(
          *MBBInfo.MBB, MBBI);
      InsertedCFIInstr = true;
      PrevMBBInfo = &MBBInfo;
      continue;
    }

    for (unsigned i = 0; i < PrevMBBInfo->OutgoingCSRLocations.size(); ++i) {
      const CSRSavedLocation &PrevOutgoingCSRLoc =
          PrevMBBInfo->OutgoingCSRLocations[i];
      const CSRSavedLocation &HasToBeCSRLoc = MBBInfo.IncomingCSRLocations[i];
      // Ignore non-callee-saved registers, they remain uninitialized (invalid).
      if (!HasToBeCSRLoc.isValid())
        continue;
      if (HasToBeCSRLoc == PrevOutgoingCSRLoc)
        continue;

      unsigned CFIIndex = (unsigned)(-1);
      if (HasToBeCSRLoc.K == CSRSavedLocation::Kind::CFA_OFFSET &&
          HasToBeCSRLoc.Offset != PrevOutgoingCSRLoc.Offset) {
        CFIIndex = MF.addFrameInst(
            MCCFIInstruction::createOffset(nullptr, i, HasToBeCSRLoc.Offset));
      } else if (HasToBeCSRLoc.K == CSRSavedLocation::Kind::REGISTER &&
                 (HasToBeCSRLoc.Reg != PrevOutgoingCSRLoc.Reg)) {
        unsigned NewReg = HasToBeCSRLoc.Reg;
        unsigned DwarfEHReg = i;
        if (NewReg == DwarfEHReg) {
          CFIIndex = MF.addFrameInst(
              MCCFIInstruction::createRestore(nullptr, DwarfEHReg));
        } else {
          CFIIndex = MF.addFrameInst(
              MCCFIInstruction::createRegister(nullptr, i, HasToBeCSRLoc.Reg));
        }
      } else
        llvm_unreachable("Unexpected CSR location.");
      BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }

    PrevMBBInfo = &MBBInfo;
  }
  return InsertedCFIInstr;
}

void CFIInstrInserter::reportCFAError(const MBBCFAInfo &Pred,
                                      const MBBCFAInfo &Succ) {
  errs() << "*** Inconsistent CFA register and/or offset between pred and succ "
            "***\n";
  errs() << "Pred: " << Pred.MBB->getName() << " #" << Pred.MBB->getNumber()
         << " in " << Pred.MBB->getParent()->getName()
         << " outgoing CFA Reg:" << Pred.OutgoingCFARegister << "\n";
  errs() << "Pred: " << Pred.MBB->getName() << " #" << Pred.MBB->getNumber()
         << " in " << Pred.MBB->getParent()->getName()
         << " outgoing CFA Offset:" << Pred.OutgoingCFAOffset << "\n";
  errs() << "Succ: " << Succ.MBB->getName() << " #" << Succ.MBB->getNumber()
         << " incoming CFA Reg:" << Succ.IncomingCFARegister << "\n";
  errs() << "Succ: " << Succ.MBB->getName() << " #" << Succ.MBB->getNumber()
         << " incoming CFA Offset:" << Succ.IncomingCFAOffset << "\n";
}

void CFIInstrInserter::reportCSRError(const MBBCFAInfo &Pred,
                                      const MBBCFAInfo &Succ) {
  errs() << "*** Inconsistent CSR Saved between pred and succ in function "
         << Pred.MBB->getParent()->getName() << " ***\n";
  errs() << "Pred: " << Pred.MBB->getName() << " #" << Pred.MBB->getNumber()
         << " outgoing CSR Saved: ";
  for (const CSRSavedLocation &OutgoingCSRLocation :
       Pred.OutgoingCSRLocations) {
    if (OutgoingCSRLocation.isValid()) {
      OutgoingCSRLocation.dump(errs());
      errs() << " ";
    }
  }
  errs() << "\n";
  errs() << "Succ: " << Succ.MBB->getName() << " #" << Succ.MBB->getNumber()
         << " incoming CSR Saved: ";
  for (const CSRSavedLocation &IncomingCSRLocation :
       Succ.IncomingCSRLocations) {
    if (IncomingCSRLocation.isValid()) {
      IncomingCSRLocation.dump(errs());
      errs() << " ";
    }
  }
  errs() << "\n";
}

unsigned CFIInstrInserter::verify(MachineFunction &MF) {
  unsigned ErrorNum = 0;
  for (auto *CurrMBB : depth_first(&MF)) {
    const MBBCFAInfo &CurrMBBInfo = MBBVector[CurrMBB->getNumber()];
    for (MachineBasicBlock *Succ : CurrMBB->successors()) {
      const MBBCFAInfo &SuccMBBInfo = MBBVector[Succ->getNumber()];
      // Check that incoming offset and register values of successors match the
      // outgoing offset and register values of CurrMBB
      if (SuccMBBInfo.IncomingCFAOffset != CurrMBBInfo.OutgoingCFAOffset ||
          SuccMBBInfo.IncomingCFARegister != CurrMBBInfo.OutgoingCFARegister) {
        // Inconsistent offsets/registers are ok for 'noreturn' blocks because
        // we don't generate epilogues inside such blocks.
        if (SuccMBBInfo.MBB->succ_empty() && !SuccMBBInfo.MBB->isReturnBlock())
          continue;
        reportCFAError(CurrMBBInfo, SuccMBBInfo);
        ErrorNum++;
      }
      // Check that IncomingCSRSaved of every successor matches the
      // OutgoingCSRSaved of CurrMBB
      for (unsigned i = 0; i < CurrMBBInfo.OutgoingCSRLocations.size(); ++i) {
        if (!(CurrMBBInfo.OutgoingCSRLocations[i] ==
              SuccMBBInfo.IncomingCSRLocations[i])) {
          reportCSRError(CurrMBBInfo, SuccMBBInfo);
          ErrorNum++;
        }
      }
    }
  }
  return ErrorNum;
}
