//===------ RISCVCFIInstrInserter.cpp - Insert additional CFI instructions
//-----===//
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

#include "RISCV.h"
#include "RISCVMachineFunctionInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-cfi-inserter"

// static cl::opt<bool> VerifyCFI("verify-cfiinstrs",
//     cl::desc("Verify Call Frame Information instructions"),
//     cl::init(false),
//     cl::Hidden);

namespace {
class RISCVCFIInstrInserter : public MachineFunctionPass {
public:
  static char ID;

  RISCVCFIInstrInserter() : MachineFunctionPass(ID) {
    initializeRISCVCFIInstrInserterPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.needsFrameMoves())
      return false;

    if (!MF.getSubtarget().doCSRSavesInRA())
      return false;

    RVFI = MF.getInfo<RISCVMachineFunctionInfo>();
    MBBVector.resize(MF.getNumBlockIDs());
    calculateCFAInfo(MF);

    // if (VerifyCFI) {
    //   if (unsigned ErrorNum = verify(MF))
    //     report_fatal_error("Found " + Twine(ErrorNum) +
    //                        " in/out CFI information errors.");
    // }
    bool insertedCFI = insertCFIInstrs(MF);
    MBBVector.clear();
    return insertedCFI;
  }

private:
#define INVALID_REG UINT_MAX
#define INVALID_OFFSET INT_MAX
  /// contains the location where CSR register is saved.
  /// Registers are recorded by their Dwarf numbers.
  struct CSRLocation {
    bool IsReg = true;
    int Reg = 0;
    int FrameReg = 0;
    int Offset = 0;
    bool isEqual(const CSRLocation &Other) const {
      if (IsReg)
        return Other.IsReg ? (Reg == Other.Reg) : false;
      return !Other.IsReg
                 ? ((Offset == Other.Offset) && FrameReg == Other.FrameReg)
                 : false;
    }
  };

  struct MBBCFAInfo {
    MachineBasicBlock *MBB;
    /// Value of cfa offset valid at basic block entry.
    int IncomingCFAOffset = -1;
    /// Value of cfa offset valid at basic block exit.
    int OutgoingCFAOffset = -1;
    /// Value of cfa register valid at basic block entry.
    int IncomingCFARegister = 0;
    /// Value of cfa register valid at basic block exit.
    int OutgoingCFARegister = 0;
    /// Set of callee saved registers saved at basic block entry.
    SmallVector<CSRLocation> IncomingCSRLocations;
    /// Set of callee saved registers saved at basic block exit.
    SmallVector<CSRLocation> OutgoingCSRLocations;
    /// If in/out cfa offset and register values for this block have already
    /// been set or not.
    bool Processed = false;
  };

  RISCVMachineFunctionInfo *RVFI;
  /// Contains cfa offset and register values valid at entry and exit of basic
  /// blocks.
  std::vector<MBBCFAInfo> MBBVector;

  /// Calculate cfa offset and register values valid at entry and exit for all
  /// basic blocks in a function.
  void calculateCFAInfo(MachineFunction &MF);
  /// Calculate cfa offset and register values valid at basic block exit by
  /// checking the block for CFI instructions. Block's incoming CFA info remains
  /// the same.
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
  /// if needed. The negated value is needed when creating CFI instructions that
  /// set absolute offset.
  int getCorrectCFAOffset(MachineBasicBlock *MBB) {
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
} // namespace

char RISCVCFIInstrInserter::ID = 0;
INITIALIZE_PASS(RISCVCFIInstrInserter, "cfi-instr-inserter",
                "Check CFA info and insert CFI instructions if needed", false,
                false)
FunctionPass *llvm::createRISCVCFIInstrInserter() {
  return new RISCVCFIInstrInserter();
}

void RISCVCFIInstrInserter::calculateCFAInfo(MachineFunction &MF) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  // Initial CFA offset value i.e. the one valid at the beginning of the
  // function.
  int InitialOffset =
      MF.getSubtarget().getFrameLowering()->getInitialCFAOffset(MF);
  // Initial CFA register value i.e. the one valid at the beginning of the
  // function.
  int InitialRegister = TRI.getDwarfRegNum(
      MF.getSubtarget().getFrameLowering()->getInitialCFARegister(MF), true);
  unsigned NumRegs = TRI.getNumSupportedRegs(MF);

  // Initialize MBBMap.
  for (MachineBasicBlock &MBB : MF) {
    MBBCFAInfo &MBBInfo = MBBVector[MBB.getNumber()];
    MBBInfo.MBB = &MBB;
    MBBInfo.IncomingCFAOffset = InitialOffset;
    MBBInfo.OutgoingCFAOffset = InitialOffset;
    MBBInfo.IncomingCFARegister = InitialRegister;
    MBBInfo.OutgoingCFARegister = InitialRegister;
    MBBInfo.IncomingCSRLocations.resize(NumRegs);
    MBBInfo.OutgoingCSRLocations.resize(NumRegs);
  }

  MBBCFAInfo &EntryMBBInfo = MBBVector[MF.front().getNumber()];
  const MCPhysReg *CSRegs = MF.getRegInfo().getCalleeSavedRegs();
  for (int i = 0; CSRegs[i]; ++i) {
    unsigned Reg = TRI.getDwarfRegNum(CSRegs[i], true);
    CSRLocation &CSRLoc = EntryMBBInfo.IncomingCSRLocations[Reg];
    CSRLoc.IsReg = true;
    CSRLoc.Reg = Reg;
  }
  // Set in/out cfa info for all blocks in the function. This traversal is based
  // on the assumption that the first block in the function is the entry block
  // i.e. that it has initial cfa offset and register values as incoming CFA
  // information.
  updateSuccCFAInfo(MBBVector[MF.front().getNumber()]);

  LLVM_DEBUG(dbgs() << "Calculated CFI info for " << MF.getName() << "\n";
             for (MachineBasicBlock &MBB : MF) {
               dbgs() << "BasicBlock: " << MBB.getNumber() << " "
                      << MBB.getName() << "\n";
               dbgs() << "IncomingCSRLocations:\n";
               for (int i = 0; CSRegs[i]; ++i) {
                 int Reg = TRI.getDwarfRegNum(CSRegs[i], true);
                 const CSRLocation &CSRLoc =
                     MBBVector[MBB.getNumber()].IncomingCSRLocations[Reg];
                 dbgs() << "CSR: " << Reg << ", Location: {";
                 dbgs() << "IsReg: " << CSRLoc.IsReg << ", ";
                 dbgs() << "Reg: " << CSRLoc.Reg << ", ";
                 dbgs() << "FrameReg: " << CSRLoc.FrameReg << ", ";
                 dbgs() << "Offset: " << CSRLoc.Offset << "}\n";
               }
               dbgs() << "OutgoingCSRLocations:\n";
               for (int i = 0; CSRegs[i]; ++i) {
                 int Reg = TRI.getDwarfRegNum(CSRegs[i], true);
                 const CSRLocation &CSRLoc =
                     MBBVector[MBB.getNumber()].OutgoingCSRLocations[Reg];
                 dbgs() << "CSR: " << Reg << ", Location: {";
                 dbgs() << "IsReg: " << CSRLoc.IsReg << ", ";
                 dbgs() << "Reg: " << CSRLoc.Reg << ", ";
                 dbgs() << "FrameReg: " << CSRLoc.FrameReg << ", ";
                 dbgs() << "Offset: " << CSRLoc.Offset << "}\n";
               }
             });
}

void RISCVCFIInstrInserter::calculateOutgoingCFAInfo(MBBCFAInfo &MBBInfo) {
  MachineFunction *MF = MBBInfo.MBB->getParent();
  const std::vector<MCCFIInstruction> &Instrs = MF->getFrameInstructions();

  int &OutgoingCFAOffset = MBBInfo.OutgoingCFAOffset;
  int &OutgoingCFARegister = MBBInfo.OutgoingCFARegister;
  SmallVector<CSRLocation> &OutgoingCSRLocations = MBBInfo.OutgoingCSRLocations;

  OutgoingCSRLocations = MBBInfo.IncomingCSRLocations;
  // Determine cfa offset and register set by the block.
  for (MachineInstr &MI : *MBBInfo.MBB) {
    if (MI.isCFIInstruction()) {
      unsigned CFIIndex = MI.getOperand(0).getCFIIndex();
      const MCCFIInstruction &CFI = Instrs[CFIIndex];
      switch (CFI.getOperation()) {
      case MCCFIInstruction::OpDefCfaRegister: {
        int Reg = CFI.getRegister();
        assert(Reg >= 0 && "Negative dwarf register number!");
        OutgoingCFARegister = Reg;
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
        int Reg = CFI.getRegister();
        assert(Reg >= 0 && "Negative dwarf register number!");
        OutgoingCFARegister = Reg;
        OutgoingCFAOffset = CFI.getOffset();
        break;
      }
      case MCCFIInstruction::OpOffset: {
        int Reg = CFI.getRegister();
        assert(Reg >= 0 && "Negative dwarf register number!");
        OutgoingCSRLocations[Reg].Offset = CFI.getOffset();
        OutgoingCSRLocations[Reg].FrameReg = CFI.getOffset();
        OutgoingCSRLocations[Reg].IsReg = false;
        break;
      }
      case MCCFIInstruction::OpEscape: {
        int Reg;
        int FrameReg;
        int64_t Offset;
        bool isRegPlusOffset = RVFI->getCFIInfo(&MI, Reg, FrameReg, Offset);
        if (!isRegPlusOffset) {
          break;
        }
        assert(Reg >= 0 && "Negative dwarf register number!");
        assert(FrameReg >= 0 && "Negative dwarf register number!");
        OutgoingCSRLocations[Reg].IsReg = false;
        OutgoingCSRLocations[Reg].Offset = Offset;
        OutgoingCSRLocations[Reg].FrameReg = FrameReg;
        break;
      }
      case MCCFIInstruction::OpRegister: {
        int Reg = CFI.getRegister();
        assert(Reg >= 0 && "Negative dwarf register number!");
        int Reg2 = CFI.getRegister();
        assert(Reg2 >= 0 && "Negative dwarf register number!");
        OutgoingCSRLocations[Reg].Reg = Reg2;
        OutgoingCSRLocations[Reg].IsReg = true;
        break;
      }
      case MCCFIInstruction::OpRelOffset:
        report_fatal_error(
            "Support for .cfi_rel_offset not implemented! Value of CFA "
            "may be incorrect!\n");
        break;
      case MCCFIInstruction::OpRestore:
        report_fatal_error(
            "Support for .cfi_restore not implemented! Value of CFA "
            "may be incorrect!\n");
        break;
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
        report_fatal_error(
            "Support for cfi_remember_state not implemented! Value of CFA "
            "may be incorrect!\n");
#endif
        break;
      case MCCFIInstruction::OpRestoreState:
        // TODO: Add support for handling cfi_restore_state.
#ifndef NDEBUG
        report_fatal_error(
            "Support for cfi_restore_state not implemented! Value of CFA may "
            "be incorrect!\n");
#endif
        break;
      case MCCFIInstruction::OpUndefined:
      case MCCFIInstruction::OpSameValue:
      case MCCFIInstruction::OpWindowSave:
      case MCCFIInstruction::OpNegateRAState:
      case MCCFIInstruction::OpGnuArgsSize:
        break;
      }
    }
  }

  MBBInfo.Processed = true;
}

void RISCVCFIInstrInserter::updateSuccCFAInfo(MBBCFAInfo &MBBInfo) {
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

bool RISCVCFIInstrInserter::insertCFIInstrs(MachineFunction &MF) {
  const MBBCFAInfo *PrevMBBInfo = &MBBVector[MF.front().getNumber()];
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool InsertedCFIInstr = false;

  BitVector SetDifference;
  for (MachineBasicBlock &MBB : MF) {
    // Skip the first MBB in a function
    if (MBB.getNumber() == MF.front().getNumber())
      continue;

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
      const CSRLocation &OutgoingCSRLoc = PrevMBBInfo->OutgoingCSRLocations[i];
      const CSRLocation &IncomingCSRLoc = MBBInfo.IncomingCSRLocations[i];
      if (IncomingCSRLoc.IsReg && (IncomingCSRLoc.Reg == 0))
        continue;
      if (MBBInfo.IncomingCSRLocations[i].isEqual(OutgoingCSRLoc))
        continue;
      unsigned CFIIndex;
      if (IncomingCSRLoc.IsReg) {
        CFIIndex = MF.addFrameInst(
            MCCFIInstruction::createRegister(nullptr, i, IncomingCSRLoc.Reg));
      } else {
        // CFIIndex = MF.addFrameInst(
        //   MCCFIInstruction::createOffset(nullptr, i, IncomingCSRLoc.Offset)
        //);
        std::string CommentBuffer;
        llvm::raw_string_ostream Comment(CommentBuffer);
        int DwarfRegSP = IncomingCSRLoc.FrameReg;
        int DwarfEHRegNum = i;
        int64_t FixedOffset = IncomingCSRLoc.Offset;
        // Build up the expression (SP + FixedOffset)
        SmallString<64> Expr;
        uint8_t Buffer[16];

        Comment << FixedOffset;
        // 0x11
        Expr.push_back(dwarf::DW_OP_consts);
        Expr.append(Buffer, Buffer + encodeSLEB128(FixedOffset, Buffer));

        // 0x92
        Expr.push_back((uint8_t)dwarf::DW_OP_bregx);
        // 0x02
        Expr.append(Buffer, Buffer + encodeULEB128(DwarfRegSP, Buffer));
        Expr.push_back(0);

        // 0x22
        Expr.push_back((uint8_t)dwarf::DW_OP_plus);
        // Wrap this into DW_CFA_def_cfa.
        SmallString<64> DefCfaExpr;
        // 0x10
        DefCfaExpr.push_back(dwarf::DW_CFA_expression);
        DefCfaExpr.append(Buffer,
                          Buffer + encodeULEB128(DwarfEHRegNum, Buffer));
        DefCfaExpr.append(Buffer, Buffer + encodeULEB128(Expr.size(), Buffer));
        DefCfaExpr.append(Expr.str());
        CFIIndex = MF.addFrameInst(MCCFIInstruction::createEscape(
            nullptr, DefCfaExpr.str(), SMLoc(), Comment.str()));
      }
      BuildMI(*MBBInfo.MBB, MBBI, DL, TII->get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
      InsertedCFIInstr = true;
    }
    // BitVector::apply([](auto x, auto y) { return x & ~y; }, SetDifference,
    //                  PrevMBBInfo->OutgoingCSRSaved,
    //                  MBBInfo.IncomingCSRSaved);
    // for (int Reg : SetDifference.set_bits()) {
    //   unsigned CFIIndex =
    //       MF.addFrameInst(MCCFIInstruction::createRestore(nullptr, Reg));
    //   BuildMI(*MBBInfo.MBB, MBBI, DL,
    //   TII->get(TargetOpcode::CFI_INSTRUCTION))
    //       .addCFIIndex(CFIIndex);
    //   InsertedCFIInstr = true;
    // }

    // BitVector::apply([](auto x, auto y) { return x & ~y; }, SetDifference,
    //                  MBBInfo.IncomingCSRSaved,
    //                  PrevMBBInfo->OutgoingCSRSaved);
    // for (int Reg : SetDifference.set_bits()) {
    //   auto it = CSRLocMap.find(Reg);
    //   assert(it != CSRLocMap.end() && "Reg should have an entry in
    //   CSRLocMap"); unsigned CFIIndex; CSRSavedLocation RO = it->second; if
    //   (!RO.Reg && RO.Offset) {
    //     CFIIndex = MF.addFrameInst(
    //         MCCFIInstruction::createOffset(nullptr, Reg, *RO.Offset));
    //   } else if (RO.Reg && !RO.Offset) {
    //     CFIIndex = MF.addFrameInst(
    //         MCCFIInstruction::createRegister(nullptr, Reg, *RO.Reg));
    //   } else {
    //     llvm_unreachable("RO.Reg and RO.Offset cannot both be
    //     valid/invalid");
    //   }
    //   BuildMI(*MBBInfo.MBB, MBBI, DL,
    //   TII->get(TargetOpcode::CFI_INSTRUCTION))
    //       .addCFIIndex(CFIIndex);
    //   InsertedCFIInstr = true;
    // }

    PrevMBBInfo = &MBBInfo;
  }
  return InsertedCFIInstr;
}

// void RISCVCFIInstrInserter::reportCFAError(const MBBCFAInfo &Pred,
//                                       const MBBCFAInfo &Succ) {
//   errs() << "*** Inconsistent CFA register and/or offset between pred and
//   succ "
//             "***\n";
//   errs() << "Pred: " << Pred.MBB->getName() << " #" << Pred.MBB->getNumber()
//          << " in " << Pred.MBB->getParent()->getName()
//          << " outgoing CFA Reg:" << Pred.OutgoingCFARegister << "\n";
//   errs() << "Pred: " << Pred.MBB->getName() << " #" << Pred.MBB->getNumber()
//          << " in " << Pred.MBB->getParent()->getName()
//          << " outgoing CFA Offset:" << Pred.OutgoingCFAOffset << "\n";
//   errs() << "Succ: " << Succ.MBB->getName() << " #" << Succ.MBB->getNumber()
//          << " incoming CFA Reg:" << Succ.IncomingCFARegister << "\n";
//   errs() << "Succ: " << Succ.MBB->getName() << " #" << Succ.MBB->getNumber()
//          << " incoming CFA Offset:" << Succ.IncomingCFAOffset << "\n";
// }
//
// void RISCVCFIInstrInserter::reportCSRError(const MBBCFAInfo &Pred,
//                                       const MBBCFAInfo &Succ) {
//   errs() << "*** Inconsistent CSR Saved between pred and succ in function "
//          << Pred.MBB->getParent()->getName() << " ***\n";
//   errs() << "Pred: " << Pred.MBB->getName() << " #" << Pred.MBB->getNumber()
//          << " outgoing CSR Saved: ";
//   for (int Reg : Pred.OutgoingCSRSaved.set_bits())
//     errs() << Reg << " ";
//   errs() << "\n";
//   errs() << "Succ: " << Succ.MBB->getName() << " #" << Succ.MBB->getNumber()
//          << " incoming CSR Saved: ";
//   for (int Reg : Succ.IncomingCSRSaved.set_bits())
//     errs() << Reg << " ";
//   errs() << "\n";
// }

// unsigned RISCVCFIInstrInserter::verify(MachineFunction &MF) {
//   unsigned ErrorNum = 0;
//   for (auto *CurrMBB : depth_first(&MF)) {
//     const MBBCFAInfo &CurrMBBInfo = MBBVector[CurrMBB->getNumber()];
//     for (MachineBasicBlock *Succ : CurrMBB->successors()) {
//       const MBBCFAInfo &SuccMBBInfo = MBBVector[Succ->getNumber()];
//       // Check that incoming offset and register values of successors match
//       the
//       // outgoing offset and register values of CurrMBB
//       if (SuccMBBInfo.IncomingCFAOffset != CurrMBBInfo.OutgoingCFAOffset ||
//           SuccMBBInfo.IncomingCFARegister != CurrMBBInfo.OutgoingCFARegister)
//           {
//         // Inconsistent offsets/registers are ok for 'noreturn' blocks
//         because
//         // we don't generate epilogues inside such blocks.
//         if (SuccMBBInfo.MBB->succ_empty() &&
//         !SuccMBBInfo.MBB->isReturnBlock())
//           continue;
//         reportCFAError(CurrMBBInfo, SuccMBBInfo);
//         ErrorNum++;
//       }
//       // Check that IncomingCSRSaved of every successor matches the
//       // OutgoingCSRSaved of CurrMBB
//       if (SuccMBBInfo.IncomingCSRSaved != CurrMBBInfo.OutgoingCSRSaved) {
//         reportCSRError(CurrMBBInfo, SuccMBBInfo);
//         ErrorNum++;
//       }
//     }
//   }
//   return ErrorNum;
// }
