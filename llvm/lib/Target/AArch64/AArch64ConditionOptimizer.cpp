//=- AArch64ConditionOptimizer.cpp - Remove useless comparisons for AArch64 -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// This pass tries to make consecutive comparisons of values use the same
// operands to allow the CSE pass to remove duplicate instructions. It adjusts
// comparisons with immediate values by converting between inclusive and
// exclusive forms (GE <-> GT, LE <-> LT) and correcting immediate values to
// make them equal.
//
// The pass handles:
//  * Cross-block: SUBS/ADDS followed by conditional branches
//  * Intra-block: CSINC conditional instructions
//
//
// Consider the following example in C:
//
//   if ((a < 5 && ...) || (a > 5 && ...)) {
//        ~~~~~             ~~~~~
//          ^                 ^
//          x                 y
//
// Here both "x" and "y" expressions compare "a" with "5".  When "x" evaluates
// to "false", "y" can just check flags set by the first comparison.  As a
// result of the canonicalization employed by
// SelectionDAGBuilder::visitSwitchCase, DAGCombine, and other target-specific
// code, assembly ends up in the form that is not CSE friendly:
//
//     ...
//     cmp      w8, #4
//     b.gt     .LBB0_3
//     ...
//   .LBB0_3:
//     cmp      w8, #6
//     b.lt     .LBB0_6
//     ...
//
// Same assembly after the pass:
//
//     ...
//     cmp      w8, #5
//     b.ge     .LBB0_3
//     ...
//   .LBB0_3:
//     cmp      w8, #5     // <-- CSE pass removes this instruction
//     b.le     .LBB0_6
//     ...
//
// See optimizeCrossBlock() and optimizeIntraBlock() for implementation details.
//
// TODO: maybe handle TBNZ/TBZ the same way as CMP when used instead for "a < 0"
// TODO: For cross-block:
//   - handle other conditional instructions (e.g. CSET)
//   - allow second branching to be anything if it doesn't require adjusting
// TODO: For intra-block:
//   - handle CINC and CSET (CSINC aliases) as their conditions are inverted
//   compared to CSINC.
//   - handle other non-CSINC conditional instructions
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>

using namespace llvm;

#define DEBUG_TYPE "aarch64-condopt"

STATISTIC(NumConditionsAdjusted, "Number of conditions adjusted");

namespace {

/// Represents the result of adjusting a comparison instruction.
struct CmpAdjustment {
  int Imm;
  unsigned Opc;
  AArch64CC::CondCode CC;
};

class AArch64ConditionOptimizer : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineDominatorTree *DomTree;
  const MachineRegisterInfo *MRI;

public:
  static char ID;

  AArch64ConditionOptimizer() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool canAdjustCmp(MachineInstr &CmpMI);
  bool registersMatch(MachineInstr *FirstMI, MachineInstr *SecondMI);
  bool nzcvLivesOut(MachineBasicBlock *MBB);
  MachineInstr *getBccTerminator(MachineBasicBlock *MBB);
  MachineInstr *findAdjustableCmp(MachineInstr *CondMI);
  CmpAdjustment getAdjustedCmpInfo(MachineInstr *CmpMI,
                                   AArch64CC::CondCode Cmp);
  void updateCmpInstr(MachineInstr *CmpMI, int NewImm, unsigned NewOpc);
  void updateCondInstr(MachineInstr *CondMI, AArch64CC::CondCode NewCC);
  bool adjustTo(MachineInstr *CmpMI, AArch64CC::CondCode Cmp, MachineInstr *To,
                int ToImm);
  bool optimizeIntraBlock(MachineBasicBlock &MBB);
  bool optimizeCrossBlock(MachineBasicBlock &HBB);
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Condition Optimizer";
  }
};

} // end anonymous namespace

char AArch64ConditionOptimizer::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64ConditionOptimizer, "aarch64-condopt",
                      "AArch64 CondOpt Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AArch64ConditionOptimizer, "aarch64-condopt",
                    "AArch64 CondOpt Pass", false, false)

FunctionPass *llvm::createAArch64ConditionOptimizerPass() {
  return new AArch64ConditionOptimizer();
}

void AArch64ConditionOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

// Verify that the MI's immediate is adjustable and it only sets flags (pure
// cmp)
bool AArch64ConditionOptimizer::canAdjustCmp(MachineInstr &CmpMI) {
  unsigned ShiftAmt = AArch64_AM::getShiftValue(CmpMI.getOperand(3).getImm());
  if (!CmpMI.getOperand(2).isImm()) {
    LLVM_DEBUG(dbgs() << "Immediate of cmp is symbolic, " << CmpMI << '\n');
    return false;
  } else if (CmpMI.getOperand(2).getImm() << ShiftAmt >= 0xfff) {
    LLVM_DEBUG(dbgs() << "Immediate of cmp may be out of range, " << CmpMI
                      << '\n');
    return false;
  } else if (!MRI->use_nodbg_empty(CmpMI.getOperand(0).getReg())) {
    LLVM_DEBUG(dbgs() << "Destination of cmp is not dead, " << CmpMI << '\n');
    return false;
  }

  return true;
}

// Ensure both compare MIs use the same register, tracing through copies.
bool AArch64ConditionOptimizer::registersMatch(MachineInstr *FirstMI,
                                               MachineInstr *SecondMI) {
  Register FirstReg = FirstMI->getOperand(1).getReg();
  Register SecondReg = SecondMI->getOperand(1).getReg();
  Register FirstCmpReg =
      FirstReg.isVirtual() ? TRI->lookThruCopyLike(FirstReg, MRI) : FirstReg;
  Register SecondCmpReg =
      SecondReg.isVirtual() ? TRI->lookThruCopyLike(SecondReg, MRI) : SecondReg;
  if (FirstCmpReg != SecondCmpReg) {
    LLVM_DEBUG(dbgs() << "CMPs compare different registers\n");
    return false;
  }

  return true;
}

// Check if NZCV lives out to any successor block.
bool AArch64ConditionOptimizer::nzcvLivesOut(MachineBasicBlock *MBB) {
  for (auto *SuccBB : MBB->successors()) {
    if (SuccBB->isLiveIn(AArch64::NZCV)) {
      LLVM_DEBUG(dbgs() << "NZCV live into successor "
                        << printMBBReference(*SuccBB) << " from "
                        << printMBBReference(*MBB) << '\n');
      return true;
    }
  }
  return false;
}

// Returns true if the opcode is a comparison instruction (CMP/CMN).
// CMP is an alias for SUBS with a dead destination register.
// CMN is an alias for ADDS with a dead destination register.
static bool isCmpInstruction(unsigned Opc) {
  switch (Opc) {
  case AArch64::SUBSWri:
  case AArch64::SUBSXri:
  case AArch64::ADDSWri:
  case AArch64::ADDSXri:
    return true;
  default:
    return false;
  }
}

static bool isCSINCInstruction(unsigned Opc) {
  return Opc == AArch64::CSINCWr || Opc == AArch64::CSINCXr;
}

// Returns the Bcc terminator if present, otherwise nullptr.
MachineInstr *
AArch64ConditionOptimizer::getBccTerminator(MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator Term = MBB->getFirstTerminator();
  if (Term == MBB->end()) {
    LLVM_DEBUG(dbgs() << "No terminator in " << printMBBReference(*MBB)
                      << '\n');
    return nullptr;
  }

  if (Term->getOpcode() != AArch64::Bcc) {
    LLVM_DEBUG(dbgs() << "Non-Bcc terminator in " << printMBBReference(*MBB)
                      << ": " << *Term);
    return nullptr;
  }

  return &*Term;
}

// Find the CMP instruction controlling the given conditional instruction
// (Bcc or CSINC) that can be adjusted for CSE optimization. Searches backward
// from CondMI, ensuring no NZCV interference. Returns nullptr if no suitable
// CMP is found or if adjustments are not safe.
MachineInstr *
AArch64ConditionOptimizer::findAdjustableCmp(MachineInstr *CondMI) {
  assert(CondMI && "CondMI cannot be null");

  MachineBasicBlock *MBB = CondMI->getParent();

  // Search backward from the conditional to find the instruction controlling
  // it.
  for (MachineBasicBlock::iterator B = MBB->begin(),
                                   It = MachineBasicBlock::iterator(CondMI);
       It != B;) {
    It = prev_nodbg(It, B);
    MachineInstr &I = *It;
    assert(!I.isTerminator() && "Spurious terminator");

    // Check if there is any use of NZCV between CMP and conditional.
    if (I.readsRegister(AArch64::NZCV, /*TRI=*/nullptr))
      return nullptr;

    if (isCmpInstruction(I.getOpcode())) {
      if (!canAdjustCmp(I)) {
        return nullptr;
      }
      return &I;
    }

    // Check for non-cmp instructions that modify NZCV
    if (I.modifiesRegister(AArch64::NZCV, /*TRI=*/nullptr))
      return nullptr;
  }
  LLVM_DEBUG(dbgs() << "Flags not defined in " << printMBBReference(*MBB)
                    << '\n');
  return nullptr;
}

// Changes opcode adds <-> subs considering register operand width.
static int getComplementOpc(int Opc) {
  switch (Opc) {
  case AArch64::ADDSWri:
    return AArch64::SUBSWri;
  case AArch64::ADDSXri:
    return AArch64::SUBSXri;
  case AArch64::SUBSWri:
    return AArch64::ADDSWri;
  case AArch64::SUBSXri:
    return AArch64::ADDSXri;
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

// Changes form of comparison inclusive <-> exclusive.
static AArch64CC::CondCode getAdjustedCmp(AArch64CC::CondCode Cmp) {
  switch (Cmp) {
  case AArch64CC::GT:
    return AArch64CC::GE;
  case AArch64CC::GE:
    return AArch64CC::GT;
  case AArch64CC::LT:
    return AArch64CC::LE;
  case AArch64CC::LE:
    return AArch64CC::LT;
  default:
    llvm_unreachable("Unexpected condition code");
  }
}

// Returns the adjusted immediate, opcode, and condition code for switching
// between inclusive/exclusive forms (GT <-> GE, LT <-> LE).
CmpAdjustment
AArch64ConditionOptimizer::getAdjustedCmpInfo(MachineInstr *CmpMI,
                                              AArch64CC::CondCode Cmp) {
  unsigned Opc = CmpMI->getOpcode();

  // CMN (compare with negative immediate) is an alias to ADDS (as
  // "operand - negative" == "operand + positive")
  bool Negative = (Opc == AArch64::ADDSWri || Opc == AArch64::ADDSXri);

  int Correction = (Cmp == AArch64CC::GT) ? 1 : -1;
  // Negate Correction value for comparison with negative immediate (CMN).
  if (Negative) {
    Correction = -Correction;
  }

  const int OldImm = (int)CmpMI->getOperand(2).getImm();
  const int NewImm = std::abs(OldImm + Correction);

  // Handle +0 -> -1 and -0 -> +1 (CMN with 0 immediate) transitions by
  // adjusting compare instruction opcode.
  if (OldImm == 0 &&
      ((Negative && Correction == 1) || (!Negative && Correction == -1))) {
    Opc = getComplementOpc(Opc);
  }

  return {NewImm, Opc, getAdjustedCmp(Cmp)};
}

// Modifies a comparison instruction's immediate and opcode.
void AArch64ConditionOptimizer::updateCmpInstr(MachineInstr *CmpMI, int NewImm,
                                               unsigned NewOpc) {
  CmpMI->getOperand(2).setImm(NewImm);
  CmpMI->setDesc(TII->get(NewOpc));
}

// Modifies the condition code of a conditional instruction.
void AArch64ConditionOptimizer::updateCondInstr(MachineInstr *CondMI,
                                                AArch64CC::CondCode NewCC) {
  unsigned CondOpIdx;
  switch (CondMI->getOpcode()) {
  case AArch64::Bcc:
    CondOpIdx = 0;
    break;
  case AArch64::CSINCWr:
  case AArch64::CSINCXr:
    CondOpIdx = 3;
    break;
  default:
    llvm_unreachable("Unsupported conditional instruction");
  }
  CondMI->getOperand(CondOpIdx).setImm(NewCC);
  ++NumConditionsAdjusted;
}

// Extracts the condition code from the result of analyzeBranch.
// Returns the CondCode or Invalid if parsing fails.
static AArch64CC::CondCode parseCondCode(ArrayRef<MachineOperand> Cond) {
  // A normal br.cond simply has the condition code.
  if (Cond[0].getImm() != -1) {
    assert(Cond.size() == 1 && "Unknown Cond array format");
    return (AArch64CC::CondCode)(int)Cond[0].getImm();
  }
  return AArch64CC::CondCode::Invalid;
}

// Adjusts one cmp instruction to another one if result of adjustment will allow
// CSE.  Returns true if compare instruction was changed, otherwise false is
// returned.
bool AArch64ConditionOptimizer::adjustTo(MachineInstr *CmpMI,
                                         AArch64CC::CondCode Cmp,
                                         MachineInstr *To, int ToImm) {
  CmpAdjustment Adj = getAdjustedCmpInfo(CmpMI, Cmp);
  if (Adj.Imm == ToImm && Adj.Opc == To->getOpcode()) {
    MachineBasicBlock *MBB = CmpMI->getParent();
    updateCmpInstr(CmpMI, Adj.Imm, Adj.Opc);
    MachineInstr &BrMI = *MBB->getFirstTerminator();
    updateCondInstr(&BrMI, Adj.CC);
    return true;
  }
  return false;
}

// This function transforms two CMP+CSINC pairs within the same basic block
// when both conditions are the same (GT/GT or LT/LT) and immediates differ
// by 1.
//
// Example transformation:
//   cmp  w8, #10
//   csinc w9, w0, w1, gt     ; w9 = (w8 > 10) ? w0 : w1+1
//   cmp  w8, #9
//   csinc w10, w0, w1, gt    ; w10 = (w8 > 9) ? w0 : w1+1
//
// Into:
//   cmp  w8, #10
//   csinc w9, w0, w1, gt     ; w9 = (w8 > 10) ? w0 : w1+1
//   csinc w10, w0, w1, ge    ; w10 = (w8 >= 10) ? w0 : w1+1
//
// The second CMP is eliminated, enabling CSE to remove the redundant
// comparison.
bool AArch64ConditionOptimizer::optimizeIntraBlock(MachineBasicBlock &MBB) {
  MachineInstr *FirstCSINC = nullptr;
  MachineInstr *SecondCSINC = nullptr;

  // Find two CSINC instructions
  for (MachineInstr &MI : MBB) {
    if (isCSINCInstruction(MI.getOpcode())) {
      if (!FirstCSINC) {
        FirstCSINC = &MI;
      } else if (!SecondCSINC) {
        SecondCSINC = &MI;
        break; // Found both
      }
    }
  }

  if (!FirstCSINC || !SecondCSINC) {
    return false;
  }

  // Since we may modify cmps in this MBB, make sure NZCV does not live out.
  if (nzcvLivesOut(&MBB))
    return false;

  // Find the CMPs controlling each CSINC
  MachineInstr *FirstCmpMI = findAdjustableCmp(FirstCSINC);
  MachineInstr *SecondCmpMI = findAdjustableCmp(SecondCSINC);
  if (!FirstCmpMI || !SecondCmpMI)
    return false;

  // Ensure we have two distinct CMPs
  if (FirstCmpMI == SecondCmpMI) {
    LLVM_DEBUG(dbgs() << "Both CSINCs controlled by same CMP\n");
    return false;
  }

  if (!registersMatch(FirstCmpMI, SecondCmpMI))
    return false;

  // Ensure NZCV isn't clobbered between first CMP and second conditional
  // (Second CMP is allowed to set NZCV since it'll be eliminated)
  for (auto It = std::next(MachineBasicBlock::iterator(FirstCmpMI));
       It != std::next(MachineBasicBlock::iterator(SecondCSINC)); ++It) {
    if (&*It != SecondCmpMI &&
        It->modifiesRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
      LLVM_DEBUG(dbgs() << "Flags modified between CMPs by: " << *It << '\n');
      return false;
    }
  }

  // Ensure NZCV isn't read after second conditional within the same block
  for (auto It = std::next(MachineBasicBlock::iterator(SecondCSINC));
       It != MBB.end(); ++It) {
    if (It->readsRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
      LLVM_DEBUG(dbgs() << "Flags read after second CSINC by: " << *It << '\n');
      return false;
    }
  }

  // Extract condition codes from both CSINCs (operand 3)
  AArch64CC::CondCode FirstCond =
      (AArch64CC::CondCode)(int)FirstCSINC->getOperand(3).getImm();
  AArch64CC::CondCode SecondCond =
      (AArch64CC::CondCode)(int)SecondCSINC->getOperand(3).getImm();

  const int FirstImm = (int)FirstCmpMI->getOperand(2).getImm();
  const int SecondImm = (int)SecondCmpMI->getOperand(2).getImm();

  LLVM_DEBUG(dbgs() << "Comparing intra-block CSINCs: "
                    << AArch64CC::getCondCodeName(FirstCond) << " #" << FirstImm
                    << " and " << AArch64CC::getCondCodeName(SecondCond) << " #"
                    << SecondImm << '\n');

  // Check if both conditions are the same and immediates differ by 1
  if (((FirstCond == AArch64CC::GT && SecondCond == AArch64CC::GT) ||
       (FirstCond == AArch64CC::LT && SecondCond == AArch64CC::LT)) &&
      std::abs(SecondImm - FirstImm) == 1) {
    // Pick which comparison to adjust to match the other
    // For GT: adjust the one with smaller immediate
    // For LT: adjust the one with larger immediate
    bool adjustFirst = (FirstImm < SecondImm);
    if (FirstCond == AArch64CC::LT) {
      adjustFirst = !adjustFirst;
    }

    MachineInstr *CmpToAdjust = adjustFirst ? FirstCmpMI : SecondCmpMI;
    MachineInstr *CSINCToAdjust = adjustFirst ? FirstCSINC : SecondCSINC;
    AArch64CC::CondCode CondToAdjust = adjustFirst ? FirstCond : SecondCond;
    int TargetImm = adjustFirst ? SecondImm : FirstImm;

    CmpAdjustment Adj = getAdjustedCmpInfo(CmpToAdjust, CondToAdjust);

    if (Adj.Imm == TargetImm &&
        Adj.Opc == (adjustFirst ? SecondCmpMI : FirstCmpMI)->getOpcode()) {
      LLVM_DEBUG(dbgs() << "Successfully optimizing intra-block CSINC pair\n");

      // Modify the selected CMP and CSINC
      updateCmpInstr(CmpToAdjust, Adj.Imm, Adj.Opc);
      updateCondInstr(CSINCToAdjust, Adj.CC);

      return true;
    }
  }

  return false;
}

// Optimize across blocks
bool AArch64ConditionOptimizer::optimizeCrossBlock(MachineBasicBlock &HBB) {
  SmallVector<MachineOperand, 4> HeadCond;
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  if (TII->analyzeBranch(HBB, TBB, FBB, HeadCond)) {
    return false;
  }

  // Equivalence check is to skip loops.
  if (!TBB || TBB == &HBB) {
    return false;
  }

  SmallVector<MachineOperand, 4> TrueCond;
  MachineBasicBlock *TBB_TBB = nullptr, *TBB_FBB = nullptr;
  if (TII->analyzeBranch(*TBB, TBB_TBB, TBB_FBB, TrueCond)) {
    return false;
  }

  if (HeadCond.empty() || TrueCond.empty())
    return false;

  MachineInstr *HeadBrMI = getBccTerminator(&HBB);
  MachineInstr *TrueBrMI = getBccTerminator(TBB);
  if (!HeadBrMI || !TrueBrMI)
    return false;

  // Since we may modify cmps in these blocks, make sure NZCV does not live out.
  if (nzcvLivesOut(&HBB) || nzcvLivesOut(TBB))
    return false;

  MachineInstr *HeadCmpMI = findAdjustableCmp(HeadBrMI);
  MachineInstr *TrueCmpMI = findAdjustableCmp(TrueBrMI);
  if (!HeadCmpMI || !TrueCmpMI)
    return false;

  if (!registersMatch(HeadCmpMI, TrueCmpMI)) {
    return false;
  }

  AArch64CC::CondCode HeadCmp = parseCondCode(HeadCond);
  AArch64CC::CondCode TrueCmp = parseCondCode(TrueCond);
  if (HeadCmp == AArch64CC::CondCode::Invalid ||
      TrueCmp == AArch64CC::CondCode::Invalid) {
    return false;
  }

  const int HeadImm = (int)HeadCmpMI->getOperand(2).getImm();
  const int TrueImm = (int)TrueCmpMI->getOperand(2).getImm();

  LLVM_DEBUG(dbgs() << "Head branch:\n");
  LLVM_DEBUG(dbgs() << "\tcondition: " << AArch64CC::getCondCodeName(HeadCmp)
                    << '\n');
  LLVM_DEBUG(dbgs() << "\timmediate: " << HeadImm << '\n');

  LLVM_DEBUG(dbgs() << "True branch:\n");
  LLVM_DEBUG(dbgs() << "\tcondition: " << AArch64CC::getCondCodeName(TrueCmp)
                    << '\n');
  LLVM_DEBUG(dbgs() << "\timmediate: " << TrueImm << '\n');

  if (((HeadCmp == AArch64CC::GT && TrueCmp == AArch64CC::LT) ||
       (HeadCmp == AArch64CC::LT && TrueCmp == AArch64CC::GT)) &&
      std::abs(TrueImm - HeadImm) == 2) {
    // This branch transforms machine instructions that correspond to
    //
    // 1) (a > {TrueImm} && ...) || (a < {HeadImm} && ...)
    // 2) (a < {TrueImm} && ...) || (a > {HeadImm} && ...)
    //
    // into
    //
    // 1) (a >= {NewImm} && ...) || (a <= {NewImm} && ...)
    // 2) (a <= {NewImm} && ...) || (a >= {NewImm} && ...)

    CmpAdjustment HeadAdj = getAdjustedCmpInfo(HeadCmpMI, HeadCmp);
    CmpAdjustment TrueAdj = getAdjustedCmpInfo(TrueCmpMI, TrueCmp);
    if (HeadAdj.Imm == TrueAdj.Imm && HeadAdj.Opc == TrueAdj.Opc) {
      updateCmpInstr(HeadCmpMI, HeadAdj.Imm, HeadAdj.Opc);
      updateCondInstr(HeadBrMI, HeadAdj.CC);
      updateCmpInstr(TrueCmpMI, TrueAdj.Imm, TrueAdj.Opc);
      updateCondInstr(TrueBrMI, TrueAdj.CC);
      return true;
    }
  } else if (((HeadCmp == AArch64CC::GT && TrueCmp == AArch64CC::GT) ||
              (HeadCmp == AArch64CC::LT && TrueCmp == AArch64CC::LT)) &&
             std::abs(TrueImm - HeadImm) == 1) {
    // This branch transforms machine instructions that correspond to
    //
    // 1) (a > {TrueImm} && ...) || (a > {HeadImm} && ...)
    // 2) (a < {TrueImm} && ...) || (a < {HeadImm} && ...)
    //
    // into
    //
    // 1) (a <= {NewImm} && ...) || (a >  {NewImm} && ...)
    // 2) (a <  {NewImm} && ...) || (a >= {NewImm} && ...)

    // GT -> GE transformation increases immediate value, so picking the
    // smaller one; LT -> LE decreases immediate value so invert the choice.
    bool adjustHeadCond = (HeadImm < TrueImm);
    if (HeadCmp == AArch64CC::LT) {
      adjustHeadCond = !adjustHeadCond;
    }

    if (adjustHeadCond) {
      return adjustTo(HeadCmpMI, HeadCmp, TrueCmpMI, TrueImm);
    } else {
      return adjustTo(TrueCmpMI, TrueCmp, HeadCmpMI, HeadImm);
    }
  }
  // Other transformation cases almost never occur due to generation of < or >
  // comparisons instead of <= and >=.

  return false;
}

bool AArch64ConditionOptimizer::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** AArch64 Conditional Compares **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  if (skipFunction(MF.getFunction()))
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MRI = &MF.getRegInfo();

  bool Changed = false;

  // Visit blocks in dominator tree pre-order. The pre-order enables multiple
  // cmp-conversions from the same head block.
  // Note that updateDomTree() modifies the children of the DomTree node
  // currently being visited. The df_iterator supports that; it doesn't look at
  // child_begin() / child_end() until after a node has been visited.
  for (MachineDomTreeNode *I : depth_first(DomTree)) {
    MachineBasicBlock *HBB = I->getBlock();
    Changed |= optimizeIntraBlock(*HBB);
    Changed |= optimizeCrossBlock(*HBB);
  }

  return Changed;
}
