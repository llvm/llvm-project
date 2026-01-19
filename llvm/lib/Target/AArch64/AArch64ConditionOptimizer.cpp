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
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "aarch64-condopt"

STATISTIC(NumConditionsAdjusted, "Number of conditions adjusted");

namespace {

class AArch64ConditionOptimizer : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineDominatorTree *DomTree;
  const MachineRegisterInfo *MRI;

public:
  // Stores immediate, compare instruction opcode and branch condition (in this
  // order) of adjusted comparison.
  using CmpInfo = std::tuple<int, unsigned, AArch64CC::CondCode>;

  static char ID;

  AArch64ConditionOptimizer() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  MachineInstr *findSuitableCompare(MachineBasicBlock *MBB);
  CmpInfo adjustCmp(MachineInstr *CmpMI, AArch64CC::CondCode Cmp);
  void modifyCmp(MachineInstr *CmpMI, const CmpInfo &Info);
  bool adjustTo(MachineInstr *CmpMI, AArch64CC::CondCode Cmp, MachineInstr *To,
                int ToImm);
  bool isPureCmp(MachineInstr &CmpMI);
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

// Finds compare instruction that corresponds to supported types of branching.
// Returns the instruction or nullptr on failures or detecting unsupported
// instructions.
MachineInstr *AArch64ConditionOptimizer::findSuitableCompare(
    MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator Term = MBB->getFirstTerminator();
  if (Term == MBB->end())
    return nullptr;

  if (Term->getOpcode() != AArch64::Bcc)
    return nullptr;

  // Since we may modify cmp of this MBB, make sure NZCV does not live out.
  for (auto *SuccBB : MBB->successors())
    if (SuccBB->isLiveIn(AArch64::NZCV))
      return nullptr;

  // Now find the instruction controlling the terminator.
  for (MachineBasicBlock::iterator B = MBB->begin(), It = Term; It != B;) {
    It = prev_nodbg(It, B);
    MachineInstr &I = *It;
    assert(!I.isTerminator() && "Spurious terminator");
    // Check if there is any use of NZCV between CMP and Bcc.
    if (I.readsRegister(AArch64::NZCV, /*TRI=*/nullptr))
      return nullptr;
    switch (I.getOpcode()) {
    // cmp is an alias for subs with a dead destination register.
    case AArch64::SUBSWri:
    case AArch64::SUBSXri:
    // cmn is an alias for adds with a dead destination register.
    case AArch64::ADDSWri:
    case AArch64::ADDSXri: {
      unsigned ShiftAmt = AArch64_AM::getShiftValue(I.getOperand(3).getImm());
      if (!I.getOperand(2).isImm()) {
        LLVM_DEBUG(dbgs() << "Immediate of cmp is symbolic, " << I << '\n');
        return nullptr;
      } else if (I.getOperand(2).getImm() << ShiftAmt >= 0xfff) {
        LLVM_DEBUG(dbgs() << "Immediate of cmp may be out of range, " << I
                          << '\n');
        return nullptr;
      } else if (!MRI->use_nodbg_empty(I.getOperand(0).getReg())) {
        LLVM_DEBUG(dbgs() << "Destination of cmp is not dead, " << I << '\n');
        return nullptr;
      }
      return &I;
    }
    // Prevent false positive case like:
    // cmp      w19, #0
    // cinc     w0, w19, gt
    // ...
    // fcmp     d8, #0.0
    // b.gt     .LBB0_5
    case AArch64::FCMPDri:
    case AArch64::FCMPSri:
    case AArch64::FCMPESri:
    case AArch64::FCMPEDri:

    case AArch64::SUBSWrr:
    case AArch64::SUBSXrr:
    case AArch64::ADDSWrr:
    case AArch64::ADDSXrr:
    case AArch64::FCMPSrr:
    case AArch64::FCMPDrr:
    case AArch64::FCMPESrr:
    case AArch64::FCMPEDrr:
      // Skip comparison instructions without immediate operands.
      return nullptr;
    }
  }
  LLVM_DEBUG(dbgs() << "Flags not defined in " << printMBBReference(*MBB)
                    << '\n');
  return nullptr;
}

// Changes opcode adds <-> subs considering register operand width.
static int getComplementOpc(int Opc) {
  switch (Opc) {
  case AArch64::ADDSWri: return AArch64::SUBSWri;
  case AArch64::ADDSXri: return AArch64::SUBSXri;
  case AArch64::SUBSWri: return AArch64::ADDSWri;
  case AArch64::SUBSXri: return AArch64::ADDSXri;
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

// Changes form of comparison inclusive <-> exclusive.
static AArch64CC::CondCode getAdjustedCmp(AArch64CC::CondCode Cmp) {
  switch (Cmp) {
  case AArch64CC::GT: return AArch64CC::GE;
  case AArch64CC::GE: return AArch64CC::GT;
  case AArch64CC::LT: return AArch64CC::LE;
  case AArch64CC::LE: return AArch64CC::LT;
  default:
    llvm_unreachable("Unexpected condition code");
  }
}

// Transforms GT -> GE, GE -> GT, LT -> LE, LE -> LT by updating comparison
// operator and condition code.
AArch64ConditionOptimizer::CmpInfo AArch64ConditionOptimizer::adjustCmp(
    MachineInstr *CmpMI, AArch64CC::CondCode Cmp) {
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
  if (OldImm == 0 && ((Negative && Correction == 1) ||
                      (!Negative && Correction == -1))) {
    Opc = getComplementOpc(Opc);
  }

  return CmpInfo(NewImm, Opc, getAdjustedCmp(Cmp));
}

// Applies changes to comparison instruction suggested by adjustCmp().
void AArch64ConditionOptimizer::modifyCmp(MachineInstr *CmpMI,
    const CmpInfo &Info) {
  int Imm;
  unsigned Opc;
  AArch64CC::CondCode Cmp;
  std::tie(Imm, Opc, Cmp) = Info;

  MachineBasicBlock *const MBB = CmpMI->getParent();

  // Change immediate in comparison instruction (ADDS or SUBS).
  BuildMI(*MBB, CmpMI, CmpMI->getDebugLoc(), TII->get(Opc))
      .add(CmpMI->getOperand(0))
      .add(CmpMI->getOperand(1))
      .addImm(Imm)
      .add(CmpMI->getOperand(3));
  CmpMI->eraseFromParent();

  // The fact that this comparison was picked ensures that it's related to the
  // first terminator instruction.
  MachineInstr &BrMI = *MBB->getFirstTerminator();

  // Change condition in branch instruction.
  BuildMI(*MBB, BrMI, BrMI.getDebugLoc(), TII->get(AArch64::Bcc))
      .addImm(Cmp)
      .add(BrMI.getOperand(1));
  BrMI.eraseFromParent();

  ++NumConditionsAdjusted;
}

// Parse a condition code returned by analyzeBranch, and compute the CondCode
// corresponding to TBB.
// Returns true if parsing was successful, otherwise false is returned.
static bool parseCond(ArrayRef<MachineOperand> Cond, AArch64CC::CondCode &CC) {
  // A normal br.cond simply has the condition code.
  if (Cond[0].getImm() != -1) {
    assert(Cond.size() == 1 && "Unknown Cond array format");
    CC = (AArch64CC::CondCode)(int)Cond[0].getImm();
    return true;
  }
  return false;
}

// Adjusts one cmp instruction to another one if result of adjustment will allow
// CSE.  Returns true if compare instruction was changed, otherwise false is
// returned.
bool AArch64ConditionOptimizer::adjustTo(MachineInstr *CmpMI,
  AArch64CC::CondCode Cmp, MachineInstr *To, int ToImm)
{
  CmpInfo Info = adjustCmp(CmpMI, Cmp);
  if (std::get<0>(Info) == ToImm && std::get<1>(Info) == To->getOpcode()) {
    modifyCmp(CmpMI, Info);
    return true;
  }
  return false;
}

bool AArch64ConditionOptimizer::isPureCmp(MachineInstr &CmpMI) {
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
  MachineInstr *FirstCmp = nullptr;
  MachineInstr *FirstCSINC = nullptr;
  MachineInstr *SecondCmp = nullptr;
  MachineInstr *SecondCSINC = nullptr;

  // Find two CMP + CSINC pairs
  for (MachineInstr &MI : MBB) {
    switch (MI.getOpcode()) {
    // cmp is an alias for subs with a dead destination register.
    case AArch64::SUBSWri:
    case AArch64::SUBSXri:
    // cmn is an alias for adds with a dead destination register.
    case AArch64::ADDSWri:
    case AArch64::ADDSXri: {
      if (!FirstCmp) {
        FirstCmp = &MI;
      } else if (FirstCSINC && !SecondCmp) {
        SecondCmp = &MI;
      }
      break;
    }

    case AArch64::CSINCWr:
    case AArch64::CSINCXr: {
      // Found a CSINC, ensure it comes after the corresponding comparison
      if (FirstCmp && !FirstCSINC) {
        FirstCSINC = &MI;
      } else if (SecondCmp && !SecondCSINC) {
        SecondCSINC = &MI;
      }
      break;
    }
    }

    if (SecondCSINC)
      break;
  }

  if (!SecondCmp || !SecondCSINC) {
    LLVM_DEBUG(dbgs() << "Didn't find two CMP+CSINC pairs\n");
    return false;
  }

  if (FirstCmp->getOperand(1).getReg() != SecondCmp->getOperand(1).getReg()) {
    LLVM_DEBUG(dbgs() << "CMPs compare different registers\n");
    return false;
  }

  if (!isPureCmp(*FirstCmp) || !isPureCmp(*SecondCmp)) {
    LLVM_DEBUG(dbgs() << "One or both CMPs are not pure\n");
    return false;
  }

  // Check that nothing else modifies the flags between the first CMP and second
  // conditional
  for (auto It = std::next(MachineBasicBlock::iterator(FirstCmp));
       It != std::next(MachineBasicBlock::iterator(SecondCSINC)); ++It) {
    if (&*It != SecondCmp &&
        It->modifiesRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
      LLVM_DEBUG(dbgs() << "Flags modified between CMPs by: " << *It << '\n');
      return false;
    }
  }

  // Check flags aren't read after second conditional within the same block
  for (auto It = std::next(MachineBasicBlock::iterator(SecondCSINC));
       It != MBB.end(); ++It) {
    if (It->readsRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
      LLVM_DEBUG(dbgs() << "Flags read after second CSINC by: " << *It << '\n');
      return false;
    }
  }

  // Since we may modify a cmp in this MBB, make sure NZCV does not live out.
  for (auto *SuccBB : MBB.successors())
    if (SuccBB->isLiveIn(AArch64::NZCV))
      return false;

  // Extract condition codes from both CSINCs (operand 3)
  AArch64CC::CondCode FirstCond =
      (AArch64CC::CondCode)(int)FirstCSINC->getOperand(3).getImm();
  AArch64CC::CondCode SecondCond =
      (AArch64CC::CondCode)(int)SecondCSINC->getOperand(3).getImm();

  const int FirstImm = (int)FirstCmp->getOperand(2).getImm();
  const int SecondImm = (int)SecondCmp->getOperand(2).getImm();

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

    MachineInstr *CmpToAdjust = adjustFirst ? FirstCmp : SecondCmp;
    MachineInstr *CSINCToAdjust = adjustFirst ? FirstCSINC : SecondCSINC;
    AArch64CC::CondCode CondToAdjust = adjustFirst ? FirstCond : SecondCond;
    int TargetImm = adjustFirst ? SecondImm : FirstImm;

    CmpInfo AdjustedInfo = adjustCmp(CmpToAdjust, CondToAdjust);

    if (std::get<0>(AdjustedInfo) == TargetImm &&
        std::get<1>(AdjustedInfo) ==
            (adjustFirst ? SecondCmp : FirstCmp)->getOpcode()) {
      LLVM_DEBUG(dbgs() << "Successfully optimizing intra-block CSINC pair\n");

      // Modify the selected CMP and CSINC
      CmpToAdjust->getOperand(2).setImm(std::get<0>(AdjustedInfo));
      CmpToAdjust->setDesc(TII->get(std::get<1>(AdjustedInfo)));
      CSINCToAdjust->getOperand(3).setImm(std::get<2>(AdjustedInfo));

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

  MachineInstr *HeadCmpMI = findSuitableCompare(&HBB);
  if (!HeadCmpMI) {
    return false;
  }

  MachineInstr *TrueCmpMI = findSuitableCompare(TBB);
  if (!TrueCmpMI) {
    return false;
  }

  AArch64CC::CondCode HeadCmp;
  if (HeadCond.empty() || !parseCond(HeadCond, HeadCmp)) {
    return false;
  }

  AArch64CC::CondCode TrueCmp;
  if (TrueCond.empty() || !parseCond(TrueCond, TrueCmp)) {
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

    CmpInfo HeadCmpInfo = adjustCmp(HeadCmpMI, HeadCmp);
    CmpInfo TrueCmpInfo = adjustCmp(TrueCmpMI, TrueCmp);
    if (std::get<0>(HeadCmpInfo) == std::get<0>(TrueCmpInfo) &&
        std::get<1>(HeadCmpInfo) == std::get<1>(TrueCmpInfo)) {
      modifyCmp(HeadCmpMI, HeadCmpInfo);
      modifyCmp(TrueCmpMI, TrueCmpInfo);
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
