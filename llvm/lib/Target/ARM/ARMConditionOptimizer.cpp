//=- ARMConditionOptimizer.cpp - Remove useless comparisons for ARM -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass tries to make consecutive compares of values use same operands to
// allow CSE pass to remove duplicated instructions.  For this it analyzes
// branches and adjusts comparisons with immediate values by converting:
//  * GE -> GT
//  * GT -> GE
//  * LT -> LE
//  * LE -> LT
// and adjusting immediate values appropriately.  It basically corrects two
// immediate values towards each other to make them equal.
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
//     cmp      r8, #4
//     bgt     .LBB0_3
//     ...
//   .LBB0_3:
//     cmp      r8, #6
//     blt     .LBB0_6
//     ...
//
// Same assembly after the pass:
//
//     ...
//     cmp      r8, #5
//     bge     .LBB0_3
//     ...
//   .LBB0_3:
//     cmp      r8, #5     // <-- CSE pass removes this instruction
//     ble     .LBB0_6
//     ...
//
// Currently only CMP and CMN followed by branches are supported.
//
// TODO: maybe deal with predicated instructions
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMSubtarget.h"
#include "MCTargetDesc/ARMAddressingModes.h"
#include "Utils/ARMBaseInfo.h"
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

#define DEBUG_TYPE "arm-condopt"

STATISTIC(NumConditionsAdjusted, "Number of conditions adjusted");

namespace {

class ARMConditionOptimizer : public MachineFunctionPass {
  const TargetInstrInfo *TII;
  MachineDominatorTree *DomTree;
  const MachineRegisterInfo *MRI;

public:
  // Stores immediate, compare instruction opcode and branch condition (in this
  // order) of adjusted comparison.
  using CmpInfo = std::tuple<int, unsigned, ARMCC::CondCodes>;

  static char ID;

  ARMConditionOptimizer() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  MachineInstr *findSuitableCompare(MachineBasicBlock *MBB);
  CmpInfo adjustCmp(MachineInstr *CmpMI, ARMCC::CondCodes Cmp);
  void modifyCmp(MachineInstr *CmpMI, const CmpInfo &Info);
  bool adjustTo(MachineInstr *CmpMI, ARMCC::CondCodes Cmp, MachineInstr *To,
                int ToImm);
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "ARM Condition Optimizer"; }
};

} // end anonymous namespace

char ARMConditionOptimizer::ID = 0;

INITIALIZE_PASS_BEGIN(ARMConditionOptimizer, "ARM-condopt", "ARM CondOpt Pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(ARMConditionOptimizer, "ARM-condopt", "ARM CondOpt Pass",
                    false, false)

FunctionPass *llvm::createARMConditionOptimizerPass() {
  return new ARMConditionOptimizer();
}

void ARMConditionOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

// Finds compare instruction that corresponds to supported types of branching.
// Returns the instruction or nullptr on failures or detecting unsupported
// instructions.
MachineInstr *
ARMConditionOptimizer::findSuitableCompare(MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator Term = MBB->getFirstTerminator();
  if (Term == MBB->end())
    return nullptr;

  // Accept ARM, Thumb, and Thumb2 conditional branches
  if (Term->getOpcode() != ARM::Bcc && Term->getOpcode() != ARM::tBcc &&
      Term->getOpcode() != ARM::t2Bcc)
    return nullptr;

  // Since we may modify cmp of this MBB, make sure NZCV does not live out.
  for (auto *SuccBB : MBB->successors())
    if (SuccBB->isLiveIn(ARM::CPSR))
      return nullptr;

  // Now find the instruction controlling the terminator.
  for (MachineBasicBlock::iterator B = MBB->begin(), It = Term; It != B;) {
    It = prev_nodbg(It, B);
    MachineInstr &I = *It;
    assert(!I.isTerminator() && "Spurious terminator");
    // Check if there is any use of CPSR between CMP and Bcc.
    if (I.readsRegister(ARM::CPSR, /*TRI=*/nullptr))
      return nullptr;
    switch (I.getOpcode()) {
    // Thumb-1, Thumb-2, and ARM CMP instructions - immediate variants only
    case ARM::tCMPi8:
    case ARM::t2CMPri:
    case ARM::t2CMNri:
    case ARM::CMPri:
    case ARM::CMNri:
      // Only handle unpredicated CMP/CMN instructions
      // ARM and Thumb2 instructions can be predicated, Thumb-1 cannot.
      if (I.getOpcode() != ARM::tCMPi8) {
        int PIdx = I.findFirstPredOperandIdx();
        if (PIdx != -1 && I.getOperand(PIdx).getImm() != (int64_t)ARMCC::AL) {
          LLVM_DEBUG(dbgs()
                     << "Skipping predicated instruction: " << I << '\n');
          return nullptr;
        }
      }

      // Check that the immediate operand is valid
      if (!I.getOperand(1).isImm()) {
        LLVM_DEBUG(dbgs() << "Immediate of cmp/cmn is symbolic, " << I << '\n');
        return nullptr;
      }
      return &I;
    default:
      if (I.modifiesRegister(ARM::CPSR, /*TRI=*/nullptr))
        return nullptr;
    }
  }
  LLVM_DEBUG(dbgs() << "Flags not defined in " << printMBBReference(*MBB)
                    << '\n');
  return nullptr;
}

// Changes opcode cmp <-> cmn considering register operand width.
static int getComplementOpc(int Opc) {
  switch (Opc) {
  // ARM CMN/CMP immediate instructions
  case ARM::CMNri:
    return ARM::CMPri;
  case ARM::CMPri:
    return ARM::CMNri;
  // Thumb CMP immediate instructions - NOTE: Thumb1 doesn't have CMN!
  case ARM::tCMPi8:
    return ARM::INSTRUCTION_LIST_END; // No complement for Thumb1
  // Thumb2 CMN/CMP immediate instructions
  case ARM::t2CMPri:
    return ARM::t2CMNri;
  case ARM::t2CMNri:
    return ARM::t2CMPri;
  default:
    llvm_unreachable("Unexpected opcode");
  }
}

// Changes form of comparison inclusive <-> exclusive.
static ARMCC::CondCodes getAdjustedCmp(ARMCC::CondCodes Cmp) {
  switch (Cmp) {
  case ARMCC::GT:
    return ARMCC::GE;
  case ARMCC::GE:
    return ARMCC::GT;
  case ARMCC::LT:
    return ARMCC::LE;
  case ARMCC::LE:
    return ARMCC::LT;
  case ARMCC::HI:
    return ARMCC::HS;
  case ARMCC::HS:
    return ARMCC::HI;
  case ARMCC::LO:
    return ARMCC::LS;
  case ARMCC::LS:
    return ARMCC::LO;
  default:
    llvm_unreachable("Unexpected condition code");
  }
}

// Transforms GT -> GE, GE -> GT, LT -> LE, LE -> LT by updating comparison
// operator and condition code.
ARMConditionOptimizer::CmpInfo
ARMConditionOptimizer::adjustCmp(MachineInstr *CmpMI, ARMCC::CondCodes Cmp) {
  unsigned Opc = CmpMI->getOpcode();
  unsigned OldOpc = Opc;

  bool IsSigned = Cmp == ARMCC::GT || Cmp == ARMCC::GE || Cmp == ARMCC::LT ||
                  Cmp == ARMCC::LE;

  // CMN (compare with negative immediate) is an alias to ADDS (as
  // "operand - negative" == "operand + positive")
  bool Negative = (Opc == ARM::CMNri || Opc == ARM::t2CMNri);

  int Correction = (Cmp == ARMCC::GT || Cmp == ARMCC::HI) ? 1 : -1;
  // Negate Correction value for comparison with negative immediate (CMN).
  if (Negative) {
    Correction = -Correction;
  }

  const int OldImm = (int)CmpMI->getOperand(1).getImm();
  const int NewImm = std::abs(OldImm + Correction);

  // Handle cmn 1 -> cmp 0, transitions by adjusting compare instruction opcode.
  if (OldImm == 1 && Negative && Correction == -1) {
    // If we are adjusting from -1 to 0, we need to change the opcode.
    Opc = getComplementOpc(Opc);
  }

  // Handle +0 -> -1 transitions by adjusting compare instruction opcode.
  if (OldImm == 0 && Correction == -1) {
    Opc = getComplementOpc(Opc);
  }

  // If we change opcodes, this means we did an unsigned wrap, so return the old
  // cmp for unsigned comparisons.

  // If we have an invalid value, return the old cmp
  if (Opc == ARM::INSTRUCTION_LIST_END || (!IsSigned && Opc != OldOpc))
    return CmpInfo(OldImm, OldOpc, Cmp);

  return CmpInfo(NewImm, Opc, getAdjustedCmp(Cmp));
}

// Applies changes to comparison instruction suggested by adjustCmp().
void ARMConditionOptimizer::modifyCmp(MachineInstr *CmpMI,
                                      const CmpInfo &Info) {
  int Imm;
  unsigned Opc;
  ARMCC::CondCodes Cmp;
  std::tie(Imm, Opc, Cmp) = Info;
  if (Imm == 0) {
    if (Cmp == ARMCC::GE)
      Cmp = ARMCC::PL;
    if (Cmp == ARMCC::LT)
      Cmp = ARMCC::MI;
  }

  MachineBasicBlock *const MBB = CmpMI->getParent();

  // Build the new instruction with the correct format for the target opcode.
  MachineInstrBuilder MIB = BuildMI(*MBB, CmpMI, CmpMI->getDebugLoc(),
                                    TII->get(Opc))
                                .add(CmpMI->getOperand(0)) // Rn
                                .addImm(Imm);              // Immediate

  // Add predicate operands for all CMP/CMN instructions.
  // Even Thumb-1 CMP instructions have predicate operands.
  MIB.add(predOps(ARMCC::AL));

  CmpMI->eraseFromParent();

  // The fact that this comparison was picked ensures that it's related to the
  // first terminator instruction.
  MachineInstr &BrMI = *MBB->getFirstTerminator();

  // Change condition in branch instruction.
  // Rebuild the branch instruction correctly for all subtargets.
  unsigned BranchOpc = BrMI.getOpcode();
  MachineInstrBuilder BranchMIB =
      BuildMI(*MBB, BrMI, BrMI.getDebugLoc(), TII->get(BranchOpc))
          .add(BrMI.getOperand(0)); // Target MBB

  // Add the new condition code.
  BranchMIB.addImm(Cmp);

  // Add the predicate register operand for all branch types.
  // All ARM/Thumb/Thumb2 conditional branches need this.
  BranchMIB.add(BrMI.getOperand(2));

  BrMI.eraseFromParent();

  ++NumConditionsAdjusted;
}

// Parse a condition code returned by analyzeBranch, and compute the CondCode
// corresponding to TBB.
// Returns true if parsing was successful, otherwise false is returned.
static bool parseCond(ArrayRef<MachineOperand> Cond, ARMCC::CondCodes &CC) {
  // A normal br.cond simply has the condition code (size == 2 for ARM/Thumb)
  if (Cond.size() == 2 && Cond[0].isImm()) {
    CC = (ARMCC::CondCodes)(int)Cond[0].getImm();
    return true;
  }
  return false;
}

// Adjusts one cmp instruction to another one if result of adjustment will allow
// CSE.  Returns true if compare instruction was changed, otherwise false is
// returned.
bool ARMConditionOptimizer::adjustTo(MachineInstr *CmpMI, ARMCC::CondCodes Cmp,
                                     MachineInstr *To, int ToImm) {
  CmpInfo Info = adjustCmp(CmpMI, Cmp);
  if (std::get<0>(Info) == ToImm && std::get<1>(Info) == To->getOpcode()) {
    modifyCmp(CmpMI, Info);
    return true;
  }
  return false;
}

static bool isGreaterThan(ARMCC::CondCodes Cmp) {
  return Cmp == ARMCC::GT || Cmp == ARMCC::HI;
}

static bool isLessThan(ARMCC::CondCodes Cmp) {
  return Cmp == ARMCC::LT || Cmp == ARMCC::LO;
}

bool ARMConditionOptimizer::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** ARM Conditional Compares **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  if (skipFunction(MF.getFunction()))
    return false;

  TII = MF.getSubtarget<ARMSubtarget>().getInstrInfo();
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

    SmallVector<MachineOperand, 4> HeadCond;
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    if (TII->analyzeBranch(*HBB, TBB, FBB, HeadCond)) {
      continue;
    }

    // Equivalence check is to skip loops.
    if (!TBB || TBB == HBB) {
      continue;
    }

    SmallVector<MachineOperand, 4> TrueCond;
    MachineBasicBlock *TBB_TBB = nullptr, *TBB_FBB = nullptr;
    if (TII->analyzeBranch(*TBB, TBB_TBB, TBB_FBB, TrueCond)) {
      continue;
    }

    MachineInstr *HeadCmpMI = findSuitableCompare(HBB);
    if (!HeadCmpMI) {
      continue;
    }

    ARMCC::CondCodes HeadCmp;
    if (HeadCond.empty() || !parseCond(HeadCond, HeadCmp)) {
      continue;
    }

    ARMCC::CondCodes TrueCmp;
    if (TrueCond.empty() || !parseCond(TrueCond, TrueCmp)) {
      continue;
    }

    const int HeadImm = (int)HeadCmpMI->getOperand(1).getImm();

    // Convert PL/MI to GE/LT for comparisons with 0.
    if (HeadImm == 0) {
      if (HeadCmp == ARMCC::PL)
        HeadCmp = ARMCC::GE;
      if (HeadCmp == ARMCC::MI)
        HeadCmp = ARMCC::LT;
    }

    int HeadImmTrueValue = HeadImm;

    unsigned HeadOpc = HeadCmpMI->getOpcode();
    if (HeadOpc == ARM::CMNri || HeadOpc == ARM::t2CMNri)
      HeadImmTrueValue = -HeadImmTrueValue;

    // Try to find a suitable compare in TBB, but don't require it yet
    MachineInstr *TrueCmpMI = findSuitableCompare(TBB);

    // If we have a suitable compare in TBB, try the optimization
    if (TrueCmpMI) {
      const int TrueImm = (int)TrueCmpMI->getOperand(1).getImm();

      // Special Case 0.
      if (TrueImm == 0) {
        if (TrueCmp == ARMCC::PL)
          TrueCmp = ARMCC::GE;
        if (TrueCmp == ARMCC::MI)
          TrueCmp = ARMCC::LT;
      }

      int TrueImmTrueValue = TrueImm;

      unsigned TrueOpc = TrueCmpMI->getOpcode();
      if (TrueOpc == ARM::CMNri || TrueOpc == ARM::t2CMNri)
        TrueImmTrueValue = -TrueImmTrueValue;

      if (((isGreaterThan(HeadCmp) && isLessThan(TrueCmp)) ||
           (isLessThan(HeadCmp) && isGreaterThan(TrueCmp))) &&
          std::abs(TrueImmTrueValue - HeadImmTrueValue) == 2) {
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
          Changed = true;
        }
      } else if (((isGreaterThan(HeadCmp) && isGreaterThan(TrueCmp)) ||
                  (isLessThan(HeadCmp) && isLessThan(TrueCmp))) &&
                 std::abs(TrueImmTrueValue - HeadImmTrueValue) == 1) {
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
        bool adjustHeadCond = (HeadImmTrueValue < TrueImmTrueValue);
        if (isLessThan(HeadCmp)) {
          adjustHeadCond = !adjustHeadCond;
        }

        if (adjustHeadCond) {
          Changed |= adjustTo(HeadCmpMI, HeadCmp, TrueCmpMI, TrueImm);
        } else {
          Changed |= adjustTo(TrueCmpMI, TrueCmp, HeadCmpMI, HeadImm);
        }
      }
    }
    // Other transformation cases almost never occur due to generation of < or >
    // comparisons instead of <= and >=.
  }

  return Changed;
}
