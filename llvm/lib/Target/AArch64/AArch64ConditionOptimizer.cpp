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
//  * Intra-block: Select-family conditional instructions
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
//   - allow second branching to be anything if it doesn't require adjusting
//
// Cross-block optimizeCrossBlock() handles four head/true-successor
// combinations:
//   Bcc    (head) + Bcc    (true) -- original case
//   Select (head) + Bcc    (true) -- head ends with CSEL/CSET/etc.
//   Bcc    (head) + Select (true) -- true-successor ends with CSEL/CSET/etc.
//   Select (head) + Select (true) -- both blocks end with a select
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
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

/// Bundles the parameters needed to adjust a comparison instruction.
struct CmpInfo {
  int Imm;
  unsigned Opc;
  AArch64CC::CondCode CC;
};

class AArch64ConditionOptimizerImpl {
  /// Represents a comparison instruction paired with its consuming
  /// conditional instruction
  struct CmpCondPair {
    MachineInstr *CmpMI;
    MachineInstr *CondMI;
    AArch64CC::CondCode CC;

    int getImm() const { return CmpMI->getOperand(2).getImm(); }
    unsigned getOpc() const { return CmpMI->getOpcode(); }
  };

  const AArch64InstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineDominatorTree *DomTree;
  const MachineRegisterInfo *MRI;

public:
  bool run(MachineFunction &MF, MachineDominatorTree &MDT);

private:
  bool canAdjustCmp(MachineInstr &CmpMI);
  bool registersMatch(MachineInstr *FirstMI, MachineInstr *SecondMI);
  bool nzcvLivesOut(MachineBasicBlock *MBB);
  MachineInstr *getBccTerminator(MachineBasicBlock *MBB);
  MachineInstr *findAdjustableCmp(MachineInstr *CondMI);
  CmpInfo getAdjustedCmpInfo(MachineInstr *CmpMI, AArch64CC::CondCode Cmp);
  void updateCmpInstr(MachineInstr *CmpMI, int NewImm, unsigned NewOpc);
  void updateCondInstr(MachineInstr *CondMI, AArch64CC::CondCode NewCC);
  void applyCmpAdjustment(CmpCondPair &Pair, const CmpInfo &Info);
  bool commitPendingPair(std::optional<CmpCondPair> &PendingPair,
                         SmallDenseMap<Register, CmpCondPair> &PairsByReg);
  bool tryOptimizePair(CmpCondPair &First, CmpCondPair &Second);
  bool optimizeIntraBlock(MachineBasicBlock &MBB);
  bool optimizeCrossBlock(MachineBasicBlock &HBB);
  std::pair<MachineInstr *, AArch64CC::CondCode>
  findCondConsumer(MachineBasicBlock *MBB);
};

class AArch64ConditionOptimizerLegacy : public MachineFunctionPass {
public:
  static char ID;
  AArch64ConditionOptimizerLegacy() : MachineFunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AArch64 Condition Optimizer";
  }
};

} // end anonymous namespace

char AArch64ConditionOptimizerLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64ConditionOptimizerLegacy, "aarch64-condopt",
                      "AArch64 CondOpt Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(AArch64ConditionOptimizerLegacy, "aarch64-condopt",
                    "AArch64 CondOpt Pass", false, false)

FunctionPass *llvm::createAArch64ConditionOptimizerLegacyPass() {
  return new AArch64ConditionOptimizerLegacy();
}

void AArch64ConditionOptimizerLegacy::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

// Verify that the MI's immediate is adjustable and it only sets flags (pure
// cmp)
bool AArch64ConditionOptimizerImpl::canAdjustCmp(MachineInstr &CmpMI) {
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
bool AArch64ConditionOptimizerImpl::registersMatch(MachineInstr *FirstMI,
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
bool AArch64ConditionOptimizerImpl::nzcvLivesOut(MachineBasicBlock *MBB) {
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
static bool isCmpInstruction(unsigned Opc) {
  switch (Opc) {
  // cmp is an alias for SUBS with a dead destination register.
  case AArch64::SUBSWri:
  case AArch64::SUBSXri:
  // cmp is an alias for ADDS with a dead destination register.
  case AArch64::ADDSWri:
  case AArch64::ADDSXri:
    return true;
  default:
    return false;
  }
}

// Returns the Bcc terminator if present, otherwise nullptr.
MachineInstr *
AArch64ConditionOptimizerImpl::getBccTerminator(MachineBasicBlock *MBB) {
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

// Find the CMP instruction controlling the given conditional instruction and
// ensure it can be adjusted for CSE optimization. Searches backward from
// CondMI, ensuring no NZCV interference. Returns nullptr if no suitable CMP
// is found or if adjustments are not safe.
MachineInstr *
AArch64ConditionOptimizerImpl::findAdjustableCmp(MachineInstr *CondMI) {
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
    // Ensure there is no use of NZCV between CMP and conditional.
    if (I.readsRegister(AArch64::NZCV, /*TRI=*/nullptr))
      return nullptr;

    if (isCmpInstruction(I.getOpcode())) {
      if (!canAdjustCmp(I)) {
        return nullptr;
      }
      return &I;
    }

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
  case AArch64CC::GT:
    return AArch64CC::GE;
  case AArch64CC::GE:
    return AArch64CC::GT;
  case AArch64CC::LT:
    return AArch64CC::LE;
  case AArch64CC::LE:
    return AArch64CC::LT;
  case AArch64CC::HI:
    return AArch64CC::HS;
  case AArch64CC::HS:
    return AArch64CC::HI;
  case AArch64CC::LO:
    return AArch64CC::LS;
  case AArch64CC::LS:
    return AArch64CC::LO;
  default:
    llvm_unreachable("Unexpected condition code");
  }
}

// Returns the adjusted immediate, opcode, and condition code for switching
// between inclusive/exclusive forms (GT <-> GE, LT <-> LE).
CmpInfo
AArch64ConditionOptimizerImpl::getAdjustedCmpInfo(MachineInstr *CmpMI,
                                                  AArch64CC::CondCode Cmp) {
  unsigned Opc = CmpMI->getOpcode();

  bool IsSigned = Cmp == AArch64CC::GT || Cmp == AArch64CC::GE ||
                  Cmp == AArch64CC::LT || Cmp == AArch64CC::LE;

  // CMN (compare with negative immediate) is an alias to ADDS (as
  // "operand - negative" == "operand + positive")
  bool Negative = (Opc == AArch64::ADDSWri || Opc == AArch64::ADDSXri);

  int Correction = (Cmp == AArch64CC::GT || Cmp == AArch64CC::HI) ? 1 : -1;
  // Negate Correction value for comparison with negative immediate (CMN).
  if (Negative) {
    Correction = -Correction;
  }

  const int OldImm = (int)CmpMI->getOperand(2).getImm();
  const int NewImm = std::abs(OldImm + Correction);

  // Bail out on cmn 0 (ADDS with immediate 0). It is a valid instruction but
  // doesn't set flags in a way we can safely transform, so skip optimization.
  if (OldImm == 0 && Negative)
    return {OldImm, Opc, Cmp};

  if ((OldImm == 1 && Negative && Correction == -1) ||
      (OldImm == 0 && Correction == -1)) {
    // If we change opcodes for unsigned comparisons, this means we did an
    // unsigned wrap (e.g., 0 wrapping to 0xFFFFFFFF), so return the old cmp.
    // Note: For signed comparisons, opcode changes (cmn 1 ↔ cmp 0) are valid.
    if (!IsSigned)
      return {OldImm, Opc, Cmp};
    Opc = getComplementOpc(Opc);
  }

  return {NewImm, Opc, getAdjustedCmp(Cmp)};
}

// Modifies a comparison instruction's immediate and opcode.
void AArch64ConditionOptimizerImpl::updateCmpInstr(MachineInstr *CmpMI,
                                                   int NewImm,
                                                   unsigned NewOpc) {
  CmpMI->getOperand(2).setImm(NewImm);
  CmpMI->setDesc(TII->get(NewOpc));
}

// Modifies the condition code of a conditional instruction.
void AArch64ConditionOptimizerImpl::updateCondInstr(MachineInstr *CondMI,
                                                    AArch64CC::CondCode NewCC) {
  int CCOpIdx =
      AArch64InstrInfo::findCondCodeUseOperandIdxForBranchOrSelect(*CondMI);
  assert(CCOpIdx >= 0 && "Unsupported conditional instruction");
  CondMI->getOperand(CCOpIdx).setImm(NewCC);
  ++NumConditionsAdjusted;
}

// Applies a comparison adjustment to a cmp/cond instruction pair.
void AArch64ConditionOptimizerImpl::applyCmpAdjustment(CmpCondPair &Pair,
                                                       const CmpInfo &Info) {
  updateCmpInstr(Pair.CmpMI, Info.Imm, Info.Opc);
  updateCondInstr(Pair.CondMI, Info.CC);
  Pair.CC = Info.CC;
}

// Extracts the condition code from the result of analyzeBranch.
// Returns the CondCode or Invalid if the format is not a simple br.cond.
static AArch64CC::CondCode parseCondCode(ArrayRef<MachineOperand> Cond) {
  assert(!Cond.empty() && "Expected non-empty condition from analyzeBranch");
  // A normal br.cond simply has the condition code.
  if (Cond[0].getImm() != -1) {
    assert(Cond.size() == 1 && "Unknown Cond array format");
    return (AArch64CC::CondCode)(int)Cond[0].getImm();
  }
  return AArch64CC::CondCode::Invalid;
}

static bool isGreaterThan(AArch64CC::CondCode Cmp) {
  return Cmp == AArch64CC::GT || Cmp == AArch64CC::HI;
}

static bool isLessThan(AArch64CC::CondCode Cmp) {
  return Cmp == AArch64CC::LT || Cmp == AArch64CC::LO;
}

bool AArch64ConditionOptimizerImpl::tryOptimizePair(CmpCondPair &First,
                                                    CmpCondPair &Second) {
  if (!((isGreaterThan(First.CC) || isLessThan(First.CC)) &&
        (isGreaterThan(Second.CC) || isLessThan(Second.CC))))
    return false;

  int FirstImmTrueValue = First.getImm();
  int SecondImmTrueValue = Second.getImm();

  // Normalize immediate of CMN (ADDS) instructions
  if (First.getOpc() == AArch64::ADDSWri || First.getOpc() == AArch64::ADDSXri)
    FirstImmTrueValue = -FirstImmTrueValue;
  if (Second.getOpc() == AArch64::ADDSWri ||
      Second.getOpc() == AArch64::ADDSXri)
    SecondImmTrueValue = -SecondImmTrueValue;

  CmpInfo FirstAdj = getAdjustedCmpInfo(First.CmpMI, First.CC);
  CmpInfo SecondAdj = getAdjustedCmpInfo(Second.CmpMI, Second.CC);

  if (((isGreaterThan(First.CC) && isLessThan(Second.CC)) ||
       (isLessThan(First.CC) && isGreaterThan(Second.CC))) &&
      std::abs(SecondImmTrueValue - FirstImmTrueValue) == 2) {
    // This branch transforms machine instructions that correspond to
    //
    // 1) (a > {SecondImm} && ...) || (a < {FirstImm} && ...)
    // 2) (a < {SecondImm} && ...) || (a > {FirstImm} && ...)
    //
    // into
    //
    // 1) (a >= {NewImm} && ...) || (a <= {NewImm} && ...)
    // 2) (a <= {NewImm} && ...) || (a >= {NewImm} && ...)

    // Verify both adjustments converge to identical comparisons (same
    // immediate and opcode). This ensures CSE can eliminate the duplicate.
    if (FirstAdj.Imm != SecondAdj.Imm || FirstAdj.Opc != SecondAdj.Opc)
      return false;

    LLVM_DEBUG(dbgs() << "Optimized (opposite): "
                      << AArch64CC::getCondCodeName(First.CC) << " #"
                      << First.getImm() << ", "
                      << AArch64CC::getCondCodeName(Second.CC) << " #"
                      << Second.getImm() << " -> "
                      << AArch64CC::getCondCodeName(FirstAdj.CC) << " #"
                      << FirstAdj.Imm << ", "
                      << AArch64CC::getCondCodeName(SecondAdj.CC) << " #"
                      << SecondAdj.Imm << '\n');
    applyCmpAdjustment(First, FirstAdj);
    applyCmpAdjustment(Second, SecondAdj);
    return true;

  } else if (((isGreaterThan(First.CC) && isGreaterThan(Second.CC)) ||
              (isLessThan(First.CC) && isLessThan(Second.CC))) &&
             std::abs(SecondImmTrueValue - FirstImmTrueValue) == 1) {
    // This branch transforms machine instructions that correspond to
    //
    // 1) (a > {SecondImm} && ...) || (a > {FirstImm} && ...)
    // 2) (a < {SecondImm} && ...) || (a < {FirstImm} && ...)
    //
    // into
    //
    // 1) (a <= {NewImm} && ...) || (a >  {NewImm} && ...)
    // 2) (a <  {NewImm} && ...) || (a >= {NewImm} && ...)

    // GT -> GE transformation increases immediate value, so picking the
    // smaller one; LT -> LE decreases immediate value so invert the choice.
    bool AdjustFirst = (FirstImmTrueValue < SecondImmTrueValue);
    if (isLessThan(First.CC))
      AdjustFirst = !AdjustFirst;

    CmpCondPair &Target = AdjustFirst ? Second : First;
    CmpCondPair &ToChange = AdjustFirst ? First : Second;
    CmpInfo &Adj = AdjustFirst ? FirstAdj : SecondAdj;

    // Verify the adjustment converges to the target's comparison (same
    // immediate and opcode). This ensures CSE can eliminate the duplicate.
    if (Adj.Imm != Target.getImm() || Adj.Opc != Target.getOpc())
      return false;

    LLVM_DEBUG(dbgs() << "Optimized (same-direction): "
                      << AArch64CC::getCondCodeName(ToChange.CC) << " #"
                      << ToChange.getImm() << " -> "
                      << AArch64CC::getCondCodeName(Adj.CC) << " #" << Adj.Imm
                      << '\n');
    applyCmpAdjustment(ToChange, Adj);
    return true;
  }

  // Other transformation cases almost never occur due to generation of < or >
  // comparisons instead of <= and >=.
  return false;
}

bool AArch64ConditionOptimizerImpl::commitPendingPair(
    std::optional<CmpCondPair> &PendingPair,
    SmallDenseMap<Register, CmpCondPair> &PairsByReg) {
  if (!PendingPair)
    return false;

  Register Reg = PendingPair->CmpMI->getOperand(1).getReg();
  Register Key = Reg.isVirtual() ? TRI->lookThruCopyLike(Reg, MRI) : Reg;

  auto MatchingPair = PairsByReg.find(Key);
  bool Changed = MatchingPair != PairsByReg.end() &&
                 tryOptimizePair(MatchingPair->second, *PendingPair);

  PairsByReg[Key] = *PendingPair;
  PendingPair = std::nullopt;
  return Changed;
}

// This function transforms cmps and their consuming conditionals (CmpCondPairs)
// 1. Same direction: when both conditions are the same (e.g. GT/GT or LT/LT)
//    and immediates differ by 1
// 2. Opposite direction: when both conditions are adjustable to a common middle
//    (e.g., GT/LT) and immediates differ by 2.
// The compare instructions are made to match to enable CSE.
// All cmp/cond pairs within a basic block are examined
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
//   cmp w8, #10              ; <- CSE can remove the redundant cmp
//   csinc w10, w0, w1, ge    ; w10 = (w8 >= 10) ? w0 : w1+1
//
bool AArch64ConditionOptimizerImpl::optimizeIntraBlock(MachineBasicBlock &MBB) {
  SmallDenseMap<Register, CmpCondPair> PairsByReg;
  std::optional<CmpCondPair> PendingPair;
  MachineInstr *ActiveCmp = nullptr;
  bool Changed = false;

  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;

    if (isCmpInstruction(MI.getOpcode()) && canAdjustCmp(MI)) {
      Changed |= commitPendingPair(PendingPair, PairsByReg);
      ActiveCmp = &MI;
      continue;
    }

    if (MI.modifiesRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
      // Non-CMP clobber: commit any pending pair and reset all state, since
      // unknown flag state at this point invalidates all prior pairs
      Changed |= commitPendingPair(PendingPair, PairsByReg);
      ActiveCmp = nullptr;
      PairsByReg.clear();
      continue;
    }

    if (AArch64InstrInfo::findCondCodeUseOperandIdxForBranchOrSelect(MI) >= 0 &&
        !MI.isBranch()) {
      if (PendingPair) {
        // A second conditional consuming the same CMP would invalidate any
        // optimization: modifying the CMP would silently change what both
        // consumers compare against. Mark the CMP spent.
        PendingPair = std::nullopt;
        ActiveCmp = nullptr;
      } else if (ActiveCmp) {
        int CCOpIdx =
            AArch64InstrInfo::findCondCodeUseOperandIdxForBranchOrSelect(MI);
        assert(CCOpIdx >= 0 && "Unsupported conditional instruction");
        AArch64CC::CondCode CC =
            (AArch64CC::CondCode)(int)MI.getOperand(CCOpIdx).getImm();
        PendingPair = CmpCondPair{ActiveCmp, &MI, CC};
      }
      continue;
    }

    if (MI.readsRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
      ActiveCmp = nullptr;
      PendingPair = std::nullopt;
      continue;
    }
  }

  // Only commit the final pending pair if NZCV doesn't live out: a cross-block
  // consumer would be affected by any CMP adjustment we make.
  if (!nzcvLivesOut(&MBB))
    Changed |= commitPendingPair(PendingPair, PairsByReg);

  return Changed;
}

// Finds the last valid conditional consumer in MBB and returns it together
// with its condition code. Handles two cases:
//
//  1. Bcc terminator: if the block ends with a Bcc, analyzeBranch extracts
//     the condition code directly from the branch operands.
//
//  2. Select-family instruction (CSET/CSEL/CSINC/CSINV/CSNEG): scans
//     backward past terminators to find the sole non-branch NZCV consumer,
//     verifying there is no interfering NZCV read or write between it and
//     the CMP that produces the flags.
//
// Returns {nullptr, Invalid} if no suitable consumer is found or if any
// safety check fails.
std::pair<MachineInstr *, AArch64CC::CondCode>
AArch64ConditionOptimizerImpl::findCondConsumer(MachineBasicBlock *MBB) {
  // Case 1: block ends with a Bcc terminator.
  if (MachineInstr *BrMI = getBccTerminator(MBB)) {
    SmallVector<MachineOperand, 4> CondOperands;
    MachineBasicBlock *TBBDest = nullptr, *FBBDest = nullptr;
    if (TII->analyzeBranch(*MBB, TBBDest, FBBDest, CondOperands))
      return {nullptr, AArch64CC::Invalid};
    AArch64CC::CondCode CC = parseCondCode(CondOperands);
    if (CC == AArch64CC::Invalid)
      return {nullptr, AArch64CC::Invalid};
    return {BrMI, CC};
  }

  // Case 2: no Bcc terminator — scan backward for a select-family instruction
  // (CSET/CSEL/CSINC/CSINV/CSNEG) that is the sole NZCV consumer in the block.
  MachineInstr *Found = nullptr;
  AArch64CC::CondCode FoundCC = AArch64CC::Invalid;

  for (MachineInstr &MI : reverse(*MBB)) {
    // Skip terminators (e.g. an unconditional branch at the end of the block)
    // and debug instructions, which carry no real semantics.
    if (MI.isTerminator() || MI.isDebugInstr())
      continue;

    if (!Found) {
      // We have not yet found the select. Keep scanning backward.

      // If something writes NZCV before we find a select, the flags at that
      // point are not from the CMP we are looking for. Stop searching.
      if (MI.modifiesRegister(AArch64::NZCV, /*TRI=*/nullptr))
        return {nullptr, AArch64CC::Invalid};

      // findCondCodeUseOperandIdxForBranchOrSelect returns the operand index
      // of the condition code for any branch or select-family instruction, or
      // -1 if the instruction does not use a condition code.
      // We exclude branches because getBccTerminator already handles those;
      // we only want non-branch conditionals: CSET, CSEL, CSINC, CSINV, CSNEG.
      int CCOpIdx =
          AArch64InstrInfo::findCondCodeUseOperandIdxForBranchOrSelect(MI);
      if (CCOpIdx >= 0 && !MI.isBranch()) {
        Found = &MI;
        FoundCC = (AArch64CC::CondCode)(int)MI.getOperand(CCOpIdx).getImm();
        continue;
      }

      // Any other instruction that reads NZCV (but is not a select) means the
      // flags are consumed by something we do not understand. Stop searching.
      if (MI.readsRegister(AArch64::NZCV, /*TRI=*/nullptr))
        return {nullptr, AArch64CC::Invalid};

    } else {
      // We already found a select. Now verify there is no second NZCV reader
      // between the found select and the CMP. If there is, the CMP feeds two
      // consumers and cannot be safely adjusted.
      if (MI.readsRegister(AArch64::NZCV, /*TRI=*/nullptr))
        return {nullptr, AArch64CC::Invalid};

      if (MI.modifiesRegister(AArch64::NZCV, /*TRI=*/nullptr)) {
        // A CMP instruction is the flag producer we are looking for; stop
        // scanning. findAdjustableCmp will locate it from CondMI.
        if (isCmpInstruction(MI.getOpcode()))
          break;
        // Any other NZCV writer means the select is not reading from the CMP
        // we would find further back.
        return {nullptr, AArch64CC::Invalid};
      }
    }
  }
  return {Found, FoundCC};
}

// Optimizes CMP+conditional pairs across two basic blocks in the dominator
// tree. The conditional consumer in each block may be a Bcc terminator or a
// select-family instruction (CSEL/CSET/CSINC/CSINV/CSNEG).
bool AArch64ConditionOptimizerImpl::optimizeCrossBlock(MachineBasicBlock &HBB) {
  SmallVector<MachineOperand, 4> HeadCondOperands;
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  if (TII->analyzeBranch(HBB, TBB, FBB, HeadCondOperands)) {
    return false;
  }

  // Equivalence check is to skip loops.
  if (!TBB || TBB == &HBB) {
    return false;
  }

  // Find the conditional consumer(Bcc or select-family) and its condition
  // code in each block. findCondConsumer() handles both cases uniformly.
  auto [HeadCondMI, HeadCondCode] = findCondConsumer(&HBB);
  if (!HeadCondMI)
    return false;

  auto [TrueCondMI, TrueCondCode] = findCondConsumer(TBB);
  if (!TrueCondMI)
    return false;

  // Since we may modify cmps in these blocks, make sure NZCV does not live out.
  if (nzcvLivesOut(&HBB) || nzcvLivesOut(TBB))
    return false;

  // Find the CMPs controlling each conditional.
  MachineInstr *HeadCmpMI = findAdjustableCmp(HeadCondMI);
  MachineInstr *TrueCmpMI = findAdjustableCmp(TrueCondMI);
  if (!HeadCmpMI || !TrueCmpMI)
    return false;

  if (!registersMatch(HeadCmpMI, TrueCmpMI))
    return false;

  LLVM_DEBUG(dbgs() << "Checking cross-block pair: "
                    << AArch64CC::getCondCodeName(HeadCondCode) << " #"
                    << HeadCmpMI->getOperand(2).getImm() << ", "
                    << AArch64CC::getCondCodeName(TrueCondCode) << " #"
                    << TrueCmpMI->getOperand(2).getImm() << '\n');

  CmpCondPair Head{HeadCmpMI, HeadCondMI, HeadCondCode};
  CmpCondPair True{TrueCmpMI, TrueCondMI, TrueCondCode};

  return tryOptimizePair(Head, True);
}

bool AArch64ConditionOptimizerLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  MachineDominatorTree &MDT =
      getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  return AArch64ConditionOptimizerImpl().run(MF, MDT);
}

bool AArch64ConditionOptimizerImpl::run(MachineFunction &MF,
                                        MachineDominatorTree &MDT) {
  LLVM_DEBUG(dbgs() << "********** AArch64 Conditional Compares **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  TII = static_cast<const AArch64InstrInfo *>(MF.getSubtarget().getInstrInfo());
  TRI = MF.getSubtarget().getRegisterInfo();
  DomTree = &MDT;
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

PreservedAnalyses
AArch64ConditionOptimizerPass::run(MachineFunction &MF,
                                   MachineFunctionAnalysisManager &MFAM) {
  auto &MDT = MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  bool Changed = AArch64ConditionOptimizerImpl().run(MF, MDT);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
