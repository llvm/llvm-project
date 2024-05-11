//=========-- X86ConditionalCompares.cpp --- CCMP formation for X86 -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the X86ConditionalCompares pass which reduces
// branching by using the conditional compare instructions CCMP, CTEST.
//
// The CFG transformations for forming conditional compares are very similar to
// if-conversion, and this pass should run immediately before the early
// if-conversion pass.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "x86-ccmp"

// Absolute maximum number of instructions allowed per speculated block.
// This bypasses all other heuristics, so it should be set fairly high.
static cl::opt<unsigned> BlockInstrLimit(
    "x86-ccmp-limit", cl::init(30), cl::Hidden,
    cl::desc("Maximum number of instructions per speculated block."));

STATISTIC(NumConsidered, "Number of ccmps considered");
STATISTIC(NumPhiRejs, "Number of ccmps rejected (PHI)");
STATISTIC(NumPhysRejs, "Number of ccmps rejected (Physregs)");
STATISTIC(NumPhi2Rejs, "Number of ccmps rejected (PHI2)");
STATISTIC(NumHeadBranchRejs, "Number of ccmps rejected (Head branch)");
STATISTIC(NumCmpBranchRejs, "Number of ccmps rejected (CmpBB branch)");
STATISTIC(NumCmpTermRejs, "Number of ccmps rejected (CmpBB is cbz...)");
STATISTIC(NumMultEFLAGSUses, "Number of ccmps rejected (EFLAGS used)");
STATISTIC(NumUnknEFLAGSDefs, "Number of ccmps rejected (EFLAGS def unknown)");
STATISTIC(NumSpeculateRejs, "Number of ccmps rejected (Can't speculate)");
STATISTIC(NumConverted, "Number of ccmp instructions created");

//===----------------------------------------------------------------------===//
//                                 SSACCmpConv
//===----------------------------------------------------------------------===//
//
// The SSACCmpConv class performs ccmp-conversion on SSA form machine code
// after determining if it is possible. The class contains no heuristics;
// external code should be used to determine when ccmp-conversion is a good
// idea.
//
// CCmp-formation works on a CFG representing chained conditions, typically
// from C's short-circuit || and && operators:
//
//   From:         Head            To:         Head
//                 / |                         CmpBB
//                /  |                         / |
//               |  CmpBB                     /  |
//               |  / |                    Tail  |
//               | /  |                      |   |
//              Tail  |                      |   |
//                |   |                      |   |
//               ... ...                    ... ...
//
// The Head block is terminated by a br.cond instruction, and the CmpBB block
// contains compare + br.cond. Tail must be a successor of both.
//
// The cmp-conversion turns the compare instruction in CmpBB into a conditional
// compare, and merges CmpBB into Head, speculatively executing its
// instructions. The X86 conditional compare instructions have an operand that
// specifies the conditional flags to set values when the condition is false and
// the compare isn't executed. This makes it possible to chain compares with
// different condition codes.
//
// Example:
//
// void f(int a, int b) {
//   if (a == 5 || b == 17)
//     foo();
// }
//
//    Head:
//      cmpl  $5, $edi
//      je Tail
//    CmpBB:
//      cmpl  $17, $esi
//      je Tail
//    ...
//    Tail:
//      call foo
//
//  Becomes:
//
//    Head:
//      cmpl  $5, $edi
//      ccmpel {dfv=zf} $17, $edi
//      je Tail
//    ...
//    Tail:
//      call foo
//
// The ccmp condition code is the one that would cause the Head terminator to
// branch to CmpBB.

namespace {
class SSACCmpConv {
  MachineFunction *MF;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  const MachineBranchProbabilityInfo *MBPI;

public:
  /// The first block containing a conditional branch, dominating everything
  /// else.
  MachineBasicBlock *Head;

  /// The block containing cmp+br.cond with a successor shared with Head.
  MachineBasicBlock *CmpBB;

  /// The common successor for Head and CmpBB.
  MachineBasicBlock *Tail;

  /// The compare instruction in CmpBB that can be converted to a ccmp.
  MachineInstr *CmpMI;

private:
  /// The branch condition in Head as determined by analyzeBranch.
  SmallVector<MachineOperand, 4> HeadCond;

  /// The condition code that makes Head branch to CmpBB.
  X86::CondCode HeadCmpBBCC;

  /// The branch condition in CmpBB.
  SmallVector<MachineOperand, 4> CmpBBCond;

  /// The condition code that makes CmpBB branch to Tail.
  X86::CondCode CmpBBTailCC;

  /// Check if the Tail PHIs are trivially convertible.
  bool trivialTailPHIs();

  /// Remove CmpBB from the Tail PHIs.
  void updateTailPHIs();

  /// Check if an operand defining DstReg is dead.
  bool isDeadDef(unsigned DstReg);

  /// Find the compare instruction in MBB that controls the conditional branch.
  /// Return NULL if a convertible instruction can't be found.
  MachineInstr *findConvertibleCompare(MachineBasicBlock *MBB);

  /// Return true if all non-terminator instructions in MBB can be safely
  /// speculated.
  bool canSpeculateInstrs(MachineBasicBlock *MBB, const MachineInstr *CmpMI);

public:
  /// runOnMachineFunction - Initialize per-function data structures.
  void runOnMachineFunction(MachineFunction &MF,
                            const MachineBranchProbabilityInfo *MBPI) {
    this->MF = &MF;
    this->MBPI = MBPI;
    TII = MF.getSubtarget().getInstrInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    MRI = &MF.getRegInfo();
  }

  /// If the sub-CFG headed by MBB can be cmp-converted, initialize the
  /// internal state, and return true.
  bool canConvert(MachineBasicBlock *MBB);

  /// Cmo-convert the last block passed to canConvertCmp(), assuming
  /// it is possible. Add any erased blocks to RemovedBlocks.
  void convert(SmallVectorImpl<MachineBasicBlock *> &RemovedBlocks);
};
} // end anonymous namespace

// Check that all PHIs in Tail are selecting the same value from Head and CmpBB.
// This means that no if-conversion is required when merging CmpBB into Head.
bool SSACCmpConv::trivialTailPHIs() {
  for (auto &I : Tail->phis()) {
    unsigned HeadReg = 0, CmpBBReg = 0;
    // PHI operands come in (VReg, MBB) pairs.
    for (unsigned Idx = 1, End = I.getNumOperands(); Idx != End; Idx += 2) {
      MachineBasicBlock *MBB = I.getOperand(Idx + 1).getMBB();
      Register Reg = I.getOperand(Idx).getReg();
      if (MBB == Head) {
        assert((!HeadReg || HeadReg == Reg) && "Inconsistent PHI operands");
        HeadReg = Reg;
      }
      if (MBB == CmpBB) {
        assert((!CmpBBReg || CmpBBReg == Reg) && "Inconsistent PHI operands");
        CmpBBReg = Reg;
      }
    }
    if (HeadReg != CmpBBReg)
      return false;
  }
  return true;
}

// Assuming that trivialTailPHIs() is true, update the Tail PHIs by simply
// removing the CmpBB operands. The Head operands will be identical.
void SSACCmpConv::updateTailPHIs() {
  for (auto &I : Tail->phis()) {
    // I is a PHI. It can have multiple entries for CmpBB.
    for (unsigned Idx = I.getNumOperands(); Idx > 2; Idx -= 2) {
      // PHI operands are (Reg, MBB) at (Idx-2, Idx-1).
      if (I.getOperand(Idx - 1).getMBB() == CmpBB) {
        I.removeOperand(Idx - 1);
        I.removeOperand(Idx - 2);
      }
    }
  }
}

bool SSACCmpConv::isDeadDef(unsigned DstReg) {
  if (!Register::isVirtualRegister(DstReg))
    return false;
  // A virtual register def without any uses will be marked dead later, and
  // eventually replaced by the zero register.
  return MRI->use_nodbg_empty(DstReg);
}

MachineInstr *SSACCmpConv::findConvertibleCompare(MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator I = MBB->getFirstTerminator();
  if (I == MBB->end())
    return nullptr;
  // The terminator must be controlled by the flags.
  if (!I->readsRegister(X86::EFLAGS, TRI)) {
    ++NumCmpTermRejs;
    LLVM_DEBUG(dbgs() << "Flags not used by terminator: " << *I);
    return nullptr;
  }

  // Now find the instruction controlling the terminator.
  for (MachineBasicBlock::iterator B = MBB->begin(); I != B;) {
    I = prev_nodbg(I, MBB->begin());
    assert(!I->isTerminator() && "Spurious terminator");

    switch (I->getOpcode()) {
    // This pass run before peephole optimization, so the SUB has not been
    // optimized to CMP yet.
    case X86::SUB8rr:
    case X86::SUB16rr:
    case X86::SUB32rr:
    case X86::SUB64rr:
    case X86::SUB8ri:
    case X86::SUB16ri:
    case X86::SUB32ri:
    case X86::SUB64ri32:
    case X86::SUB8rr_ND:
    case X86::SUB16rr_ND:
    case X86::SUB32rr_ND:
    case X86::SUB64rr_ND:
    case X86::SUB8ri_ND:
    case X86::SUB16ri_ND:
    case X86::SUB32ri_ND:
    case X86::SUB64ri32_ND: {
      if (!isDeadDef(I->getOperand(0).getReg()))
        return nullptr;
      return &*I;
    }
    case X86::CMP8rr:
    case X86::CMP16rr:
    case X86::CMP32rr:
    case X86::CMP64rr:
    case X86::CMP8ri:
    case X86::CMP16ri:
    case X86::CMP32ri:
    case X86::CMP64ri32:
    case X86::TEST8rr:
    case X86::TEST16rr:
    case X86::TEST32rr:
    case X86::TEST64rr:
    case X86::TEST8ri:
    case X86::TEST16ri:
    case X86::TEST32ri:
    case X86::TEST64ri32:
      return &*I;
    default:
      break;
    }

    // Check for flag reads and clobbers.
    PhysRegInfo PRI = AnalyzePhysRegInBundle(*I, X86::EFLAGS, TRI);

    if (PRI.Read) {
      // The ccmp doesn't produce exactly the same flags as the original
      // compare, so reject the transform if there are uses of the flags
      // besides the terminators.
      LLVM_DEBUG(dbgs() << "Can't create ccmp with multiple uses: " << *I);
      ++NumMultEFLAGSUses;
      return nullptr;
    }

    if (PRI.Defined || PRI.Clobbered) {
      LLVM_DEBUG(dbgs() << "Not convertible compare: " << *I);
      ++NumUnknEFLAGSDefs;
      return nullptr;
    }
  }
  LLVM_DEBUG(dbgs() << "Flags not defined in " << printMBBReference(*MBB)
                    << '\n');
  return nullptr;
}

/// Determine if all the instructions in MBB can safely
/// be speculated. The terminators are not considered.
///
/// Only CmpMI is allowed to clobber the flags.
///
bool SSACCmpConv::canSpeculateInstrs(MachineBasicBlock *MBB,
                                     const MachineInstr *CmpMI) {
  // Reject any live-in physregs. It's very hard to get right.
  if (!MBB->livein_empty()) {
    LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has live-ins.\n");
    return false;
  }

  unsigned InstrCount = 0;

  // Check all instructions, except the terminators. It is assumed that
  // terminators never have side effects or define any used register values.
  for (auto &I : make_range(MBB->begin(), MBB->getFirstTerminator())) {
    if (I.isDebugInstr())
      continue;

    if (++InstrCount > BlockInstrLimit) {
      LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has more than "
                        << BlockInstrLimit << " instructions.\n");
      return false;
    }

    // There shouldn't normally be any phis in a single-predecessor block.
    if (I.isPHI()) {
      LLVM_DEBUG(dbgs() << "Can't hoist: " << I);
      return false;
    }

    // Don't speculate loads. Note that it may be possible and desirable to
    // speculate GOT or constant pool loads that are guaranteed not to trap,
    // but we don't support that for now.
    if (I.mayLoad()) {
      LLVM_DEBUG(dbgs() << "Won't speculate load: " << I);
      return false;
    }

    // We never speculate stores, so an AA pointer isn't necessary.
    bool DontMoveAcrossStore = true;
    if (!I.isSafeToMove(nullptr, DontMoveAcrossStore)) {
      LLVM_DEBUG(dbgs() << "Can't speculate: " << I);
      return false;
    }

    // Only CmpMI is allowed to clobber the flags.
    if (&I != CmpMI && I.modifiesRegister(X86::EFLAGS, TRI)) {
      LLVM_DEBUG(dbgs() << "Clobbers flags: " << I);
      return false;
    }
  }
  return true;
}

// Parse a condition code returned by analyzeBranch, and compute the CondCode
// corresponding to TBB.
static bool parseCond(ArrayRef<MachineOperand> Cond, X86::CondCode &CC,
                      ArrayRef<X86::CondCode> UnsupportedCCs = {}) {
  if (Cond.size() != 1)
    return false;

  CC = static_cast<X86::CondCode>(Cond[0].getImm());

  for (const auto &UnsupportedCC : UnsupportedCCs) {
    if (CC == UnsupportedCC)
      return false;
  }

  return CC != X86::COND_INVALID;
}

static unsigned getNumOfJcc(const MachineBasicBlock *MBB) {
  unsigned NumOfJcc = 0;
  for (auto It = MBB->rbegin(); It != MBB->rend(); ++It) {
    if (!It->isTerminator())
      return NumOfJcc;
    if (It->getOpcode() == X86::JCC_1)
      ++NumOfJcc;
  }
  return NumOfJcc;
}

/// Analyze the sub-cfg rooted in MBB, and return true if it is a potential
/// candidate for cmp-conversion. Fill out the internal state.
///
bool SSACCmpConv::canConvert(MachineBasicBlock *MBB) {
  Head = MBB;
  Tail = CmpBB = nullptr;

  if (Head->succ_size() != 2)
    return false;
  MachineBasicBlock *Succ0 = Head->succ_begin()[0];
  MachineBasicBlock *Succ1 = Head->succ_begin()[1];

  // CmpBB can only have a single predecessor. Tail is allowed many.
  if (Succ0->pred_size() != 1)
    std::swap(Succ0, Succ1);

  // Succ0 is our candidate for CmpBB.
  if (Succ0->pred_size() != 1 || Succ0->succ_size() != 2)
    return false;

  CmpBB = Succ0;
  Tail = Succ1;

  if (!CmpBB->isSuccessor(Tail))
    return false;

  // The CFG topology checks out.
  LLVM_DEBUG(dbgs() << "\nTriangle: " << printMBBReference(*Head) << " -> "
                    << printMBBReference(*CmpBB) << " -> "
                    << printMBBReference(*Tail) << '\n');
  ++NumConsidered;

  // Tail is allowed to have many predecessors, but we can't handle PHIs yet.
  //
  // FIXME: Real PHIs could be if-converted as long as the CmpBB values are
  // defined before The CmpBB cmp clobbers the flags. Alternatively, it should
  // always be safe to sink the ccmp down to immediately before the CmpBB
  // terminators.
  if (!trivialTailPHIs()) {
    LLVM_DEBUG(dbgs() << "Can't handle phis in Tail.\n");
    ++NumPhiRejs;
    return false;
  }

  if (!Tail->livein_empty()) {
    LLVM_DEBUG(dbgs() << "Can't handle live-in physregs in Tail.\n");
    ++NumPhysRejs;
    return false;
  }

  // CmpBB should never have PHIs since Head is its only predecessor.
  // FIXME: Clean them up if it happens.
  if (!CmpBB->empty() && CmpBB->front().isPHI()) {
    LLVM_DEBUG(dbgs() << "Can't handle phis in CmpBB.\n");
    ++NumPhi2Rejs;
    return false;
  }

  if (!CmpBB->livein_empty()) {
    LLVM_DEBUG(dbgs() << "Can't handle live-in physregs in CmpBB.\n");
    ++NumPhysRejs;
    return false;
  }

  // The branch we're looking to eliminate must be analyzable.
  HeadCond.clear();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
  if (TII->analyzeBranch(*Head, TBB, FBB, HeadCond)) {
    LLVM_DEBUG(dbgs() << "Head branch not analyzable.\n");
    ++NumHeadBranchRejs;
    return false;
  }

  // CCMP/CTEST resets all the bits of EFLAGS
  unsigned NumOfJcc = getNumOfJcc(Head);
  if (NumOfJcc > 1) {
    LLVM_DEBUG(dbgs() << "More than one Jcc in Head.\n");
    ++NumHeadBranchRejs;
    return false;
  }

  // This is weird, probably some sort of degenerate CFG, or an edge to a
  // landing pad.
  if (!TBB || HeadCond.empty()) {
    LLVM_DEBUG(
        dbgs() << "analyzeBranch didn't find conditional branch in Head.\n");
    ++NumHeadBranchRejs;
    return false;
  }

  if (!parseCond(HeadCond, HeadCmpBBCC, {X86::COND_P, X86::COND_NP})) {
    LLVM_DEBUG(dbgs() << "Unsupported branch type on Head\n");
    ++NumHeadBranchRejs;
    return false;
  }

  // Make sure the branch direction is right.
  if (TBB != CmpBB) {
    assert(TBB == Tail && "Unexpected TBB");
    HeadCmpBBCC = X86::GetOppositeBranchCondition(HeadCmpBBCC);
  }

  CmpBBCond.clear();
  TBB = FBB = nullptr;
  if (TII->analyzeBranch(*CmpBB, TBB, FBB, CmpBBCond)) {
    LLVM_DEBUG(dbgs() << "CmpBB branch not analyzable.\n");
    ++NumCmpBranchRejs;
    return false;
  }

  if (!TBB || CmpBBCond.empty()) {
    LLVM_DEBUG(
        dbgs() << "analyzeBranch didn't find conditional branch in CmpBB.\n");
    ++NumCmpBranchRejs;
    return false;
  }

  if (!parseCond(CmpBBCond, CmpBBTailCC)) {
    LLVM_DEBUG(dbgs() << "Unsupported branch type on CmpBB\n");
    ++NumCmpBranchRejs;
    return false;
  }

  if (TBB != Tail)
    CmpBBTailCC = X86::GetOppositeBranchCondition(CmpBBTailCC);

  CmpMI = findConvertibleCompare(CmpBB);
  if (!CmpMI)
    return false;

  if (!canSpeculateInstrs(CmpBB, CmpMI)) {
    ++NumSpeculateRejs;
    return false;
  }

  return true;
}

void SSACCmpConv::convert(SmallVectorImpl<MachineBasicBlock *> &RemovedBlocks) {
  LLVM_DEBUG(dbgs() << "Merging " << printMBBReference(*CmpBB) << " into "
                    << printMBBReference(*Head) << ":\n"
                    << *CmpBB);

  // All CmpBB instructions are moved into Head, and CmpBB is deleted.
  // Update the CFG first.
  updateTailPHIs();

  // Save successor probabilties before removing CmpBB and Tail from their
  // parents.
  BranchProbability Head2CmpBB = MBPI->getEdgeProbability(Head, CmpBB);
  BranchProbability CmpBB2Tail = MBPI->getEdgeProbability(CmpBB, Tail);

  Head->removeSuccessor(CmpBB);
  CmpBB->removeSuccessor(Tail);

  // If Head and CmpBB had successor probabilties, udpate the probabilities to
  // reflect the ccmp-conversion.
  if (Head->hasSuccessorProbabilities() && CmpBB->hasSuccessorProbabilities()) {

    // Head is allowed two successors. We've removed CmpBB, so the remaining
    // successor is Tail. We need to increase the successor probability for
    // Tail to account for the CmpBB path we removed.
    //
    // Pr(Tail|Head) += Pr(CmpBB|Head) * Pr(Tail|CmpBB).
    assert(*Head->succ_begin() == Tail && "Head successor is not Tail");
    BranchProbability Head2Tail = MBPI->getEdgeProbability(Head, Tail);
    Head->setSuccProbability(Head->succ_begin(),
                             Head2Tail + Head2CmpBB * CmpBB2Tail);

    // We will transfer successors of CmpBB to Head in a moment without
    // normalizing the successor probabilities. Set the successor probabilites
    // before doing so.
    //
    // Pr(I|Head) = Pr(CmpBB|Head) * Pr(I|CmpBB).
    for (auto I = CmpBB->succ_begin(), E = CmpBB->succ_end(); I != E; ++I) {
      BranchProbability CmpBB2I = MBPI->getEdgeProbability(CmpBB, *I);
      CmpBB->setSuccProbability(I, Head2CmpBB * CmpBB2I);
    }
  }

  Head->transferSuccessorsAndUpdatePHIs(CmpBB);
  TII->removeBranch(*Head);

  Head->splice(Head->end(), CmpBB, CmpBB->begin(), CmpBB->end());

  // Now replace CmpMI with a ccmp instruction that also considers the incoming
  // flags.
  unsigned Opc = [=]() {
    switch (CmpMI->getOpcode()) {
    default:
      llvm_unreachable("Unknown compare opcode");
    case X86::SUB8rr:
    case X86::SUB8rr_ND:
      return X86::CCMP8rr;
    case X86::SUB16rr:
    case X86::SUB16rr_ND:
      return X86::CCMP16rr;
    case X86::SUB32rr:
    case X86::SUB32rr_ND:
      return X86::CCMP32rr;
    case X86::SUB64rr:
    case X86::SUB64rr_ND:
      return X86::CCMP64rr;
    case X86::SUB8ri:
    case X86::SUB8ri_ND:
      return X86::CCMP8ri;
    case X86::SUB16ri:
    case X86::SUB16ri_ND:
      return X86::CCMP16ri;
    case X86::SUB32ri:
    case X86::SUB32ri_ND:
      return X86::CCMP32ri;
    case X86::SUB64ri32:
    case X86::SUB64ri32_ND:
      return X86::CCMP64ri32;
    case X86::CMP8rr:
      return X86::CCMP8rr;
    case X86::CMP16rr:
      return X86::CCMP16rr;
    case X86::CMP32rr:
      return X86::CCMP32rr;
    case X86::CMP64rr:
      return X86::CCMP64rr;
    case X86::CMP8ri:
      return X86::CCMP8ri;
    case X86::CMP16ri:
      return X86::CCMP16ri;
    case X86::CMP32ri:
      return X86::CCMP32ri;
    case X86::CMP64ri32:
      return X86::CCMP64ri32;
    case X86::TEST8rr:
      return X86::CTEST8rr;
    case X86::TEST16rr:
      return X86::CTEST16rr;
    case X86::TEST32rr:
      return X86::CTEST32rr;
    case X86::TEST64rr:
      return X86::CTEST64rr;
    case X86::TEST8ri:
      return X86::CTEST8ri;
    case X86::TEST16ri:
      return X86::CTEST16ri;
    case X86::TEST32ri:
      return X86::CTEST32ri;
    case X86::TEST64ri32:
      return X86::CTEST64ri32;
    }
  }();
  const MCInstrDesc &MCID = TII->get(Opc);
  unsigned NumDefs = CmpMI->getDesc().getNumDefs();
  MachineOperand Op0 = CmpMI->getOperand(NumDefs);
  MachineOperand Op1 = CmpMI->getOperand(NumDefs + 1);
  BuildMI(*Head, CmpMI, CmpMI->getDebugLoc(), MCID)
      .add(Op0)
      .add(Op1)
      .addImm(X86::getCondFlagsFromCondCode(CmpBBTailCC))
      .addImm(HeadCmpBBCC);
  CmpMI->eraseFromParent();
  Head->updateTerminator(CmpBB->getNextNode());

  RemovedBlocks.push_back(CmpBB);
  CmpBB->eraseFromParent();
  LLVM_DEBUG(dbgs() << "Result:\n" << *Head);
  ++NumConverted;
}

//===----------------------------------------------------------------------===//
//                       X86ConditionalCompares Pass
//===----------------------------------------------------------------------===//

namespace {
class X86ConditionalCompares : public MachineFunctionPass {
  const X86Subtarget *STI = nullptr;
  const MachineBranchProbabilityInfo *MBPI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineDominatorTree *DomTree = nullptr;
  MachineLoopInfo *Loops = nullptr;
  MachineTraceMetrics *Traces = nullptr;
  SSACCmpConv CmpConv;

public:
  static char ID;
  X86ConditionalCompares() : MachineFunctionPass(ID) {
    initializeX86ConditionalComparesPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override { return "X86 Conditional Compares"; }

private:
  bool tryConvert(MachineBasicBlock *MBB);
  void updateDomTree(ArrayRef<MachineBasicBlock *> Removed);
  void updateLoops(ArrayRef<MachineBasicBlock *> Removed);
  void invalidateTraces();
  bool shouldConvert();
};
} // end anonymous namespace

char X86ConditionalCompares::ID = 0;

INITIALIZE_PASS_BEGIN(X86ConditionalCompares, "x86-ccmp", "X86 CCMP Pass",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineTraceMetrics)
INITIALIZE_PASS_END(X86ConditionalCompares, "x86-ccmp", "X86 CCMP Pass", false,
                    false)

FunctionPass *llvm::createX86ConditionalCompares() {
  return new X86ConditionalCompares();
}

void X86ConditionalCompares::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
  AU.addRequired<MachineTraceMetrics>();
  AU.addPreserved<MachineTraceMetrics>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

/// Update the dominator tree after if-conversion erased some blocks.
void X86ConditionalCompares::updateDomTree(
    ArrayRef<MachineBasicBlock *> Removed) {
  // convert() removes CmpBB which was previously dominated by Head.
  // CmpBB children should be transferred to Head.
  MachineDomTreeNode *HeadNode = DomTree->getNode(CmpConv.Head);
  for (MachineBasicBlock *RemovedMBB : Removed) {
    MachineDomTreeNode *Node = DomTree->getNode(RemovedMBB);
    assert(Node != HeadNode && "Cannot erase the head node");
    assert(Node->getIDom() == HeadNode && "CmpBB should be dominated by Head");
    while (Node->getNumChildren())
      DomTree->changeImmediateDominator(Node->back(), HeadNode);
    DomTree->eraseNode(RemovedMBB);
  }
}

/// Update LoopInfo after if-conversion.
void X86ConditionalCompares::updateLoops(
    ArrayRef<MachineBasicBlock *> Removed) {
  if (!Loops)
    return;
  for (MachineBasicBlock *RemovedMBB : Removed)
    Loops->removeBlock(RemovedMBB);
}

/// Invalidate MachineTraceMetrics before if-conversion.
void X86ConditionalCompares::invalidateTraces() {
  Traces->invalidate(CmpConv.Head);
  Traces->invalidate(CmpConv.CmpBB);
}

/// Apply cost model and heuristics to the if-conversion in IfConv.
/// Return true if the conversion is a good idea.
///
bool X86ConditionalCompares::shouldConvert() { return true; }

bool X86ConditionalCompares::tryConvert(MachineBasicBlock *MBB) {
  bool Changed = false;
  while (CmpConv.canConvert(MBB) && shouldConvert()) {
    invalidateTraces();
    SmallVector<MachineBasicBlock *, 4> RemovedBlocks;
    CmpConv.convert(RemovedBlocks);
    Changed = true;
    updateDomTree(RemovedBlocks);
    updateLoops(RemovedBlocks);
  }
  return Changed;
}

bool X86ConditionalCompares::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** X86 Conditional Compares **********\n"
                    << "********** Function: " << MF.getName() << '\n');
  if (skipFunction(MF.getFunction()))
    return false;

  STI = &MF.getSubtarget<X86Subtarget>();
  if (!STI->hasCCMP())
    return false;

  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  DomTree = &getAnalysis<MachineDominatorTree>();
  Loops = getAnalysisIfAvailable<MachineLoopInfo>();
  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  Traces = &getAnalysis<MachineTraceMetrics>();

  bool Changed = false;
  CmpConv.runOnMachineFunction(MF, MBPI);

  // Visit blocks in dominator tree pre-order. The pre-order enables multiple
  // cmp-conversions from the same head block.
  // Note that updateDomTree() modifies the children of the DomTree node
  // currently being visited. The df_iterator supports that; it doesn't look at
  // child_begin() / child_end() until after a node has been visited.
  for (auto *I : depth_first(DomTree))
    if (tryConvert(I->getBlock()))
      Changed = true;

  return Changed;
}
