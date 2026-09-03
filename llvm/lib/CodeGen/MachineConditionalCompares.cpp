//===-- MachineConditionalCompares.cpp --- CCMP formation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the target-independent MachineConditionalCompares pass
// which reduces branching and code size by using the conditional-compare
// instructions (CCMP/CTEST on X86; CCMP/CCMN/FCCMP on AArch64).
//
// The CFG transformations for forming conditional compares are very similar to
// if-conversion, and this pass should run immediately before the early
// if-conversion pass. The transform itself is target-independent; the
// recognition of a convertible compare and the emission of the conditional
// compare are delegated to the target via TargetInstrInfo hooks
// (getConditionalCompareFlagReg / canConvertToCCMP / convertToCCMP) and the
// pass is gated per target via TargetSubtargetInfo::enableCCMPFormation.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineConditionalCompares.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineInstrBundle.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineTraceMetrics.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "machine-ccmp"

// Absolute maximum number of instructions allowed per speculated block.
// This bypasses all other heuristics, so it should be set fairly high.
static cl::opt<unsigned> BlockInstrLimit(
    "machine-ccmp-limit", cl::init(30), cl::Hidden,
    cl::desc("Maximum number of instructions per speculated block."));

// Stress testing mode - disable heuristics.
static cl::opt<bool> Stress("stress-machine-ccmp", cl::Hidden,
                            cl::desc("Turn all knobs to 11"));

STATISTIC(NumConsidered, "Number of ccmps considered");
STATISTIC(NumPhiRejs, "Number of ccmps rejected (PHI)");
STATISTIC(NumPhysRejs, "Number of ccmps rejected (Physregs)");
STATISTIC(NumPhi2Rejs, "Number of ccmps rejected (PHI2)");
STATISTIC(NumHeadBranchRejs, "Number of ccmps rejected (Head branch)");
STATISTIC(NumCmpBranchRejs, "Number of ccmps rejected (CmpBB branch)");
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
// instructions. The AArch64 conditional compare instructions have an immediate
// operand that specifies the NZCV flag values when the condition is false and
// the compare isn't executed. This makes it possible to chain compares with
// different condition codes.
//
// Example:
//
//    if (a == 5 || b == 17)
//      foo();
//
//    Head:
//       cmp  w0, #5
//       b.eq Tail
//    CmpBB:
//       cmp  w1, #17
//       b.eq Tail
//    ...
//    Tail:
//      bl _foo
//
//  Becomes:
//
//    Head:
//       cmp  w0, #5
//       ccmp w1, #17, 4, ne  ; 4 = nZcv
//       b.eq Tail
//    ...
//    Tail:
//      bl _foo
//
// The ccmp condition code is the one that would cause the Head terminator to
// branch to CmpBB.
//
// FIXME: It should also be possible to speculate a block on the critical edge
// between Head and Tail, just like if-converting a diamond.
//
// FIXME: Handle PHIs in Tail by turning them into selects (if-conversion).
//
namespace {
class SSACCmpConv {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  const MachineBranchProbabilityInfo *MBPI;
  MachineOptimizationRemarkEmitter *ORE;

  /// The physical flag/status register clobbered by ccmp candidates.
  MCRegister FlagReg;

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
  /// The branch condition in Head as determined by analyzeBranch. Stored
  /// opaquely; only round-tripped through the target hooks.
  SmallVector<MachineOperand, 4> HeadCond;

  /// The branch condition in CmpBB as determined by analyzeBranch. Stored
  /// opaquely; only round-tripped through the target hooks.
  SmallVector<MachineOperand, 4> CmpBBCond;

  /// Target-owned recognition/emission state produced by canConvertToCCMP().
  TargetInstrInfo::CCmpConvInfo Info;

  /// Check if the Tail PHIs are trivially convertible.
  bool trivialTailPHIs();

  /// Remove CmpBB from the Tail PHIs.
  void updateTailPHIs();

  /// Return true if all non-terminator instructions in MBB can be safely
  /// speculated.
  bool canSpeculateInstrs(MachineBasicBlock *MBB, const MachineInstr *CmpMI);

public:
  /// Initialize per-function data structures.
  void init(MachineFunction &MF, const MachineBranchProbabilityInfo *MBPI,
            MachineOptimizationRemarkEmitter *ORE) {
    this->MBPI = MBPI;
    this->ORE = ORE;
    TII = MF.getSubtarget().getInstrInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    MRI = &MF.getRegInfo();
    FlagReg = TII->getConditionalCompareFlagReg();
  }

  /// If the sub-CFG headed by MBB can be cmp-converted, initialize the internal
  /// state, and return true.
  bool canConvert(MachineBasicBlock *MBB);

  /// Cmp-convert the last block passed to canConvert(), assuming it is
  /// possible. Add any erased blocks to RemovedBlocks.
  void convert(SmallVectorImpl<MachineBasicBlock *> &RemovedBlocks);

  /// Return the expected code size delta if the conversion into a conditional
  /// compare is performed, as computed by canConvertToCCMP().
  int expectedCodeSizeDelta() const { return Info.CodeSizeDelta; }
};
} // end anonymous namespace

// Detect a chain of vreg-to-vreg copies feeding Reg, returning the original
// value. This lets trivialTailPHIs() see copy-equivalent PHI operands as equal.
static Register lookThroughCopies(Register Reg, MachineRegisterInfo *MRI) {
  MachineInstr *MI;
  while ((MI = MRI->getUniqueVRegDef(Reg)) &&
         MI->getOpcode() == TargetOpcode::COPY) {
    if (MI->getOperand(1).getReg().isPhysical())
      break;
    Reg = MI->getOperand(1).getReg();
  }
  return Reg;
}

// Check that all PHIs in Tail are selecting the same value from Head and CmpBB.
// This means that no if-conversion is required when merging CmpBB into Head.
bool SSACCmpConv::trivialTailPHIs() {
  for (auto &I : Tail->phis()) {
    unsigned HeadReg = 0, CmpBBReg = 0;
    // PHI operands come in (VReg, MBB) pairs.
    for (unsigned Idx = 1, End = I.getNumOperands(); Idx != End; Idx += 2) {
      MachineBasicBlock *MBB = I.getOperand(Idx + 1).getMBB();
      Register Reg = lookThroughCopies(I.getOperand(Idx).getReg(), MRI);
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

/// Determine if all the instructions in MBB can safely be speculated. The
/// terminators are not considered. Only CmpMI is allowed to clobber the flags.
bool SSACCmpConv::canSpeculateInstrs(MachineBasicBlock *MBB,
                                     const MachineInstr *CmpMI) {
  // Reject any live-in physregs. It's probably NZCV/EFLAGS, and very hard to
  // get right.
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

    if (++InstrCount > BlockInstrLimit && !Stress) {
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
    if (!I.isSafeToMove(DontMoveAcrossStore)) {
      LLVM_DEBUG(dbgs() << "Can't speculate: " << I);
      return false;
    }

    // Only CmpMI is allowed to clobber the flags.
    if (&I != CmpMI && I.modifiesRegister(FlagReg, TRI)) {
      LLVM_DEBUG(dbgs() << "Clobbers flags: " << I);
      return false;
    }
  }
  return true;
}

/// Analyze the sub-cfg rooted in MBB, and return true if it is a potential
/// candidate for cmp-conversion. Fill out the internal state.
bool SSACCmpConv::canConvert(MachineBasicBlock *MBB) {
  Head = MBB;
  Tail = CmpBB = nullptr;
  CmpMI = nullptr;
  Info = TargetInstrInfo::CCmpConvInfo();

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
  MachineBasicBlock *HeadTBB = nullptr, *HeadFBB = nullptr;
  if (TII->analyzeBranch(*Head, HeadTBB, HeadFBB, HeadCond)) {
    LLVM_DEBUG(dbgs() << "Head branch not analyzable.\n");
    ++NumHeadBranchRejs;
    return false;
  }

  // This is weird, probably some sort of degenerate CFG, or an edge to a
  // landing pad.
  if (!HeadTBB || HeadCond.empty()) {
    LLVM_DEBUG(
        dbgs() << "analyzeBranch didn't find conditional branch in Head.\n");
    ++NumHeadBranchRejs;
    return false;
  }

  // Analyze the branch in CmpBB.
  CmpBBCond.clear();
  MachineBasicBlock *CmpBBTBB = nullptr, *CmpBBFBB = nullptr;
  if (TII->analyzeBranch(*CmpBB, CmpBBTBB, CmpBBFBB, CmpBBCond)) {
    LLVM_DEBUG(dbgs() << "CmpBB branch not analyzable.\n");
    ++NumCmpBranchRejs;
    return false;
  }

  if (!CmpBBTBB || CmpBBCond.empty()) {
    LLVM_DEBUG(
        dbgs() << "analyzeBranch didn't find conditional branch in CmpBB.\n");
    ++NumCmpBranchRejs;
    return false;
  }

  // Delegate target-specific recognition: condition-code parsing/reject-lists,
  // any target-specific terminator constraints, and finding the convertible
  // compare. The condition arrays are passed opaquely; the booleans tell the
  // target whether analyzeBranch's TBB is the desired successor so it can apply
  // its own condition-code inversion.
  if (!TII->canConvertToCCMP(*Head, *CmpBB, HeadCond, HeadTBB == CmpBB,
                             CmpBBCond, CmpBBTBB == Tail, *MRI, Info))
    return false;
  CmpMI = Info.CmpMI;

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

  // Save successor probabilities before removing CmpBB and Tail from their
  // parents.
  BranchProbability Head2CmpBB = MBPI->getEdgeProbability(Head, CmpBB);
  BranchProbability CmpBB2Tail = MBPI->getEdgeProbability(CmpBB, Tail);

  Head->removeSuccessor(CmpBB);
  CmpBB->removeSuccessor(Tail);

  // If Head and CmpBB had successor probabilities, update the probabilities to
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
    // normalizing the successor probabilities. Set the successor probabilities
    // before doing so.
    //
    // Pr(I|Head) = Pr(CmpBB|Head) * Pr(I|CmpBB).
    for (auto I = CmpBB->succ_begin(), E = CmpBB->succ_end(); I != E; ++I) {
      BranchProbability CmpBB2I = MBPI->getEdgeProbability(CmpBB, *I);
      CmpBB->setSuccProbability(I, Head2CmpBB * CmpBB2I);
    }
  }

  Head->transferSuccessorsAndUpdatePHIs(CmpBB);
  DebugLoc HeadTermDL = Head->getFirstTerminator()->getDebugLoc();
  TII->removeBranch(*Head);

  // Remember the splice boundary: the first CmpBB instruction, which after the
  // splice below becomes the insertion point in Head for any synthesized Head
  // compare (e.g. a cbz/cbnz head).
  assert(!CmpBB->empty() && "CmpBB unexpectedly empty");
  MachineInstr *SpliceStart = &CmpBB->front();

  Head->splice(Head->end(), CmpBB, CmpBB->begin(), CmpBB->end());

  // Let the target emit the conditional compare that replaces CmpMI (and any
  // Head-terminator fixup). It builds before CmpMI, which now lives in Head.
  TII->convertToCCMP(*Head, SpliceStart->getIterator(), HeadTermDL, HeadCond,
                     Info, *MRI);

  if (ORE)
    ORE->emit([&]() {
      MachineOptimizationRemark R(DEBUG_TYPE, "ConvertedCMP",
                                  CmpMI->getDebugLoc(), CmpBB);
      R << "convert CMP into conditional CMP";
      return R;
    });

  CmpMI->eraseFromParent();
  Head->updateTerminator(CmpBB->getNextNode());

  RemovedBlocks.push_back(CmpBB);
  LLVM_DEBUG(dbgs() << "Result:\n" << *Head);
  ++NumConverted;
}

//===----------------------------------------------------------------------===//
//                     MachineConditionalCompares Pass
//===----------------------------------------------------------------------===//

namespace {
class MachineConditionalCompares {
  const TargetSubtargetInfo *STI = nullptr;
  const MachineBranchProbabilityInfo *MBPI = nullptr;
  MachineDominatorTree *DomTree = nullptr;
  MachineLoopInfo *Loops = nullptr;
  MachineTraceMetrics *Traces = nullptr;
  MachineTraceMetrics::Ensemble *MinInstr = nullptr;
  MachineOptimizationRemarkEmitter *ORE = nullptr;
  bool MinSize = false;
  SSACCmpConv CmpConv;

public:
  MachineConditionalCompares(const MachineBranchProbabilityInfo *MBPI,
                             MachineDominatorTree *DomTree,
                             MachineLoopInfo *Loops,
                             MachineTraceMetrics *Traces,
                             MachineOptimizationRemarkEmitter *ORE)
      : MBPI(MBPI), DomTree(DomTree), Loops(Loops), Traces(Traces), ORE(ORE) {}

  bool run(MachineFunction &MF);

private:
  bool tryConvert(MachineBasicBlock *MBB);
  void updateDomTree(ArrayRef<MachineBasicBlock *> Removed);
  void updateLoops(ArrayRef<MachineBasicBlock *> Removed);
  void invalidateTraces();
  bool shouldConvert();
};
} // end anonymous namespace

/// Update the dominator tree after if-conversion erased some blocks.
void MachineConditionalCompares::updateDomTree(
    ArrayRef<MachineBasicBlock *> Removed) {
  // convert() removes CmpBB which was previously dominated by Head.
  // CmpBB children should be transferred to Head.
  MachineDomTreeNode *HeadNode = DomTree->getNode(CmpConv.Head);
  for (MachineBasicBlock *RemovedMBB : Removed) {
    MachineDomTreeNode *Node = DomTree->getNode(RemovedMBB);
    assert(Node != HeadNode && "Cannot erase the head node");
    assert(Node->getIDom() == HeadNode && "CmpBB should be dominated by Head");
    while (!Node->isLeaf())
      DomTree->changeImmediateDominator(*Node->begin(), HeadNode);
    DomTree->eraseNode(RemovedMBB);
  }
}

/// Update LoopInfo after if-conversion.
void MachineConditionalCompares::updateLoops(
    ArrayRef<MachineBasicBlock *> Removed) {
  if (!Loops)
    return;
  for (MachineBasicBlock *RemovedMBB : Removed)
    Loops->removeBlock(RemovedMBB);
}

/// Invalidate MachineTraceMetrics before if-conversion.
void MachineConditionalCompares::invalidateTraces() {
  Traces->invalidate(CmpConv.Head);
  Traces->invalidate(CmpConv.CmpBB);
}

/// Apply the cost model to the candidate conversion in CmpConv. Return true if
/// the conversion is a good idea.
bool MachineConditionalCompares::shouldConvert() {
  // Stress testing mode disables all cost considerations.
  if (Stress)
    return true;

  if (!MinInstr)
    MinInstr = Traces->getEnsemble(MachineTraceStrategy::TS_MinInstrCount);

  // Head dominates CmpBB, so it is always included in its trace.
  MachineTraceMetrics::Trace Trace = MinInstr->getTrace(CmpConv.CmpBB);

  // If code size is the main concern, decide by the code-size delta reported
  // by the target. Targets that do not model it report 0, which falls through
  // to the regular heuristics below.
  if (MinSize) {
    int CodeSizeDelta = CmpConv.expectedCodeSizeDelta();
    LLVM_DEBUG(dbgs() << "Code size delta:  " << CodeSizeDelta << '\n');
    // If we are minimizing the code size, do the conversion whatever the cost
    // is.
    if (CodeSizeDelta < 0)
      return true;
    if (CodeSizeDelta > 0) {
      LLVM_DEBUG(dbgs() << "Code size is increasing, give up on this one.\n");
      return false;
    }
    // CodeSizeDelta == 0, continue with the regular heuristics.
  }

  // Heuristic: The compare conversion delays the execution of the branch
  // instruction because we must wait for the inputs to the second compare as
  // well. The branch has no dependent instructions, but delaying it increases
  // the cost of a misprediction.
  //
  // Set a limit on the delay we will accept.
  unsigned DelayLimit = STI->getMispredictionPenalty() * 3 / 4;

  // Instruction depths can be computed for all trace instructions above CmpBB.
  unsigned HeadDepth =
      Trace.getInstrCycles(*CmpConv.Head->getFirstTerminator()).Depth;

  // The conversion delays the branch because it must also wait for the inputs
  // to the second compare. The branch has no dependent instructions, but
  // delaying it increases the cost of a misprediction, so cap the delay at 3/4
  // of the misprediction penalty.
  unsigned CmpBBDepth =
      Trace.getInstrCycles(*CmpConv.CmpBB->getFirstTerminator()).Depth;
  LLVM_DEBUG(dbgs() << "Head depth:  " << HeadDepth << "\nCmpBB depth: "
                    << CmpBBDepth << "\nDelay limit: " << DelayLimit << '\n');
  if (CmpBBDepth > HeadDepth + DelayLimit) {
    LLVM_DEBUG(dbgs() << "Branch delay would be larger than " << DelayLimit
                      << " cycles.\n");
    return false;
  }

  // Check the resource depth at the bottom of CmpBB - these instructions will
  // be speculated.
  unsigned ResDepth = Trace.getResourceDepth(true);
  LLVM_DEBUG(dbgs() << "Resources:   " << ResDepth << '\n');

  // Heuristic: The speculatively executed instructions must all be able to
  // merge into the Head block. The Head critical path should dominate the
  // resource cost of the speculated instructions.
  if (ResDepth > HeadDepth) {
    LLVM_DEBUG(dbgs() << "Too many instructions to speculate.\n");
    return false;
  }

  return true;
}

bool MachineConditionalCompares::tryConvert(MachineBasicBlock *MBB) {
  bool Changed = false;
  while (CmpConv.canConvert(MBB) && shouldConvert()) {
    invalidateTraces();
    SmallVector<MachineBasicBlock *, 4> RemovedBlocks;
    CmpConv.convert(RemovedBlocks);
    Changed = true;
    updateDomTree(RemovedBlocks);
    updateLoops(RemovedBlocks);
    for (MachineBasicBlock *RemovedMBB : RemovedBlocks)
      RemovedMBB->eraseFromParent();
  }
  return Changed;
}

bool MachineConditionalCompares::run(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********** Machine Conditional Compares **********\n"
                    << "********** Function: " << MF.getName() << '\n');

  STI = &MF.getSubtarget();
  MinInstr = nullptr;
  MinSize = MF.getFunction().hasMinSize();

  bool Changed = false;
  CmpConv.init(MF, MBPI, ORE);

  // Visit blocks in dominator tree pre-order. The pre-order enables multiple
  // cmp-conversions from the same head block.
  // Note that updateDomTree() modifies the children of the DomTree node
  // currently being visited. The df_iterator supports that; it doesn't look at
  // child_begin() / child_end() until after a node has been visited.
  for (auto *I : depth_first(DomTree))
    if (tryConvert(I->getBlock()))
      Changed = true;

  if (Changed && ORE) {
    ORE->emit([&]() {
      MachineOptimizationRemarkAnalysis R(DEBUG_TYPE, "NumOfCCMP",
                                          MF.getFunction().getSubprogram(),
                                          &MF.front());
      R << "converted compare(s) to CCMP in function "
        << ore::NV("Function", MF.getName());
      return R;
    });
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
//                            Pass wrappers
//===----------------------------------------------------------------------===//

namespace {
class MachineConditionalComparesLegacy : public MachineFunctionPass {
public:
  static char ID;
  MachineConditionalComparesLegacy() : MachineFunctionPass(ID) {
    initializeMachineConditionalComparesLegacyPass(
        *PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  StringRef getPassName() const override {
    return "Machine Conditional Compares";
  }
};
} // end anonymous namespace

char MachineConditionalComparesLegacy::ID = 0;
char &llvm::MachineConditionalComparesLegacyID =
    MachineConditionalComparesLegacy::ID;

INITIALIZE_PASS_BEGIN(MachineConditionalComparesLegacy, DEBUG_TYPE,
                      "Machine Conditional Compares", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineTraceMetricsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(MachineConditionalComparesLegacy, DEBUG_TYPE,
                    "Machine Conditional Compares", false, false)

void MachineConditionalComparesLegacy::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  AU.addRequired<MachineTraceMetricsWrapperPass>();
  AU.addPreserved<MachineTraceMetricsWrapperPass>();
  AU.addRequired<MachineOptimizationRemarkEmitterPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineConditionalComparesLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  if (!MF.getSubtarget().enableCCMPFormation())
    return false;

  const MachineBranchProbabilityInfo *MBPI =
      &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();
  MachineDominatorTree *DomTree =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  MachineLoopInfo *Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MachineTraceMetrics *Traces =
      &getAnalysis<MachineTraceMetricsWrapperPass>().getMTM();
  MachineOptimizationRemarkEmitter *ORE =
      &getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();

  MachineConditionalCompares Impl(MBPI, DomTree, Loops, Traces, ORE);
  return Impl.run(MF);
}

PreservedAnalyses
MachineConditionalComparesPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &MFAM) {
  if (!MF.getSubtarget().enableCCMPFormation())
    return PreservedAnalyses::all();

  const MachineBranchProbabilityInfo *MBPI =
      &MFAM.getResult<MachineBranchProbabilityAnalysis>(MF);
  MachineDominatorTree *DomTree =
      &MFAM.getResult<MachineDominatorTreeAnalysis>(MF);
  MachineLoopInfo *Loops = &MFAM.getResult<MachineLoopAnalysis>(MF);
  MachineTraceMetrics *Traces =
      &MFAM.getResult<MachineTraceMetricsAnalysis>(MF);
  MachineOptimizationRemarkEmitter *ORE =
      &MFAM.getResult<MachineOptimizationRemarkEmitterAnalysis>(MF);

  MachineConditionalCompares Impl(MBPI, DomTree, Loops, Traces, ORE);
  bool Changed = Impl.run(MF);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserve<MachineDominatorTreeAnalysis>();
  PA.preserve<MachineLoopAnalysis>();
  PA.preserve<MachineTraceMetricsAnalysis>();
  return PA;
}
