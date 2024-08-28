#include "llvm/CodeGen/SSAIfConv.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "early-ifcvt"

using namespace llvm;

// Stress testing mode - disable heuristics.
static cl::opt<bool> Stress("stress-early-ifcvt", cl::Hidden,
                            cl::desc("Turn all knobs to 11"));

// Absolute maximum number of instructions allowed per speculated block.
// This bypasses all other heuristics, so it should be set fairly high.
static cl::opt<unsigned> BlockInstrLimit(
    "early-ifcvt-limit", cl::init(30), cl::Hidden,
    cl::desc("Maximum number of instructions per speculated block."));

STATISTIC(NumDiamondsSeen, "Number of diamonds");
STATISTIC(NumDiamondsConv, "Number of diamonds converted");
STATISTIC(NumTrianglesSeen, "Number of triangles");
STATISTIC(NumTrianglesConv, "Number of triangles converted");

SSAIfConv::SSAIfConv(PredicationStrategyBase &Predicate, MachineFunction &MF,
                     MachineDominatorTree *DomTree, MachineLoopInfo *Loops,
                     MachineTraceMetrics *Traces)
    : DomTree(DomTree), Loops(Loops), Traces(Traces), Predicate(Predicate) {
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  LiveRegUnits.clear();
  LiveRegUnits.setUniverse(TRI->getNumRegUnits());
  ClobberedRegUnits.clear();
  ClobberedRegUnits.resize(TRI->getNumRegUnits());
}

MachineTraceMetrics::Ensemble *SSAIfConv::getEnsemble(MachineTraceStrategy S) {
  return Traces ? Traces->getEnsemble(S) : nullptr;
}

/// Check that there is no dependencies preventing if conversion.
///
/// If instruction uses any values that are defined in the head basic block,
/// the defining instructions are added to InsertAfter.
bool SSAIfConv::InstrDependenciesAllowIfConv(MachineInstr *I) {
  for (const MachineOperand &MO : I->operands()) {
    if (MO.isRegMask()) {
      LLVM_DEBUG(dbgs() << "Won't speculate regmask: " << *I);
      return false;
    }
    if (!MO.isReg())
      continue;
    Register Reg = MO.getReg();

    // Remember clobbered regunits.
    if (MO.isDef() && Reg.isPhysical())
      for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg()))
        ClobberedRegUnits.set(Unit);

    if (!MO.readsReg() || !Reg.isVirtual())
      continue;
    MachineInstr *DefMI = MRI->getVRegDef(Reg);
    if (!DefMI || DefMI->getParent() != Head)
      continue;
    if (InsertAfter.insert(DefMI).second)
      LLVM_DEBUG(dbgs() << printMBBReference(*I->getParent()) << " depends on "
                        << *DefMI);
    if (DefMI->isTerminator()) {
      LLVM_DEBUG(dbgs() << "Can't insert instructions below terminator.\n");
      return false;
    }
  }
  return true;
}

/// canPredicateInstrs - Returns true if all the instructions in MBB can safely
/// be predicates. The terminators are not considered.
///
/// If instructions use any values that are defined in the head basic block,
/// the defining instructions are added to InsertAfter.
///
/// Any clobbered regunits are added to ClobberedRegUnits.
///
bool SSAIfConv::canPredicateInstrs(MachineBasicBlock *MBB) {
  // Reject any live-in physregs. It's probably CPSR/EFLAGS, and very hard to
  // get right.
  if (!MBB->livein_empty()) {
    LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has live-ins.\n");
    return false;
  }

  unsigned InstrCount = 0;

  // Check all instructions, except the terminators. It is assumed that
  // terminators never have side effects or define any used register values.
  for (MachineBasicBlock::iterator I = MBB->begin(),
                                   E = MBB->getFirstTerminator();
       I != E; ++I) {
    if (I->isDebugInstr())
      continue;

    if (++InstrCount > BlockInstrLimit && !Stress) {
      LLVM_DEBUG(dbgs() << printMBBReference(*MBB) << " has more than "
                        << BlockInstrLimit << " instructions.\n");
      return false;
    }

    // There shouldn't normally be any phis in a single-predecessor block.
    if (I->isPHI()) {
      LLVM_DEBUG(dbgs() << "Can't predicate: " << *I);
      return false;
    }

    if (!Predicate.canPredicateInstr(*I))
      return false;

    // Check for any dependencies on Head instructions.
    if (!InstrDependenciesAllowIfConv(&(*I)))
      return false;
  }
  return true;
}

/// Find an insertion point in Head for the speculated instructions. The
/// insertion point must be:
///
/// 1. Before any terminators.
/// 2. After any instructions in InsertAfter.
/// 3. Not have any clobbered regunits live.
///
/// This function sets InsertionPoint and returns true when successful, it
/// returns false if no valid insertion point could be found.
///
bool SSAIfConv::findInsertionPoint() {
  // Keep track of live regunits before the current position.
  // Only track RegUnits that are also in ClobberedRegUnits.
  LiveRegUnits.clear();
  SmallVector<MCRegister, 8> Reads;
  MachineBasicBlock::iterator FirstTerm = Head->getFirstTerminator();
  MachineBasicBlock::iterator I = Head->end();
  MachineBasicBlock::iterator B = Head->begin();
  while (I != B) {
    --I;
    // Some of the conditional code depends in I.
    if (InsertAfter.count(&*I)) {
      LLVM_DEBUG(dbgs() << "Can't insert code after " << *I);
      return false;
    }

    // Update live regunits.
    for (const MachineOperand &MO : I->operands()) {
      // We're ignoring regmask operands. That is conservatively correct.
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg.isPhysical())
        continue;
      // I clobbers Reg, so it isn't live before I.
      if (MO.isDef())
        for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg()))
          LiveRegUnits.erase(Unit);
      // Unless I reads Reg.
      if (MO.readsReg())
        Reads.push_back(Reg.asMCReg());
    }
    // Anything read by I is live before I.
    while (!Reads.empty())
      for (MCRegUnit Unit : TRI->regunits(Reads.pop_back_val()))
        if (ClobberedRegUnits.test(Unit))
          LiveRegUnits.insert(Unit);

    // We can't insert before a terminator.
    if (I != FirstTerm && I->isTerminator())
      continue;

    // Some of the clobbered registers are live before I, not a valid insertion
    // point.
    if (!LiveRegUnits.empty()) {
      LLVM_DEBUG({
        dbgs() << "Would clobber";
        for (unsigned LRU : LiveRegUnits)
          dbgs() << ' ' << printRegUnit(LRU, TRI);
        dbgs() << " live before " << *I;
      });
      continue;
    }

    // This is a valid insertion point.
    InsertionPoint = I;
    LLVM_DEBUG(dbgs() << "Can insert before " << *I);
    return true;
  }
  LLVM_DEBUG(dbgs() << "No legal insertion point found.\n");
  return false;
}

/// canConvertIf - analyze the sub-cfg rooted in MBB, and return true if it is
/// a potential candidate for if-conversion. Fill out the internal state.
///
bool SSAIfConv::canConvertIf(MachineBasicBlock *MBB) {
  Head = MBB;
  TBB = FBB = Tail = nullptr;

  if (Head->succ_size() != 2)
    return false;
  MachineBasicBlock *Succ0 = Head->succ_begin()[0];
  MachineBasicBlock *Succ1 = Head->succ_begin()[1];

  // Canonicalize so Succ0 has MBB as its single predecessor.
  if (Succ0->pred_size() != 1)
    std::swap(Succ0, Succ1);

  if (Succ0->pred_size() != 1 || Succ0->succ_size() != 1)
    return false;

  Tail = Succ0->succ_begin()[0];

  // This is not a triangle.
  if (Tail != Succ1) {
    // Check for a diamond. We won't deal with any critical edges.
    if (Succ1->pred_size() != 1 || Succ1->succ_size() != 1 ||
        Succ1->succ_begin()[0] != Tail)
      return false;
    LLVM_DEBUG(dbgs() << "\nDiamond: " << printMBBReference(*Head) << " -> "
                      << printMBBReference(*Succ0) << "/"
                      << printMBBReference(*Succ1) << " -> "
                      << printMBBReference(*Tail) << '\n');

    // Live-in physregs are tricky to get right when speculating code.
    if (!Tail->livein_empty()) {
      LLVM_DEBUG(dbgs() << "Tail has live-ins.\n");
      return false;
    }
  } else {
    LLVM_DEBUG(dbgs() << "\nTriangle: " << printMBBReference(*Head) << " -> "
                      << printMBBReference(*Succ0) << " -> "
                      << printMBBReference(*Tail) << '\n');
  }

  // The branch we're looking to eliminate must be analyzable.
  Cond.clear();
  if (TII->analyzeBranch(*Head, TBB, FBB, Cond)) {
    LLVM_DEBUG(dbgs() << "Branch not analyzable.\n");
    return false;
  }

  if (!Predicate.canConvertIf(Head, TBB, FBB, Tail, Cond)) {
    return false;
  }

  // This is weird, probably some sort of degenerate CFG.
  if (!TBB) {
    LLVM_DEBUG(dbgs() << "analyzeBranch didn't find conditional branch.\n");
    return false;
  }

  // Make sure the analyzed branch is conditional; one of the successors
  // could be a landing pad. (Empty landing pads can be generated on Windows.)
  if (Cond.empty()) {
    LLVM_DEBUG(dbgs() << "analyzeBranch found an unconditional branch.\n");
    return false;
  }

  // analyzeBranch doesn't set FBB on a fall-through branch.
  // Make sure it is always set.
  FBB = TBB == Succ0 ? Succ1 : Succ0;

  // Any phis in the tail block must be convertible to selects.
  PHIs.clear();
  MachineBasicBlock *TPred = getTPred();
  MachineBasicBlock *FPred = getFPred();
  for (MachineBasicBlock::iterator I = Tail->begin(), E = Tail->end();
       I != E && I->isPHI(); ++I) {
    PHIs.push_back(&*I);
    PHIInfo &PI = PHIs.back();
    // Find PHI operands corresponding to TPred and FPred.
    for (unsigned i = 1; i != PI.PHI->getNumOperands(); i += 2) {
      if (PI.PHI->getOperand(i + 1).getMBB() == TPred)
        PI.TReg = PI.PHI->getOperand(i).getReg();
      if (PI.PHI->getOperand(i + 1).getMBB() == FPred)
        PI.FReg = PI.PHI->getOperand(i).getReg();
    }
    assert(Register::isVirtualRegister(PI.TReg) && "Bad PHI");
    assert(Register::isVirtualRegister(PI.FReg) && "Bad PHI");

    // Get target information.
    if (!TII->canInsertSelect(*Head, Cond, PI.PHI->getOperand(0).getReg(),
                              PI.TReg, PI.FReg, PI.CondCycles, PI.TCycles,
                              PI.FCycles)) {
      LLVM_DEBUG(dbgs() << "Can't convert: " << *PI.PHI);
      return false;
    }
  }

  // Check that the conditional instructions can be speculated.
  InsertAfter.clear();
  ClobberedRegUnits.reset();
  for (MachineBasicBlock *MBB : {TBB, FBB})
    if (MBB != Tail && !canPredicateInstrs(MBB))
      return false;

  // Try to find a valid insertion point for the speculated instructions in the
  // head basic block.
  if (!findInsertionPoint())
    return false;

  if (isTriangle())
    ++NumTrianglesSeen;
  else
    ++NumDiamondsSeen;
  return true;
}

/// \return true iff the two registers are known to have the same value.
bool hasSameValue(const MachineRegisterInfo &MRI, const TargetInstrInfo *TII,
                  Register TReg, Register FReg) {
  if (TReg == FReg)
    return true;

  if (!TReg.isVirtual() || !FReg.isVirtual())
    return false;

  const MachineInstr *TDef = MRI.getUniqueVRegDef(TReg);
  const MachineInstr *FDef = MRI.getUniqueVRegDef(FReg);
  if (!TDef || !FDef)
    return false;

  // If there are side-effects, all bets are off.
  if (TDef->hasUnmodeledSideEffects())
    return false;

  // If the instruction could modify memory, or there may be some intervening
  // store between the two, we can't consider them to be equal.
  if (TDef->mayLoadOrStore() && !TDef->isDereferenceableInvariantLoad())
    return false;

  // We also can't guarantee that they are the same if, for example, the
  // instructions are both a copy from a physical reg, because some other
  // instruction may have modified the value in that reg between the two
  // defining insts.
  if (any_of(TDef->uses(), [](const MachineOperand &MO) {
        return MO.isReg() && MO.getReg().isPhysical();
      }))
    return false;

  // Check whether the two defining instructions produce the same value(s).
  if (!TII->produceSameValue(*TDef, *FDef, &MRI))
    return false;

  // Further, check that the two defs come from corresponding operands.
  int TIdx = TDef->findRegisterDefOperandIdx(TReg, /*TRI=*/nullptr);
  int FIdx = FDef->findRegisterDefOperandIdx(FReg, /*TRI=*/nullptr);
  if (TIdx == -1 || FIdx == -1)
    return false;

  return TIdx == FIdx;
}

/// replacePHIInstrs - Completely replace PHI instructions with selects.
/// This is possible when the only Tail predecessors are the if-converted
/// blocks.
void SSAIfConv::replacePHIInstrs() {
  assert(Tail->pred_size() == 2 && "Cannot replace PHIs");
  MachineBasicBlock::iterator FirstTerm = Head->getFirstTerminator();
  assert(FirstTerm != Head->end() && "No terminators");
  DebugLoc HeadDL = FirstTerm->getDebugLoc();

  // Convert all PHIs to select instructions inserted before FirstTerm.
  for (PHIInfo &PI : PHIs) {
    LLVM_DEBUG(dbgs() << "If-converting " << *PI.PHI);
    Register DstReg = PI.PHI->getOperand(0).getReg();
    if (hasSameValue(*MRI, TII, PI.TReg, PI.FReg)) {
      // We do not need the select instruction if both incoming values are
      // equal, but we do need a COPY.
      BuildMI(*Head, FirstTerm, HeadDL, TII->get(TargetOpcode::COPY), DstReg)
          .addReg(PI.TReg);
    } else {
      TII->insertSelect(*Head, FirstTerm, HeadDL, DstReg, Cond, PI.TReg,
                        PI.FReg);
    }
    LLVM_DEBUG(dbgs() << "          --> " << *std::prev(FirstTerm));
    PI.PHI->eraseFromParent();
    PI.PHI = nullptr;
  }
}

/// rewritePHIOperands - When there are additional Tail predecessors, insert
/// select instructions in Head and rewrite PHI operands to use the selects.
/// Keep the PHI instructions in Tail to handle the other predecessors.
void SSAIfConv::rewritePHIOperands() {
  MachineBasicBlock::iterator FirstTerm = Head->getFirstTerminator();
  assert(FirstTerm != Head->end() && "No terminators");
  DebugLoc HeadDL = FirstTerm->getDebugLoc();

  // Convert all PHIs to select instructions inserted before FirstTerm.
  for (PHIInfo &PI : PHIs) {
    unsigned DstReg = 0;

    LLVM_DEBUG(dbgs() << "If-converting " << *PI.PHI);
    if (hasSameValue(*MRI, TII, PI.TReg, PI.FReg)) {
      // We do not need the select instruction if both incoming values are
      // equal.
      DstReg = PI.TReg;
    } else {
      Register PHIDst = PI.PHI->getOperand(0).getReg();
      DstReg = MRI->createVirtualRegister(MRI->getRegClass(PHIDst));
      TII->insertSelect(*Head, FirstTerm, HeadDL, DstReg, Cond, PI.TReg,
                        PI.FReg);
      LLVM_DEBUG(dbgs() << "          --> " << *std::prev(FirstTerm));
    }

    // Rewrite PHI operands TPred -> (DstReg, Head), remove FPred.
    for (unsigned i = PI.PHI->getNumOperands(); i != 1; i -= 2) {
      MachineBasicBlock *MBB = PI.PHI->getOperand(i - 1).getMBB();
      if (MBB == getTPred()) {
        PI.PHI->getOperand(i - 1).setMBB(Head);
        PI.PHI->getOperand(i - 2).setReg(DstReg);
      } else if (MBB == getFPred()) {
        PI.PHI->removeOperand(i - 1);
        PI.PHI->removeOperand(i - 2);
      }
    }
    LLVM_DEBUG(dbgs() << "          --> " << *PI.PHI);
  }
}

/// convertIf - Execute the if conversion after canConvertIf has determined the
/// feasibility.
///
/// Any basic blocks that need to be erased will be added to RemoveBlocks.
///
void SSAIfConv::convertIf(SmallVectorImpl<MachineBasicBlock *> &RemoveBlocks) {
  assert(Head && Tail && TBB && FBB && "Call canConvertIf first.");

  // Update statistics.
  if (isTriangle())
    ++NumTrianglesConv;
  else
    ++NumDiamondsConv;

  // Move all instructions into Head, except for the terminators.
  for (MachineBasicBlock *MBB : {TBB, FBB}) {
    if (MBB != Tail) {
      // reverse the condition for the false bb
      Predicate.predicateBlock(MBB, Cond, MBB == FBB);
      Head->splice(InsertionPoint, MBB, MBB->begin(),
                   MBB->getFirstTerminator());
    }
  }

  // Are there extra Tail predecessors?
  bool ExtraPreds = Tail->pred_size() != 2;
  if (ExtraPreds)
    rewritePHIOperands();
  else
    replacePHIInstrs();

  // Fix up the CFG, temporarily leave Head without any successors.
  Head->removeSuccessor(TBB);
  Head->removeSuccessor(FBB, true);
  if (TBB != Tail)
    TBB->removeSuccessor(Tail, true);
  if (FBB != Tail)
    FBB->removeSuccessor(Tail, true);

  // Fix up Head's terminators.
  // It should become a single branch or a fallthrough.
  DebugLoc HeadDL = Head->getFirstTerminator()->getDebugLoc();
  TII->removeBranch(*Head);

  // Mark the now empty conditional blocks for removal and move them to the end.
  // It is likely that Head can fall
  // through to Tail, and we can join the two blocks.
  if (TBB != Tail) {
    RemoveBlocks.push_back(TBB);
    if (TBB != &TBB->getParent()->back())
      TBB->moveAfter(&TBB->getParent()->back());
  }
  if (FBB != Tail) {
    RemoveBlocks.push_back(FBB);
    if (FBB != &FBB->getParent()->back())
      FBB->moveAfter(&FBB->getParent()->back());
  }

  assert(Head->succ_empty() && "Additional head successors?");
  if (!ExtraPreds && Head->isLayoutSuccessor(Tail)) {
    // Splice Tail onto the end of Head.
    LLVM_DEBUG(dbgs() << "Joining tail " << printMBBReference(*Tail)
                      << " into head " << printMBBReference(*Head) << '\n');
    Head->splice(Head->end(), Tail, Tail->begin(), Tail->end());
    Head->transferSuccessorsAndUpdatePHIs(Tail);
    RemoveBlocks.push_back(Tail);
    if (Tail != &Tail->getParent()->back())
      Tail->moveAfter(&Tail->getParent()->back());
  } else {
    // We need a branch to Tail, let code placement work it out later.
    LLVM_DEBUG(dbgs() << "Converting to unconditional branch.\n");
    SmallVector<MachineOperand, 0> EmptyCond;
    TII->insertBranch(*Head, Tail, nullptr, EmptyCond, HeadDL);
    Head->addSuccessor(Tail);
  }
  LLVM_DEBUG(dbgs() << *Head);
}

void SSAIfConv::updateDomTree(ArrayRef<MachineBasicBlock *> Removed) {
  // convertIf can remove TBB, FBB, and Tail can be merged into Head.
  // TBB and FBB should not dominate any blocks.
  // Tail children should be transferred to Head.
  MachineDomTreeNode *HeadNode = DomTree->getNode(Head);
  for (auto *B : Removed) {
    MachineDomTreeNode *Node = DomTree->getNode(B);
    assert(Node != HeadNode && "Cannot erase the head node");
    while (Node->getNumChildren()) {
      assert(Node->getBlock() == Tail && "Unexpected children");
      DomTree->changeImmediateDominator(Node->back(), HeadNode);
    }
    DomTree->eraseNode(B);
  }
}

void SSAIfConv::updateLoops(ArrayRef<MachineBasicBlock *> Removed) {
  // If-conversion doesn't change loop structure, and it doesn't mess with back
  // edges, so updating LoopInfo is simply removing the dead blocks.
  for (auto *B : Removed)
    Loops->removeBlock(B);
}

void SSAIfConv::invalidateTraces() {
  if (!Traces)
    return;
  Traces->verifyAnalysis();
  Traces->invalidate(Head);
  Traces->invalidate(Tail);
  Traces->invalidate(TBB);
  Traces->invalidate(FBB);
  Traces->verifyAnalysis();
}

// Visit blocks in dominator tree post-order. The post-order enables nested
// if-conversion in a single pass. The tryConvertIf() function may erase
// blocks, but only blocks dominated by the head block. This makes it safe to
// update the dominator tree while the post-order iterator is still active.
bool SSAIfConv::run() {
  bool Changed = false;
  for (auto *DomNode : post_order(DomTree))
    if (tryConvertIf(DomNode->getBlock()))
      Changed = true;
  return Changed;
}

bool SSAIfConv::tryConvertIf(MachineBasicBlock *MBB) {
  bool Changed = false;
  while (canConvertIf(MBB) && (Stress || Predicate.shouldConvertIf(*this))) {
    // If-convert MBB and update analyses.
    invalidateTraces();
    SmallVector<MachineBasicBlock *, 4> RemoveBlocks;
    convertIf(RemoveBlocks);
    Changed = true;
    updateDomTree(RemoveBlocks);
    for (MachineBasicBlock *MBB : RemoveBlocks)
      MBB->eraseFromParent();
    updateLoops(RemoveBlocks);
  }
  return Changed;
}