//===- LoopRotation.cpp - Loop Rotation Pass ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements Loop Rotation Pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/LoopRotation.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopPassManager.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
using namespace llvm;

#define DEBUG_TYPE "loop-rotate"

static cl::opt<unsigned> DefaultRotationThreshold(
    "rotation-max-header-size", cl::init(16), cl::Hidden,
    cl::desc("The default maximum header size for automatic loop rotation"));

STATISTIC(NumRotated, "Number of loops rotated");

/// A simple loop rotation transformation.
class LoopRotate {
  const unsigned MaxHeaderSize;
  LoopInfo *LI;
  const TargetTransformInfo *TTI;
  AssumptionCache *AC;
  DominatorTree *DT;
  ScalarEvolution *SE;

public:
  LoopRotate(unsigned MaxHeaderSize, LoopInfo *LI,
             const TargetTransformInfo *TTI, AssumptionCache *AC,
             DominatorTree *DT, ScalarEvolution *SE)
      : MaxHeaderSize(MaxHeaderSize), LI(LI), TTI(TTI), AC(AC), DT(DT), SE(SE) {
  }
  bool processLoop(Loop *L);

private:
  bool rotateLoop(Loop *L, bool SimplifiedLatch);
  bool simplifyLoopLatch(Loop *L);
};

/// RewriteUsesOfClonedInstructions - We just cloned the instructions from the
/// old header into the preheader.  If there were uses of the values produced by
/// these instruction that were outside of the loop, we have to insert PHI nodes
/// to merge the two values.  Do this now.
static void RewriteUsesOfClonedInstructions(BasicBlock *OrigHeader,
                                            BasicBlock *OrigPreheader,
                                            ValueToValueMapTy &ValueMap) {
  // Remove PHI node entries that are no longer live.
  BasicBlock::iterator I, E = OrigHeader->end();
  for (I = OrigHeader->begin(); PHINode *PN = dyn_cast<PHINode>(I); ++I)
    PN->removeIncomingValue(PN->getBasicBlockIndex(OrigPreheader));

  // Now fix up users of the instructions in OrigHeader, inserting PHI nodes
  // as necessary.
  SSAUpdater SSA;
  for (I = OrigHeader->begin(); I != E; ++I) {
    Value *OrigHeaderVal = &*I;

    // If there are no uses of the value (e.g. because it returns void), there
    // is nothing to rewrite.
    if (OrigHeaderVal->use_empty())
      continue;

    Value *OrigPreHeaderVal = ValueMap.lookup(OrigHeaderVal);

    // The value now exits in two versions: the initial value in the preheader
    // and the loop "next" value in the original header.
    SSA.Initialize(OrigHeaderVal->getType(), OrigHeaderVal->getName());
    SSA.AddAvailableValue(OrigHeader, OrigHeaderVal);
    SSA.AddAvailableValue(OrigPreheader, OrigPreHeaderVal);

    // Visit each use of the OrigHeader instruction.
    for (Value::use_iterator UI = OrigHeaderVal->use_begin(),
                             UE = OrigHeaderVal->use_end();
         UI != UE;) {
      // Grab the use before incrementing the iterator.
      Use &U = *UI;

      // Increment the iterator before removing the use from the list.
      ++UI;

      // SSAUpdater can't handle a non-PHI use in the same block as an
      // earlier def. We can easily handle those cases manually.
      Instruction *UserInst = cast<Instruction>(U.getUser());
      if (!isa<PHINode>(UserInst)) {
        BasicBlock *UserBB = UserInst->getParent();

        // The original users in the OrigHeader are already using the
        // original definitions.
        if (UserBB == OrigHeader)
          continue;

        // Users in the OrigPreHeader need to use the value to which the
        // original definitions are mapped.
        if (UserBB == OrigPreheader) {
          U = OrigPreHeaderVal;
          continue;
        }
      }

      // Anything else can be handled by SSAUpdater.
      SSA.RewriteUse(U);
    }

    // Replace MetadataAsValue(ValueAsMetadata(OrigHeaderVal)) uses in debug
    // intrinsics.
    LLVMContext &C = OrigHeader->getContext();
    if (auto *VAM = ValueAsMetadata::getIfExists(OrigHeaderVal)) {
      if (auto *MAV = MetadataAsValue::getIfExists(C, VAM)) {
        for (auto UI = MAV->use_begin(), E = MAV->use_end(); UI != E;) {
          // Grab the use before incrementing the iterator. Otherwise, altering
          // the Use will invalidate the iterator.
          Use &U = *UI++;
          DbgInfoIntrinsic *UserInst = dyn_cast<DbgInfoIntrinsic>(U.getUser());
          if (!UserInst)
            continue;

          // The original users in the OrigHeader are already using the original
          // definitions.
          BasicBlock *UserBB = UserInst->getParent();
          if (UserBB == OrigHeader)
            continue;

          // Users in the OrigPreHeader need to use the value to which the
          // original definitions are mapped and anything else can be handled by
          // the SSAUpdater. To avoid adding PHINodes, check if the value is
          // available in UserBB, if not substitute undef.
          Value *NewVal;
          if (UserBB == OrigPreheader)
            NewVal = OrigPreHeaderVal;
          else if (SSA.HasValueForBlock(UserBB))
            NewVal = SSA.GetValueInMiddleOfBlock(UserBB);
          else
            NewVal = UndefValue::get(OrigHeaderVal->getType());
          U = MetadataAsValue::get(C, ValueAsMetadata::get(NewVal));
        }
      }
    }
  }
}

/// Rotate loop LP. Return true if the loop is rotated.
///
/// \param SimplifiedLatch is true if the latch was just folded into the final
/// loop exit. In this case we may want to rotate even though the new latch is
/// now an exiting branch. This rotation would have happened had the latch not
/// been simplified. However, if SimplifiedLatch is false, then we avoid
/// rotating loops in which the latch exits to avoid excessive or endless
/// rotation. LoopRotate should be repeatable and converge to a canonical
/// form. This property is satisfied because simplifying the loop latch can only
/// happen once across multiple invocations of the LoopRotate pass.
bool LoopRotate::rotateLoop(Loop *L, bool SimplifiedLatch) {
  // If the loop has only one block then there is not much to rotate.
  if (L->getBlocks().size() == 1)
    return false;

  BasicBlock *OrigHeader = L->getHeader();
  BasicBlock *OrigLatch = L->getLoopLatch();

  BranchInst *BI = dyn_cast<BranchInst>(OrigHeader->getTerminator());
  if (!BI || BI->isUnconditional())
    return false;

  // If the loop header is not one of the loop exiting blocks then
  // either this loop is already rotated or it is not
  // suitable for loop rotation transformations.
  if (!L->isLoopExiting(OrigHeader))
    return false;

  // If the loop latch already contains a branch that leaves the loop then the
  // loop is already rotated.
  if (!OrigLatch)
    return false;

  // Rotate if either the loop latch does *not* exit the loop, or if the loop
  // latch was just simplified.
  if (L->isLoopExiting(OrigLatch) && !SimplifiedLatch)
    return false;

  // Check size of original header and reject loop if it is very big or we can't
  // duplicate blocks inside it.
  {
    SmallPtrSet<const Value *, 32> EphValues;
    CodeMetrics::collectEphemeralValues(L, AC, EphValues);

    CodeMetrics Metrics;
    Metrics.analyzeBasicBlock(OrigHeader, *TTI, EphValues);
    if (Metrics.notDuplicatable) {
      DEBUG(dbgs() << "LoopRotation: NOT rotating - contains non-duplicatable"
                   << " instructions: ";
            L->dump());
      return false;
    }
    if (Metrics.convergent) {
      DEBUG(dbgs() << "LoopRotation: NOT rotating - contains convergent "
                      "instructions: ";
            L->dump());
      return false;
    }
    if (Metrics.NumInsts > MaxHeaderSize)
      return false;
  }

  // Now, this loop is suitable for rotation.
  BasicBlock *OrigPreheader = L->getLoopPreheader();

  // If the loop could not be converted to canonical form, it must have an
  // indirectbr in it, just give up.
  if (!OrigPreheader)
    return false;

  // Anything ScalarEvolution may know about this loop or the PHI nodes
  // in its header will soon be invalidated.
  if (SE)
    SE->forgetLoop(L);

  DEBUG(dbgs() << "LoopRotation: rotating "; L->dump());

  // Find new Loop header. NewHeader is a Header's one and only successor
  // that is inside loop.  Header's other successor is outside the
  // loop.  Otherwise loop is not suitable for rotation.
  BasicBlock *Exit = BI->getSuccessor(0);
  BasicBlock *NewHeader = BI->getSuccessor(1);
  if (L->contains(Exit))
    std::swap(Exit, NewHeader);
  assert(NewHeader && "Unable to determine new loop header");
  assert(L->contains(NewHeader) && !L->contains(Exit) &&
         "Unable to determine loop header and exit blocks");

  // This code assumes that the new header has exactly one predecessor.
  // Remove any single-entry PHI nodes in it.
  assert(NewHeader->getSinglePredecessor() &&
         "New header doesn't have one pred!");
  FoldSingleEntryPHINodes(NewHeader);

  // Begin by walking OrigHeader and populating ValueMap with an entry for
  // each Instruction.
  BasicBlock::iterator I = OrigHeader->begin(), E = OrigHeader->end();
  ValueToValueMapTy ValueMap;

  // For PHI nodes, the value available in OldPreHeader is just the
  // incoming value from OldPreHeader.
  for (; PHINode *PN = dyn_cast<PHINode>(I); ++I)
    ValueMap[PN] = PN->getIncomingValueForBlock(OrigPreheader);

  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();

  // For the rest of the instructions, either hoist to the OrigPreheader if
  // possible or create a clone in the OldPreHeader if not.
  TerminatorInst *LoopEntryBranch = OrigPreheader->getTerminator();
  while (I != E) {
    Instruction *Inst = &*I++;

    // If the instruction's operands are invariant and it doesn't read or write
    // memory, then it is safe to hoist.  Doing this doesn't change the order of
    // execution in the preheader, but does prevent the instruction from
    // executing in each iteration of the loop.  This means it is safe to hoist
    // something that might trap, but isn't safe to hoist something that reads
    // memory (without proving that the loop doesn't write).
    if (L->hasLoopInvariantOperands(Inst) && !Inst->mayReadFromMemory() &&
        !Inst->mayWriteToMemory() && !isa<TerminatorInst>(Inst) &&
        !isa<DbgInfoIntrinsic>(Inst) && !isa<AllocaInst>(Inst)) {
      Inst->moveBefore(LoopEntryBranch);
      continue;
    }

    // Otherwise, create a duplicate of the instruction.
    Instruction *C = Inst->clone();

    // Eagerly remap the operands of the instruction.
    RemapInstruction(C, ValueMap,
                     RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

    // With the operands remapped, see if the instruction constant folds or is
    // otherwise simplifyable.  This commonly occurs because the entry from PHI
    // nodes allows icmps and other instructions to fold.
    // FIXME: Provide TLI, DT, AC to SimplifyInstruction.
    Value *V = SimplifyInstruction(C, DL);
    if (V && LI->replacementPreservesLCSSAForm(C, V)) {
      // If so, then delete the temporary instruction and stick the folded value
      // in the map.
      ValueMap[Inst] = V;
      if (!C->mayHaveSideEffects()) {
        delete C;
        C = nullptr;
      }
    } else {
      ValueMap[Inst] = C;
    }
    if (C) {
      // Otherwise, stick the new instruction into the new block!
      C->setName(Inst->getName());
      C->insertBefore(LoopEntryBranch);
    }
  }

  // Along with all the other instructions, we just cloned OrigHeader's
  // terminator into OrigPreHeader. Fix up the PHI nodes in each of OrigHeader's
  // successors by duplicating their incoming values for OrigHeader.
  TerminatorInst *TI = OrigHeader->getTerminator();
  for (BasicBlock *SuccBB : TI->successors())
    for (BasicBlock::iterator BI = SuccBB->begin();
         PHINode *PN = dyn_cast<PHINode>(BI); ++BI)
      PN->addIncoming(PN->getIncomingValueForBlock(OrigHeader), OrigPreheader);

  // Now that OrigPreHeader has a clone of OrigHeader's terminator, remove
  // OrigPreHeader's old terminator (the original branch into the loop), and
  // remove the corresponding incoming values from the PHI nodes in OrigHeader.
  LoopEntryBranch->eraseFromParent();

  // If there were any uses of instructions in the duplicated block outside the
  // loop, update them, inserting PHI nodes as required
  RewriteUsesOfClonedInstructions(OrigHeader, OrigPreheader, ValueMap);

  // NewHeader is now the header of the loop.
  L->moveToHeader(NewHeader);
  assert(L->getHeader() == NewHeader && "Latch block is our new header");

  // At this point, we've finished our major CFG changes.  As part of cloning
  // the loop into the preheader we've simplified instructions and the
  // duplicated conditional branch may now be branching on a constant.  If it is
  // branching on a constant and if that constant means that we enter the loop,
  // then we fold away the cond branch to an uncond branch.  This simplifies the
  // loop in cases important for nested loops, and it also means we don't have
  // to split as many edges.
  BranchInst *PHBI = cast<BranchInst>(OrigPreheader->getTerminator());
  assert(PHBI->isConditional() && "Should be clone of BI condbr!");
  if (!isa<ConstantInt>(PHBI->getCondition()) ||
      PHBI->getSuccessor(cast<ConstantInt>(PHBI->getCondition())->isZero()) !=
          NewHeader) {
    // The conditional branch can't be folded, handle the general case.
    // Update DominatorTree to reflect the CFG change we just made.  Then split
    // edges as necessary to preserve LoopSimplify form.
    if (DT) {
      // Everything that was dominated by the old loop header is now dominated
      // by the original loop preheader. Conceptually the header was merged
      // into the preheader, even though we reuse the actual block as a new
      // loop latch.
      DomTreeNode *OrigHeaderNode = DT->getNode(OrigHeader);
      SmallVector<DomTreeNode *, 8> HeaderChildren(OrigHeaderNode->begin(),
                                                   OrigHeaderNode->end());
      DomTreeNode *OrigPreheaderNode = DT->getNode(OrigPreheader);
      for (unsigned I = 0, E = HeaderChildren.size(); I != E; ++I)
        DT->changeImmediateDominator(HeaderChildren[I], OrigPreheaderNode);

      assert(DT->getNode(Exit)->getIDom() == OrigPreheaderNode);
      assert(DT->getNode(NewHeader)->getIDom() == OrigPreheaderNode);

      // Update OrigHeader to be dominated by the new header block.
      DT->changeImmediateDominator(OrigHeader, OrigLatch);
    }

    // Right now OrigPreHeader has two successors, NewHeader and ExitBlock, and
    // thus is not a preheader anymore.
    // Split the edge to form a real preheader.
    BasicBlock *NewPH = SplitCriticalEdge(
        OrigPreheader, NewHeader,
        CriticalEdgeSplittingOptions(DT, LI).setPreserveLCSSA());
    NewPH->setName(NewHeader->getName() + ".lr.ph");

    // Preserve canonical loop form, which means that 'Exit' should have only
    // one predecessor. Note that Exit could be an exit block for multiple
    // nested loops, causing both of the edges to now be critical and need to
    // be split.
    SmallVector<BasicBlock *, 4> ExitPreds(pred_begin(Exit), pred_end(Exit));
    bool SplitLatchEdge = false;
    for (BasicBlock *ExitPred : ExitPreds) {
      // We only need to split loop exit edges.
      Loop *PredLoop = LI->getLoopFor(ExitPred);
      if (!PredLoop || PredLoop->contains(Exit))
        continue;
      if (isa<IndirectBrInst>(ExitPred->getTerminator()))
        continue;
      SplitLatchEdge |= L->getLoopLatch() == ExitPred;
      BasicBlock *ExitSplit = SplitCriticalEdge(
          ExitPred, Exit,
          CriticalEdgeSplittingOptions(DT, LI).setPreserveLCSSA());
      ExitSplit->moveBefore(Exit);
    }
    assert(SplitLatchEdge &&
           "Despite splitting all preds, failed to split latch exit?");
  } else {
    // We can fold the conditional branch in the preheader, this makes things
    // simpler. The first step is to remove the extra edge to the Exit block.
    Exit->removePredecessor(OrigPreheader, true /*preserve LCSSA*/);
    BranchInst *NewBI = BranchInst::Create(NewHeader, PHBI);
    NewBI->setDebugLoc(PHBI->getDebugLoc());
    PHBI->eraseFromParent();

    // With our CFG finalized, update DomTree if it is available.
    if (DT) {
      // Update OrigHeader to be dominated by the new header block.
      DT->changeImmediateDominator(NewHeader, OrigPreheader);
      DT->changeImmediateDominator(OrigHeader, OrigLatch);

      // Brute force incremental dominator tree update. Call
      // findNearestCommonDominator on all CFG predecessors of each child of the
      // original header.
      DomTreeNode *OrigHeaderNode = DT->getNode(OrigHeader);
      SmallVector<DomTreeNode *, 8> HeaderChildren(OrigHeaderNode->begin(),
                                                   OrigHeaderNode->end());
      bool Changed;
      do {
        Changed = false;
        for (unsigned I = 0, E = HeaderChildren.size(); I != E; ++I) {
          DomTreeNode *Node = HeaderChildren[I];
          BasicBlock *BB = Node->getBlock();

          pred_iterator PI = pred_begin(BB);
          BasicBlock *NearestDom = *PI;
          for (pred_iterator PE = pred_end(BB); PI != PE; ++PI)
            NearestDom = DT->findNearestCommonDominator(NearestDom, *PI);

          // Remember if this changes the DomTree.
          if (Node->getIDom()->getBlock() != NearestDom) {
            DT->changeImmediateDominator(BB, NearestDom);
            Changed = true;
          }
        }

        // If the dominator changed, this may have an effect on other
        // predecessors, continue until we reach a fixpoint.
      } while (Changed);
    }
  }

  assert(L->getLoopPreheader() && "Invalid loop preheader after loop rotation");
  assert(L->getLoopLatch() && "Invalid loop latch after loop rotation");

  // Now that the CFG and DomTree are in a consistent state again, try to merge
  // the OrigHeader block into OrigLatch.  This will succeed if they are
  // connected by an unconditional branch.  This is just a cleanup so the
  // emitted code isn't too gross in this common case.
  MergeBlockIntoPredecessor(OrigHeader, DT, LI);

  DEBUG(dbgs() << "LoopRotation: into "; L->dump());

  ++NumRotated;
  return true;
}

/// Determine whether the instructions in this range may be safely and cheaply
/// speculated. This is not an important enough situation to develop complex
/// heuristics. We handle a single arithmetic instruction along with any type
/// conversions.
static bool shouldSpeculateInstrs(BasicBlock::iterator Begin,
                                  BasicBlock::iterator End, Loop *L) {
  bool seenIncrement = false;
  bool MultiExitLoop = false;

  if (!L->getExitingBlock())
    MultiExitLoop = true;

  for (BasicBlock::iterator I = Begin; I != End; ++I) {

    if (!isSafeToSpeculativelyExecute(&*I))
      return false;

    if (isa<DbgInfoIntrinsic>(I))
      continue;

    switch (I->getOpcode()) {
    default:
      return false;
    case Instruction::GetElementPtr:
      // GEPs are cheap if all indices are constant.
      if (!cast<GEPOperator>(I)->hasAllConstantIndices())
        return false;
    // fall-thru to increment case
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr: {
      Value *IVOpnd =
          !isa<Constant>(I->getOperand(0))
              ? I->getOperand(0)
              : !isa<Constant>(I->getOperand(1)) ? I->getOperand(1) : nullptr;
      if (!IVOpnd)
        return false;

      // If increment operand is used outside of the loop, this speculation
      // could cause extra live range interference.
      if (MultiExitLoop) {
        for (User *UseI : IVOpnd->users()) {
          auto *UserInst = cast<Instruction>(UseI);
          if (!L->contains(UserInst))
            return false;
        }
      }

      if (seenIncrement)
        return false;
      seenIncrement = true;
      break;
    }
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
      // ignore type conversions
      break;
    }
  }
  return true;
}

/// Fold the loop tail into the loop exit by speculating the loop tail
/// instructions. Typically, this is a single post-increment. In the case of a
/// simple 2-block loop, hoisting the increment can be much better than
/// duplicating the entire loop header. In the case of loops with early exits,
/// rotation will not work anyway, but simplifyLoopLatch will put the loop in
/// canonical form so downstream passes can handle it.
///
/// I don't believe this invalidates SCEV.
bool LoopRotate::simplifyLoopLatch(Loop *L) {
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch || Latch->hasAddressTaken())
    return false;

  BranchInst *Jmp = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!Jmp || !Jmp->isUnconditional())
    return false;

  BasicBlock *LastExit = Latch->getSinglePredecessor();
  if (!LastExit || !L->isLoopExiting(LastExit))
    return false;

  BranchInst *BI = dyn_cast<BranchInst>(LastExit->getTerminator());
  if (!BI)
    return false;

  if (!shouldSpeculateInstrs(Latch->begin(), Jmp->getIterator(), L))
    return false;

  DEBUG(dbgs() << "Folding loop latch " << Latch->getName() << " into "
               << LastExit->getName() << "\n");

  // Hoist the instructions from Latch into LastExit.
  LastExit->getInstList().splice(BI->getIterator(), Latch->getInstList(),
                                 Latch->begin(), Jmp->getIterator());

  unsigned FallThruPath = BI->getSuccessor(0) == Latch ? 0 : 1;
  BasicBlock *Header = Jmp->getSuccessor(0);
  assert(Header == L->getHeader() && "expected a backward branch");

  // Remove Latch from the CFG so that LastExit becomes the new Latch.
  BI->setSuccessor(FallThruPath, Header);
  Latch->replaceSuccessorsPhiUsesWith(LastExit);
  Jmp->eraseFromParent();

  // Nuke the Latch block.
  assert(Latch->empty() && "unable to evacuate Latch");
  LI->removeBlock(Latch);
  if (DT)
    DT->eraseNode(Latch);
  Latch->eraseFromParent();
  return true;
}

/// Rotate \c L, and return true if any modification was made.
bool LoopRotate::processLoop(Loop *L) {
  // Save the loop metadata.
  MDNode *LoopMD = L->getLoopID();

  // Simplify the loop latch before attempting to rotate the header
  // upward. Rotation may not be needed if the loop tail can be folded into the
  // loop exit.
  bool SimplifiedLatch = simplifyLoopLatch(L);

  bool MadeChange = rotateLoop(L, SimplifiedLatch);
  assert((!MadeChange || L->isLoopExiting(L->getLoopLatch())) &&
         "Loop latch should be exiting after loop-rotate.");

  // Restore the loop metadata.
  // NB! We presume LoopRotation DOESN'T ADD its own metadata.
  if ((MadeChange || SimplifiedLatch) && LoopMD)
    L->setLoopID(LoopMD);

  return MadeChange;
}

LoopRotatePass::LoopRotatePass() {}

PreservedAnalyses LoopRotatePass::run(Loop &L, AnalysisManager<Loop> &AM) {
  auto &FAM = AM.getResult<FunctionAnalysisManagerLoopProxy>(L).getManager();
  Function *F = L.getHeader()->getParent();

  auto *LI = FAM.getCachedResult<LoopAnalysis>(*F);
  const auto *TTI = FAM.getCachedResult<TargetIRAnalysis>(*F);
  auto *AC = FAM.getCachedResult<AssumptionAnalysis>(*F);
  assert((LI && TTI && AC) && "Analyses for loop rotation not available");

  // Optional analyses.
  auto *DT = FAM.getCachedResult<DominatorTreeAnalysis>(*F);
  auto *SE = FAM.getCachedResult<ScalarEvolutionAnalysis>(*F);
  LoopRotate LR(DefaultRotationThreshold, LI, TTI, AC, DT, SE);

  bool Changed = LR.processLoop(&L);
  if (!Changed)
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}

namespace {

class LoopRotateLegacyPass : public LoopPass {
  unsigned MaxHeaderSize;

public:
  static char ID; // Pass ID, replacement for typeid
  LoopRotateLegacyPass(int SpecifiedMaxHeaderSize = -1) : LoopPass(ID) {
    initializeLoopRotateLegacyPassPass(*PassRegistry::getPassRegistry());
    if (SpecifiedMaxHeaderSize == -1)
      MaxHeaderSize = DefaultRotationThreshold;
    else
      MaxHeaderSize = unsigned(SpecifiedMaxHeaderSize);
  }

  // LCSSA form makes instruction renaming easier.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
    getLoopAnalysisUsage(AU);
  }

  bool runOnLoop(Loop *L, LPPassManager &LPM) override {
    if (skipLoop(L))
      return false;
    Function &F = *L->getHeader()->getParent();

    auto *LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    const auto *TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    auto *AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    auto *DTWP = getAnalysisIfAvailable<DominatorTreeWrapperPass>();
    auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
    auto *SEWP = getAnalysisIfAvailable<ScalarEvolutionWrapperPass>();
    auto *SE = SEWP ? &SEWP->getSE() : nullptr;
    LoopRotate LR(MaxHeaderSize, LI, TTI, AC, DT, SE);
    return LR.processLoop(L);
  }
};
}

char LoopRotateLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(LoopRotateLegacyPass, "loop-rotate", "Rotate Loops",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LoopPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(LoopRotateLegacyPass, "loop-rotate", "Rotate Loops", false,
                    false)

Pass *llvm::createLoopRotatePass(int MaxHeaderSize) {
  return new LoopRotateLegacyPass(MaxHeaderSize);
}
