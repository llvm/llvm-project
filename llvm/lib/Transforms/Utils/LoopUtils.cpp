//===-- LoopUtils.cpp - Loop Utility functions -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines common loop utility functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionAliasAnalysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-utils"

static const char *LLVMLoopDisableNonforced = "llvm.loop.disable_nonforced";
static const char *LLVMLoopDisableLICM = "llvm.licm.disable";

bool llvm::formDedicatedExitBlocks(Loop *L, DominatorTree *DT, LoopInfo *LI,
                                   MemorySSAUpdater *MSSAU,
                                   bool PreserveLCSSA) {
  bool Changed = false;

  // We re-use a vector for the in-loop predecesosrs.
  SmallVector<BasicBlock *, 4> InLoopPredecessors;

  auto RewriteExit = [&](BasicBlock *BB) {
    assert(InLoopPredecessors.empty() &&
           "Must start with an empty predecessors list!");
    auto Cleanup = make_scope_exit([&] { InLoopPredecessors.clear(); });

    // See if there are any non-loop predecessors of this exit block and
    // keep track of the in-loop predecessors.
    bool IsDedicatedExit = true;
    for (auto *PredBB : predecessors(BB))
      if (L->contains(PredBB)) {
        if (isa<IndirectBrInst>(PredBB->getTerminator()))
          // We cannot rewrite exiting edges from an indirectbr.
          return false;
        if (isa<CallBrInst>(PredBB->getTerminator()))
          // We cannot rewrite exiting edges from a callbr.
          return false;

        InLoopPredecessors.push_back(PredBB);
      } else {
        IsDedicatedExit = false;
      }

    assert(!InLoopPredecessors.empty() && "Must have *some* loop predecessor!");

    // Nothing to do if this is already a dedicated exit.
    if (IsDedicatedExit)
      return false;

    auto *NewExitBB = SplitBlockPredecessors(
        BB, InLoopPredecessors, ".loopexit", DT, LI, MSSAU, PreserveLCSSA);

    if (!NewExitBB)
      LLVM_DEBUG(
          dbgs() << "WARNING: Can't create a dedicated exit block for loop: "
                 << *L << "\n");
    else
      LLVM_DEBUG(dbgs() << "LoopSimplify: Creating dedicated exit block "
                        << NewExitBB->getName() << "\n");
    return true;
  };

  // Walk the exit blocks directly rather than building up a data structure for
  // them, but only visit each one once.
  SmallPtrSet<BasicBlock *, 4> Visited;
  for (auto *BB : L->blocks())
    for (auto *SuccBB : successors(BB)) {
      // We're looking for exit blocks so skip in-loop successors.
      if (L->contains(SuccBB))
        continue;

      // Visit each exit block exactly once.
      if (!Visited.insert(SuccBB).second)
        continue;

      Changed |= RewriteExit(SuccBB);
    }

  return Changed;
}

/// Returns the instructions that use values defined in the loop.
SmallVector<Instruction *, 8> llvm::findDefsUsedOutsideOfLoop(Loop *L) {
  SmallVector<Instruction *, 8> UsedOutside;

  for (auto *Block : L->getBlocks())
    // FIXME: I believe that this could use copy_if if the Inst reference could
    // be adapted into a pointer.
    for (auto &Inst : *Block) {
      auto Users = Inst.users();
      if (any_of(Users, [&](User *U) {
            auto *Use = cast<Instruction>(U);
            return !L->contains(Use->getParent());
          }))
        UsedOutside.push_back(&Inst);
    }

  return UsedOutside;
}

void llvm::getLoopAnalysisUsage(AnalysisUsage &AU) {
  // By definition, all loop passes need the LoopInfo analysis and the
  // Dominator tree it depends on. Because they all participate in the loop
  // pass manager, they must also preserve these.
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();

  // We must also preserve LoopSimplify and LCSSA. We locally access their IDs
  // here because users shouldn't directly get them from this header.
  extern char &LoopSimplifyID;
  extern char &LCSSAID;
  AU.addRequiredID(LoopSimplifyID);
  AU.addPreservedID(LoopSimplifyID);
  AU.addRequiredID(LCSSAID);
  AU.addPreservedID(LCSSAID);
  // This is used in the LPPassManager to perform LCSSA verification on passes
  // which preserve lcssa form
  AU.addRequired<LCSSAVerificationPass>();
  AU.addPreserved<LCSSAVerificationPass>();

  // Loop passes are designed to run inside of a loop pass manager which means
  // that any function analyses they require must be required by the first loop
  // pass in the manager (so that it is computed before the loop pass manager
  // runs) and preserved by all loop pasess in the manager. To make this
  // reasonably robust, the set needed for most loop passes is maintained here.
  // If your loop pass requires an analysis not listed here, you will need to
  // carefully audit the loop pass manager nesting structure that results.
  AU.addRequired<AAResultsWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
  AU.addPreserved<SCEVAAWrapperPass>();
  AU.addRequired<ScalarEvolutionWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  // FIXME: When all loop passes preserve MemorySSA, it can be required and
  // preserved here instead of the individual handling in each pass.
}

/// Manually defined generic "LoopPass" dependency initialization. This is used
/// to initialize the exact set of passes from above in \c
/// getLoopAnalysisUsage. It can be used within a loop pass's initialization
/// with:
///
///   INITIALIZE_PASS_DEPENDENCY(LoopPass)
///
/// As-if "LoopPass" were a pass.
void llvm::initializeLoopPassPass(PassRegistry &Registry) {
  INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
  INITIALIZE_PASS_DEPENDENCY(LCSSAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(BasicAAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(GlobalsAAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(SCEVAAWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(ScalarEvolutionWrapperPass)
  INITIALIZE_PASS_DEPENDENCY(MemorySSAWrapperPass)
}

/// Create MDNode for input string.
static MDNode *createStringMetadata(Loop *TheLoop, StringRef Name, unsigned V) {
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  Metadata *MDs[] = {
      MDString::get(Context, Name),
      ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Context), V))};
  return MDNode::get(Context, MDs);
}

/// Set input string into loop metadata by keeping other values intact.
/// If the string is already in loop metadata update value if it is
/// different.
void llvm::addStringMetadataToLoop(Loop *TheLoop, const char *StringMD,
                                   unsigned V) {
  SmallVector<Metadata *, 4> MDs(1);
  // If the loop already has metadata, retain it.
  MDNode *LoopID = TheLoop->getLoopID();
  if (LoopID) {
    for (unsigned i = 1, ie = LoopID->getNumOperands(); i < ie; ++i) {
      MDNode *Node = cast<MDNode>(LoopID->getOperand(i));
      // If it is of form key = value, try to parse it.
      if (Node->getNumOperands() == 2) {
        MDString *S = dyn_cast<MDString>(Node->getOperand(0));
        if (S && S->getString().equals(StringMD)) {
          ConstantInt *IntMD =
              mdconst::extract_or_null<ConstantInt>(Node->getOperand(1));
          if (IntMD && IntMD->getSExtValue() == V)
            // It is already in place. Do nothing.
            return;
          // We need to update the value, so just skip it here and it will
          // be added after copying other existed nodes.
          continue;
        }
      }
      MDs.push_back(Node);
    }
  }
  // Add new metadata.
  MDs.push_back(createStringMetadata(TheLoop, StringMD, V));
  // Replace current metadata node with new one.
  LLVMContext &Context = TheLoop->getHeader()->getContext();
  MDNode *NewLoopID = MDNode::get(Context, MDs);
  // Set operand 0 to refer to the loop id itself.
  NewLoopID->replaceOperandWith(0, NewLoopID);
  TheLoop->setLoopID(NewLoopID);
}

Optional<ElementCount>
llvm::getOptionalElementCountLoopAttribute(const Loop *TheLoop) {
  Optional<int> Width =
      getOptionalIntLoopAttribute(TheLoop, "llvm.loop.vectorize.width");

  if (Width.hasValue()) {
    Optional<int> IsScalable = getOptionalIntLoopAttribute(
        TheLoop, "llvm.loop.vectorize.scalable.enable");
    return ElementCount::get(*Width, IsScalable.getValueOr(false));
  }

  return None;
}

Optional<MDNode *> llvm::makeFollowupLoopID(
    MDNode *OrigLoopID, ArrayRef<StringRef> FollowupOptions,
    const char *InheritOptionsExceptPrefix, bool AlwaysNew) {
  if (!OrigLoopID) {
    if (AlwaysNew)
      return nullptr;
    return None;
  }

  assert(OrigLoopID->getOperand(0) == OrigLoopID);

  bool InheritAllAttrs = !InheritOptionsExceptPrefix;
  bool InheritSomeAttrs =
      InheritOptionsExceptPrefix && InheritOptionsExceptPrefix[0] != '\0';
  SmallVector<Metadata *, 8> MDs;
  MDs.push_back(nullptr);

  bool Changed = false;
  if (InheritAllAttrs || InheritSomeAttrs) {
    for (const MDOperand &Existing : drop_begin(OrigLoopID->operands())) {
      MDNode *Op = cast<MDNode>(Existing.get());

      auto InheritThisAttribute = [InheritSomeAttrs,
                                   InheritOptionsExceptPrefix](MDNode *Op) {
        if (!InheritSomeAttrs)
          return false;

        // Skip malformatted attribute metadata nodes.
        if (Op->getNumOperands() == 0)
          return true;
        Metadata *NameMD = Op->getOperand(0).get();
        if (!isa<MDString>(NameMD))
          return true;
        StringRef AttrName = cast<MDString>(NameMD)->getString();

        // Do not inherit excluded attributes.
        return !AttrName.startswith(InheritOptionsExceptPrefix);
      };

      if (InheritThisAttribute(Op))
        MDs.push_back(Op);
      else
        Changed = true;
    }
  } else {
    // Modified if we dropped at least one attribute.
    Changed = OrigLoopID->getNumOperands() > 1;
  }

  bool HasAnyFollowup = false;
  for (StringRef OptionName : FollowupOptions) {
    MDNode *FollowupNode = findOptionMDForLoopID(OrigLoopID, OptionName);
    if (!FollowupNode)
      continue;

    HasAnyFollowup = true;
    for (const MDOperand &Option : drop_begin(FollowupNode->operands())) {
      MDs.push_back(Option.get());
      Changed = true;
    }
  }

  // Attributes of the followup loop not specified explicity, so signal to the
  // transformation pass to add suitable attributes.
  if (!AlwaysNew && !HasAnyFollowup)
    return None;

  // If no attributes were added or remove, the previous loop Id can be reused.
  if (!AlwaysNew && !Changed)
    return OrigLoopID;

  // No attributes is equivalent to having no !llvm.loop metadata at all.
  if (MDs.size() == 1)
    return nullptr;

  // Build the new loop ID.
  MDTuple *FollowupLoopID = MDNode::get(OrigLoopID->getContext(), MDs);
  FollowupLoopID->replaceOperandWith(0, FollowupLoopID);
  return FollowupLoopID;
}

bool llvm::hasDisableAllTransformsHint(const Loop *L) {
  return getBooleanLoopAttribute(L, LLVMLoopDisableNonforced);
}

bool llvm::hasDisableLICMTransformsHint(const Loop *L) {
  return getBooleanLoopAttribute(L, LLVMLoopDisableLICM);
}

TransformationMode llvm::hasUnrollTransformation(const Loop *L) {
  if (getBooleanLoopAttribute(L, "llvm.loop.unroll.disable"))
    return TM_SuppressedByUser;

  Optional<int> Count =
      getOptionalIntLoopAttribute(L, "llvm.loop.unroll.count");
  if (Count.hasValue())
    return Count.getValue() == 1 ? TM_SuppressedByUser : TM_ForcedByUser;

  if (getBooleanLoopAttribute(L, "llvm.loop.unroll.enable"))
    return TM_ForcedByUser;

  if (getBooleanLoopAttribute(L, "llvm.loop.unroll.full"))
    return TM_ForcedByUser;

  if (hasDisableAllTransformsHint(L))
    return TM_Disable;

  return TM_Unspecified;
}

TransformationMode llvm::hasUnrollAndJamTransformation(const Loop *L) {
  if (getBooleanLoopAttribute(L, "llvm.loop.unroll_and_jam.disable"))
    return TM_SuppressedByUser;

  Optional<int> Count =
      getOptionalIntLoopAttribute(L, "llvm.loop.unroll_and_jam.count");
  if (Count.hasValue())
    return Count.getValue() == 1 ? TM_SuppressedByUser : TM_ForcedByUser;

  if (getBooleanLoopAttribute(L, "llvm.loop.unroll_and_jam.enable"))
    return TM_ForcedByUser;

  if (hasDisableAllTransformsHint(L))
    return TM_Disable;

  return TM_Unspecified;
}

TransformationMode llvm::hasVectorizeTransformation(const Loop *L) {
  Optional<bool> Enable =
      getOptionalBoolLoopAttribute(L, "llvm.loop.vectorize.enable");

  if (Enable == false)
    return TM_SuppressedByUser;

  Optional<ElementCount> VectorizeWidth =
      getOptionalElementCountLoopAttribute(L);
  Optional<int> InterleaveCount =
      getOptionalIntLoopAttribute(L, "llvm.loop.interleave.count");

  // 'Forcing' vector width and interleave count to one effectively disables
  // this tranformation.
  if (Enable == true && VectorizeWidth && VectorizeWidth->isScalar() &&
      InterleaveCount == 1)
    return TM_SuppressedByUser;

  if (getBooleanLoopAttribute(L, "llvm.loop.isvectorized"))
    return TM_Disable;

  if (Enable == true)
    return TM_ForcedByUser;

  if ((VectorizeWidth && VectorizeWidth->isScalar()) && InterleaveCount == 1)
    return TM_Disable;

  if ((VectorizeWidth && VectorizeWidth->isVector()) || InterleaveCount > 1)
    return TM_Enable;

  if (hasDisableAllTransformsHint(L))
    return TM_Disable;

  return TM_Unspecified;
}

TransformationMode llvm::hasDistributeTransformation(const Loop *L) {
  if (getBooleanLoopAttribute(L, "llvm.loop.distribute.enable"))
    return TM_ForcedByUser;

  if (hasDisableAllTransformsHint(L))
    return TM_Disable;

  return TM_Unspecified;
}

TransformationMode llvm::hasLICMVersioningTransformation(const Loop *L) {
  if (getBooleanLoopAttribute(L, "llvm.loop.licm_versioning.disable"))
    return TM_SuppressedByUser;

  if (hasDisableAllTransformsHint(L))
    return TM_Disable;

  return TM_Unspecified;
}

/// Does a BFS from a given node to all of its children inside a given loop.
/// The returned vector of nodes includes the starting point.
SmallVector<DomTreeNode *, 16>
llvm::collectChildrenInLoop(DomTreeNode *N, const Loop *CurLoop) {
  SmallVector<DomTreeNode *, 16> Worklist;
  auto AddRegionToWorklist = [&](DomTreeNode *DTN) {
    // Only include subregions in the top level loop.
    BasicBlock *BB = DTN->getBlock();
    if (CurLoop->contains(BB))
      Worklist.push_back(DTN);
  };

  AddRegionToWorklist(N);

  for (size_t I = 0; I < Worklist.size(); I++) {
    for (DomTreeNode *Child : Worklist[I]->children())
      AddRegionToWorklist(Child);
  }

  return Worklist;
}

void llvm::deleteDeadLoop(Loop *L, DominatorTree *DT, ScalarEvolution *SE,
                          LoopInfo *LI, MemorySSA *MSSA) {
  assert((!DT || L->isLCSSAForm(*DT)) && "Expected LCSSA!");
  auto *Preheader = L->getLoopPreheader();
  assert(Preheader && "Preheader should exist!");

  std::unique_ptr<MemorySSAUpdater> MSSAU;
  if (MSSA)
    MSSAU = std::make_unique<MemorySSAUpdater>(MSSA);

  // Now that we know the removal is safe, remove the loop by changing the
  // branch from the preheader to go to the single exit block.
  //
  // Because we're deleting a large chunk of code at once, the sequence in which
  // we remove things is very important to avoid invalidation issues.

  // Tell ScalarEvolution that the loop is deleted. Do this before
  // deleting the loop so that ScalarEvolution can look at the loop
  // to determine what it needs to clean up.
  if (SE)
    SE->forgetLoop(L);

  auto *OldBr = dyn_cast<BranchInst>(Preheader->getTerminator());
  assert(OldBr && "Preheader must end with a branch");
  assert(OldBr->isUnconditional() && "Preheader must have a single successor");
  // Connect the preheader to the exit block. Keep the old edge to the header
  // around to perform the dominator tree update in two separate steps
  // -- #1 insertion of the edge preheader -> exit and #2 deletion of the edge
  // preheader -> header.
  //
  //
  // 0.  Preheader          1.  Preheader           2.  Preheader
  //        |                    |   |                   |
  //        V                    |   V                   |
  //      Header <--\            | Header <--\           | Header <--\
  //       |  |     |            |  |  |     |           |  |  |     |
  //       |  V     |            |  |  V     |           |  |  V     |
  //       | Body --/            |  | Body --/           |  | Body --/
  //       V                     V  V                    V  V
  //      Exit                   Exit                    Exit
  //
  // By doing this is two separate steps we can perform the dominator tree
  // update without using the batch update API.
  //
  // Even when the loop is never executed, we cannot remove the edge from the
  // source block to the exit block. Consider the case where the unexecuted loop
  // branches back to an outer loop. If we deleted the loop and removed the edge
  // coming to this inner loop, this will break the outer loop structure (by
  // deleting the backedge of the outer loop). If the outer loop is indeed a
  // non-loop, it will be deleted in a future iteration of loop deletion pass.
  IRBuilder<> Builder(OldBr);

  auto *ExitBlock = L->getUniqueExitBlock();
  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  if (ExitBlock) {
    assert(ExitBlock && "Should have a unique exit block!");
    assert(L->hasDedicatedExits() && "Loop should have dedicated exits!");

    Builder.CreateCondBr(Builder.getFalse(), L->getHeader(), ExitBlock);
    // Remove the old branch. The conditional branch becomes a new terminator.
    OldBr->eraseFromParent();

    // Rewrite phis in the exit block to get their inputs from the Preheader
    // instead of the exiting block.
    for (PHINode &P : ExitBlock->phis()) {
      // Set the zero'th element of Phi to be from the preheader and remove all
      // other incoming values. Given the loop has dedicated exits, all other
      // incoming values must be from the exiting blocks.
      int PredIndex = 0;
      P.setIncomingBlock(PredIndex, Preheader);
      // Removes all incoming values from all other exiting blocks (including
      // duplicate values from an exiting block).
      // Nuke all entries except the zero'th entry which is the preheader entry.
      // NOTE! We need to remove Incoming Values in the reverse order as done
      // below, to keep the indices valid for deletion (removeIncomingValues
      // updates getNumIncomingValues and shifts all values down into the
      // operand being deleted).
      for (unsigned i = 0, e = P.getNumIncomingValues() - 1; i != e; ++i)
        P.removeIncomingValue(e - i, false);

      assert((P.getNumIncomingValues() == 1 &&
              P.getIncomingBlock(PredIndex) == Preheader) &&
             "Should have exactly one value and that's from the preheader!");
    }

    if (DT) {
      DTU.applyUpdates({{DominatorTree::Insert, Preheader, ExitBlock}});
      if (MSSA) {
        MSSAU->applyUpdates({{DominatorTree::Insert, Preheader, ExitBlock}},
                            *DT);
        if (VerifyMemorySSA)
          MSSA->verifyMemorySSA();
      }
    }

    // Disconnect the loop body by branching directly to its exit.
    Builder.SetInsertPoint(Preheader->getTerminator());
    Builder.CreateBr(ExitBlock);
    // Remove the old branch.
    Preheader->getTerminator()->eraseFromParent();
  } else {
    assert(L->hasNoExitBlocks() &&
           "Loop should have either zero or one exit blocks.");

    Builder.SetInsertPoint(OldBr);
    Builder.CreateUnreachable();
    Preheader->getTerminator()->eraseFromParent();
  }

  if (DT) {
    DTU.applyUpdates({{DominatorTree::Delete, Preheader, L->getHeader()}});
    if (MSSA) {
      MSSAU->applyUpdates({{DominatorTree::Delete, Preheader, L->getHeader()}},
                          *DT);
      SmallSetVector<BasicBlock *, 8> DeadBlockSet(L->block_begin(),
                                                   L->block_end());
      MSSAU->removeBlocks(DeadBlockSet);
      if (VerifyMemorySSA)
        MSSA->verifyMemorySSA();
    }
  }

  // Use a map to unique and a vector to guarantee deterministic ordering.
  llvm::SmallDenseSet<std::pair<DIVariable *, DIExpression *>, 4> DeadDebugSet;
  llvm::SmallVector<DbgVariableIntrinsic *, 4> DeadDebugInst;

  if (ExitBlock) {
    // Given LCSSA form is satisfied, we should not have users of instructions
    // within the dead loop outside of the loop. However, LCSSA doesn't take
    // unreachable uses into account. We handle them here.
    // We could do it after drop all references (in this case all users in the
    // loop will be already eliminated and we have less work to do but according
    // to API doc of User::dropAllReferences only valid operation after dropping
    // references, is deletion. So let's substitute all usages of
    // instruction from the loop with undef value of corresponding type first.
    for (auto *Block : L->blocks())
      for (Instruction &I : *Block) {
        auto *Undef = UndefValue::get(I.getType());
        for (Use &U : llvm::make_early_inc_range(I.uses())) {
          if (auto *Usr = dyn_cast<Instruction>(U.getUser()))
            if (L->contains(Usr->getParent()))
              continue;
          // If we have a DT then we can check that uses outside a loop only in
          // unreachable block.
          if (DT)
            assert(!DT->isReachableFromEntry(U) &&
                   "Unexpected user in reachable block");
          U.set(Undef);
        }
        auto *DVI = dyn_cast<DbgVariableIntrinsic>(&I);
        if (!DVI)
          continue;
        auto Key =
            DeadDebugSet.find({DVI->getVariable(), DVI->getExpression()});
        if (Key != DeadDebugSet.end())
          continue;
        DeadDebugSet.insert({DVI->getVariable(), DVI->getExpression()});
        DeadDebugInst.push_back(DVI);
      }

    // After the loop has been deleted all the values defined and modified
    // inside the loop are going to be unavailable.
    // Since debug values in the loop have been deleted, inserting an undef
    // dbg.value truncates the range of any dbg.value before the loop where the
    // loop used to be. This is particularly important for constant values.
    DIBuilder DIB(*ExitBlock->getModule());
    Instruction *InsertDbgValueBefore = ExitBlock->getFirstNonPHI();
    assert(InsertDbgValueBefore &&
           "There should be a non-PHI instruction in exit block, else these "
           "instructions will have no parent.");
    for (auto *DVI : DeadDebugInst)
      DIB.insertDbgValueIntrinsic(UndefValue::get(Builder.getInt32Ty()),
                                  DVI->getVariable(), DVI->getExpression(),
                                  DVI->getDebugLoc(), InsertDbgValueBefore);
  }

  // Remove the block from the reference counting scheme, so that we can
  // delete it freely later.
  for (auto *Block : L->blocks())
    Block->dropAllReferences();

  if (MSSA && VerifyMemorySSA)
    MSSA->verifyMemorySSA();

  if (LI) {
    // Erase the instructions and the blocks without having to worry
    // about ordering because we already dropped the references.
    // NOTE: This iteration is safe because erasing the block does not remove
    // its entry from the loop's block list.  We do that in the next section.
    for (BasicBlock *BB : L->blocks())
      BB->eraseFromParent();

    // Finally, the blocks from loopinfo.  This has to happen late because
    // otherwise our loop iterators won't work.

    SmallPtrSet<BasicBlock *, 8> blocks;
    blocks.insert(L->block_begin(), L->block_end());
    for (BasicBlock *BB : blocks)
      LI->removeBlock(BB);

    // The last step is to update LoopInfo now that we've eliminated this loop.
    // Note: LoopInfo::erase remove the given loop and relink its subloops with
    // its parent. While removeLoop/removeChildLoop remove the given loop but
    // not relink its subloops, which is what we want.
    if (Loop *ParentLoop = L->getParentLoop()) {
      Loop::iterator I = find(*ParentLoop, L);
      assert(I != ParentLoop->end() && "Couldn't find loop");
      ParentLoop->removeChildLoop(I);
    } else {
      Loop::iterator I = find(*LI, L);
      assert(I != LI->end() && "Couldn't find loop");
      LI->removeLoop(I);
    }
    LI->destroy(L);
  }
}

static Loop *getOutermostLoop(Loop *L) {
  while (Loop *Parent = L->getParentLoop())
    L = Parent;
  return L;
}

void llvm::breakLoopBackedge(Loop *L, DominatorTree &DT, ScalarEvolution &SE,
                             LoopInfo &LI, MemorySSA *MSSA) {
  auto *Latch = L->getLoopLatch();
  assert(Latch && "multiple latches not yet supported");
  auto *Header = L->getHeader();
  Loop *OutermostLoop = getOutermostLoop(L);

  SE.forgetLoop(L);

  std::unique_ptr<MemorySSAUpdater> MSSAU;
  if (MSSA)
    MSSAU = std::make_unique<MemorySSAUpdater>(MSSA);

  // Update the CFG and domtree.  We chose to special case a couple of
  // of common cases for code quality and test readability reasons.
  [&]() -> void {
    if (auto *BI = dyn_cast<BranchInst>(Latch->getTerminator())) {
      if (!BI->isConditional()) {
        DomTreeUpdater DTU(&DT, DomTreeUpdater::UpdateStrategy::Eager);
        (void)changeToUnreachable(BI, /*PreserveLCSSA*/ true, &DTU,
                                  MSSAU.get());
        return;
      }

      // Conditional latch/exit - note that latch can be shared by inner
      // and outer loop so the other target doesn't need to an exit
      if (L->isLoopExiting(Latch)) {
        // TODO: Generalize ConstantFoldTerminator so that it can be used
        // here without invalidating LCSSA or MemorySSA.  (Tricky case for
        // LCSSA: header is an exit block of a preceeding sibling loop w/o
        // dedicated exits.)
        const unsigned ExitIdx = L->contains(BI->getSuccessor(0)) ? 1 : 0;
        BasicBlock *ExitBB = BI->getSuccessor(ExitIdx);

        DomTreeUpdater DTU(&DT, DomTreeUpdater::UpdateStrategy::Eager);
        Header->removePredecessor(Latch, true);

        IRBuilder<> Builder(BI);
        auto *NewBI = Builder.CreateBr(ExitBB);
        // Transfer the metadata to the new branch instruction (minus the
        // loop info since this is no longer a loop)
        NewBI->copyMetadata(*BI, {LLVMContext::MD_dbg,
                                  LLVMContext::MD_annotation});

        BI->eraseFromParent();
        DTU.applyUpdates({{DominatorTree::Delete, Latch, Header}});
        if (MSSA)
          MSSAU->applyUpdates({{DominatorTree::Delete, Latch, Header}}, DT);
        return;
      }
    }

    // General case.  By splitting the backedge, and then explicitly making it
    // unreachable we gracefully handle corner cases such as switch and invoke
    // termiantors.
    auto *BackedgeBB = SplitEdge(Latch, Header, &DT, &LI, MSSAU.get());

    DomTreeUpdater DTU(&DT, DomTreeUpdater::UpdateStrategy::Eager);
    (void)changeToUnreachable(BackedgeBB->getTerminator(),
                              /*PreserveLCSSA*/ true, &DTU, MSSAU.get());
  }();

  // Erase (and destroy) this loop instance.  Handles relinking sub-loops
  // and blocks within the loop as needed.
  LI.erase(L);

  // If the loop we broke had a parent, then changeToUnreachable might have
  // caused a block to be removed from the parent loop (see loop_nest_lcssa
  // test case in zero-btc.ll for an example), thus changing the parent's
  // exit blocks.  If that happened, we need to rebuild LCSSA on the outermost
  // loop which might have a had a block removed.
  if (OutermostLoop != L)
    formLCSSARecursively(*OutermostLoop, DT, &LI, &SE);
}


/// Checks if \p L has an exiting latch branch.  There may also be other
/// exiting blocks.  Returns branch instruction terminating the loop
/// latch if above check is successful, nullptr otherwise.
static BranchInst *getExpectedExitLoopLatchBranch(Loop *L) {
  BasicBlock *Latch = L->getLoopLatch();
  if (!Latch)
    return nullptr;

  BranchInst *LatchBR = dyn_cast<BranchInst>(Latch->getTerminator());
  if (!LatchBR || LatchBR->getNumSuccessors() != 2 || !L->isLoopExiting(Latch))
    return nullptr;

  assert((LatchBR->getSuccessor(0) == L->getHeader() ||
          LatchBR->getSuccessor(1) == L->getHeader()) &&
         "At least one edge out of the latch must go to the header");

  return LatchBR;
}

/// Return the estimated trip count for any exiting branch which dominates
/// the loop latch.
static Optional<uint64_t>
getEstimatedTripCount(BranchInst *ExitingBranch, Loop *L,
                      uint64_t &OrigExitWeight) {
  // To estimate the number of times the loop body was executed, we want to
  // know the number of times the backedge was taken, vs. the number of times
  // we exited the loop.
  uint64_t LoopWeight, ExitWeight;
  if (!ExitingBranch->extractProfMetadata(LoopWeight, ExitWeight))
    return None;

  if (L->contains(ExitingBranch->getSuccessor(1)))
    std::swap(LoopWeight, ExitWeight);

  if (!ExitWeight)
    // Don't have a way to return predicated infinite
    return None;

  OrigExitWeight = ExitWeight;

  // Estimated exit count is a ratio of the loop weight by the weight of the
  // edge exiting the loop, rounded to nearest.
  uint64_t ExitCount = llvm::divideNearest(LoopWeight, ExitWeight);
  // Estimated trip count is one plus estimated exit count.
  return ExitCount + 1;
}

Optional<unsigned>
llvm::getLoopEstimatedTripCount(Loop *L,
                                unsigned *EstimatedLoopInvocationWeight) {
  // Currently we take the estimate exit count only from the loop latch,
  // ignoring other exiting blocks.  This can overestimate the trip count
  // if we exit through another exit, but can never underestimate it.
  // TODO: incorporate information from other exits
  if (BranchInst *LatchBranch = getExpectedExitLoopLatchBranch(L)) {
    uint64_t ExitWeight;
    if (Optional<uint64_t> EstTripCount =
        getEstimatedTripCount(LatchBranch, L, ExitWeight)) {
      if (EstimatedLoopInvocationWeight)
        *EstimatedLoopInvocationWeight = ExitWeight;
      return *EstTripCount;
    }
  }
  return None;
}

bool llvm::setLoopEstimatedTripCount(Loop *L, unsigned EstimatedTripCount,
                                     unsigned EstimatedloopInvocationWeight) {
  // At the moment, we currently support changing the estimate trip count of
  // the latch branch only.  We could extend this API to manipulate estimated
  // trip counts for any exit.
  BranchInst *LatchBranch = getExpectedExitLoopLatchBranch(L);
  if (!LatchBranch)
    return false;

  // Calculate taken and exit weights.
  unsigned LatchExitWeight = 0;
  unsigned BackedgeTakenWeight = 0;

  if (EstimatedTripCount > 0) {
    LatchExitWeight = EstimatedloopInvocationWeight;
    BackedgeTakenWeight = (EstimatedTripCount - 1) * LatchExitWeight;
  }

  // Make a swap if back edge is taken when condition is "false".
  if (LatchBranch->getSuccessor(0) != L->getHeader())
    std::swap(BackedgeTakenWeight, LatchExitWeight);

  MDBuilder MDB(LatchBranch->getContext());

  // Set/Update profile metadata.
  LatchBranch->setMetadata(
      LLVMContext::MD_prof,
      MDB.createBranchWeights(BackedgeTakenWeight, LatchExitWeight));

  return true;
}

bool llvm::hasIterationCountInvariantInParent(Loop *InnerLoop,
                                              ScalarEvolution &SE) {
  Loop *OuterL = InnerLoop->getParentLoop();
  if (!OuterL)
    return true;

  // Get the backedge taken count for the inner loop
  BasicBlock *InnerLoopLatch = InnerLoop->getLoopLatch();
  const SCEV *InnerLoopBECountSC = SE.getExitCount(InnerLoop, InnerLoopLatch);
  if (isa<SCEVCouldNotCompute>(InnerLoopBECountSC) ||
      !InnerLoopBECountSC->getType()->isIntegerTy())
    return false;

  // Get whether count is invariant to the outer loop
  ScalarEvolution::LoopDisposition LD =
      SE.getLoopDisposition(InnerLoopBECountSC, OuterL);
  if (LD != ScalarEvolution::LoopInvariant)
    return false;

  return true;
}

CmpInst::Predicate llvm::getMinMaxReductionPredicate(RecurKind RK) {
  switch (RK) {
  default:
    llvm_unreachable("Unknown min/max recurrence kind");
  case RecurKind::UMin:
    return CmpInst::ICMP_ULT;
  case RecurKind::UMax:
    return CmpInst::ICMP_UGT;
  case RecurKind::SMin:
    return CmpInst::ICMP_SLT;
  case RecurKind::SMax:
    return CmpInst::ICMP_SGT;
  case RecurKind::FMin:
    return CmpInst::FCMP_OLT;
  case RecurKind::FMax:
    return CmpInst::FCMP_OGT;
  }
}

Value *llvm::createSelectCmpOp(IRBuilderBase &Builder, Value *StartVal,
                               RecurKind RK, Value *Left, Value *Right) {
  if (auto VTy = dyn_cast<VectorType>(Left->getType()))
    StartVal = Builder.CreateVectorSplat(VTy->getElementCount(), StartVal);
  Value *Cmp =
      Builder.CreateCmp(CmpInst::ICMP_NE, Left, StartVal, "rdx.select.cmp");
  return Builder.CreateSelect(Cmp, Left, Right, "rdx.select");
}

Value *llvm::createMinMaxOp(IRBuilderBase &Builder, RecurKind RK, Value *Left,
                            Value *Right) {
  CmpInst::Predicate Pred = getMinMaxReductionPredicate(RK);
  Value *Cmp = Builder.CreateCmp(Pred, Left, Right, "rdx.minmax.cmp");
  Value *Select = Builder.CreateSelect(Cmp, Left, Right, "rdx.minmax.select");
  return Select;
}

// Helper to generate an ordered reduction.
Value *llvm::getOrderedReduction(IRBuilderBase &Builder, Value *Acc, Value *Src,
                                 unsigned Op, RecurKind RdxKind) {
  unsigned VF = cast<FixedVectorType>(Src->getType())->getNumElements();

  // Extract and apply reduction ops in ascending order:
  // e.g. ((((Acc + Scl[0]) + Scl[1]) + Scl[2]) + ) ... + Scl[VF-1]
  Value *Result = Acc;
  for (unsigned ExtractIdx = 0; ExtractIdx != VF; ++ExtractIdx) {
    Value *Ext =
        Builder.CreateExtractElement(Src, Builder.getInt32(ExtractIdx));

    if (Op != Instruction::ICmp && Op != Instruction::FCmp) {
      Result = Builder.CreateBinOp((Instruction::BinaryOps)Op, Result, Ext,
                                   "bin.rdx");
    } else {
      assert(RecurrenceDescriptor::isMinMaxRecurrenceKind(RdxKind) &&
             "Invalid min/max");
      Result = createMinMaxOp(Builder, RdxKind, Result, Ext);
    }
  }

  return Result;
}

// Helper to generate a log2 shuffle reduction.
Value *llvm::getShuffleReduction(IRBuilderBase &Builder, Value *Src,
                                 unsigned Op, RecurKind RdxKind) {
  unsigned VF = cast<FixedVectorType>(Src->getType())->getNumElements();
  // VF is a power of 2 so we can emit the reduction using log2(VF) shuffles
  // and vector ops, reducing the set of values being computed by half each
  // round.
  assert(isPowerOf2_32(VF) &&
         "Reduction emission only supported for pow2 vectors!");
  // Note: fast-math-flags flags are controlled by the builder configuration
  // and are assumed to apply to all generated arithmetic instructions.  Other
  // poison generating flags (nsw/nuw/inbounds/inrange/exact) are not part
  // of the builder configuration, and since they're not passed explicitly,
  // will never be relevant here.  Note that it would be generally unsound to
  // propagate these from an intrinsic call to the expansion anyways as we/
  // change the order of operations.
  Value *TmpVec = Src;
  SmallVector<int, 32> ShuffleMask(VF);
  for (unsigned i = VF; i != 1; i >>= 1) {
    // Move the upper half of the vector to the lower half.
    for (unsigned j = 0; j != i / 2; ++j)
      ShuffleMask[j] = i / 2 + j;

    // Fill the rest of the mask with undef.
    std::fill(&ShuffleMask[i / 2], ShuffleMask.end(), -1);

    Value *Shuf = Builder.CreateShuffleVector(TmpVec, ShuffleMask, "rdx.shuf");

    if (Op != Instruction::ICmp && Op != Instruction::FCmp) {
      TmpVec = Builder.CreateBinOp((Instruction::BinaryOps)Op, TmpVec, Shuf,
                                   "bin.rdx");
    } else {
      assert(RecurrenceDescriptor::isMinMaxRecurrenceKind(RdxKind) &&
             "Invalid min/max");
      TmpVec = createMinMaxOp(Builder, RdxKind, TmpVec, Shuf);
    }
  }
  // The result is in the first element of the vector.
  return Builder.CreateExtractElement(TmpVec, Builder.getInt32(0));
}

Value *llvm::createSelectCmpTargetReduction(IRBuilderBase &Builder,
                                            const TargetTransformInfo *TTI,
                                            Value *Src,
                                            const RecurrenceDescriptor &Desc,
                                            PHINode *OrigPhi) {
  assert(RecurrenceDescriptor::isSelectCmpRecurrenceKind(
             Desc.getRecurrenceKind()) &&
         "Unexpected reduction kind");
  Value *InitVal = Desc.getRecurrenceStartValue();
  Value *NewVal = nullptr;

  // First use the original phi to determine the new value we're trying to
  // select from in the loop.
  SelectInst *SI = nullptr;
  for (auto *U : OrigPhi->users()) {
    if ((SI = dyn_cast<SelectInst>(U)))
      break;
  }
  assert(SI && "One user of the original phi should be a select");

  if (SI->getTrueValue() == OrigPhi)
    NewVal = SI->getFalseValue();
  else {
    assert(SI->getFalseValue() == OrigPhi &&
           "At least one input to the select should be the original Phi");
    NewVal = SI->getTrueValue();
  }

  // Create a splat vector with the new value and compare this to the vector
  // we want to reduce.
  ElementCount EC = cast<VectorType>(Src->getType())->getElementCount();
  Value *Right = Builder.CreateVectorSplat(EC, InitVal);
  Value *Cmp =
      Builder.CreateCmp(CmpInst::ICMP_NE, Src, Right, "rdx.select.cmp");

  // If any predicate is true it means that we want to select the new value.
  Cmp = Builder.CreateOrReduce(Cmp);
  return Builder.CreateSelect(Cmp, NewVal, InitVal, "rdx.select");
}

Value *llvm::createSimpleTargetReduction(IRBuilderBase &Builder,
                                         const TargetTransformInfo *TTI,
                                         Value *Src, RecurKind RdxKind) {
  auto *SrcVecEltTy = cast<VectorType>(Src->getType())->getElementType();
  switch (RdxKind) {
  case RecurKind::Add:
    return Builder.CreateAddReduce(Src);
  case RecurKind::Mul:
    return Builder.CreateMulReduce(Src);
  case RecurKind::And:
    return Builder.CreateAndReduce(Src);
  case RecurKind::Or:
    return Builder.CreateOrReduce(Src);
  case RecurKind::Xor:
    return Builder.CreateXorReduce(Src);
  case RecurKind::FMulAdd:
  case RecurKind::FAdd:
    return Builder.CreateFAddReduce(ConstantFP::getNegativeZero(SrcVecEltTy),
                                    Src);
  case RecurKind::FMul:
    return Builder.CreateFMulReduce(ConstantFP::get(SrcVecEltTy, 1.0), Src);
  case RecurKind::SMax:
    return Builder.CreateIntMaxReduce(Src, true);
  case RecurKind::SMin:
    return Builder.CreateIntMinReduce(Src, true);
  case RecurKind::UMax:
    return Builder.CreateIntMaxReduce(Src, false);
  case RecurKind::UMin:
    return Builder.CreateIntMinReduce(Src, false);
  case RecurKind::FMax:
    return Builder.CreateFPMaxReduce(Src);
  case RecurKind::FMin:
    return Builder.CreateFPMinReduce(Src);
  default:
    llvm_unreachable("Unhandled opcode");
  }
}

Value *llvm::createTargetReduction(IRBuilderBase &B,
                                   const TargetTransformInfo *TTI,
                                   const RecurrenceDescriptor &Desc, Value *Src,
                                   PHINode *OrigPhi) {
  // TODO: Support in-order reductions based on the recurrence descriptor.
  // All ops in the reduction inherit fast-math-flags from the recurrence
  // descriptor.
  IRBuilderBase::FastMathFlagGuard FMFGuard(B);
  B.setFastMathFlags(Desc.getFastMathFlags());

  RecurKind RK = Desc.getRecurrenceKind();
  if (RecurrenceDescriptor::isSelectCmpRecurrenceKind(RK))
    return createSelectCmpTargetReduction(B, TTI, Src, Desc, OrigPhi);

  return createSimpleTargetReduction(B, TTI, Src, RK);
}

Value *llvm::createOrderedReduction(IRBuilderBase &B,
                                    const RecurrenceDescriptor &Desc,
                                    Value *Src, Value *Start) {
  assert((Desc.getRecurrenceKind() == RecurKind::FAdd ||
          Desc.getRecurrenceKind() == RecurKind::FMulAdd) &&
         "Unexpected reduction kind");
  assert(Src->getType()->isVectorTy() && "Expected a vector type");
  assert(!Start->getType()->isVectorTy() && "Expected a scalar type");

  return B.CreateFAddReduce(Start, Src);
}

void llvm::propagateIRFlags(Value *I, ArrayRef<Value *> VL, Value *OpValue) {
  auto *VecOp = dyn_cast<Instruction>(I);
  if (!VecOp)
    return;
  auto *Intersection = (OpValue == nullptr) ? dyn_cast<Instruction>(VL[0])
                                            : dyn_cast<Instruction>(OpValue);
  if (!Intersection)
    return;
  const unsigned Opcode = Intersection->getOpcode();
  VecOp->copyIRFlags(Intersection);
  for (auto *V : VL) {
    auto *Instr = dyn_cast<Instruction>(V);
    if (!Instr)
      continue;
    if (OpValue == nullptr || Opcode == Instr->getOpcode())
      VecOp->andIRFlags(V);
  }
}

bool llvm::isKnownNegativeInLoop(const SCEV *S, const Loop *L,
                                 ScalarEvolution &SE) {
  const SCEV *Zero = SE.getZero(S->getType());
  return SE.isAvailableAtLoopEntry(S, L) &&
         SE.isLoopEntryGuardedByCond(L, ICmpInst::ICMP_SLT, S, Zero);
}

bool llvm::isKnownNonNegativeInLoop(const SCEV *S, const Loop *L,
                                    ScalarEvolution &SE) {
  const SCEV *Zero = SE.getZero(S->getType());
  return SE.isAvailableAtLoopEntry(S, L) &&
         SE.isLoopEntryGuardedByCond(L, ICmpInst::ICMP_SGE, S, Zero);
}

bool llvm::cannotBeMinInLoop(const SCEV *S, const Loop *L, ScalarEvolution &SE,
                             bool Signed) {
  unsigned BitWidth = cast<IntegerType>(S->getType())->getBitWidth();
  APInt Min = Signed ? APInt::getSignedMinValue(BitWidth) :
    APInt::getMinValue(BitWidth);
  auto Predicate = Signed ? ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT;
  return SE.isAvailableAtLoopEntry(S, L) &&
         SE.isLoopEntryGuardedByCond(L, Predicate, S,
                                     SE.getConstant(Min));
}

bool llvm::cannotBeMaxInLoop(const SCEV *S, const Loop *L, ScalarEvolution &SE,
                             bool Signed) {
  unsigned BitWidth = cast<IntegerType>(S->getType())->getBitWidth();
  APInt Max = Signed ? APInt::getSignedMaxValue(BitWidth) :
    APInt::getMaxValue(BitWidth);
  auto Predicate = Signed ? ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT;
  return SE.isAvailableAtLoopEntry(S, L) &&
         SE.isLoopEntryGuardedByCond(L, Predicate, S,
                                     SE.getConstant(Max));
}

//===----------------------------------------------------------------------===//
// rewriteLoopExitValues - Optimize IV users outside the loop.
// As a side effect, reduces the amount of IV processing within the loop.
//===----------------------------------------------------------------------===//

static bool hasHardUserWithinLoop(const Loop *L, const Instruction *I) {
  SmallPtrSet<const Instruction *, 8> Visited;
  SmallVector<const Instruction *, 8> WorkList;
  Visited.insert(I);
  WorkList.push_back(I);
  while (!WorkList.empty()) {
    const Instruction *Curr = WorkList.pop_back_val();
    // This use is outside the loop, nothing to do.
    if (!L->contains(Curr))
      continue;
    // Do we assume it is a "hard" use which will not be eliminated easily?
    if (Curr->mayHaveSideEffects())
      return true;
    // Otherwise, add all its users to worklist.
    for (auto U : Curr->users()) {
      auto *UI = cast<Instruction>(U);
      if (Visited.insert(UI).second)
        WorkList.push_back(UI);
    }
  }
  return false;
}

// Collect information about PHI nodes which can be transformed in
// rewriteLoopExitValues.
struct RewritePhi {
  PHINode *PN;               // For which PHI node is this replacement?
  unsigned Ith;              // For which incoming value?
  const SCEV *ExpansionSCEV; // The SCEV of the incoming value we are rewriting.
  Instruction *ExpansionPoint; // Where we'd like to expand that SCEV?
  bool HighCost;               // Is this expansion a high-cost?

  RewritePhi(PHINode *P, unsigned I, const SCEV *Val, Instruction *ExpansionPt,
             bool H)
      : PN(P), Ith(I), ExpansionSCEV(Val), ExpansionPoint(ExpansionPt),
        HighCost(H) {}
};

// Check whether it is possible to delete the loop after rewriting exit
// value. If it is possible, ignore ReplaceExitValue and do rewriting
// aggressively.
static bool canLoopBeDeleted(Loop *L, SmallVector<RewritePhi, 8> &RewritePhiSet) {
  BasicBlock *Preheader = L->getLoopPreheader();
  // If there is no preheader, the loop will not be deleted.
  if (!Preheader)
    return false;

  // In LoopDeletion pass Loop can be deleted when ExitingBlocks.size() > 1.
  // We obviate multiple ExitingBlocks case for simplicity.
  // TODO: If we see testcase with multiple ExitingBlocks can be deleted
  // after exit value rewriting, we can enhance the logic here.
  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);
  if (ExitBlocks.size() != 1 || ExitingBlocks.size() != 1)
    return false;

  BasicBlock *ExitBlock = ExitBlocks[0];
  BasicBlock::iterator BI = ExitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    Value *Incoming = P->getIncomingValueForBlock(ExitingBlocks[0]);

    // If the Incoming value of P is found in RewritePhiSet, we know it
    // could be rewritten to use a loop invariant value in transformation
    // phase later. Skip it in the loop invariant check below.
    bool found = false;
    for (const RewritePhi &Phi : RewritePhiSet) {
      unsigned i = Phi.Ith;
      if (Phi.PN == P && (Phi.PN)->getIncomingValue(i) == Incoming) {
        found = true;
        break;
      }
    }

    Instruction *I;
    if (!found && (I = dyn_cast<Instruction>(Incoming)))
      if (!L->hasLoopInvariantOperands(I))
        return false;

    ++BI;
  }

  for (auto *BB : L->blocks())
    if (llvm::any_of(*BB, [](Instruction &I) {
          return I.mayHaveSideEffects();
        }))
      return false;

  return true;
}

int llvm::rewriteLoopExitValues(Loop *L, LoopInfo *LI, TargetLibraryInfo *TLI,
                                ScalarEvolution *SE,
                                const TargetTransformInfo *TTI,
                                SCEVExpander &Rewriter, DominatorTree *DT,
                                ReplaceExitVal ReplaceExitValue,
                                SmallVector<WeakTrackingVH, 16> &DeadInsts) {
  // Check a pre-condition.
  assert(L->isRecursivelyLCSSAForm(*DT, *LI) &&
         "Indvars did not preserve LCSSA!");

  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getUniqueExitBlocks(ExitBlocks);

  SmallVector<RewritePhi, 8> RewritePhiSet;
  // Find all values that are computed inside the loop, but used outside of it.
  // Because of LCSSA, these values will only occur in LCSSA PHI Nodes.  Scan
  // the exit blocks of the loop to find them.
  for (BasicBlock *ExitBB : ExitBlocks) {
    // If there are no PHI nodes in this exit block, then no values defined
    // inside the loop are used on this path, skip it.
    PHINode *PN = dyn_cast<PHINode>(ExitBB->begin());
    if (!PN) continue;

    unsigned NumPreds = PN->getNumIncomingValues();

    // Iterate over all of the PHI nodes.
    BasicBlock::iterator BBI = ExitBB->begin();
    while ((PN = dyn_cast<PHINode>(BBI++))) {
      if (PN->use_empty())
        continue; // dead use, don't replace it

      if (!SE->isSCEVable(PN->getType()))
        continue;

      // Iterate over all of the values in all the PHI nodes.
      for (unsigned i = 0; i != NumPreds; ++i) {
        // If the value being merged in is not integer or is not defined
        // in the loop, skip it.
        Value *InVal = PN->getIncomingValue(i);
        if (!isa<Instruction>(InVal))
          continue;

        // If this pred is for a subloop, not L itself, skip it.
        if (LI->getLoopFor(PN->getIncomingBlock(i)) != L)
          continue; // The Block is in a subloop, skip it.

        // Check that InVal is defined in the loop.
        Instruction *Inst = cast<Instruction>(InVal);
        if (!L->contains(Inst))
          continue;

        // Okay, this instruction has a user outside of the current loop
        // and varies predictably *inside* the loop.  Evaluate the value it
        // contains when the loop exits, if possible.  We prefer to start with
        // expressions which are true for all exits (so as to maximize
        // expression reuse by the SCEVExpander), but resort to per-exit
        // evaluation if that fails.
        const SCEV *ExitValue = SE->getSCEVAtScope(Inst, L->getParentLoop());
        if (isa<SCEVCouldNotCompute>(ExitValue) ||
            !SE->isLoopInvariant(ExitValue, L) ||
            !isSafeToExpand(ExitValue, *SE)) {
          // TODO: This should probably be sunk into SCEV in some way; maybe a
          // getSCEVForExit(SCEV*, L, ExitingBB)?  It can be generalized for
          // most SCEV expressions and other recurrence types (e.g. shift
          // recurrences).  Is there existing code we can reuse?
          const SCEV *ExitCount = SE->getExitCount(L, PN->getIncomingBlock(i));
          if (isa<SCEVCouldNotCompute>(ExitCount))
            continue;
          if (auto *AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(Inst)))
            if (AddRec->getLoop() == L)
              ExitValue = AddRec->evaluateAtIteration(ExitCount, *SE);
          if (isa<SCEVCouldNotCompute>(ExitValue) ||
              !SE->isLoopInvariant(ExitValue, L) ||
              !isSafeToExpand(ExitValue, *SE))
            continue;
        }

        // Computing the value outside of the loop brings no benefit if it is
        // definitely used inside the loop in a way which can not be optimized
        // away. Avoid doing so unless we know we have a value which computes
        // the ExitValue already. TODO: This should be merged into SCEV
        // expander to leverage its knowledge of existing expressions.
        if (ReplaceExitValue != AlwaysRepl && !isa<SCEVConstant>(ExitValue) &&
            !isa<SCEVUnknown>(ExitValue) && hasHardUserWithinLoop(L, Inst))
          continue;

        // Check if expansions of this SCEV would count as being high cost.
        bool HighCost = Rewriter.isHighCostExpansion(
            ExitValue, L, SCEVCheapExpansionBudget, TTI, Inst);

        // Note that we must not perform expansions until after
        // we query *all* the costs, because if we perform temporary expansion
        // inbetween, one that we might not intend to keep, said expansion
        // *may* affect cost calculation of the the next SCEV's we'll query,
        // and next SCEV may errneously get smaller cost.

        // Collect all the candidate PHINodes to be rewritten.
        RewritePhiSet.emplace_back(PN, i, ExitValue, Inst, HighCost);
      }
    }
  }

  // TODO: evaluate whether it is beneficial to change how we calculate
  // high-cost: if we have SCEV 'A' which we know we will expand, should we
  // calculate the cost of other SCEV's after expanding SCEV 'A', thus
  // potentially giving cost bonus to those other SCEV's?

  bool LoopCanBeDel = canLoopBeDeleted(L, RewritePhiSet);
  int NumReplaced = 0;

  // Transformation.
  for (const RewritePhi &Phi : RewritePhiSet) {
    PHINode *PN = Phi.PN;

    // Only do the rewrite when the ExitValue can be expanded cheaply.
    // If LoopCanBeDel is true, rewrite exit value aggressively.
    if (ReplaceExitValue == OnlyCheapRepl && !LoopCanBeDel && Phi.HighCost)
      continue;

    Value *ExitVal = Rewriter.expandCodeFor(
        Phi.ExpansionSCEV, Phi.PN->getType(), Phi.ExpansionPoint);

    LLVM_DEBUG(dbgs() << "rewriteLoopExitValues: AfterLoopVal = " << *ExitVal
                      << '\n'
                      << "  LoopVal = " << *(Phi.ExpansionPoint) << "\n");

#ifndef NDEBUG
    // If we reuse an instruction from a loop which is neither L nor one of
    // its containing loops, we end up breaking LCSSA form for this loop by
    // creating a new use of its instruction.
    if (auto *ExitInsn = dyn_cast<Instruction>(ExitVal))
      if (auto *EVL = LI->getLoopFor(ExitInsn->getParent()))
        if (EVL != L)
          assert(EVL->contains(L) && "LCSSA breach detected!");
#endif

    NumReplaced++;
    Instruction *Inst = cast<Instruction>(PN->getIncomingValue(Phi.Ith));
    PN->setIncomingValue(Phi.Ith, ExitVal);
    // It's necessary to tell ScalarEvolution about this explicitly so that
    // it can walk the def-use list and forget all SCEVs, as it may not be
    // watching the PHI itself. Once the new exit value is in place, there
    // may not be a def-use connection between the loop and every instruction
    // which got a SCEVAddRecExpr for that loop.
    SE->forgetValue(PN);

    // If this instruction is dead now, delete it. Don't do it now to avoid
    // invalidating iterators.
    if (isInstructionTriviallyDead(Inst, TLI))
      DeadInsts.push_back(Inst);

    // Replace PN with ExitVal if that is legal and does not break LCSSA.
    if (PN->getNumIncomingValues() == 1 &&
        LI->replacementPreservesLCSSAForm(PN, ExitVal)) {
      PN->replaceAllUsesWith(ExitVal);
      PN->eraseFromParent();
    }
  }

  // The insertion point instruction may have been deleted; clear it out
  // so that the rewriter doesn't trip over it later.
  Rewriter.clearInsertPoint();
  return NumReplaced;
}

/// Set weights for \p UnrolledLoop and \p RemainderLoop based on weights for
/// \p OrigLoop.
void llvm::setProfileInfoAfterUnrolling(Loop *OrigLoop, Loop *UnrolledLoop,
                                        Loop *RemainderLoop, uint64_t UF) {
  assert(UF > 0 && "Zero unrolled factor is not supported");
  assert(UnrolledLoop != RemainderLoop &&
         "Unrolled and Remainder loops are expected to distinct");

  // Get number of iterations in the original scalar loop.
  unsigned OrigLoopInvocationWeight = 0;
  Optional<unsigned> OrigAverageTripCount =
      getLoopEstimatedTripCount(OrigLoop, &OrigLoopInvocationWeight);
  if (!OrigAverageTripCount)
    return;

  // Calculate number of iterations in unrolled loop.
  unsigned UnrolledAverageTripCount = *OrigAverageTripCount / UF;
  // Calculate number of iterations for remainder loop.
  unsigned RemainderAverageTripCount = *OrigAverageTripCount % UF;

  setLoopEstimatedTripCount(UnrolledLoop, UnrolledAverageTripCount,
                            OrigLoopInvocationWeight);
  setLoopEstimatedTripCount(RemainderLoop, RemainderAverageTripCount,
                            OrigLoopInvocationWeight);
}

/// Utility that implements appending of loops onto a worklist.
/// Loops are added in preorder (analogous for reverse postorder for trees),
/// and the worklist is processed LIFO.
template <typename RangeT>
void llvm::appendReversedLoopsToWorklist(
    RangeT &&Loops, SmallPriorityWorklist<Loop *, 4> &Worklist) {
  // We use an internal worklist to build up the preorder traversal without
  // recursion.
  SmallVector<Loop *, 4> PreOrderLoops, PreOrderWorklist;

  // We walk the initial sequence of loops in reverse because we generally want
  // to visit defs before uses and the worklist is LIFO.
  for (Loop *RootL : Loops) {
    assert(PreOrderLoops.empty() && "Must start with an empty preorder walk.");
    assert(PreOrderWorklist.empty() &&
           "Must start with an empty preorder walk worklist.");
    PreOrderWorklist.push_back(RootL);
    do {
      Loop *L = PreOrderWorklist.pop_back_val();
      PreOrderWorklist.append(L->begin(), L->end());
      PreOrderLoops.push_back(L);
    } while (!PreOrderWorklist.empty());

    Worklist.insert(std::move(PreOrderLoops));
    PreOrderLoops.clear();
  }
}

template <typename RangeT>
void llvm::appendLoopsToWorklist(RangeT &&Loops,
                                 SmallPriorityWorklist<Loop *, 4> &Worklist) {
  appendReversedLoopsToWorklist(reverse(Loops), Worklist);
}

template void llvm::appendLoopsToWorklist<ArrayRef<Loop *> &>(
    ArrayRef<Loop *> &Loops, SmallPriorityWorklist<Loop *, 4> &Worklist);

template void
llvm::appendLoopsToWorklist<Loop &>(Loop &L,
                                    SmallPriorityWorklist<Loop *, 4> &Worklist);

void llvm::appendLoopsToWorklist(LoopInfo &LI,
                                 SmallPriorityWorklist<Loop *, 4> &Worklist) {
  appendReversedLoopsToWorklist(LI, Worklist);
}

Loop *llvm::cloneLoop(Loop *L, Loop *PL, ValueToValueMapTy &VM,
                      LoopInfo *LI, LPPassManager *LPM) {
  Loop &New = *LI->AllocateLoop();
  if (PL)
    PL->addChildLoop(&New);
  else
    LI->addTopLevelLoop(&New);

  if (LPM)
    LPM->addLoop(New);

  // Add all of the blocks in L to the new loop.
  for (BasicBlock *BB : L->blocks())
    if (LI->getLoopFor(BB) == L)
      New.addBasicBlockToLoop(cast<BasicBlock>(VM[BB]), *LI);

  // Add all of the subloops to the new loop.
  for (Loop *I : *L)
    cloneLoop(I, &New, VM, LI, LPM);

  return &New;
}

/// IR Values for the lower and upper bounds of a pointer evolution.  We
/// need to use value-handles because SCEV expansion can invalidate previously
/// expanded values.  Thus expansion of a pointer can invalidate the bounds for
/// a previous one.
struct PointerBounds {
  TrackingVH<Value> Start;
  TrackingVH<Value> End;
};

/// Expand code for the lower and upper bound of the pointer group \p CG
/// in \p TheLoop.  \return the values for the bounds.
static PointerBounds expandBounds(const RuntimeCheckingPtrGroup *CG,
                                  Loop *TheLoop, Instruction *Loc,
                                  SCEVExpander &Exp) {
  LLVMContext &Ctx = Loc->getContext();
  Type *PtrArithTy = Type::getInt8PtrTy(Ctx, CG->AddressSpace);

  Value *Start = nullptr, *End = nullptr;
  LLVM_DEBUG(dbgs() << "LAA: Adding RT check for range:\n");
  Start = Exp.expandCodeFor(CG->Low, PtrArithTy, Loc);
  End = Exp.expandCodeFor(CG->High, PtrArithTy, Loc);
  LLVM_DEBUG(dbgs() << "Start: " << *CG->Low << " End: " << *CG->High << "\n");
  return {Start, End};
}

/// Turns a collection of checks into a collection of expanded upper and
/// lower bounds for both pointers in the check.
static SmallVector<std::pair<PointerBounds, PointerBounds>, 4>
expandBounds(const SmallVectorImpl<RuntimePointerCheck> &PointerChecks, Loop *L,
             Instruction *Loc, SCEVExpander &Exp) {
  SmallVector<std::pair<PointerBounds, PointerBounds>, 4> ChecksWithBounds;

  // Here we're relying on the SCEV Expander's cache to only emit code for the
  // same bounds once.
  transform(PointerChecks, std::back_inserter(ChecksWithBounds),
            [&](const RuntimePointerCheck &Check) {
              PointerBounds First = expandBounds(Check.first, L, Loc, Exp),
                            Second = expandBounds(Check.second, L, Loc, Exp);
              return std::make_pair(First, Second);
            });

  return ChecksWithBounds;
}

Value *llvm::addRuntimeChecks(
    Instruction *Loc, Loop *TheLoop,
    const SmallVectorImpl<RuntimePointerCheck> &PointerChecks,
    SCEVExpander &Exp) {
  // TODO: Move noalias annotation code from LoopVersioning here and share with LV if possible.
  // TODO: Pass  RtPtrChecking instead of PointerChecks and SE separately, if possible
  auto ExpandedChecks = expandBounds(PointerChecks, TheLoop, Loc, Exp);

  LLVMContext &Ctx = Loc->getContext();
  IRBuilder<> ChkBuilder(Loc);
  // Our instructions might fold to a constant.
  Value *MemoryRuntimeCheck = nullptr;

  for (const auto &Check : ExpandedChecks) {
    const PointerBounds &A = Check.first, &B = Check.second;
    // Check if two pointers (A and B) conflict where conflict is computed as:
    // start(A) <= end(B) && start(B) <= end(A)
    unsigned AS0 = A.Start->getType()->getPointerAddressSpace();
    unsigned AS1 = B.Start->getType()->getPointerAddressSpace();

    assert((AS0 == B.End->getType()->getPointerAddressSpace()) &&
           (AS1 == A.End->getType()->getPointerAddressSpace()) &&
           "Trying to bounds check pointers with different address spaces");

    Type *PtrArithTy0 = Type::getInt8PtrTy(Ctx, AS0);
    Type *PtrArithTy1 = Type::getInt8PtrTy(Ctx, AS1);

    Value *Start0 = ChkBuilder.CreateBitCast(A.Start, PtrArithTy0, "bc");
    Value *Start1 = ChkBuilder.CreateBitCast(B.Start, PtrArithTy1, "bc");
    Value *End0 = ChkBuilder.CreateBitCast(A.End, PtrArithTy1, "bc");
    Value *End1 = ChkBuilder.CreateBitCast(B.End, PtrArithTy0, "bc");

    // [A|B].Start points to the first accessed byte under base [A|B].
    // [A|B].End points to the last accessed byte, plus one.
    // There is no conflict when the intervals are disjoint:
    // NoConflict = (B.Start >= A.End) || (A.Start >= B.End)
    //
    // bound0 = (B.Start < A.End)
    // bound1 = (A.Start < B.End)
    //  IsConflict = bound0 & bound1
    Value *Cmp0 = ChkBuilder.CreateICmpULT(Start0, End1, "bound0");
    Value *Cmp1 = ChkBuilder.CreateICmpULT(Start1, End0, "bound1");
    Value *IsConflict = ChkBuilder.CreateAnd(Cmp0, Cmp1, "found.conflict");
    if (MemoryRuntimeCheck) {
      IsConflict =
          ChkBuilder.CreateOr(MemoryRuntimeCheck, IsConflict, "conflict.rdx");
    }
    MemoryRuntimeCheck = IsConflict;
  }

  return MemoryRuntimeCheck;
}

Optional<IVConditionInfo> llvm::hasPartialIVCondition(Loop &L,
                                                      unsigned MSSAThreshold,
                                                      MemorySSA &MSSA,
                                                      AAResults &AA) {
  auto *TI = dyn_cast<BranchInst>(L.getHeader()->getTerminator());
  if (!TI || !TI->isConditional())
    return {};

  auto *CondI = dyn_cast<CmpInst>(TI->getCondition());
  // The case with the condition outside the loop should already be handled
  // earlier.
  if (!CondI || !L.contains(CondI))
    return {};

  SmallVector<Instruction *> InstToDuplicate;
  InstToDuplicate.push_back(CondI);

  SmallVector<Value *, 4> WorkList;
  WorkList.append(CondI->op_begin(), CondI->op_end());

  SmallVector<MemoryAccess *, 4> AccessesToCheck;
  SmallVector<MemoryLocation, 4> AccessedLocs;
  while (!WorkList.empty()) {
    Instruction *I = dyn_cast<Instruction>(WorkList.pop_back_val());
    if (!I || !L.contains(I))
      continue;

    // TODO: support additional instructions.
    if (!isa<LoadInst>(I) && !isa<GetElementPtrInst>(I))
      return {};

    // Do not duplicate volatile and atomic loads.
    if (auto *LI = dyn_cast<LoadInst>(I))
      if (LI->isVolatile() || LI->isAtomic())
        return {};

    InstToDuplicate.push_back(I);
    if (MemoryAccess *MA = MSSA.getMemoryAccess(I)) {
      if (auto *MemUse = dyn_cast_or_null<MemoryUse>(MA)) {
        // Queue the defining access to check for alias checks.
        AccessesToCheck.push_back(MemUse->getDefiningAccess());
        AccessedLocs.push_back(MemoryLocation::get(I));
      } else {
        // MemoryDefs may clobber the location or may be atomic memory
        // operations. Bail out.
        return {};
      }
    }
    WorkList.append(I->op_begin(), I->op_end());
  }

  if (InstToDuplicate.empty())
    return {};

  SmallVector<BasicBlock *, 4> ExitingBlocks;
  L.getExitingBlocks(ExitingBlocks);
  auto HasNoClobbersOnPath =
      [&L, &AA, &AccessedLocs, &ExitingBlocks, &InstToDuplicate,
       MSSAThreshold](BasicBlock *Succ, BasicBlock *Header,
                      SmallVector<MemoryAccess *, 4> AccessesToCheck)
      -> Optional<IVConditionInfo> {
    IVConditionInfo Info;
    // First, collect all blocks in the loop that are on a patch from Succ
    // to the header.
    SmallVector<BasicBlock *, 4> WorkList;
    WorkList.push_back(Succ);
    WorkList.push_back(Header);
    SmallPtrSet<BasicBlock *, 4> Seen;
    Seen.insert(Header);
    Info.PathIsNoop &=
        all_of(*Header, [](Instruction &I) { return !I.mayHaveSideEffects(); });

    while (!WorkList.empty()) {
      BasicBlock *Current = WorkList.pop_back_val();
      if (!L.contains(Current))
        continue;
      const auto &SeenIns = Seen.insert(Current);
      if (!SeenIns.second)
        continue;

      Info.PathIsNoop &= all_of(
          *Current, [](Instruction &I) { return !I.mayHaveSideEffects(); });
      WorkList.append(succ_begin(Current), succ_end(Current));
    }

    // Require at least 2 blocks on a path through the loop. This skips
    // paths that directly exit the loop.
    if (Seen.size() < 2)
      return {};

    // Next, check if there are any MemoryDefs that are on the path through
    // the loop (in the Seen set) and they may-alias any of the locations in
    // AccessedLocs. If that is the case, they may modify the condition and
    // partial unswitching is not possible.
    SmallPtrSet<MemoryAccess *, 4> SeenAccesses;
    while (!AccessesToCheck.empty()) {
      MemoryAccess *Current = AccessesToCheck.pop_back_val();
      auto SeenI = SeenAccesses.insert(Current);
      if (!SeenI.second || !Seen.contains(Current->getBlock()))
        continue;

      // Bail out if exceeded the threshold.
      if (SeenAccesses.size() >= MSSAThreshold)
        return {};

      // MemoryUse are read-only accesses.
      if (isa<MemoryUse>(Current))
        continue;

      // For a MemoryDef, check if is aliases any of the location feeding
      // the original condition.
      if (auto *CurrentDef = dyn_cast<MemoryDef>(Current)) {
        if (any_of(AccessedLocs, [&AA, CurrentDef](MemoryLocation &Loc) {
              return isModSet(
                  AA.getModRefInfo(CurrentDef->getMemoryInst(), Loc));
            }))
          return {};
      }

      for (Use &U : Current->uses())
        AccessesToCheck.push_back(cast<MemoryAccess>(U.getUser()));
    }

    // We could also allow loops with known trip counts without mustprogress,
    // but ScalarEvolution may not be available.
    Info.PathIsNoop &= isMustProgress(&L);

    // If the path is considered a no-op so far, check if it reaches a
    // single exit block without any phis. This ensures no values from the
    // loop are used outside of the loop.
    if (Info.PathIsNoop) {
      for (auto *Exiting : ExitingBlocks) {
        if (!Seen.contains(Exiting))
          continue;
        for (auto *Succ : successors(Exiting)) {
          if (L.contains(Succ))
            continue;

          Info.PathIsNoop &= llvm::empty(Succ->phis()) &&
                             (!Info.ExitForPath || Info.ExitForPath == Succ);
          if (!Info.PathIsNoop)
            break;
          assert((!Info.ExitForPath || Info.ExitForPath == Succ) &&
                 "cannot have multiple exit blocks");
          Info.ExitForPath = Succ;
        }
      }
    }
    if (!Info.ExitForPath)
      Info.PathIsNoop = false;

    Info.InstToDuplicate = InstToDuplicate;
    return Info;
  };

  // If we branch to the same successor, partial unswitching will not be
  // beneficial.
  if (TI->getSuccessor(0) == TI->getSuccessor(1))
    return {};

  if (auto Info = HasNoClobbersOnPath(TI->getSuccessor(0), L.getHeader(),
                                      AccessesToCheck)) {
    Info->KnownValue = ConstantInt::getTrue(TI->getContext());
    return Info;
  }
  if (auto Info = HasNoClobbersOnPath(TI->getSuccessor(1), L.getHeader(),
                                      AccessesToCheck)) {
    Info->KnownValue = ConstantInt::getFalse(TI->getContext());
    return Info;
  }

  return {};
}
