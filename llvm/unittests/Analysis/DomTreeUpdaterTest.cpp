//===- DomTreeUpdaterTest.cpp - DomTreeUpdater unit tests -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <algorithm>

using namespace llvm;

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              StringRef ModuleStr) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

TEST(DomTreeUpdater, EagerUpdateBasicOperations) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                          define i32 @f(i32 %i, i32 *%p) {
                          bb0:
                             store i32 %i, i32 *%p
                             switch i32 %i, label %bb1 [
                               i32 1, label %bb2
                               i32 2, label %bb3
                             ]
                          bb1:
                             ret i32 1
                          bb2:
                             ret i32 2
                          bb3:
                             ret i32 3
                          })";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DomTreeUpdater.
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);

  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_TRUE(DTU.isEager());
  ASSERT_FALSE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());
  ASSERT_FALSE(DTU.hasPendingUpdates());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;
  BasicBlock *BB2 = &*FI++;
  BasicBlock *BB3 = &*FI++;
  SwitchInst *SI = dyn_cast<SwitchInst>(BB0->getTerminator());
  ASSERT_NE(SI, nullptr) << "Couldn't get SwitchInst.";

  DTU.insertEdgeRelaxed(BB0, BB0);
  DTU.deleteEdgeRelaxed(BB0, BB0);

  // Delete edge bb0 -> bb3 and push the update twice to verify duplicate
  // entries are discarded.
  std::vector<DominatorTree::UpdateType> Updates;
  Updates.reserve(4);
  Updates.push_back({DominatorTree::Delete, BB0, BB3});
  Updates.push_back({DominatorTree::Delete, BB0, BB3});

  // Invalid Insert: no edge bb1 -> bb2 after change to bb0.
  Updates.push_back({DominatorTree::Insert, BB1, BB2});
  // Invalid Delete: edge exists bb0 -> bb1 after change to bb0.
  Updates.push_back({DominatorTree::Delete, BB0, BB1});

  // CFG Change: remove edge bb0 -> bb3.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 3u);
  BB3->removePredecessor(BB0);
  for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
    if (i->getCaseSuccessor() == BB3) {
      SI->removeCase(i);
      break;
    }
  }
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);
  // Deletion of a BasicBlock is an immediate event. We remove all uses to the
  // contained Instructions and change the Terminator to "unreachable" when
  // queued for deletion.
  ASSERT_FALSE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB3));
  DTU.applyUpdates(Updates, /*ForceRemoveDuplicates*/ true);
  ASSERT_FALSE(DTU.hasPendingUpdates());

  // Invalid Insert: no edge bb1 -> bb2 after change to bb0.
  DTU.insertEdgeRelaxed(BB1, BB2);
  // Invalid Delete: edge exists bb0 -> bb1 after change to bb0.
  DTU.deleteEdgeRelaxed(BB0, BB1);

  // DTU working with Eager UpdateStrategy does not need to flush.
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());

  // Test callback utils.
  ASSERT_EQ(BB3->getParent(), F);
  DTU.callbackDeleteBB(BB3,
                       [&F](BasicBlock *BB) { ASSERT_NE(BB->getParent(), F); });

  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());
  ASSERT_FALSE(DTU.hasPendingUpdates());

  // Unnecessary flush() test
  DTU.flush();
  EXPECT_TRUE(DT.verify());
  EXPECT_TRUE(PDT.verify());

  // Remove all case branch to BB2 to test Eager recalculation.
  // Code section from llvm::ConstantFoldTerminator
  for (auto i = SI->case_begin(), e = SI->case_end(); i != e;) {
    if (i->getCaseSuccessor() == BB2) {
      // Remove this entry.
      BB2->removePredecessor(BB0);
      i = SI->removeCase(i);
      e = SI->case_end();
    } else
      ++i;
  }
  ASSERT_FALSE(DT.verify());
  ASSERT_FALSE(PDT.verify());
  DTU.recalculate(*F);
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());
}

TEST(DomTreeUpdater, EagerUpdateReplaceEntryBB) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f() {
                           bb0:
                              br label %bb1
                            bb1:
                              ret i32 1
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DTU.
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_TRUE(DTU.isEager());
  ASSERT_FALSE(DTU.isLazy());
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;

  // Add a block as the new function entry BB. We also link it to BB0.
  BasicBlock *NewEntry =
      BasicBlock::Create(F->getContext(), "new_entry", F, BB0);
  BranchInst::Create(BB0, NewEntry);
  EXPECT_EQ(F->begin()->getName(), NewEntry->getName());
  EXPECT_TRUE(&F->getEntryBlock() == NewEntry);

  DTU.insertEdgeRelaxed(NewEntry, BB0);

  // Changing the Entry BB requires a full recalculation of DomTree.
  DTU.recalculate(*F);
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());

  // CFG Change: remove new_edge -> bb0 and redirect to new_edge -> bb1.
  EXPECT_EQ(NewEntry->getTerminator()->getNumSuccessors(), 1u);
  NewEntry->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, NewEntry);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 1u);

  // Update the DTU. At this point bb0 now has no predecessors but is still a
  // Child of F.
  DTU.applyUpdates({{DominatorTree::Delete, NewEntry, BB0},
                    {DominatorTree::Insert, NewEntry, BB1}});
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());

  // Now remove bb0 from F.
  ASSERT_FALSE(isa<UnreachableInst>(BB0->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB0));
  DTU.deleteBB(BB0);
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());
}

TEST(DomTreeUpdater, LazyUpdateDTBasicOperations) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f(i32 %i, i32 *%p) {
                            bb0:
                              store i32 %i, i32 *%p
                              switch i32 %i, label %bb1 [
                                i32 0, label %bb2
                                i32 1, label %bb2
                                i32 2, label %bb3
                              ]
                            bb1:
                              ret i32 1
                            bb2:
                              ret i32 2
                            bb3:
                              ret i32 3
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DTU.
  DominatorTree DT(*F);
  PostDominatorTree *PDT = nullptr;
  DomTreeUpdater DTU(&DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_FALSE(DTU.hasPostDomTree());
  ASSERT_FALSE(DTU.isEager());
  ASSERT_TRUE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;
  BasicBlock *BB2 = &*FI++;
  BasicBlock *BB3 = &*FI++;

  // Test discards of self-domination update.
  DTU.deleteEdge(BB0, BB0);
  ASSERT_FALSE(DTU.hasPendingDomTreeUpdates());

  // Delete edge bb0 -> bb3 and push the update twice to verify duplicate
  // entries are discarded.
  std::vector<DominatorTree::UpdateType> Updates;
  Updates.reserve(4);
  Updates.push_back({DominatorTree::Delete, BB0, BB3});
  Updates.push_back({DominatorTree::Delete, BB0, BB3});

  // Invalid Insert: no edge bb1 -> bb2 after change to bb0.
  Updates.push_back({DominatorTree::Insert, BB1, BB2});
  // Invalid Delete: edge exists bb0 -> bb1 after change to bb0.
  Updates.push_back({DominatorTree::Delete, BB0, BB1});

  // CFG Change: remove edge bb0 -> bb3 and one duplicate edge bb0 -> bb2.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 4u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, BB2, ConstantInt::getTrue(F->getContext()), BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);

  // Verify. Updates to DTU must be applied *after* all changes to the CFG
  // (including block deletion).
  DTU.applyUpdates(Updates);
  ASSERT_TRUE(DTU.getDomTree().verify());

  // Deletion of a BasicBlock is an immediate event. We remove all uses to the
  // contained Instructions and change the Terminator to "unreachable" when
  // queued for deletion. Its parent is still F until all the pending updates
  // are applied to all trees held by the DomTreeUpdater (DomTree/PostDomTree).
  // We don't defer this action because it can cause problems for other
  // transforms or analysis as it's part of the actual CFG. We only defer
  // updates to the DominatorTrees. This code will crash if it is placed before
  // the BranchInst::Create() call above. After a deletion of a BasicBlock. Only
  // an explicit flush event can trigger the flushing of deleteBBs. Because some
  // passes using Lazy UpdateStrategy rely on this behavior.

  ASSERT_FALSE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB3));
  EXPECT_FALSE(DTU.hasPendingDeletedBB());
  DTU.deleteBB(BB3);
  EXPECT_TRUE(DTU.isBBPendingDeletion(BB3));
  EXPECT_TRUE(DTU.hasPendingDeletedBB());
  ASSERT_TRUE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_EQ(BB3->getParent(), F);
  DTU.recalculate(*F);
  EXPECT_FALSE(DTU.hasPendingDeletedBB());
}

TEST(DomTreeUpdater, LazyUpdateDTInheritedPreds) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f(i32 %i, i32 *%p) {
                            bb0:
                              store i32 %i, i32 *%p
                              switch i32 %i, label %bb1 [
                                i32 2, label %bb2
                                i32 3, label %bb3
                              ]
                            bb1:
                              br label %bb3
                            bb2:
                              br label %bb3
                            bb3:
                              ret i32 3
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DTU.
  DominatorTree DT(*F);
  PostDominatorTree *PDT = nullptr;
  DomTreeUpdater DTU(&DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_FALSE(DTU.hasPostDomTree());
  ASSERT_FALSE(DTU.isEager());
  ASSERT_TRUE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;
  BasicBlock *BB2 = &*FI++;
  BasicBlock *BB3 = &*FI++;

  // There are several CFG locations where we have:
  //
  //   pred1..predN
  //    |        |
  //    +> curr <+    converted into:   pred1..predN curr
  //        |                            |        |
  //        v                            +> succ <+
  //       succ
  //
  // There is a specific shape of this we have to be careful of:
  //
  //   pred1..predN
  //   ||        |
  //   |+> curr <+    converted into:   pred1..predN curr
  //   |    |                            |        |
  //   |    v                            +> succ <+
  //   +-> succ
  //
  // While the final CFG form is functionally identical the updates to
  // DTU are not. In the first case we must have DTU.insertEdge(Pred1, Succ)
  // while in the latter case we must *NOT* have DTU.insertEdge(Pred1, Succ).

  // CFG Change: bb0 now only has bb0 -> bb1 and bb0 -> bb3. We are preparing to
  // remove bb2.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 3u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, BB3, ConstantInt::getTrue(F->getContext()), BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);

  // Test callback utils.
  std::vector<BasicBlock *> BasicBlocks;
  BasicBlocks.push_back(BB1);
  BasicBlocks.push_back(BB2);
  auto Eraser = [&](BasicBlock *BB) {
    BasicBlocks.erase(
        std::remove_if(BasicBlocks.begin(), BasicBlocks.end(),
                       [&](const BasicBlock *i) { return i == BB; }),
        BasicBlocks.end());
  };
  ASSERT_EQ(BasicBlocks.size(), static_cast<size_t>(2));
  // Remove bb2 from F. This has to happen before the call to applyUpdates() for
  // DTU to detect there is no longer an edge between bb2 -> bb3. The deleteBB()
  // method converts bb2's TI into "unreachable".
  ASSERT_FALSE(isa<UnreachableInst>(BB2->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB2));
  DTU.callbackDeleteBB(BB2, Eraser);
  EXPECT_TRUE(DTU.isBBPendingDeletion(BB2));
  ASSERT_TRUE(isa<UnreachableInst>(BB2->getTerminator()));
  EXPECT_EQ(BB2->getParent(), F);

  // Queue up the DTU updates.
  std::vector<DominatorTree::UpdateType> Updates;
  Updates.reserve(4);
  Updates.push_back({DominatorTree::Delete, BB0, BB2});
  Updates.push_back({DominatorTree::Delete, BB2, BB3});

  // Handle the specific shape case next.
  // CFG Change: bb0 now only branches to bb3. We are preparing to remove bb1.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB3, BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 1u);

  // Remove bb1 from F. This has to happen before the call to applyUpdates() for
  // DTU to detect there is no longer an edge between bb1 -> bb3. The deleteBB()
  // method converts bb1's TI into "unreachable".
  ASSERT_FALSE(isa<UnreachableInst>(BB1->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB1));
  DTU.callbackDeleteBB(BB1, Eraser);
  EXPECT_TRUE(DTU.isBBPendingDeletion(BB1));
  ASSERT_TRUE(isa<UnreachableInst>(BB1->getTerminator()));
  EXPECT_EQ(BB1->getParent(), F);

  // Update the DTU. In this case we don't call DTU.insertEdge(BB0, BB3) because
  // the edge previously existed at the start of this test when DT was first
  // created.
  Updates.push_back({DominatorTree::Delete, BB0, BB1});
  Updates.push_back({DominatorTree::Delete, BB1, BB3});

  // Verify everything.
  DTU.applyUpdates(Updates);
  ASSERT_EQ(BasicBlocks.size(), static_cast<size_t>(2));
  DTU.flush();
  ASSERT_EQ(BasicBlocks.size(), static_cast<size_t>(0));
  ASSERT_TRUE(DT.verify());
}

TEST(DomTreeUpdater, LazyUpdateBasicOperations) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f(i32 %i, i32 *%p) {
                            bb0:
                              store i32 %i, i32 *%p
                              switch i32 %i, label %bb1 [
                                i32 0, label %bb2
                                i32 1, label %bb2
                                i32 2, label %bb3
                              ]
                            bb1:
                              ret i32 1
                            bb2:
                              ret i32 2
                            bb3:
                              ret i32 3
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DTU.
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  DomTreeUpdater DTU(&DT, &PDT, DomTreeUpdater::UpdateStrategy::Lazy);
  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_FALSE(DTU.isEager());
  ASSERT_TRUE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;
  BasicBlock *BB2 = &*FI++;
  BasicBlock *BB3 = &*FI++;
  // Test discards of self-domination update.
  DTU.deleteEdge(BB0, BB0);

  // Delete edge bb0 -> bb3 and push the update twice to verify duplicate
  // entries are discarded.
  std::vector<DominatorTree::UpdateType> Updates;
  Updates.reserve(4);
  Updates.push_back({DominatorTree::Delete, BB0, BB3});
  Updates.push_back({DominatorTree::Delete, BB0, BB3});

  // Unnecessary Insert: no edge bb1 -> bb2 after change to bb0.
  Updates.push_back({DominatorTree::Insert, BB1, BB2});
  // Unnecessary Delete: edge exists bb0 -> bb1 after change to bb0.
  Updates.push_back({DominatorTree::Delete, BB0, BB1});

  // CFG Change: remove edge bb0 -> bb3 and one duplicate edge bb0 -> bb2.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 4u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, BB2, ConstantInt::getTrue(F->getContext()), BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);

  // Deletion of a BasicBlock is an immediate event. We remove all uses to the
  // contained Instructions and change the Terminator to "unreachable" when
  // queued for deletion. Its parent is still F until DTU.flushDomTree is
  // called. We don't defer this action because it can cause problems for other
  // transforms or analysis as it's part of the actual CFG. We only defer
  // updates to the DominatorTree. This code will crash if it is placed before
  // the BranchInst::Create() call above.
  bool CallbackFlag = false;
  ASSERT_FALSE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB3));
  DTU.callbackDeleteBB(BB3, [&](BasicBlock *) { CallbackFlag = true; });
  EXPECT_TRUE(DTU.isBBPendingDeletion(BB3));
  ASSERT_TRUE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_EQ(BB3->getParent(), F);

  // Verify. Updates to DTU must be applied *after* all changes to the CFG
  // (including block deletion).
  DTU.applyUpdates(Updates);
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.hasPendingUpdates());
  ASSERT_TRUE(DTU.hasPendingPostDomTreeUpdates());
  ASSERT_FALSE(DTU.hasPendingDomTreeUpdates());
  ASSERT_TRUE(DTU.hasPendingDeletedBB());
  ASSERT_TRUE(DTU.getPostDomTree().verify());
  ASSERT_FALSE(DTU.hasPendingUpdates());
  ASSERT_FALSE(DTU.hasPendingPostDomTreeUpdates());
  ASSERT_FALSE(DTU.hasPendingDomTreeUpdates());
  ASSERT_FALSE(DTU.hasPendingDeletedBB());
  ASSERT_EQ(CallbackFlag, true);
}

TEST(DomTreeUpdater, LazyUpdateReplaceEntryBB) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f() {
                           bb0:
                              br label %bb1
                            bb1:
                              ret i32 1
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DTU.
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_FALSE(DTU.isEager());
  ASSERT_TRUE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;

  // Add a block as the new function entry BB. We also link it to BB0.
  BasicBlock *NewEntry =
      BasicBlock::Create(F->getContext(), "new_entry", F, BB0);
  BranchInst::Create(BB0, NewEntry);
  EXPECT_EQ(F->begin()->getName(), NewEntry->getName());
  EXPECT_TRUE(&F->getEntryBlock() == NewEntry);

  // Insert the new edge between new_entry -> bb0. Without this the
  // recalculate() call below will not actually recalculate the DT as there
  // are no changes pending and no blocks deleted.
  DTU.insertEdge(NewEntry, BB0);

  // Changing the Entry BB requires a full recalculation.
  DTU.recalculate(*F);
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());

  // CFG Change: remove new_edge -> bb0 and redirect to new_edge -> bb1.
  EXPECT_EQ(NewEntry->getTerminator()->getNumSuccessors(), 1u);
  NewEntry->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, NewEntry);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 1u);

  // Update the DTU. At this point bb0 now has no predecessors but is still a
  // Child of F.
  DTU.applyUpdates({{DominatorTree::Delete, NewEntry, BB0},
                    {DominatorTree::Insert, NewEntry, BB1}});
  DTU.flush();
  ASSERT_TRUE(DT.verify());
  ASSERT_TRUE(PDT.verify());

  // Now remove bb0 from F.
  ASSERT_FALSE(isa<UnreachableInst>(BB0->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB0));
  DTU.deleteBB(BB0);
  EXPECT_TRUE(DTU.isBBPendingDeletion(BB0));
  ASSERT_TRUE(isa<UnreachableInst>(BB0->getTerminator()));
  EXPECT_EQ(BB0->getParent(), F);

  // Perform a full recalculation of the DTU. It is not necessary here but we
  // do this to test the case when there are no pending DT updates but there are
  // pending deleted BBs.
  ASSERT_TRUE(DTU.hasPendingDeletedBB());
  DTU.recalculate(*F);
  ASSERT_FALSE(DTU.hasPendingDeletedBB());
}

TEST(DomTreeUpdater, LazyUpdateStepTest) {
  // This test focus on testing a DTU holding both trees applying multiple
  // updates and DT/PDT not flushed together.
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f(i32 %i, i32 *%p) {
                            bb0:
                              store i32 %i, i32 *%p
                              switch i32 %i, label %bb1 [
                                i32 0, label %bb1
                                i32 1, label %bb2
                                i32 2, label %bb3
                                i32 3, label %bb1
                              ]
                            bb1:
                              ret i32 1
                            bb2:
                              ret i32 2
                            bb3:
                              ret i32 3
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DomTreeUpdater.
  DominatorTree DT(*F);
  PostDominatorTree PDT(*F);
  DomTreeUpdater DTU(DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);

  ASSERT_TRUE(DTU.hasDomTree());
  ASSERT_TRUE(DTU.hasPostDomTree());
  ASSERT_FALSE(DTU.isEager());
  ASSERT_TRUE(DTU.isLazy());
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.getPostDomTree().verify());
  ASSERT_FALSE(DTU.hasPendingUpdates());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  FI++;
  BasicBlock *BB2 = &*FI++;
  BasicBlock *BB3 = &*FI++;
  SwitchInst *SI = dyn_cast<SwitchInst>(BB0->getTerminator());
  ASSERT_NE(SI, nullptr) << "Couldn't get SwitchInst.";

  // Delete edge bb0 -> bb3 and push the update twice to verify duplicate
  // entries are discarded.
  std::vector<DominatorTree::UpdateType> Updates;
  Updates.reserve(1);
  Updates.push_back({DominatorTree::Delete, BB0, BB3});

  // CFG Change: remove edge bb0 -> bb3.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 5u);
  BB3->removePredecessor(BB0);
  for (auto i = SI->case_begin(), e = SI->case_end(); i != e; ++i) {
    if (i->getCaseIndex() == 2) {
      SI->removeCase(i);
      break;
    }
  }
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 4u);
  // Deletion of a BasicBlock is an immediate event. We remove all uses to the
  // contained Instructions and change the Terminator to "unreachable" when
  // queued for deletion.
  ASSERT_FALSE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_FALSE(DTU.isBBPendingDeletion(BB3));
  DTU.applyUpdates(Updates);

  // Only flush DomTree.
  ASSERT_TRUE(DTU.getDomTree().verify());
  ASSERT_TRUE(DTU.hasPendingPostDomTreeUpdates());
  ASSERT_FALSE(DTU.hasPendingDomTreeUpdates());

  ASSERT_EQ(BB3->getParent(), F);
  DTU.deleteBB(BB3);

  Updates.clear();

  // Remove all case branch to BB2 to test Eager recalculation.
  // Code section from llvm::ConstantFoldTerminator
  for (auto i = SI->case_begin(), e = SI->case_end(); i != e;) {
    if (i->getCaseSuccessor() == BB2) {
      // Remove this entry.
      BB2->removePredecessor(BB0);
      i = SI->removeCase(i);
      e = SI->case_end();
      Updates.push_back({DominatorTree::Delete, BB0, BB2});
    } else
      ++i;
  }

  DTU.applyUpdates(Updates);
  // flush PostDomTree
  ASSERT_TRUE(DTU.getPostDomTree().verify());
  ASSERT_FALSE(DTU.hasPendingPostDomTreeUpdates());
  ASSERT_TRUE(DTU.hasPendingDomTreeUpdates());
  // flush both trees
  DTU.flush();
  ASSERT_TRUE(DT.verify());
}

TEST(DomTreeUpdater, NoTreeTest) {
  StringRef FuncName = "f";
  StringRef ModuleString = R"(
                           define i32 @f() {
                           bb0:
                              ret i32 0
                           }
                           )";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);

  // Make the DTU.
  DomTreeUpdater DTU(nullptr, nullptr, DomTreeUpdater::UpdateStrategy::Lazy);
  ASSERT_FALSE(DTU.hasDomTree());
  ASSERT_FALSE(DTU.hasPostDomTree());
  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  // Test whether PendingDeletedBB is flushed after the recalculation.
  DTU.deleteBB(BB0);
  ASSERT_TRUE(DTU.hasPendingDeletedBB());
  DTU.recalculate(*F);
  ASSERT_FALSE(DTU.hasPendingDeletedBB());
}
