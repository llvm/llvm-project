//===- llvm/unittests/IR/DeferredDominanceTest.cpp - DDT unit tests -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              StringRef ModuleStr) {
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(ModuleStr, Err, Context);
  assert(M && "Bad LLVM IR?");
  return M;
}

TEST(DeferredDominance, BasicOperations) {
  StringRef FuncName = "f";
  StringRef ModuleString =
      "define i32 @f(i32 %i, i32 *%p) {\n"
      " bb0:\n"
      "   store i32 %i, i32 *%p\n"
      "   switch i32 %i, label %bb1 [\n"
      "     i32 0, label %bb2\n"
      "     i32 1, label %bb2\n"
      "     i32 2, label %bb3\n"
      "   ]\n"
      " bb1:\n"
      "   ret i32 1\n"
      " bb2:\n"
      "   ret i32 2\n"
      " bb3:\n"
      "   ret i32 3\n"
      "}\n";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Couldn't get function " << FuncName << ".";

  // Make the DDT.
  DominatorTree DT(*F);
  DeferredDominance DDT(DT);
  ASSERT_TRUE(DDT.flush().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;
  BasicBlock *BB2 = &*FI++;
  BasicBlock *BB3 = &*FI++;

  // Test discards of invalid self-domination updates. These use the single
  // short-hand interface but are still queued inside DDT.
  DDT.deleteEdge(BB0, BB0);
  DDT.insertEdge(BB1, BB1);

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
  // queued for deletion. Its parent is still F until DDT.flush() is called. We
  // don't defer this action because it can cause problems for other transforms
  // or analysis as it's part of the actual CFG. We only defer updates to the
  // DominatorTree. This code will crash if it is placed before the
  // BranchInst::Create() call above.
  ASSERT_FALSE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_FALSE(DDT.pendingDeletedBB(BB3));
  DDT.deleteBB(BB3);
  EXPECT_TRUE(DDT.pendingDeletedBB(BB3));
  ASSERT_TRUE(isa<UnreachableInst>(BB3->getTerminator()));
  EXPECT_EQ(BB3->getParent(), F);

  // Verify. Updates to DDT must be applied *after* all changes to the CFG
  // (including block deletion).
  DDT.applyUpdates(Updates);
  ASSERT_TRUE(DDT.flush().verify());
}

TEST(DeferredDominance, PairedUpdate) {
  StringRef FuncName = "f";
  StringRef ModuleString =
      "define i32 @f(i32 %i, i32 *%p) {\n"
      " bb0:\n"
      "   store i32 %i, i32 *%p\n"
      "   switch i32 %i, label %bb1 [\n"
      "     i32 0, label %bb2\n"
      "     i32 1, label %bb2\n"
      "   ]\n"
      " bb1:\n"
      "   ret i32 1\n"
      " bb2:\n"
      "   ret i32 2\n"
      "}\n";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Couldn't get function " << FuncName << ".";

  // Make the DDT.
  DominatorTree DT(*F);
  DeferredDominance DDT(DT);
  ASSERT_TRUE(DDT.flush().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;
  BasicBlock *BB2 = &*FI++;

  // CFG Change: only edge from bb0 is bb0 -> bb1.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 3u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 1u);

  // Must be done after the CFG change. The applyUpdate() routine analyzes the
  // current state of the CFG.
  DDT.deleteEdge(BB0, BB2);

  // CFG Change: bb0 now has bb0 -> bb1 and bb0 -> bb2.
  // With this change no dominance has been altered from the original IR. DT
  // doesn't care if the type of TerminatorInstruction changed, only if the
  // unique edges have.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 1u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, BB2, ConstantInt::getTrue(F->getContext()), BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);

  // Must be done after the CFG change. The applyUpdate() routine analyzes the
  // current state of the CFG. This DDT update pairs with the previous one and
  // is cancelled out before ever applying updates to DT.
  DDT.insertEdge(BB0, BB2);

  // Test the empty DeletedBB list.
  EXPECT_FALSE(DDT.pendingDeletedBB(BB0));
  EXPECT_FALSE(DDT.pendingDeletedBB(BB1));
  EXPECT_FALSE(DDT.pendingDeletedBB(BB2));

  // The DT has no changes, this flush() simply returns a reference to the
  // internal DT calculated at the beginning of this test.
  ASSERT_TRUE(DDT.flush().verify());
}

TEST(DeferredDominance, ReplaceEntryBB) {
  StringRef FuncName = "f";
  StringRef ModuleString =
      "define i32 @f() {\n"
      "bb0:\n"
      "   br label %bb1\n"
      " bb1:\n"
      "   ret i32 1\n"
      "}\n";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Couldn't get function " << FuncName << ".";

  // Make the DDT.
  DominatorTree DT(*F);
  DeferredDominance DDT(DT);
  ASSERT_TRUE(DDT.flush().verify());

  Function::iterator FI = F->begin();
  BasicBlock *BB0 = &*FI++;
  BasicBlock *BB1 = &*FI++;

  // Add a block as the new function entry BB. We also link it to BB0.
  BasicBlock *NewEntry =
      BasicBlock::Create(F->getContext(), "new_entry", F, BB0);
  BranchInst::Create(BB0, NewEntry);
  EXPECT_EQ(F->begin()->getName(), NewEntry->getName());
  EXPECT_TRUE(&F->getEntryBlock() == NewEntry);

  // Insert the new edge between new_eentry -> bb0. Without this the
  // recalculate() call below will not actually recalculate the DT as there
  // are no changes pending and no blocks deleted.
  DDT.insertEdge(NewEntry, BB0);

  // Changing the Entry BB requires a full recalulation.
  DDT.recalculate(*F);
  ASSERT_TRUE(DDT.flush().verify());

  // CFG Change: remove new_edge -> bb0 and redirect to new_edge -> bb1.
  EXPECT_EQ(NewEntry->getTerminator()->getNumSuccessors(), 1u);
  NewEntry->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, NewEntry);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 1u);

  // Update the DDT. At this point bb0 now has no predecessors but is still a
  // Child of F.
  DDT.applyUpdates({{DominatorTree::Delete, NewEntry, BB0},
                    {DominatorTree::Insert, NewEntry, BB1}});
  ASSERT_TRUE(DDT.flush().verify());

  // Now remove bb0 from F.
  ASSERT_FALSE(isa<UnreachableInst>(BB0->getTerminator()));
  EXPECT_FALSE(DDT.pendingDeletedBB(BB0));
  DDT.deleteBB(BB0);
  EXPECT_TRUE(DDT.pendingDeletedBB(BB0));
  ASSERT_TRUE(isa<UnreachableInst>(BB0->getTerminator()));
  EXPECT_EQ(BB0->getParent(), F);

  // Perform a full recalculation of the DDT. It is not necessary here but we
  // do this to test the case when there are no pending DT updates but there are
  // pending deleted BBs.
  DDT.recalculate(*F);
  ASSERT_TRUE(DDT.flush().verify());
}

TEST(DeferredDominance, InheritedPreds) {
  StringRef FuncName = "f";
  StringRef ModuleString =
      "define i32 @f(i32 %i, i32 *%p) {\n"
      " bb0:\n"
      "   store i32 %i, i32 *%p\n"
      "   switch i32 %i, label %bb1 [\n"
      "     i32 2, label %bb2\n"
      "     i32 3, label %bb3\n"
      "   ]\n"
      " bb1:\n"
      "   br label %bb3\n"
      " bb2:\n"
      "   br label %bb3\n"
      " bb3:\n"
      "   ret i32 3\n"
      "}\n";
  // Make the module.
  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleString);
  Function *F = M->getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Couldn't get function " << FuncName << ".";

  // Make the DDT.
  DominatorTree DT(*F);
  DeferredDominance DDT(DT);
  ASSERT_TRUE(DDT.flush().verify());

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
  // DDT are not. In the first case we must have DDT.insertEdge(Pred1, Succ)
  // while in the latter case we must *NOT* have DDT.insertEdge(Pred1, Succ).

  // CFG Change: bb0 now only has bb0 -> bb1 and bb0 -> bb3. We are preparing to
  // remove bb2.
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 3u);
  BB0->getTerminator()->eraseFromParent();
  BranchInst::Create(BB1, BB3, ConstantInt::getTrue(F->getContext()), BB0);
  EXPECT_EQ(BB0->getTerminator()->getNumSuccessors(), 2u);

  // Remove bb2 from F. This has to happen before the call to applyUpdates() for
  // DDT to detect there is no longer an edge between bb2 -> bb3. The deleteBB()
  // method converts bb2's TI into "unreachable".
  ASSERT_FALSE(isa<UnreachableInst>(BB2->getTerminator()));
  EXPECT_FALSE(DDT.pendingDeletedBB(BB2));
  DDT.deleteBB(BB2);
  EXPECT_TRUE(DDT.pendingDeletedBB(BB2));
  ASSERT_TRUE(isa<UnreachableInst>(BB2->getTerminator()));
  EXPECT_EQ(BB2->getParent(), F);

  // Queue up the DDT updates.
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
  // DDT to detect there is no longer an edge between bb1 -> bb3. The deleteBB()
  // method converts bb1's TI into "unreachable".
  ASSERT_FALSE(isa<UnreachableInst>(BB1->getTerminator()));
  EXPECT_FALSE(DDT.pendingDeletedBB(BB1));
  DDT.deleteBB(BB1);
  EXPECT_TRUE(DDT.pendingDeletedBB(BB1));
  ASSERT_TRUE(isa<UnreachableInst>(BB1->getTerminator()));
  EXPECT_EQ(BB1->getParent(), F);

  // Update the DDT. In this case we don't call DDT.insertEdge(BB0, BB3) because
  // the edge previously existed at the start of this test when DT was first
  // created.
  Updates.push_back({DominatorTree::Delete, BB0, BB1});
  Updates.push_back({DominatorTree::Delete, BB1, BB3});

  // Verify everything.
  DDT.applyUpdates(Updates);
  ASSERT_TRUE(DDT.flush().verify());
}
