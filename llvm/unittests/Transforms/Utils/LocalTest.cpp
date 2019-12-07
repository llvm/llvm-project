//===- Local.cpp - Unit tests for Local -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(Local, RecursivelyDeleteDeadPHINodes) {
  LLVMContext C;

  IRBuilder<> builder(C);

  // Make blocks
  BasicBlock *bb0 = BasicBlock::Create(C);
  BasicBlock *bb1 = BasicBlock::Create(C);

  builder.SetInsertPoint(bb0);
  PHINode    *phi = builder.CreatePHI(Type::getInt32Ty(C), 2);
  BranchInst *br0 = builder.CreateCondBr(builder.getTrue(), bb0, bb1);

  builder.SetInsertPoint(bb1);
  BranchInst *br1 = builder.CreateBr(bb0);

  phi->addIncoming(phi, bb0);
  phi->addIncoming(phi, bb1);

  // The PHI will be removed
  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  // Make sure the blocks only contain the branches
  EXPECT_EQ(&bb0->front(), br0);
  EXPECT_EQ(&bb1->front(), br1);

  builder.SetInsertPoint(bb0);
  phi = builder.CreatePHI(Type::getInt32Ty(C), 0);

  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  builder.SetInsertPoint(bb0);
  phi = builder.CreatePHI(Type::getInt32Ty(C), 0);
  builder.CreateAdd(phi, phi);

  EXPECT_TRUE(RecursivelyDeleteDeadPHINode(phi));

  bb0->dropAllReferences();
  bb1->dropAllReferences();
  delete bb0;
  delete bb1;
}

TEST(Local, RemoveDuplicatePHINodes) {
  LLVMContext C;
  IRBuilder<> B(C);

  std::unique_ptr<Function> F(
      Function::Create(FunctionType::get(B.getVoidTy(), false),
                       GlobalValue::ExternalLinkage, "F"));
  BasicBlock *Entry(BasicBlock::Create(C, "", F.get()));
  BasicBlock *BB(BasicBlock::Create(C, "", F.get()));
  BranchInst::Create(BB, Entry);

  B.SetInsertPoint(BB);

  AssertingVH<PHINode> P1 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P1->addIncoming(B.getInt32(42), Entry);

  PHINode *P2 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P2->addIncoming(B.getInt32(42), Entry);

  AssertingVH<PHINode> P3 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P3->addIncoming(B.getInt32(42), Entry);
  P3->addIncoming(B.getInt32(23), BB);

  PHINode *P4 = B.CreatePHI(Type::getInt32Ty(C), 2);
  P4->addIncoming(B.getInt32(42), Entry);
  P4->addIncoming(B.getInt32(23), BB);

  P1->addIncoming(P3, BB);
  P2->addIncoming(P4, BB);
  BranchInst::Create(BB, BB);

  // Verify that we can eliminate PHIs that become duplicates after chaning PHIs
  // downstream.
  EXPECT_TRUE(EliminateDuplicatePHINodes(BB));
  EXPECT_EQ(3U, BB->size());
}

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("UtilsTests", errs());
  return Mod;
}

TEST(Local, ReplaceDbgDeclare) {
  LLVMContext C;

  // Original C source to get debug info for a local variable:
  // void f() { int x; }
  std::unique_ptr<Module> M = parseIR(C,
                                      R"(
      define void @f() !dbg !8 {
      entry:
        %x = alloca i32, align 4
        call void @llvm.dbg.declare(metadata i32* %x, metadata !11, metadata !DIExpression()), !dbg !13
        call void @llvm.dbg.declare(metadata i32* %x, metadata !11, metadata !DIExpression()), !dbg !13
        ret void, !dbg !14
      }
      declare void @llvm.dbg.declare(metadata, metadata, metadata)
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3, !4}
      !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
      !1 = !DIFile(filename: "t2.c", directory: "foo")
      !2 = !{}
      !3 = !{i32 2, !"Dwarf Version", i32 4}
      !4 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
      !9 = !DISubroutineType(types: !10)
      !10 = !{null}
      !11 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !12)
      !12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !13 = !DILocation(line: 2, column: 7, scope: !8)
      !14 = !DILocation(line: 3, column: 1, scope: !8)
      )");
  auto *GV = M->getNamedValue("f");
  ASSERT_TRUE(GV);
  auto *F = dyn_cast<Function>(GV);
  ASSERT_TRUE(F);
  Instruction *Inst = &F->front().front();
  auto *AI = dyn_cast<AllocaInst>(Inst);
  ASSERT_TRUE(AI);
  Inst = Inst->getNextNode()->getNextNode();
  ASSERT_TRUE(Inst);
  auto *DII = dyn_cast<DbgDeclareInst>(Inst);
  ASSERT_TRUE(DII);
  Value *NewBase = Constant::getNullValue(Type::getInt32PtrTy(C));
  DIBuilder DIB(*M);
  replaceDbgDeclare(AI, NewBase, DII, DIB, DIExpression::ApplyOffset, 0);

  // There should be exactly two dbg.declares.
  int Declares = 0;
  for (const Instruction &I : F->front())
    if (isa<DbgDeclareInst>(I))
      Declares++;
  EXPECT_EQ(2, Declares);
}

/// Build the dominator tree for the function and run the Test.
static void runWithDomTree(
    Module &M, StringRef FuncName,
    function_ref<void(Function &F, DominatorTree *DT)> Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
  // Compute the dominator tree for the function.
  DominatorTree DT(*F);
  Test(*F, &DT);
}

TEST(Local, MergeBasicBlockIntoOnlyPred) {
  LLVMContext C;
  std::unique_ptr<Module> M;
  auto resetIR = [&]() {
    M = parseIR(C,
                R"(
      define i32 @f(i8* %str) {
      entry:
        br label %bb2.i
      bb2.i:                                            ; preds = %bb4.i, %entry
        br i1 false, label %bb4.i, label %base2flt.exit204
      bb4.i:                                            ; preds = %bb2.i
        br i1 false, label %base2flt.exit204, label %bb2.i
      bb10.i196.bb7.i197_crit_edge:                     ; No predecessors!
        br label %bb7.i197
      bb7.i197:                                         ; preds = %bb10.i196.bb7.i197_crit_edge
        %.reg2mem.0 = phi i32 [ %.reg2mem.0, %bb10.i196.bb7.i197_crit_edge ]
        br i1 undef, label %base2flt.exit204, label %base2flt.exit204
      base2flt.exit204:                                 ; preds = %bb7.i197, %bb7.i197, %bb2.i, %bb4.i
        ret i32 0
      }
      )");
  };

  auto resetIRReplaceEntry = [&]() {
    M = parseIR(C,
                R"(
      define i32 @f() {
      entry:
        br label %bb2.i
      bb2.i:                                            ; preds = %entry
        ret i32 0
      }
      )");
  };

  auto Test = [&](Function &F, DomTreeUpdater &DTU) {
    for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
      BasicBlock *BB = &*I++;
      BasicBlock *SinglePred = BB->getSinglePredecessor();
      if (!SinglePred || SinglePred == BB || BB->hasAddressTaken())
        continue;
      BranchInst *Term = dyn_cast<BranchInst>(SinglePred->getTerminator());
      if (Term && !Term->isConditional())
        MergeBasicBlockIntoOnlyPred(BB, &DTU);
    }
    if (DTU.hasDomTree()) {
      EXPECT_TRUE(DTU.getDomTree().verify());
    }
    if (DTU.hasPostDomTree()) {
      EXPECT_TRUE(DTU.getPostDomTree().verify());
    }
  };

  // Test MergeBasicBlockIntoOnlyPred working under Eager UpdateStrategy with
  // both DT and PDT.
  resetIR();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Eager UpdateStrategy with
  // DT.
  resetIR();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    DomTreeUpdater DTU(*DT, DomTreeUpdater::UpdateStrategy::Eager);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Eager UpdateStrategy with
  // PDT.
  resetIR();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(PDT, DomTreeUpdater::UpdateStrategy::Eager);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Lazy UpdateStrategy with
  // both DT and PDT.
  resetIR();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Lazy UpdateStrategy with
  // PDT.
  resetIR();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(PDT, DomTreeUpdater::UpdateStrategy::Lazy);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Lazy UpdateStrategy with DT.
  resetIR();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    DomTreeUpdater DTU(*DT, DomTreeUpdater::UpdateStrategy::Lazy);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Eager UpdateStrategy with
  // both DT and PDT.
  resetIRReplaceEntry();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Eager UpdateStrategy with
  // DT.
  resetIRReplaceEntry();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    DomTreeUpdater DTU(*DT, DomTreeUpdater::UpdateStrategy::Eager);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Eager UpdateStrategy with
  // PDT.
  resetIRReplaceEntry();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(PDT, DomTreeUpdater::UpdateStrategy::Eager);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Lazy UpdateStrategy with
  // both DT and PDT.
  resetIRReplaceEntry();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Lazy UpdateStrategy with
  // PDT.
  resetIRReplaceEntry();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(PDT, DomTreeUpdater::UpdateStrategy::Lazy);
    Test(F, DTU);
  });

  // Test MergeBasicBlockIntoOnlyPred working under Lazy UpdateStrategy with DT.
  resetIRReplaceEntry();
  runWithDomTree(*M, "f", [&](Function &F, DominatorTree *DT) {
    DomTreeUpdater DTU(*DT, DomTreeUpdater::UpdateStrategy::Lazy);
    Test(F, DTU);
  });
}

TEST(Local, ConstantFoldTerminator) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(C,
                                      R"(
      define void @br_same_dest() {
      entry:
        br i1 false, label %bb0, label %bb0
      bb0:
        ret void
      }

      define void @br_different_dest() {
      entry:
        br i1 true, label %bb0, label %bb1
      bb0:
        br label %exit
      bb1:
        br label %exit
      exit:
        ret void
      }

      define void @switch_2_different_dest() {
      entry:
        switch i32 0, label %default [ i32 0, label %bb0 ]
      default:
        ret void
      bb0:
        ret void
      }
      define void @switch_2_different_dest_default() {
      entry:
        switch i32 1, label %default [ i32 0, label %bb0 ]
      default:
        ret void
      bb0:
        ret void
      }
      define void @switch_3_different_dest() {
      entry:
        switch i32 0, label %default [ i32 0, label %bb0
                                       i32 1, label %bb1 ]
      default:
        ret void
      bb0:
        ret void
      bb1:
        ret void
      }

      define void @switch_variable_2_default_dest(i32 %arg) {
      entry:
        switch i32 %arg, label %default [ i32 0, label %default ]
      default:
        ret void
      }

      define void @switch_constant_2_default_dest() {
      entry:
        switch i32 1, label %default [ i32 0, label %default ]
      default:
        ret void
      }

      define void @switch_constant_3_repeated_dest() {
      entry:
        switch i32 0, label %default [ i32 0, label %bb0
                                       i32 1, label %bb0 ]
       bb0:
         ret void
      default:
        ret void
      }

      define void @indirectbr() {
      entry:
        indirectbr i8* blockaddress(@indirectbr, %bb0), [label %bb0, label %bb1]
      bb0:
        ret void
      bb1:
        ret void
      }

      define void @indirectbr_repeated() {
      entry:
        indirectbr i8* blockaddress(@indirectbr_repeated, %bb0), [label %bb0, label %bb0]
      bb0:
        ret void
      }

      define void @indirectbr_unreachable() {
      entry:
        indirectbr i8* blockaddress(@indirectbr_unreachable, %bb0), [label %bb1]
      bb0:
        ret void
      bb1:
        ret void
      }
        )");

  auto CFAllTerminatorsEager = [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
    for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
      BasicBlock *BB = &*I++;
      ConstantFoldTerminator(BB, true, nullptr, &DTU);
    }

    EXPECT_TRUE(DTU.getDomTree().verify());
    EXPECT_TRUE(DTU.getPostDomTree().verify());
  };

  auto CFAllTerminatorsLazy = [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
    for (Function::iterator I = F.begin(), E = F.end(); I != E;) {
      BasicBlock *BB = &*I++;
      ConstantFoldTerminator(BB, true, nullptr, &DTU);
    }

    EXPECT_TRUE(DTU.getDomTree().verify());
    EXPECT_TRUE(DTU.getPostDomTree().verify());
  };

  // Test ConstantFoldTerminator under Eager UpdateStrategy.
  runWithDomTree(*M, "br_same_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "br_different_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "switch_2_different_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "switch_2_different_dest_default", CFAllTerminatorsEager);
  runWithDomTree(*M, "switch_3_different_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "switch_variable_2_default_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "switch_constant_2_default_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "switch_constant_3_repeated_dest", CFAllTerminatorsEager);
  runWithDomTree(*M, "indirectbr", CFAllTerminatorsEager);
  runWithDomTree(*M, "indirectbr_repeated", CFAllTerminatorsEager);
  runWithDomTree(*M, "indirectbr_unreachable", CFAllTerminatorsEager);

  // Test ConstantFoldTerminator under Lazy UpdateStrategy.
  runWithDomTree(*M, "br_same_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "br_different_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "switch_2_different_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "switch_2_different_dest_default", CFAllTerminatorsLazy);
  runWithDomTree(*M, "switch_3_different_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "switch_variable_2_default_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "switch_constant_2_default_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "switch_constant_3_repeated_dest", CFAllTerminatorsLazy);
  runWithDomTree(*M, "indirectbr", CFAllTerminatorsLazy);
  runWithDomTree(*M, "indirectbr_repeated", CFAllTerminatorsLazy);
  runWithDomTree(*M, "indirectbr_unreachable", CFAllTerminatorsLazy);
}

struct SalvageDebugInfoTest : ::testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;
  Function *F = nullptr;

  void SetUp() {
    M = parseIR(C,
                R"(
      define void @f() !dbg !8 {
      entry:
        %x = add i32 0, 1
        %y = add i32 %x, 2
        call void @llvm.dbg.value(metadata i32 %x, metadata !11, metadata !DIExpression()), !dbg !13
        call void @llvm.dbg.value(metadata i32 %y, metadata !11, metadata !DIExpression()), !dbg !13
        ret void, !dbg !14
      }
      declare void @llvm.dbg.value(metadata, metadata, metadata)
      !llvm.dbg.cu = !{!0}
      !llvm.module.flags = !{!3, !4}
      !0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
      !1 = !DIFile(filename: "t2.c", directory: "foo")
      !2 = !{}
      !3 = !{i32 2, !"Dwarf Version", i32 4}
      !4 = !{i32 2, !"Debug Info Version", i32 3}
      !8 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: false, unit: !0, retainedNodes: !2)
      !9 = !DISubroutineType(types: !10)
      !10 = !{null}
      !11 = !DILocalVariable(name: "x", scope: !8, file: !1, line: 2, type: !12)
      !12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
      !13 = !DILocation(line: 2, column: 7, scope: !8)
      !14 = !DILocation(line: 3, column: 1, scope: !8)
      )");

    auto *GV = M->getNamedValue("f");
    ASSERT_TRUE(GV);
    F = dyn_cast<Function>(GV);
    ASSERT_TRUE(F);
  }

  bool doesDebugValueDescribeX(const DbgValueInst &DI) {
    const auto &CI = *cast<ConstantInt>(DI.getValue());
    if (CI.isZero())
      return DI.getExpression()->getElements().equals(
          {dwarf::DW_OP_plus_uconst, 1, dwarf::DW_OP_stack_value});
    else if (CI.isOneValue())
      return DI.getExpression()->getElements().empty();
    return false;
  }

  bool doesDebugValueDescribeY(const DbgValueInst &DI) {
    const auto &CI = *cast<ConstantInt>(DI.getValue());
    if (CI.isZero())
      return DI.getExpression()->getElements().equals(
          {dwarf::DW_OP_plus_uconst, 1, dwarf::DW_OP_plus_uconst, 2,
           dwarf::DW_OP_stack_value});
    else if (CI.isOneValue())
      return DI.getExpression()->getElements().equals(
          {dwarf::DW_OP_plus_uconst, 2, dwarf::DW_OP_stack_value});
    return false;
  }

  void verifyDebugValuesAreSalvaged() {
    // Check that the debug values for %x and %y are preserved.
    bool FoundX = false;
    bool FoundY = false;
    for (const Instruction &I : F->front()) {
      auto DI = dyn_cast<DbgValueInst>(&I);
      if (!DI) {
        // The function should only contain debug values and a terminator.
        ASSERT_TRUE(I.isTerminator());
        continue;
      }
      EXPECT_EQ(DI->getVariable()->getName(), "x");
      FoundX |= doesDebugValueDescribeX(*DI);
      FoundY |= doesDebugValueDescribeY(*DI);
    }
    ASSERT_TRUE(FoundX);
    ASSERT_TRUE(FoundY);
  }
};

TEST_F(SalvageDebugInfoTest, RecursiveInstDeletion) {
  Instruction *Inst = &F->front().front();
  Inst = Inst->getNextNode(); // Get %y = add ...
  ASSERT_TRUE(Inst);
  bool Deleted = RecursivelyDeleteTriviallyDeadInstructions(Inst);
  ASSERT_TRUE(Deleted);
  verifyDebugValuesAreSalvaged();
}

TEST_F(SalvageDebugInfoTest, RecursiveBlockSimplification) {
  BasicBlock *BB = &F->front();
  ASSERT_TRUE(BB);
  bool Deleted = SimplifyInstructionsInBlock(BB);
  ASSERT_TRUE(Deleted);
  verifyDebugValuesAreSalvaged();
}

TEST(Local, ChangeToUnreachable) {
  LLVMContext Ctx;

  std::unique_ptr<Module> M = parseIR(Ctx,
                                      R"(
    define internal void @foo() !dbg !6 {
    entry:
      ret void, !dbg !8
    }

    !llvm.dbg.cu = !{!0}
    !llvm.debugify = !{!3, !4}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "test.ll", directory: "/")
    !2 = !{}
    !3 = !{i32 1}
    !4 = !{i32 0}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "foo", linkageName: "foo", scope: null, file: !1, line: 1, type: !7, isLocal: true, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !2)
    !7 = !DISubroutineType(types: !2)
    !8 = !DILocation(line: 1, column: 1, scope: !6)
  )");

  bool BrokenDebugInfo = true;
  verifyModule(*M, &errs(), &BrokenDebugInfo);
  ASSERT_FALSE(BrokenDebugInfo);

  Function &F = *cast<Function>(M->getNamedValue("foo"));

  BasicBlock &BB = F.front();
  Instruction &A = BB.front();
  DebugLoc DLA = A.getDebugLoc();

  ASSERT_TRUE(isa<ReturnInst>(&A));
  // One instruction should be affected.
  EXPECT_EQ(changeToUnreachable(&A, /*UseLLVMTrap*/false), 1U);

  Instruction &B = BB.front();

  // There should be an uncreachable instruction.
  ASSERT_TRUE(isa<UnreachableInst>(&B));

  DebugLoc DLB = B.getDebugLoc();
  EXPECT_EQ(DLA, DLB);
}

TEST(Local, ReplaceAllDbgUsesWith) {
  using namespace llvm::dwarf;

  LLVMContext Ctx;

  // Note: The datalayout simulates Darwin/x86_64.
  std::unique_ptr<Module> M = parseIR(Ctx,
                                      R"(
    target datalayout = "e-m:o-i63:64-f80:128-n8:16:32:64-S128"

    declare i32 @escape(i32)

    define void @f() !dbg !6 {
    entry:
      %a = add i32 0, 1, !dbg !15
      call void @llvm.dbg.value(metadata i32 %a, metadata !9, metadata !DIExpression()), !dbg !15

      %b = add i64 0, 1, !dbg !16
      call void @llvm.dbg.value(metadata i64 %b, metadata !11, metadata !DIExpression()), !dbg !16
      call void @llvm.dbg.value(metadata i64 %b, metadata !11, metadata !DIExpression(DW_OP_lit0, DW_OP_mul)), !dbg !16
      call void @llvm.dbg.value(metadata i64 %b, metadata !11, metadata !DIExpression(DW_OP_lit0, DW_OP_mul, DW_OP_stack_value)), !dbg !16
      call void @llvm.dbg.value(metadata i64 %b, metadata !11, metadata !DIExpression(DW_OP_LLVM_fragment, 0, 8)), !dbg !16
      call void @llvm.dbg.value(metadata i64 %b, metadata !11, metadata !DIExpression(DW_OP_lit0, DW_OP_mul, DW_OP_LLVM_fragment, 0, 8)), !dbg !16
      call void @llvm.dbg.value(metadata i64 %b, metadata !11, metadata !DIExpression(DW_OP_lit0, DW_OP_mul, DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 8)), !dbg !16

      %c = inttoptr i64 0 to i64*, !dbg !17
      call void @llvm.dbg.declare(metadata i64* %c, metadata !13, metadata !DIExpression()), !dbg !17

      %d = inttoptr i64 0 to i32*, !dbg !18
      call void @llvm.dbg.addr(metadata i32* %d, metadata !20, metadata !DIExpression()), !dbg !18

      %e = add <2 x i16> zeroinitializer, zeroinitializer
      call void @llvm.dbg.value(metadata <2 x i16> %e, metadata !14, metadata !DIExpression()), !dbg !18

      %f = call i32 @escape(i32 0)
      call void @llvm.dbg.value(metadata i32 %f, metadata !9, metadata !DIExpression()), !dbg !15

      %barrier = call i32 @escape(i32 0)

      %g = call i32 @escape(i32 %f)
      call void @llvm.dbg.value(metadata i32 %g, metadata !9, metadata !DIExpression()), !dbg !15

      ret void, !dbg !19
    }

    declare void @llvm.dbg.addr(metadata, metadata, metadata)
    declare void @llvm.dbg.declare(metadata, metadata, metadata)
    declare void @llvm.dbg.value(metadata, metadata, metadata)

    !llvm.dbg.cu = !{!0}
    !llvm.module.flags = !{!5}

    !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
    !1 = !DIFile(filename: "/Users/vsk/Desktop/foo.ll", directory: "/")
    !2 = !{}
    !5 = !{i32 2, !"Debug Info Version", i32 3}
    !6 = distinct !DISubprogram(name: "f", linkageName: "f", scope: null, file: !1, line: 1, type: !7, isLocal: false, isDefinition: true, scopeLine: 1, isOptimized: true, unit: !0, retainedNodes: !8)
    !7 = !DISubroutineType(types: !2)
    !8 = !{!9, !11, !13, !14}
    !9 = !DILocalVariable(name: "1", scope: !6, file: !1, line: 1, type: !10)
    !10 = !DIBasicType(name: "ty32", size: 32, encoding: DW_ATE_signed)
    !11 = !DILocalVariable(name: "2", scope: !6, file: !1, line: 2, type: !12)
    !12 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_signed)
    !13 = !DILocalVariable(name: "3", scope: !6, file: !1, line: 3, type: !12)
    !14 = !DILocalVariable(name: "4", scope: !6, file: !1, line: 4, type: !10)
    !15 = !DILocation(line: 1, column: 1, scope: !6)
    !16 = !DILocation(line: 2, column: 1, scope: !6)
    !17 = !DILocation(line: 3, column: 1, scope: !6)
    !18 = !DILocation(line: 4, column: 1, scope: !6)
    !19 = !DILocation(line: 5, column: 1, scope: !6)
    !20 = !DILocalVariable(name: "5", scope: !6, file: !1, line: 5, type: !10)
  )");

  bool BrokenDebugInfo = true;
  verifyModule(*M, &errs(), &BrokenDebugInfo);
  ASSERT_FALSE(BrokenDebugInfo);

  Function &F = *cast<Function>(M->getNamedValue("f"));
  DominatorTree DT{F};

  BasicBlock &BB = F.front();
  Instruction &A = BB.front();
  Instruction &B = *A.getNextNonDebugInstruction();
  Instruction &C = *B.getNextNonDebugInstruction();
  Instruction &D = *C.getNextNonDebugInstruction();
  Instruction &E = *D.getNextNonDebugInstruction();
  Instruction &F_ = *E.getNextNonDebugInstruction();
  Instruction &Barrier = *F_.getNextNonDebugInstruction();
  Instruction &G = *Barrier.getNextNonDebugInstruction();

  // Simulate i32 <-> i64* conversion. Expect no updates: the datalayout says
  // pointers are 64 bits, so the conversion would be lossy.
  EXPECT_FALSE(replaceAllDbgUsesWith(A, C, C, DT));
  EXPECT_FALSE(replaceAllDbgUsesWith(C, A, A, DT));

  // Simulate i32 <-> <2 x i16> conversion. This is unsupported.
  EXPECT_FALSE(replaceAllDbgUsesWith(E, A, A, DT));
  EXPECT_FALSE(replaceAllDbgUsesWith(A, E, E, DT));

  // Simulate i32* <-> i64* conversion.
  EXPECT_TRUE(replaceAllDbgUsesWith(D, C, C, DT));

  SmallVector<DbgVariableIntrinsic *, 2> CDbgVals;
  findDbgUsers(CDbgVals, &C);
  EXPECT_EQ(2U, CDbgVals.size());
  EXPECT_TRUE(any_of(CDbgVals, [](DbgVariableIntrinsic *DII) {
    return isa<DbgAddrIntrinsic>(DII);
  }));
  EXPECT_TRUE(any_of(CDbgVals, [](DbgVariableIntrinsic *DII) {
    return isa<DbgDeclareInst>(DII);
  }));

  EXPECT_TRUE(replaceAllDbgUsesWith(C, D, D, DT));

  SmallVector<DbgVariableIntrinsic *, 2> DDbgVals;
  findDbgUsers(DDbgVals, &D);
  EXPECT_EQ(2U, DDbgVals.size());
  EXPECT_TRUE(any_of(DDbgVals, [](DbgVariableIntrinsic *DII) {
    return isa<DbgAddrIntrinsic>(DII);
  }));
  EXPECT_TRUE(any_of(DDbgVals, [](DbgVariableIntrinsic *DII) {
    return isa<DbgDeclareInst>(DII);
  }));

  // Introduce a use-before-def. Check that the dbg.value for %a is salvaged.
  EXPECT_TRUE(replaceAllDbgUsesWith(A, F_, F_, DT));

  auto *ADbgVal = cast<DbgValueInst>(A.getNextNode());
  EXPECT_EQ(ConstantInt::get(A.getType(), 0), ADbgVal->getVariableLocation());

  // Introduce a use-before-def. Check that the dbg.values for %f become undef.
  EXPECT_TRUE(replaceAllDbgUsesWith(F_, G, G, DT));

  auto *FDbgVal = cast<DbgValueInst>(F_.getNextNode());
  EXPECT_TRUE(isa<UndefValue>(FDbgVal->getVariableLocation()));

  SmallVector<DbgValueInst *, 1> FDbgVals;
  findDbgValues(FDbgVals, &F_);
  EXPECT_EQ(0U, FDbgVals.size());

  // Simulate i32 -> i64 conversion to test sign-extension. Here are some
  // interesting cases to handle:
  //  1) debug user has empty DIExpression
  //  2) debug user has non-empty, non-stack-value'd DIExpression
  //  3) debug user has non-empty, stack-value'd DIExpression
  //  4-6) like (1-3), but with a fragment
  EXPECT_TRUE(replaceAllDbgUsesWith(B, A, A, DT));

  SmallVector<DbgValueInst *, 8> ADbgVals;
  findDbgValues(ADbgVals, &A);
  EXPECT_EQ(6U, ADbgVals.size());

  // Check that %a has a dbg.value with a DIExpression matching \p Ops.
  auto hasADbgVal = [&](ArrayRef<uint64_t> Ops) {
    return any_of(ADbgVals, [&](DbgValueInst *DVI) {
      assert(DVI->getVariable()->getName() == "2");
      return DVI->getExpression()->getElements() == Ops;
    });
  };

  // Case 1: The original expr is empty, so no deref is needed.
  EXPECT_TRUE(hasADbgVal({DW_OP_LLVM_convert, 32, DW_ATE_signed,
                         DW_OP_LLVM_convert, 64, DW_ATE_signed,
                         DW_OP_stack_value}));

  // Case 2: Perform an address calculation with the original expr, deref it,
  // then sign-extend the result.
  EXPECT_TRUE(hasADbgVal({DW_OP_lit0, DW_OP_mul, DW_OP_deref,
                         DW_OP_LLVM_convert, 32, DW_ATE_signed,
                         DW_OP_LLVM_convert, 64, DW_ATE_signed,
                         DW_OP_stack_value}));

  // Case 3: Insert the sign-extension logic before the DW_OP_stack_value.
  EXPECT_TRUE(hasADbgVal({DW_OP_lit0, DW_OP_mul, DW_OP_LLVM_convert, 32,
                         DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed,
                         DW_OP_stack_value}));

  // Cases 4-6: Just like cases 1-3, but preserve the fragment at the end.
  EXPECT_TRUE(hasADbgVal({DW_OP_LLVM_convert, 32, DW_ATE_signed,
                         DW_OP_LLVM_convert, 64, DW_ATE_signed,
                         DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 8}));

  EXPECT_TRUE(hasADbgVal({DW_OP_lit0, DW_OP_mul, DW_OP_deref,
                         DW_OP_LLVM_convert, 32, DW_ATE_signed,
                         DW_OP_LLVM_convert, 64, DW_ATE_signed,
                         DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 8}));

  EXPECT_TRUE(hasADbgVal({DW_OP_lit0, DW_OP_mul, DW_OP_LLVM_convert, 32,
                         DW_ATE_signed, DW_OP_LLVM_convert, 64, DW_ATE_signed,
                         DW_OP_stack_value, DW_OP_LLVM_fragment, 0, 8}));

  verifyModule(*M, &errs(), &BrokenDebugInfo);
  ASSERT_FALSE(BrokenDebugInfo);
}

TEST(Local, RemoveUnreachableBlocks) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(C,
                                      R"(
      define void @br_simple() {
      entry:
        br label %bb0
      bb0:
        ret void
      bb1:
        ret void
      }

      define void @br_self_loop() {
      entry:
        br label %bb0
      bb0:
        br i1 true, label %bb1, label %bb0
      bb1:
        br i1 true, label %bb0, label %bb2
      bb2:
        br label %bb2
      }

      define void @br_constant() {
      entry:
        br label %bb0
      bb0:
        br i1 true, label %bb1, label %bb2
      bb1:
        br i1 true, label %bb0, label %bb2
      bb2:
        br label %bb2
      }

      define void @br_loop() {
      entry:
        br label %bb0
      bb0:
        br label %bb0
      bb1:
        br label %bb2
      bb2:
        br label %bb1
      }

      declare i32 @__gxx_personality_v0(...)

      define void @invoke_terminator() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
      entry:
        br i1 undef, label %invoke.block, label %exit

      invoke.block:
        %cond = invoke zeroext i1 @invokable()
                to label %continue.block unwind label %lpad.block

      continue.block:
        br i1 %cond, label %if.then, label %if.end

      if.then:
        unreachable

      if.end:
        unreachable

      lpad.block:
        %lp = landingpad { i8*, i32 }
                catch i8* null
        br label %exit

      exit:
        ret void
      }

      declare i1 @invokable()
      )");

  auto runEager = [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Eager);
    removeUnreachableBlocks(F, &DTU);
    EXPECT_TRUE(DTU.getDomTree().verify());
    EXPECT_TRUE(DTU.getPostDomTree().verify());
  };

  auto runLazy = [&](Function &F, DominatorTree *DT) {
    PostDominatorTree PDT = PostDominatorTree(F);
    DomTreeUpdater DTU(*DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
    removeUnreachableBlocks(F, &DTU);
    EXPECT_TRUE(DTU.getDomTree().verify());
    EXPECT_TRUE(DTU.getPostDomTree().verify());
  };

  // Test removeUnreachableBlocks under Eager UpdateStrategy.
  runWithDomTree(*M, "br_simple", runEager);
  runWithDomTree(*M, "br_self_loop", runEager);
  runWithDomTree(*M, "br_constant", runEager);
  runWithDomTree(*M, "br_loop", runEager);
  runWithDomTree(*M, "invoke_terminator", runEager);

  // Test removeUnreachableBlocks under Lazy UpdateStrategy.
  runWithDomTree(*M, "br_simple", runLazy);
  runWithDomTree(*M, "br_self_loop", runLazy);
  runWithDomTree(*M, "br_constant", runLazy);
  runWithDomTree(*M, "br_loop", runLazy);
  runWithDomTree(*M, "invoke_terminator", runLazy);

  M = parseIR(C,
              R"(
      define void @f() {
      entry:
        ret void
      bb0:
        ret void
      }
        )");

  auto checkRUBlocksRetVal = [&](Function &F, DominatorTree *DT) {
    DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Lazy);
    EXPECT_TRUE(removeUnreachableBlocks(F, &DTU));
    EXPECT_FALSE(removeUnreachableBlocks(F, &DTU));
    EXPECT_TRUE(DTU.getDomTree().verify());
  };

  runWithDomTree(*M, "f", checkRUBlocksRetVal);
}

TEST(Local, SimplifyCFGWithNullAC) {
  LLVMContext Ctx;

  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    declare void @true_path()
    declare void @false_path()
    declare void @llvm.assume(i1 %cond);

    define i32 @foo(i1, i32) {
    entry:
      %cmp = icmp sgt i32 %1, 0
      br i1 %cmp, label %if.bb1, label %then.bb1
    if.bb1:
      call void @true_path()
      br label %test.bb
    then.bb1:
      call void @false_path()
      br label %test.bb
    test.bb:
      %phi = phi i1 [1, %if.bb1], [%0, %then.bb1]
      call void @llvm.assume(i1 %0)
      br i1 %phi, label %if.bb2, label %then.bb2
    if.bb2:
      ret i32 %1
    then.bb2:
      ret i32 0
    }
  )");

  Function &F = *cast<Function>(M->getNamedValue("foo"));
  TargetTransformInfo TTI(M->getDataLayout());

  SimplifyCFGOptions Options{};
  Options.setAssumptionCache(nullptr);

  // Obtain BasicBlock of interest to this test, %test.bb.
  BasicBlock *TestBB = nullptr;
  for (BasicBlock &BB : F) {
    if (BB.getName().equals("test.bb")) {
      TestBB = &BB;
      break;
    }
  }
  ASSERT_TRUE(TestBB);

  // %test.bb is expected to be simplified by FoldCondBranchOnPHI.
  EXPECT_TRUE(simplifyCFG(TestBB, TTI, Options));
}
