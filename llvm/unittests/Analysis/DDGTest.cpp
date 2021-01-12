//===- DDGTest.cpp - DDGAnalysis unit tests -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DDG.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

/// Build the DDG analysis for a loop and run the given test \p Test.
static void runTest(Module &M, StringRef FuncName,
                    function_ref<void(Function &F, LoopInfo &LI,
                                      DependenceInfo &DI, ScalarEvolution &SE)>
                        Test) {
  auto *F = M.getFunction(FuncName);
  ASSERT_NE(F, nullptr) << "Could not find " << FuncName;

  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(*F);
  DominatorTree DT(*F);
  LoopInfo LI(DT);
  ScalarEvolution SE(*F, TLI, AC, DT, LI);
  AAResults AA(TLI);
  DependenceInfo DI(F, &AA, &SE, &LI);
  Test(*F, LI, DI, SE);
}

static std::unique_ptr<Module> makeLLVMModule(LLVMContext &Context,
                                              const char *ModuleStr) {
  SMDiagnostic Err;
  return parseAssemblyString(ModuleStr, Err, Context);
}

TEST(DDGTest, getDependencies) {
  const char *ModuleStr =
      "target datalayout = \"e-m:e-i64:64-n32:64\"\n"
      "target triple = \"powerpc64le-unknown-linux-gnu\"\n"
      "\n"
      "define dso_local void @foo(i32 signext %n, i32* noalias %A, i32* "
      "noalias %B) {\n"
      "entry:\n"
      "   %cmp1 = icmp sgt i32 %n, 0\n"
      "   br i1 %cmp1, label %for.body.preheader, label %for.end\n"
      "\n"
      "for.body.preheader:\n"
      "   %wide.trip.count = zext i32 %n to i64\n"
      "   br label %for.body\n"
      " \n"
      " for.body:\n"
      "   %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ "
      "%indvars.iv.next, %for.body ]\n"
      "   %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv\n"
      "  %0 = trunc i64 %indvars.iv to i32\n"
      "  store i32 %0, i32* %arrayidx, align 4\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 "
      "%indvars.iv.next\n"
      "  %1 = load i32, i32* %arrayidx2, align 4\n"
      "  %add3 = add nsw i32 %1, 1\n"
      "  %arrayidx5 = getelementptr inbounds i32, i32* %B, i64 %indvars.iv\n"
      "  store i32 %add3, i32* %arrayidx5, align 4\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count\n"
      "  br i1 %exitcond, label %for.body, label %for.end.loopexit\n"
      "\n"
      "for.end.loopexit:\n"
      "  br label %for.end\n"
      "\n"
      "for.end:\n"
      "  ret void\n"
      "}\n";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runTest(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, DependenceInfo &DI, ScalarEvolution &SE) {
        Loop *L = *LI.begin();
        assert(L && "expected the loop to be identified.");

        DataDependenceGraph DDG(*L, LI, DI);

        // Collect all the nodes that have an outgoing memory edge
        // while collecting all memory edges as well. There should
        // only be one node with an outgoing memory edge and there
        // should only be one memory edge in the entire graph.
        std::vector<DDGNode *> DependenceSourceNodes;
        std::vector<DDGEdge *> MemoryEdges;
        for (DDGNode *N : DDG) {
          for (DDGEdge *E : *N) {
            bool SourceAdded = false;
            if (E->isMemoryDependence()) {
              MemoryEdges.push_back(E);
              if (!SourceAdded) {
                DependenceSourceNodes.push_back(N);
                SourceAdded = true;
              }
            }
          }
        }

        EXPECT_EQ(DependenceSourceNodes.size(), 1ull);
        EXPECT_EQ(MemoryEdges.size(), 1ull);

        DataDependenceGraph::DependenceList DL;
        DDG.getDependencies(*DependenceSourceNodes.back(),
                            MemoryEdges.back()->getTargetNode(), DL);

        EXPECT_EQ(DL.size(), 1ull);
        EXPECT_TRUE(DL.back()->isAnti());
        EXPECT_EQ(DL.back()->getLevels(), 1u);
        EXPECT_NE(DL.back()->getDistance(1), nullptr);
        EXPECT_EQ(DL.back()->getDistance(1),
                  SE.getOne(DL.back()->getDistance(1)->getType()));
      });
}

/// Test to make sure that when pi-blocks are formed, multiple edges of the same
/// kind and direction are collapsed into a single edge.
/// In the test below, %loadASubI belongs to an outside node, which has input
/// dependency with multiple load instructions in the pi-block containing
/// %loadBSubI. We expect a single memory dependence edge from the outside node
/// to this pi-block. The pi-block also contains %add and %add7 both of which
/// feed a phi in an outside node. We expect a single def-use edge from the
/// pi-block to the node containing that phi.
TEST(DDGTest, avoidDuplicateEdgesToFromPiBlocks) {
  const char *ModuleStr =
      "target datalayout = \"e-m:e-i64:64-n32:64-v256:256:256-v512:512:512\"\n"
      "\n"
      "define void @foo(float* noalias %A, float* noalias %B, float* noalias "
      "%C, float* noalias %D, i32 signext %n) {\n"
      "entry:\n"
      "  %cmp1 = icmp sgt i32 %n, 0\n"
      "  br i1 %cmp1, label %for.body.preheader, label %for.end\n"
      "\n"
      "for.body.preheader:                               ; preds = %entry\n"
      "  %wide.trip.count = zext i32 %n to i64\n"
      "  br label %for.body\n"
      "\n"
      "for.body:                                         ; preds = "
      "%for.body.preheader, %if.end\n"
      "  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, "
      "%if.end ]\n"
      "  %arrayidx = getelementptr inbounds float, float* %A, i64 %indvars.iv\n"
      "  %loadASubI = load float, float* %arrayidx, align 4\n"
      "  %arrayidx2 = getelementptr inbounds float, float* %B, i64 "
      "%indvars.iv\n"
      "  %loadBSubI = load float, float* %arrayidx2, align 4\n"
      "  %add = fadd fast float %loadASubI, %loadBSubI\n"
      "  %arrayidx4 = getelementptr inbounds float, float* %A, i64 "
      "%indvars.iv\n"
      "  store float %add, float* %arrayidx4, align 4\n"
      "  %arrayidx6 = getelementptr inbounds float, float* %A, i64 "
      "%indvars.iv\n"
      "  %0 = load float, float* %arrayidx6, align 4\n"
      "  %add7 = fadd fast float %0, 1.000000e+00\n"
      "  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1\n"
      "  %arrayidx10 = getelementptr inbounds float, float* %B, i64 "
      "%indvars.iv.next\n"
      "  store float %add7, float* %arrayidx10, align 4\n"
      "  %arrayidx12 = getelementptr inbounds float, float* %A, i64 "
      "%indvars.iv\n"
      "  %1 = load float, float* %arrayidx12, align 4\n"
      "  %cmp13 = fcmp fast ogt float %1, 1.000000e+02\n"
      "  br i1 %cmp13, label %if.then, label %if.else\n"
      "\n"
      "if.then:                                          ; preds = %for.body\n"
      "  br label %if.end\n"
      "\n"
      "if.else:                                          ; preds = %for.body\n"
      "  br label %if.end\n"
      "\n"
      "if.end:                                           ; preds = %if.else, "
      "%if.then\n"
      "  %ff.0 = phi float [ %add, %if.then ], [ %add7, %if.else ]\n"
      "  store float %ff.0, float* %C, align 4\n"
      "  %exitcond = icmp ne i64 %indvars.iv.next, %wide.trip.count\n"
      "  br i1 %exitcond, label %for.body, label %for.end.loopexit\n"
      "\n"
      "for.end.loopexit:                                 ; preds = %if.end\n"
      "  br label %for.end\n"
      "\n"
      "for.end:                                          ; preds = "
      "%for.end.loopexit, %entry\n"
      "  ret void\n"
      "}\n";

  LLVMContext Context;
  std::unique_ptr<Module> M = makeLLVMModule(Context, ModuleStr);

  runTest(
      *M, "foo",
      [&](Function &F, LoopInfo &LI, DependenceInfo &DI, ScalarEvolution &SE) {
        Loop *L = *LI.begin();
        assert(L && "expected the loop to be identified.");

        DataDependenceGraph DDG(*L, LI, DI);

        const DDGNode *LoadASubI = nullptr;
        for (DDGNode *N : DDG) {
          if (!isa<SimpleDDGNode>(N))
            continue;
          SmallVector<Instruction *, 8> IList;
          N->collectInstructions([](const Instruction *I) { return true; },
                                 IList);
          if (llvm::any_of(IList, [](Instruction *I) {
                return I->getName() == "loadASubI";
              })) {
            LoadASubI = N;
            break;
          }
        }
        assert(LoadASubI && "Did not find load of A[i]");

        const PiBlockDDGNode *PiBlockWithBSubI = nullptr;
        for (DDGNode *N : DDG) {
          if (!isa<PiBlockDDGNode>(N))
            continue;
          for (DDGNode *M : cast<PiBlockDDGNode>(N)->getNodes()) {
            SmallVector<Instruction *, 8> IList;
            M->collectInstructions([](const Instruction *I) { return true; },
                                   IList);
            if (llvm::any_of(IList, [](Instruction *I) {
                  return I->getName() == "loadBSubI";
                })) {
              PiBlockWithBSubI = static_cast<PiBlockDDGNode *>(N);
              break;
            }
          }
          if (PiBlockWithBSubI)
            break;
        }
        assert(PiBlockWithBSubI &&
               "Did not find pi-block containing load of B[i]");

        const DDGNode *FFPhi = nullptr;
        for (DDGNode *N : DDG) {
          if (!isa<SimpleDDGNode>(N))
            continue;
          SmallVector<Instruction *, 8> IList;
          N->collectInstructions([](const Instruction *I) { return true; },
                                 IList);
          if (llvm::any_of(IList, [](Instruction *I) {
                return I->getName() == "ff.0";
              })) {
            FFPhi = N;
            break;
          }
        }
        assert(FFPhi && "Did not find ff.0 phi instruction");

        // Expect a single memory edge from '%0 = A[i]' to the pi-block. This
        // means the duplicate incoming memory edges are removed during pi-block
        // formation.
        SmallVector<DDGEdge *, 4> EL;
        LoadASubI->findEdgesTo(*PiBlockWithBSubI, EL);
        unsigned NumMemoryEdges = llvm::count_if(
            EL, [](DDGEdge *Edge) { return Edge->isMemoryDependence(); });
        EXPECT_EQ(NumMemoryEdges, 1ull);

        /// Expect a single def-use edge from the pi-block to '%ff.0 = phi...`.
        /// This means the duplicate outgoing def-use edges are removed during
        /// pi-block formation.
        EL.clear();
        PiBlockWithBSubI->findEdgesTo(*FFPhi, EL);
        NumMemoryEdges =
            llvm::count_if(EL, [](DDGEdge *Edge) { return Edge->isDefUse(); });
        EXPECT_EQ(NumMemoryEdges, 1ull);
      });
}
