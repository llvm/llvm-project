//===- CallGraphSCCPassTest.cpp - Unit tests for the legacy CGPassManager -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

struct VisitCountingPass : public CallGraphSCCPass {
  static char ID;
  StringMap<unsigned> &Visits;

  VisitCountingPass(StringMap<unsigned> &Visits)
      : CallGraphSCCPass(ID), Visits(Visits) {}

  bool runOnSCC(CallGraphSCC &SCC) override {
    for (CallGraphNode *Node : SCC) {
      Function *F = Node->getFunction();
      if (F && !F->isDeclaration())
        ++Visits[F->getName()];
    }
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    CallGraphSCCPass::getAnalysisUsage(AU);
    AU.setPreservesAll();
  }
};

char VisitCountingPass::ID = 0;

// Runs the legacy CGPassManager over the module and checks that every function
// defined in it is visited exactly once.
void assertEachFunctionVisitedOnce(const char *IR,
                                   unsigned ExpectedFunctionsVisited) {
  // Arrange
  LLVMContext Context;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Context);
  ASSERT_TRUE(M) << Err.getMessage();
  initializeCallGraphWrapperPassPass(*PassRegistry::getPassRegistry());
  StringMap<unsigned> Visits;
  legacy::PassManager PM;
  PM.add(new VisitCountingPass(Visits));

  // Act
  PM.run(*M);

  // Assert
  // Check that no function was visited more than once (or not at all).
  for (const Function &F : *M) {
    if (F.isDeclaration())
      continue;
    EXPECT_EQ(1u, Visits.lookup(F.getName()))
        << "function '" << F.getName().str() << "' visited "
        << Visits.lookup(F.getName()) << " times";
  }
  ASSERT_EQ(ExpectedFunctionsVisited, Visits.size());
}

// @orphan is internal and not called, so the main traversal from the external
// calling node does not reach it. The catch-up traversal that starts at
// @orphan must not visit @shared again.
TEST(CallGraphSCCPassTest, OrphanCallingFunctionVisitedByMainTraversal) {
  assertEachFunctionVisitedOnce("define void @shared() {\n"
                                "  ret void\n"
                                "}\n"
                                "define void @foo() {\n"
                                "  call void @shared()\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @orphan() {\n"
                                "  call void @shared()\n"
                                "  ret void\n"
                                "}\n",
                                3u);
}

// Same as above, but the already visited function is reached through a chain:
// @orphan -> @mid -> @leaf, with @mid and @leaf both reached from @entry.
TEST(CallGraphSCCPassTest, OrphanReachingVisitedFunctionsThroughChain) {
  assertEachFunctionVisitedOnce("define void @leaf() {\n"
                                "  ret void\n"
                                "}\n"
                                "define void @mid() {\n"
                                "  call void @leaf()\n"
                                "  ret void\n"
                                "}\n"
                                "define void @entry() {\n"
                                "  call void @mid()\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @orphan() {\n"
                                "  call void @mid()\n"
                                "  ret void\n"
                                "}\n",
                                4u);
}

// Both functions are orphans. @orphan_leaf comes first in the module, so it
// gets its own catch-up traversal before the one that starts at
// @orphan_caller; that second traversal must not visit @orphan_leaf again.
TEST(CallGraphSCCPassTest, OrphanCallingEarlierVisitedOrphan) {
  assertEachFunctionVisitedOnce("define internal void @orphan_leaf() {\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @orphan_caller() {\n"
                                "  call void @orphan_leaf()\n"
                                "  ret void\n"
                                "}\n",
                                2u);
}

// An orphan SCC (two mutually recursive internal functions) calling a function
// that a previous orphan traversal already visited.
TEST(CallGraphSCCPassTest, OrphanCycleCallingEarlierVisitedOrphan) {
  assertEachFunctionVisitedOnce("define internal void @orphan_leaf() {\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @cycle_a() {\n"
                                "  call void @cycle_b()\n"
                                "  call void @orphan_leaf()\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @cycle_b() {\n"
                                "  call void @cycle_a()\n"
                                "  ret void\n"
                                "}\n",
                                3u);
}

// Orphans that only call each other in the order the module lists them, and a
// main-traversal function nobody else calls: nothing is reached twice. This
// passes today and guards the fix against dropping functions.
TEST(CallGraphSCCPassTest, OrphansWithoutSharedCalleesVisitedOnce) {
  assertEachFunctionVisitedOnce("define void @foo() {\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @orphan_caller() {\n"
                                "  call void @orphan_leaf()\n"
                                "  ret void\n"
                                "}\n"
                                "define internal void @orphan_leaf() {\n"
                                "  ret void\n"
                                "}\n",
                                3u);
}

} // namespace
