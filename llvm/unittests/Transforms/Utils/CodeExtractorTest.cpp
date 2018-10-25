//===- CodeExtractor.cpp - Unit tests for CodeExtractor -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/CodeExtractor.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
TEST(CodeExtractor, DISABLED_ExitStub) {
  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M(parseAssemblyString(R"invalid(
    define i32 @foo(i32 %x, i32 %y, i32 %z) {
    header:
      %0 = icmp ugt i32 %x, %y
      br i1 %0, label %body1, label %body2

    body1:
      %1 = add i32 %z, 2
      br label %notExtracted

    body2:
      %2 = mul i32 %z, 7
      br label %notExtracted

    notExtracted:
      %3 = phi i32 [ %1, %body1 ], [ %2, %body2 ]
      %4 = add i32 %3, %x
      ret i32 %4
    }
  )invalid",
                                                Err, Ctx));

  // CodeExtractor miscompiles this function. There appear to be some issues
  // with the handling of outlined regions with live output values.
  //
  // In the original function, CE adds two reloads in the codeReplacer block:
  //
  //   codeRepl:                                         ; preds = %header
  //     call void @foo_header.split(i32 %z, i32 %x, i32 %y, i32* %.loc, i32* %.loc1)
  //     %.reload = load i32, i32* %.loc
  //     %.reload2 = load i32, i32* %.loc1
  //     br label %notExtracted
  //
  // These reloads must flow into the notExtracted block:
  //
  //   notExtracted:                                     ; preds = %codeRepl
  //     %0 = phi i32 [ %.reload, %codeRepl ], [ %.reload2, %body2 ]
  //
  // The problem is that the PHI node in notExtracted now has an incoming
  // value from a BasicBlock that's in a different function.

  Function *Func = M->getFunction("foo");
  SmallVector<BasicBlock *, 3> Candidates;
  for (auto &BB : *Func) {
    if (BB.getName() == "body1")
      Candidates.push_back(&BB);
    if (BB.getName() == "body2")
      Candidates.push_back(&BB);
  }
  // CodeExtractor requires the first basic block
  // to dominate all the other ones.
  Candidates.insert(Candidates.begin(), &Func->getEntryBlock());

  DominatorTree DT(*Func);
  CodeExtractor CE(Candidates, &DT);
  EXPECT_TRUE(CE.isEligible());

  Function *Outlined = CE.extractCodeRegion();
  EXPECT_TRUE(Outlined);
  EXPECT_FALSE(verifyFunction(*Outlined));
}
} // end anonymous namespace
