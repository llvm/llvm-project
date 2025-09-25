//=======- AliasSetTrackerTest.cpp - Unit test for the Alias Set Tracker  -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(AliasSetTracker, AliasUnknownInst) {
  StringRef Assembly = R"(
    @a = common global i32 0, align 4
    @b = common global float 0.000000e+00, align 4

    ; Function Attrs: nounwind ssp uwtable
    define i32 @read_a() #0 {
      %1 = load i32, i32* @a, align 4, !tbaa !3
      ret i32 %1
    }

    ; Function Attrs: nounwind ssp uwtable
    define void @write_b() #0 {
      store float 1.000000e+01, float* @b, align 4, !tbaa !7
      ret void
    }

    ; Function Attrs: nounwind ssp uwtable
    define void @test() #0 {
      %1 = call i32 @read_a(), !tbaa !3
      call void @write_b(), !tbaa !7
      ret void
    }

    !3 = !{!4, !4, i64 0}
    !4 = !{!"int", !5, i64 0}
    !5 = !{!"omnipotent char", !6, i64 0}
    !6 = !{!"Simple C/C++ TBAA"}
    !7 = !{!8, !8, i64 0}
    !8 = !{!"float", !5, i64 0}
  )";

  // Parse the IR. The two calls in @test can not access aliasing elements.
  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  // Initialize the alias result.
  Triple Trip(M->getTargetTriple());
  TargetLibraryInfoImpl TLII(Trip);
  TargetLibraryInfo TLI(TLII);
  AAResults AA(TLI);
  TypeBasedAAResult TBAAR(false);
  AA.addAAResult(TBAAR);

  // Initialize the alias set tracker for the @test function.
  Function *Test = M->getFunction("test");
  ASSERT_NE(Test, nullptr);
  BatchAAResults BAA(AA);
  AliasSetTracker AST(BAA);
  for (auto &BB : *Test)
    AST.add(BB);
  // There should be 2 disjoint alias sets. 1 from each call. 
  ASSERT_EQ((int)AST.getAliasSets().size(), 2);

  // Directly test aliasesUnknownInst.
  // Now every call instruction should only alias one alias set.
  BatchAAResults BatchAA(AA);
  for (auto &Inst : *Test->begin()) {
    bool FoundAS = false;
    for (AliasSet &AS : AST) {
      if (!Inst.mayReadOrWriteMemory())
        continue;
      if (!isModOrRefSet(AS.aliasesUnknownInst(&Inst, BatchAA)))
        continue;
      ASSERT_NE(FoundAS, true);
      FoundAS = true;
    }
  }
}
