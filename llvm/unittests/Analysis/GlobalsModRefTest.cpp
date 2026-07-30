//===--- GlobalsModRefTest.cpp - Mixed TBAA unit tests --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(GlobalsModRef, OptNone) {
  StringRef Assembly = R"(
    define void @f1() optnone {
      ret void
    }
    define void @f2() optnone readnone {
      ret void
    }
    define void @f3() optnone readonly {
      ret void
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  const auto &funcs = M->functions();
  auto I = funcs.begin();
  ASSERT_NE(I, funcs.end());
  const Function &F1 = *I;
  ASSERT_NE(++I, funcs.end());
  const Function &F2 = *I;
  ASSERT_NE(++I, funcs.end());
  const Function &F3 = *I;
  EXPECT_EQ(++I, funcs.end());

  Triple Trip(M->getTargetTriple());
  TargetLibraryInfoImpl TLII(Trip);
  TargetLibraryInfo TLI(TLII);
  auto GetTLI = [&TLI](Function &F) -> TargetLibraryInfo & { return TLI; };
  llvm::CallGraph CG(*M);

  auto AAR = GlobalsAAResult::analyzeModule(*M, GetTLI, CG);

  EXPECT_EQ(MemoryEffects::unknown(), AAR.getMemoryEffects(&F1));
  EXPECT_EQ(MemoryEffects::none(), AAR.getMemoryEffects(&F2));
  EXPECT_EQ(MemoryEffects::readOnly(), AAR.getMemoryEffects(&F3));
}
