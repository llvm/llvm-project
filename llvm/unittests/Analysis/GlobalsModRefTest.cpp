//===--- GlobalsModRefTest.cpp - Mixed TBAA unit tests --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

TEST(GlobalsModRef, OptNone) {
  StringRef Assembly = R"(
    define void @f() optnone {
      ret void
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  auto M = parseAssemblyString(Assembly, Error, Context);
  ASSERT_TRUE(M) << "Bad assembly?";

  const auto &funcs = M->functions();
  ASSERT_NE(funcs.begin(), funcs.end());
  EXPECT_EQ(std::next(funcs.begin()), funcs.end());
  const Function &F = *funcs.begin();

  Triple Trip(M->getTargetTriple());
  TargetLibraryInfoImpl TLII(Trip);
  TargetLibraryInfo TLI(TLII);
  llvm::CallGraph CG(*M);

  auto AAR = GlobalsAAResult::analyzeModule(*M, TLI, CG);
  EXPECT_EQ(FMRB_UnknownModRefBehavior, AAR.getModRefBehavior(&F));
}
