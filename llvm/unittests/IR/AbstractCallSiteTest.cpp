//===----- AbstractCallSiteTest.cpp - AbstractCallSite Unittests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AbstractCallSiteTests", errs());
  return Mod;
}

TEST(AbstractCallSite, CallbackCall) {
  LLVMContext C;

  const char *IR =
      "define void @callback(i8* %X, i32* %A) {\n"
      "  ret void\n"
      "}\n"
      "define void @foo(i32* %A) {\n"
      "  call void (i32, void (i8*, ...)*, ...) @broker(i32 1, void (i8*, ...)* bitcast (void (i8*, i32*)* @callback to void (i8*, ...)*), i32* %A)\n"
      "  ret void\n"
      "}\n"
      "declare !callback !0 void @broker(i32, void (i8*, ...)*, ...)\n"
      "!0 = !{!1}\n"
      "!1 = !{i64 1, i64 -1, i1 true}";

  std::unique_ptr<Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  Function *Callback = M->getFunction("callback");
  ASSERT_NE(Callback, nullptr);

  const Use *CallbackUse = Callback->getSingleUndroppableUse();
  ASSERT_NE(CallbackUse, nullptr);

  AbstractCallSite ACS(CallbackUse);
  EXPECT_TRUE(ACS);
  EXPECT_TRUE(ACS.isCallbackCall());
  EXPECT_TRUE(ACS.isCallee(CallbackUse));
  EXPECT_EQ(ACS.getCalledFunction(), Callback);
}
