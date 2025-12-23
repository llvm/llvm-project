//===----- AbstractCallSiteTest.cpp - AbstractCallSite Unittests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/AbstractCallSite.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Argument.h"
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
  EXPECT_EQ(ACS.getCalleeUseForCallback(), *CallbackUse);
  EXPECT_EQ(ACS.getCalledFunction(), Callback);

  // The callback metadata {CallbackNo, Arg0No, ..., isVarArg} = {1, -1, true}
  EXPECT_EQ(ACS.getCallArgOperandNoForCallee(), 1);
  // Though the callback metadata only specifies ONE unfixed argument No, the
  // callback callee is vararg, hence the third arg is also considered as
  // another arg for the callback.
  EXPECT_EQ(ACS.getNumArgOperands(), 2u);
  Argument *Param0 = Callback->getArg(0), *Param1 = Callback->getArg(1);
  ASSERT_TRUE(Param0 && Param1);
  EXPECT_EQ(ACS.getCallArgOperandNo(*Param0), -1);
  EXPECT_EQ(ACS.getCallArgOperandNo(*Param1), 2);
}

TEST(AbstractCallSite, DirectCall) {
  LLVMContext C;

  const char *IR = "declare void @bar(i32 %x, i32 %y)\n"
                   "define void @foo() {\n"
                   "  call void @bar(i32 1, i32 2)\n"
                   "  ret void\n"
                   "}\n";

  std::unique_ptr<Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  Function *Callee = M->getFunction("bar");
  ASSERT_NE(Callee, nullptr);

  const Use *DirectCallUse = Callee->getSingleUndroppableUse();
  ASSERT_NE(DirectCallUse, nullptr);

  AbstractCallSite ACS(DirectCallUse);
  EXPECT_TRUE(ACS);
  EXPECT_TRUE(ACS.isDirectCall());
  EXPECT_TRUE(ACS.isCallee(DirectCallUse));
  EXPECT_EQ(ACS.getCalledFunction(), Callee);
  EXPECT_EQ(ACS.getNumArgOperands(), 2u);
  Argument *ArgX = Callee->getArg(0);
  ASSERT_NE(ArgX, nullptr);
  Value *CAO1 = ACS.getCallArgOperand(*ArgX);
  Value *CAO2 = ACS.getCallArgOperand(0);
  ASSERT_NE(CAO2, nullptr);
  // The two call arg operands should be the same object, since they are both
  // the first argument of the call.
  EXPECT_EQ(CAO2, CAO1);

  ConstantInt *FirstArgInt = dyn_cast<ConstantInt>(CAO2);
  ASSERT_NE(FirstArgInt, nullptr);
  EXPECT_EQ(FirstArgInt->getZExtValue(), 1ull);

  EXPECT_EQ(ACS.getCallArgOperandNo(*ArgX), 0);
  EXPECT_EQ(ACS.getCallArgOperandNo(0), 0);
  EXPECT_EQ(ACS.getCallArgOperandNo(1), 1);
}

TEST(AbstractCallSite, IndirectCall) {
  LLVMContext C;

  const char *IR = "define void @foo(ptr %0) {\n"
                   "  call void %0(i32 1, i32 2)\n"
                   "  ret void\n"
                   "}\n";

  std::unique_ptr<Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  Function *Fun = M->getFunction("foo");
  ASSERT_NE(Fun, nullptr);

  Argument *ArgAsCallee = Fun->getArg(0);
  ASSERT_NE(ArgAsCallee, nullptr);

  const Use *IndCallUse = ArgAsCallee->getSingleUndroppableUse();
  ASSERT_NE(IndCallUse, nullptr);

  AbstractCallSite ACS(IndCallUse);
  EXPECT_TRUE(ACS);
  EXPECT_TRUE(ACS.isIndirectCall());
  EXPECT_TRUE(ACS.isCallee(IndCallUse));
  EXPECT_EQ(ACS.getCalledFunction(), nullptr);
  EXPECT_EQ(ACS.getCalledOperand(), ArgAsCallee);
  EXPECT_EQ(ACS.getNumArgOperands(), 2u);
  Value *CalledOperand = ACS.getCallArgOperand(0);
  ASSERT_NE(CalledOperand, nullptr);
  ConstantInt *FirstArgInt = dyn_cast<ConstantInt>(CalledOperand);
  ASSERT_NE(FirstArgInt, nullptr);
  EXPECT_EQ(FirstArgInt->getZExtValue(), 1ull);

  EXPECT_EQ(ACS.getCallArgOperandNo(0), 0);
  EXPECT_EQ(ACS.getCallArgOperandNo(1), 1);
}
