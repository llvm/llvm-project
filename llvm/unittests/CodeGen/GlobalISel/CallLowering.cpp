//===- CallLowering.cpp - CallLowering unit tests -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {
struct TargetCallLoweringTest : public CallLowering, testing::Test {
  LLVMContext C;

public:
  TargetCallLoweringTest() : CallLowering(nullptr) {}

  std::unique_ptr<Module> parseIR(const char *IR) {
    SMDiagnostic Err;
    std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
    if (!Mod)
      Err.print("TestTargetCallLoweringTest", errs());
    return Mod;
  }
};
} // namespace

TEST_F(TargetCallLoweringTest, ArgByRef) {
  std::unique_ptr<Module> M = parseIR(R"(
define void @foo(ptr %p0, ptr byref(i32) align(4) %p1) {
  ret void
}
)");
  const DataLayout &DL = M->getDataLayout();
  Function *F = M->getFunction("foo");
  // Dummy vregs.
  SmallVector<Register, 4> VRegs(1, 1);

  CallLowering::ArgInfo Arg0(VRegs, F->getArg(0)->getType(), 0);
  setArgFlags(Arg0, AttributeList::FirstArgIndex + 0, DL, *F);
  EXPECT_TRUE(Arg0.Flags[0].isPointer());
  EXPECT_FALSE(Arg0.Flags[0].isByRef());

  CallLowering::ArgInfo Arg1(VRegs, F->getArg(1)->getType(), 1);
  setArgFlags(Arg1, AttributeList::FirstArgIndex + 1, DL, *F);
  EXPECT_TRUE(Arg1.Flags[0].isPointer());
  EXPECT_TRUE(Arg1.Flags[0].isByRef());
  EXPECT_EQ(Arg1.Flags[0].getByRefSize(), 4U);
}
