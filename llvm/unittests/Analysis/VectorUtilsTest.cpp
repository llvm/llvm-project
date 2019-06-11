//===- VectorUtilsTest.cpp - VectorUtils tests ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/VectorUtils.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/KnownBits.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class VectorUtilsTest : public testing::Test {
protected:
  void parseAssembly(const char *Assembly) {
    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);

    std::string errMsg;
    raw_string_ostream os(errMsg);
    Error.print("", os);

    // A failure here means that the test itself is buggy.
    if (!M)
      report_fatal_error(os.str());

    Function *F = M->getFunction("test");
    if (F == nullptr)
      report_fatal_error("Test must have a function named @test");

    A = nullptr;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (I->hasName()) {
        if (I->getName() == "A")
          A = &*I;
      }
    }
    if (A == nullptr)
      report_fatal_error("@test must have an instruction %A");
  }

  LLVMContext Context;
  std::unique_ptr<Module> M;
  Instruction *A;
};

} // namespace

TEST_F(VectorUtilsTest, getSplatValueElt0) {
  parseAssembly(
      "define <2 x i8> @test(i8 %x) {\n"
      "  %ins = insertelement <2 x i8> undef, i8 %x, i32 0\n"
      "  %A = shufflevector <2 x i8> %ins, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_EQ(getSplatValue(A)->getName(), "x");
}

TEST_F(VectorUtilsTest, getSplatValueEltMismatch) {
  parseAssembly(
      "define <2 x i8> @test(i8 %x) {\n"
      "  %ins = insertelement <2 x i8> undef, i8 %x, i32 1\n"
      "  %A = shufflevector <2 x i8> %ins, <2 x i8> undef, <2 x i32> zeroinitializer\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_EQ(getSplatValue(A), nullptr);
}

// TODO: This is a splat, but we don't recognize it.

TEST_F(VectorUtilsTest, getSplatValueElt1) {
  parseAssembly(
      "define <2 x i8> @test(i8 %x) {\n"
      "  %ins = insertelement <2 x i8> undef, i8 %x, i32 1\n"
      "  %A = shufflevector <2 x i8> %ins, <2 x i8> undef, <2 x i32> <i32 1, i32 1>\n"
      "  ret <2 x i8> %A\n"
      "}\n");
  EXPECT_EQ(getSplatValue(A), nullptr);
}
