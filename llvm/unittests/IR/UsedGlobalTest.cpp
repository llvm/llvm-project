//===- UsedGlobalTest.cpp - Unit tests for Module utility ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, StringRef IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("UsedGlobalTest", errs());
  return Mod;
}

static int getListSize(Module &M, StringRef Name) {
  auto *List = M.getGlobalVariable(Name);
  if (!List)
    return 0;
  auto *T = cast<ArrayType>(List->getValueType());
  return T->getNumElements();
}

TEST(UsedGlobal, AppendToUsedList1) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(
      C, R"(@x = addrspace(4) global [2 x i32] zeroinitializer, align 4)");
  SmallVector<GlobalValue *, 2> Globals;
  for (auto &G : M->globals()) {
    Globals.push_back(&G);
  }
  EXPECT_EQ(0, getListSize(*M, "llvm.compiler.used"));
  appendToCompilerUsed(*M, Globals);
  EXPECT_EQ(1, getListSize(*M, "llvm.compiler.used"));

  EXPECT_EQ(0, getListSize(*M, "llvm.used"));
  appendToUsed(*M, Globals);
  EXPECT_EQ(1, getListSize(*M, "llvm.used"));
}

TEST(UsedGlobal, AppendToUsedList2) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, R"(@x = global [2 x i32] zeroinitializer, align 4)");
  SmallVector<GlobalValue *, 2> Globals;
  for (auto &G : M->globals()) {
    Globals.push_back(&G);
  }
  EXPECT_EQ(0, getListSize(*M, "llvm.compiler.used"));
  appendToCompilerUsed(*M, Globals);
  EXPECT_EQ(1, getListSize(*M, "llvm.compiler.used"));

  EXPECT_EQ(0, getListSize(*M, "llvm.used"));
  appendToUsed(*M, Globals);
  EXPECT_EQ(1, getListSize(*M, "llvm.used"));
}

TEST(UsedGlobal, AppendToUsedList3) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(C, R"(
          @x = addrspace(1) global [2 x i32] zeroinitializer, align 4
          @y = addrspace(2) global [2 x i32] zeroinitializer, align 4
          @llvm.compiler.used = appending global [1 x ptr addrspace(3)] [ptr addrspace(3) addrspacecast (ptr addrspace(1) @x to ptr addrspace(3))]
      )");
  GlobalVariable *X = M->getNamedGlobal("x");
  GlobalVariable *Y = M->getNamedGlobal("y");
  EXPECT_EQ(1, getListSize(*M, "llvm.compiler.used"));
  appendToCompilerUsed(*M, X);
  EXPECT_EQ(1, getListSize(*M, "llvm.compiler.used"));
  appendToCompilerUsed(*M, Y);
  EXPECT_EQ(2, getListSize(*M, "llvm.compiler.used"));
}
