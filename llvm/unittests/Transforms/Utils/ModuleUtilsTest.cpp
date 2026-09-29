//===- ModuleUtilsTest.cpp - Unit tests for Module utility ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, StringRef IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("ModuleUtilsTest", errs());
  return Mod;
}

static int getListSize(Module &M, StringRef Name) {
  auto *List = M.getGlobalVariable(Name);
  if (!List)
    return 0;
  auto *T = cast<ArrayType>(List->getValueType());
  return T->getNumElements();
}

TEST(ModuleUtils, AppendToUsedList1) {
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

TEST(ModuleUtils, AppendToUsedList2) {
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

using AppendFnType = decltype(&appendToGlobalCtors);
using TransformFnType = decltype(&transformGlobalCtors);
using ParamType = std::tuple<StringRef, AppendFnType, TransformFnType>;
class ModuleUtilsTest : public testing::TestWithParam<ParamType> {
public:
  StringRef arrayName() const { return std::get<0>(GetParam()); }
  AppendFnType appendFn() const { return std::get<AppendFnType>(GetParam()); }
  TransformFnType transformFn() const {
    return std::get<TransformFnType>(GetParam());
  }
};

INSTANTIATE_TEST_SUITE_P(
    ModuleUtilsTestCtors, ModuleUtilsTest,
    ::testing::Values(ParamType{"llvm.global_ctors", &appendToGlobalCtors,
                                &transformGlobalCtors},
                      ParamType{"llvm.global_dtors", &appendToGlobalDtors,
                                &transformGlobalDtors}));

TEST_P(ModuleUtilsTest, AppendToMissingArray) {
  LLVMContext C;

  std::unique_ptr<Module> M = parseIR(C, "");

  EXPECT_EQ(0, getListSize(*M, arrayName()));
  Function *F = cast<Function>(
      M->getOrInsertFunction("ctor", Type::getVoidTy(C)).getCallee());
  appendFn()(*M, F, 11, F);
  ASSERT_EQ(1, getListSize(*M, arrayName()));

  ConstantArray *CA = dyn_cast<ConstantArray>(
      M->getGlobalVariable(arrayName())->getInitializer());
  ASSERT_NE(nullptr, CA);
  ConstantStruct *CS = dyn_cast<ConstantStruct>(CA->getOperand(0));
  ASSERT_NE(nullptr, CS);
  ConstantInt *Pri = dyn_cast<ConstantInt>(CS->getOperand(0));
  ASSERT_NE(nullptr, Pri);
  EXPECT_EQ(11u, Pri->getLimitedValue());
  EXPECT_EQ(F, dyn_cast<Function>(CS->getOperand(1)));
  EXPECT_EQ(F, CS->getOperand(2));
}

TEST_P(ModuleUtilsTest, AppendToArray) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, (R"(@)" + arrayName() +
                  R"( = appending global [2 x { i32, ptr, ptr }] [
            { i32, ptr, ptr } { i32 65535, ptr  null, ptr null },
            { i32, ptr, ptr } { i32 0, ptr  null, ptr null }]
      )")
                     .str());

  EXPECT_EQ(2, getListSize(*M, arrayName()));
  appendFn()(
      *M,
      cast<Function>(
          M->getOrInsertFunction("ctor", Type::getVoidTy(C)).getCallee()),
      11, nullptr);
  EXPECT_EQ(3, getListSize(*M, arrayName()));
}

TEST_P(ModuleUtilsTest, UpdateArray) {
  LLVMContext C;

  std::unique_ptr<Module> M =
      parseIR(C, (R"(@)" + arrayName() +
                  R"( = appending global [2 x { i32, ptr, ptr }] [
            { i32, ptr, ptr } { i32 65535, ptr  null, ptr null },
            { i32, ptr, ptr } { i32 0, ptr  null, ptr null }]
      )")
                     .str());

  EXPECT_EQ(2, getListSize(*M, arrayName()));
  transformFn()(*M, [](Constant *C) -> Constant * {
    ConstantStruct *CS = dyn_cast<ConstantStruct>(C);
    if (!CS)
      return nullptr;
    StructType *EltTy = cast<StructType>(C->getType());
    Constant *CSVals[3] = {
        ConstantInt::getSigned(CS->getOperand(0)->getType(), 12),
        CS->getOperand(1),
        CS->getOperand(2),
    };
    return ConstantStruct::get(EltTy,
                               ArrayRef(CSVals, EltTy->getNumElements()));
  });
  EXPECT_EQ(1, getListSize(*M, arrayName()));
  ConstantArray *CA = dyn_cast<ConstantArray>(
      M->getGlobalVariable(arrayName())->getInitializer());
  ASSERT_NE(nullptr, CA);
  ConstantStruct *CS = dyn_cast<ConstantStruct>(CA->getOperand(0));
  ASSERT_NE(nullptr, CS);
  ConstantInt *Pri = dyn_cast<ConstantInt>(CS->getOperand(0));
  ASSERT_NE(nullptr, Pri);
  EXPECT_EQ(12u, Pri->getLimitedValue());
}
