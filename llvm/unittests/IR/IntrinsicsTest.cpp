//===- llvm/unittest/IR/IntrinsicsTest.cpp - ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Intrinsics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

static const char *const NameTable1[] = {
    "llvm.foo", "llvm.foo.a", "llvm.foo.b", "llvm.foo.b.a", "llvm.foo.c",
};

class IntrinsicsTest : public ::testing::Test {
  LLVMContext Context;
  std::unique_ptr<Module> M;
  BasicBlock *BB = nullptr;

  void TearDown() override { M.reset(); }

  void SetUp() override {
    M = std::make_unique<Module>("Test", Context);
    auto F = M->getOrInsertFunction(
        "test", FunctionType::get(Type::getVoidTy(Context), false));
    BB = BasicBlock::Create(Context, "", cast<Function>(F.getCallee()));
    EXPECT_NE(BB, nullptr);
  }

public:
  Instruction *makeIntrinsic(Intrinsic::ID ID) const {
    IRBuilder<> Builder(BB);
    SmallVector<Value *, 4> ProcessedArgs;
    auto *Decl = Intrinsic::getDeclaration(M.get(), ID);
    for (auto *Ty : Decl->getFunctionType()->params()) {
      auto *Val = Constant::getNullValue(Ty);
      ProcessedArgs.push_back(Val);
    }
    return Builder.CreateCall(Decl, ProcessedArgs);
  }
  template <typename T> void checkIsa(const Instruction &I) {
    EXPECT_TRUE(isa<T>(I));
  }
};

TEST(IntrinsicNameLookup, Basic) {
  int I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo");
  EXPECT_EQ(0, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.f64");
  EXPECT_EQ(0, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.b");
  EXPECT_EQ(2, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.b.a");
  EXPECT_EQ(3, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.c");
  EXPECT_EQ(4, I);
  I = Intrinsic::lookupLLVMIntrinsicByName(NameTable1, "llvm.foo.c.f64");
  EXPECT_EQ(4, I);
}

TEST_F(IntrinsicsTest, InstrProfInheritance) {
  auto isInstrProfInstBase = [](const Instruction &I) {
    return isa<InstrProfInstBase>(I);
  };
#define __ISA(TYPE, PARENT)                                                    \
  auto is##TYPE = [&](const Instruction &I) -> bool {                          \
    return isa<TYPE>(I) && is##PARENT(I);                                      \
  }
  __ISA(InstrProfCntrInstBase, InstrProfInstBase);
  __ISA(InstrProfCoverInst, InstrProfCntrInstBase);
  __ISA(InstrProfIncrementInst, InstrProfCntrInstBase);
  __ISA(InstrProfIncrementInstStep, InstrProfIncrementInst);
  __ISA(InstrProfCallsite, InstrProfCntrInstBase);
  __ISA(InstrProfTimestampInst, InstrProfCntrInstBase);
  __ISA(InstrProfValueProfileInst, InstrProfCntrInstBase);
  __ISA(InstrProfMCDCBitmapInstBase, InstrProfInstBase);
  __ISA(InstrProfMCDCBitmapParameters, InstrProfMCDCBitmapInstBase);
  __ISA(InstrProfMCDCTVBitmapUpdate, InstrProfMCDCBitmapInstBase);
#undef __ISA

  std::vector<
      std::pair<Intrinsic::ID, std::function<bool(const Instruction &)>>>
      LeafIDs = {
          {Intrinsic::instrprof_cover, isInstrProfCoverInst},
          {Intrinsic::instrprof_increment, isInstrProfIncrementInst},
          {Intrinsic::instrprof_increment_step, isInstrProfIncrementInstStep},
          {Intrinsic::instrprof_callsite, isInstrProfCallsite},
          {Intrinsic::instrprof_mcdc_parameters,
           isInstrProfMCDCBitmapParameters},
          {Intrinsic::instrprof_mcdc_tvbitmap_update,
           isInstrProfMCDCTVBitmapUpdate},
          {Intrinsic::instrprof_timestamp, isInstrProfTimestampInst},
          {Intrinsic::instrprof_value_profile, isInstrProfValueProfileInst}};
  for (const auto &[ID, Checker] : LeafIDs) {
    auto *Intr = makeIntrinsic(ID);
    EXPECT_TRUE(Checker(*Intr));
  }
}

TEST(IntrinsicVerifierTest, LRound) {
  LLVMContext C;
  std::unique_ptr<Module> M = std::make_unique<Module>("M", C);
  IRBuilder<> Builder(C);

  using TypePair = std::pair<Type *, Type *>;
  Type *Int32Ty = Type::getInt32Ty(C);
  Type *Int64Ty = Type::getInt64Ty(C);
  Type *HalfTy = Type::getHalfTy(C);
  Type *FltTy = Type::getFloatTy(C);
  Type *DblTy = Type::getDoubleTy(C);
  auto Vec2xTy = [&](Type *ElemTy) {
    return VectorType::get(ElemTy, ElementCount::getFixed(2));
  };
  Type *Vec2xInt32Ty = Vec2xTy(Int32Ty);
  Type *Vec2xInt64Ty = Vec2xTy(Int64Ty);
  Type *Vec2xFltTy = Vec2xTy(FltTy);

  // Test Cases
  // Validating only a limited set of possible combinations.
  std::vector<TypePair> ValidTypes = {
      {Int32Ty, FltTy},          {Int32Ty, DblTy},  {Int64Ty, FltTy},
      {Int64Ty, DblTy},          {Int32Ty, HalfTy}, {Vec2xInt32Ty, Vec2xFltTy},
      {Vec2xInt64Ty, Vec2xFltTy}};

  // CreateIntrinsic errors out on invalid argument types.
  std::vector<TypePair> InvalidTypes = {
      {VectorType::get(Int32Ty, ElementCount::getFixed(3)), Vec2xFltTy}};

  auto testIntrinsic = [&](TypePair types, Intrinsic::ID ID, bool expectValid) {
    Function *F =
        Function::Create(FunctionType::get(types.first, {types.second}, false),
                         Function::ExternalLinkage, "lround_fn", M.get());
    BasicBlock *BB = BasicBlock::Create(C, "entry", F);
    Builder.SetInsertPoint(BB);

    Value *Arg = F->arg_begin();
    Value *Result = Builder.CreateIntrinsic(types.first, ID, {Arg});
    Builder.CreateRet(Result);

    std::string Error;
    raw_string_ostream ErrorOS(Error);
    EXPECT_EQ(expectValid, !verifyFunction(*F, &ErrorOS));
    if (!expectValid) {
      EXPECT_TRUE(StringRef(ErrorOS.str())
                      .contains("llvm.lround, llvm.llround: argument must be "
                                "same length as result"));
    }
  };

  // Run Valid Cases.
  for (auto Types : ValidTypes) {
    testIntrinsic(Types, Intrinsic::lround, true);
    testIntrinsic(Types, Intrinsic::llround, true);
  }

  // Run Invalid Cases.
  for (auto Types : InvalidTypes) {
    testIntrinsic(Types, Intrinsic::lround, false);
    testIntrinsic(Types, Intrinsic::llround, false);
  }
}
} // end namespace
