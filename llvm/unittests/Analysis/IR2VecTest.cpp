//===- IR2VecTest.cpp - Unit tests for IR2Vec -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Error.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <map>
#include <vector>

using namespace llvm;
using namespace ir2vec;
using namespace ::testing;

namespace {

class TestableEmbedder : public Embedder {
public:
  TestableEmbedder(const Function &F, const Vocab &V) : Embedder(F, V) {}
  void computeEmbeddings() const override {}
  void computeEmbeddings(const BasicBlock &BB) const override {}
  using Embedder::lookupVocab;
  static void addVectors(Embedding &Dst, const Embedding &Src) {
    Embedder::addVectors(Dst, Src);
  }
  static void addScaledVector(Embedding &Dst, const Embedding &Src,
                              float Factor) {
    Embedder::addScaledVector(Dst, Src, Factor);
  }
};

TEST(IR2VecTest, CreateSymbolicEmbedder) {
  Vocab V = {{"foo", {1.0, 2.0}}};

  LLVMContext Ctx;
  Module M("M", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
  EXPECT_TRUE(static_cast<bool>(Result));

  auto *Emb = Result->get();
  EXPECT_NE(Emb, nullptr);
}

TEST(IR2VecTest, CreateInvalidMode) {
  Vocab V = {{"foo", {1.0, 2.0}}};

  LLVMContext Ctx;
  Module M("M", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  // static_cast an invalid int to IR2VecKind
  auto Result = Embedder::create(static_cast<IR2VecKind>(-1), *F, V);
  EXPECT_FALSE(static_cast<bool>(Result));

  std::string ErrMsg;
  llvm::handleAllErrors(
      Result.takeError(),
      [&](const llvm::ErrorInfoBase &EIB) { ErrMsg = EIB.message(); });
  EXPECT_NE(ErrMsg.find("Unknown IR2VecKind"), std::string::npos);
}

TEST(IR2VecTest, AddVectors) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {0.5, 1.5, -1.0};

  TestableEmbedder::addVectors(E1, E2);
  EXPECT_THAT(E1, ElementsAre(1.5, 3.5, 2.0));

  // Check that E2 is unchanged
  EXPECT_THAT(E2, ElementsAre(0.5, 1.5, -1.0));
}

TEST(IR2VecTest, AddScaledVector) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {2.0, 0.5, -1.0};

  TestableEmbedder::addScaledVector(E1, E2, 0.5f);
  EXPECT_THAT(E1, ElementsAre(2.0, 2.25, 2.5));

  // Check that E2 is unchanged
  EXPECT_THAT(E2, ElementsAre(2.0, 0.5, -1.0));
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(IR2VecTest, MismatchedDimensionsAddVectors) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.0};
  EXPECT_DEATH(TestableEmbedder::addVectors(E1, E2),
               "Vectors must have the same dimension");
}

TEST(IR2VecTest, MismatchedDimensionsAddScaledVector) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.0};
  EXPECT_DEATH(TestableEmbedder::addScaledVector(E1, E2, 1.0f),
               "Vectors must have the same dimension");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

TEST(IR2VecTest, LookupVocab) {
  Vocab V = {{"foo", {1.0, 2.0}}, {"bar", {3.0, 4.0}}};
  LLVMContext Ctx;
  Module M("M", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  TestableEmbedder E(*F, V);
  auto V_foo = E.lookupVocab("foo");
  EXPECT_EQ(V_foo.size(), 2u);
  EXPECT_THAT(V_foo, ElementsAre(1.0, 2.0));

  auto V_missing = E.lookupVocab("missing");
  EXPECT_EQ(V_missing.size(), 2u);
  EXPECT_THAT(V_missing, ElementsAre(0.0, 0.0));
}

TEST(IR2VecTest, ZeroDimensionEmbedding) {
  Embedding E1;
  Embedding E2;
  // Should be no-op, but not crash
  TestableEmbedder::addVectors(E1, E2);
  TestableEmbedder::addScaledVector(E1, E2, 1.0f);
  EXPECT_TRUE(E1.empty());
}

TEST(IR2VecTest, IR2VecVocabResultValidity) {
  // Default constructed is invalid
  IR2VecVocabResult invalidResult;
  EXPECT_FALSE(invalidResult.isValid());
#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(invalidResult.getVocabulary(), "IR2Vec Vocabulary is invalid");
  EXPECT_DEATH(invalidResult.getDimension(), "IR2Vec Vocabulary is invalid");
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

  // Valid vocab
  Vocab V = {{"foo", {1.0, 2.0}}, {"bar", {3.0, 4.0}}};
  IR2VecVocabResult validResult(std::move(V));
  EXPECT_TRUE(validResult.isValid());
  EXPECT_EQ(validResult.getDimension(), 2u);
}

// Helper to create a minimal function and embedder for getter tests
struct GetterTestEnv {
  Vocab V = {};
  LLVMContext Ctx;
  std::unique_ptr<Module> M = nullptr;
  Function *F = nullptr;
  BasicBlock *BB = nullptr;
  Instruction *Add = nullptr;
  Instruction *Ret = nullptr;
  std::unique_ptr<Embedder> Emb = nullptr;

  GetterTestEnv() {
    V = {{"add", {1.0, 2.0}},
         {"integerTy", {0.5, 0.5}},
         {"constant", {0.2, 0.3}},
         {"variable", {0.0, 0.0}},
         {"unknownTy", {0.0, 0.0}}};

    M = std::make_unique<Module>("M", Ctx);
    FunctionType *FTy = FunctionType::get(
        Type::getInt32Ty(Ctx), {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)},
        false);
    F = Function::Create(FTy, Function::ExternalLinkage, "f", M.get());
    BB = BasicBlock::Create(Ctx, "entry", F);
    Argument *Arg = F->getArg(0);
    llvm::Value *Const = ConstantInt::get(Type::getInt32Ty(Ctx), 42);

    Add = BinaryOperator::CreateAdd(Arg, Const, "add", BB);
    Ret = ReturnInst::Create(Ctx, Add, BB);

    auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
    EXPECT_TRUE(static_cast<bool>(Result));
    Emb = std::move(*Result);
  }
};

TEST(IR2VecTest, GetInstVecMap) {
  GetterTestEnv Env;
  const auto &InstMap = Env.Emb->getInstVecMap();

  EXPECT_EQ(InstMap.size(), 2u);
  EXPECT_TRUE(InstMap.count(Env.Add));
  EXPECT_TRUE(InstMap.count(Env.Ret));

  EXPECT_EQ(InstMap.at(Env.Add).size(), 2u);
  EXPECT_EQ(InstMap.at(Env.Ret).size(), 2u);

  // Check values for add: {1.29, 2.31}
  EXPECT_THAT(InstMap.at(Env.Add),
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));

  // Check values for ret: {0.0, 0.}; Neither ret nor voidTy are present in
  // vocab
  EXPECT_THAT(InstMap.at(Env.Ret), ElementsAre(0.0, 0.0));
}

TEST(IR2VecTest, GetBBVecMap) {
  GetterTestEnv Env;
  const auto &BBMap = Env.Emb->getBBVecMap();

  EXPECT_EQ(BBMap.size(), 1u);
  EXPECT_TRUE(BBMap.count(Env.BB));
  EXPECT_EQ(BBMap.at(Env.BB).size(), 2u);

  // BB vector should be sum of add and ret: {1.29, 2.31} + {0.0, 0.0} =
  // {1.29, 2.31}
  EXPECT_THAT(BBMap.at(Env.BB),
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));
}

TEST(IR2VecTest, GetBBVector) {
  GetterTestEnv Env;
  const auto &BBVec = Env.Emb->getBBVector(*Env.BB);

  EXPECT_EQ(BBVec.size(), 2u);
  EXPECT_THAT(BBVec,
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));
}

TEST(IR2VecTest, GetFunctionVector) {
  GetterTestEnv Env;
  const auto &FuncVec = Env.Emb->getFunctionVector();

  EXPECT_EQ(FuncVec.size(), 2u);

  // Function vector should match BB vector (only one BB): {1.29, 2.31}
  EXPECT_THAT(FuncVec,
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));
}

} // end anonymous namespace
