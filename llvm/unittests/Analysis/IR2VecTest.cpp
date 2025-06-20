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
#include "llvm/Support/JSON.h"

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
};

TEST(EmbeddingTest, ConstructorsAndAccessors) {
  // Default constructor
  {
    Embedding E;
    EXPECT_TRUE(E.empty());
    EXPECT_EQ(E.size(), 0u);
  }

  // Constructor with const std::vector<double>&
  {
    std::vector<double> Data = {1.0, 2.0, 3.0};
    Embedding E(Data);
    EXPECT_FALSE(E.empty());
    ASSERT_THAT(E, SizeIs(3u));
    EXPECT_THAT(E.getData(), ElementsAre(1.0, 2.0, 3.0));
    EXPECT_EQ(E[0], 1.0);
    EXPECT_EQ(E[1], 2.0);
    EXPECT_EQ(E[2], 3.0);
  }

  // Constructor with std::vector<double>&&
  {
    Embedding E(std::vector<double>({4.0, 5.0}));
    ASSERT_THAT(E, SizeIs(2u));
    EXPECT_THAT(E.getData(), ElementsAre(4.0, 5.0));
  }

  // Constructor with std::initializer_list<double>
  {
    Embedding E({6.0, 7.0, 8.0, 9.0});
    ASSERT_THAT(E, SizeIs(4u));
    EXPECT_THAT(E.getData(), ElementsAre(6.0, 7.0, 8.0, 9.0));
    EXPECT_EQ(E[0], 6.0);
    E[0] = 6.5;
    EXPECT_EQ(E[0], 6.5);
  }

  // Constructor with size_t
  {
    Embedding E(5);
    ASSERT_THAT(E, SizeIs(5u));
    EXPECT_THAT(E.getData(), ElementsAre(0.0, 0.0, 0.0, 0.0, 0.0));
  }

  // Constructor with size_t and double
  {
    Embedding E(5, 1.5);
    ASSERT_THAT(E, SizeIs(5u));
    EXPECT_THAT(E.getData(), ElementsAre(1.5, 1.5, 1.5, 1.5, 1.5));
  }

  // Test iterators
  {
    Embedding E({6.5, 7.0, 8.0, 9.0});
    std::vector<double> VecE;
    for (double Val : E) {
      VecE.push_back(Val);
    }
    EXPECT_THAT(VecE, ElementsAre(6.5, 7.0, 8.0, 9.0));

    const Embedding CE = E;
    std::vector<double> VecCE;
    for (const double &Val : CE) {
      VecCE.push_back(Val);
    }
    EXPECT_THAT(VecCE, ElementsAre(6.5, 7.0, 8.0, 9.0));

    EXPECT_EQ(*E.begin(), 6.5);
    EXPECT_EQ(*(E.end() - 1), 9.0);
    EXPECT_EQ(*CE.cbegin(), 6.5);
    EXPECT_EQ(*(CE.cend() - 1), 9.0);
  }
}

TEST(EmbeddingTest, AddVectors) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {0.5, 1.5, -1.0};

  E1 += E2;
  EXPECT_THAT(E1, ElementsAre(1.5, 3.5, 2.0));

  // Check that E2 is unchanged
  EXPECT_THAT(E2, ElementsAre(0.5, 1.5, -1.0));
}

TEST(EmbeddingTest, SubtractVectors) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {0.5, 1.5, -1.0};

  E1 -= E2;
  EXPECT_THAT(E1, ElementsAre(0.5, 0.5, 4.0));

  // Check that E2 is unchanged
  EXPECT_THAT(E2, ElementsAre(0.5, 1.5, -1.0));
}

TEST(EmbeddingTest, AddScaledVector) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {2.0, 0.5, -1.0};

  E1.scaleAndAdd(E2, 0.5f);
  EXPECT_THAT(E1, ElementsAre(2.0, 2.25, 2.5));

  // Check that E2 is unchanged
  EXPECT_THAT(E2, ElementsAre(2.0, 0.5, -1.0));
}

TEST(EmbeddingTest, ApproximatelyEqual) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {1.0000001, 2.0000001, 3.0000001};
  EXPECT_TRUE(E1.approximatelyEquals(E2)); // Diff = 1e-7

  Embedding E3 = {1.00002, 2.00002, 3.00002}; // Diff = 2e-5
  EXPECT_FALSE(E1.approximatelyEquals(E3));
  EXPECT_TRUE(E1.approximatelyEquals(E3, 3e-5));

  Embedding E_clearly_within = {1.0000005, 2.0000005, 3.0000005}; // Diff = 5e-7
  EXPECT_TRUE(E1.approximatelyEquals(E_clearly_within));

  Embedding E_clearly_outside = {1.00001, 2.00001, 3.00001}; // Diff = 1e-5
  EXPECT_FALSE(E1.approximatelyEquals(E_clearly_outside));

  Embedding E4 = {1.0, 2.0, 3.5}; // Large diff
  EXPECT_FALSE(E1.approximatelyEquals(E4, 0.01));

  Embedding E5 = {1.0, 2.0, 3.0};
  EXPECT_TRUE(E1.approximatelyEquals(E5, 0.0));
  EXPECT_TRUE(E1.approximatelyEquals(E5));
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(EmbeddingTest, AccessOutOfBounds) {
  Embedding E = {1.0, 2.0, 3.0};
  EXPECT_DEATH(E[3], "Index out of bounds");
  EXPECT_DEATH(E[-1], "Index out of bounds");
  EXPECT_DEATH(E[4] = 4.0, "Index out of bounds");
}

TEST(EmbeddingTest, MismatchedDimensionsAddVectors) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.0};
  EXPECT_DEATH(E1 += E2, "Vectors must have the same dimension");
}

TEST(EmbeddingTest, MismatchedDimensionsSubtractVectors) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.0};
  EXPECT_DEATH(E1 -= E2, "Vectors must have the same dimension");
}

TEST(EmbeddingTest, MismatchedDimensionsAddScaledVector) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.0};
  EXPECT_DEATH(E1.scaleAndAdd(E2, 1.0f),
               "Vectors must have the same dimension");
}

TEST(EmbeddingTest, MismatchedDimensionsApproximatelyEqual) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.010};
  EXPECT_DEATH(E1.approximatelyEquals(E2),
               "Vectors must have the same dimension");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

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
  E1 += E2;
  E1 -= E2;
  E1.scaleAndAdd(E2, 1.0f);
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

// Fixture for IR2Vec tests requiring IR setup and weight management.
class IR2VecTestFixture : public ::testing::Test {
protected:
  Vocab V;
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  BasicBlock *BB = nullptr;
  Instruction *AddInst = nullptr;
  Instruction *RetInst = nullptr;

  float OriginalOpcWeight = ::OpcWeight;
  float OriginalTypeWeight = ::TypeWeight;
  float OriginalArgWeight = ::ArgWeight;

  void SetUp() override {
    V = {{"add", {1.0, 2.0}},
         {"integerTy", {0.5, 0.5}},
         {"constant", {0.2, 0.3}},
         {"variable", {0.0, 0.0}},
         {"unknownTy", {0.0, 0.0}}};

    // Setup IR
    M = std::make_unique<Module>("TestM", Ctx);
    FunctionType *FTy = FunctionType::get(
        Type::getInt32Ty(Ctx), {Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)},
        false);
    F = Function::Create(FTy, Function::ExternalLinkage, "f", M.get());
    BB = BasicBlock::Create(Ctx, "entry", F);
    Argument *Arg = F->getArg(0);
    llvm::Value *Const = ConstantInt::get(Type::getInt32Ty(Ctx), 42);

    AddInst = BinaryOperator::CreateAdd(Arg, Const, "add", BB);
    RetInst = ReturnInst::Create(Ctx, AddInst, BB);
  }

  void setWeights(float OpcWeight, float TypeWeight, float ArgWeight) {
    ::OpcWeight = OpcWeight;
    ::TypeWeight = TypeWeight;
    ::ArgWeight = ArgWeight;
  }

  void TearDown() override {
    // Restore original global weights
    ::OpcWeight = OriginalOpcWeight;
    ::TypeWeight = OriginalTypeWeight;
    ::ArgWeight = OriginalArgWeight;
  }
};

TEST_F(IR2VecTestFixture, GetInstVecMap) {
  auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Result));
  auto Emb = std::move(*Result);

  const auto &InstMap = Emb->getInstVecMap();

  EXPECT_EQ(InstMap.size(), 2u);
  EXPECT_TRUE(InstMap.count(AddInst));
  EXPECT_TRUE(InstMap.count(RetInst));

  EXPECT_EQ(InstMap.at(AddInst).size(), 2u);
  EXPECT_EQ(InstMap.at(RetInst).size(), 2u);

  // Check values for add: {1.29, 2.31}
  EXPECT_THAT(InstMap.at(AddInst),
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));

  // Check values for ret: {0.0, 0.}; Neither ret nor voidTy are present in
  // vocab
  EXPECT_THAT(InstMap.at(RetInst), ElementsAre(0.0, 0.0));
}

TEST_F(IR2VecTestFixture, GetBBVecMap) {
  auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Result));
  auto Emb = std::move(*Result);

  const auto &BBMap = Emb->getBBVecMap();

  EXPECT_EQ(BBMap.size(), 1u);
  EXPECT_TRUE(BBMap.count(BB));
  EXPECT_EQ(BBMap.at(BB).size(), 2u);

  // BB vector should be sum of add and ret: {1.29, 2.31} + {0.0, 0.0} =
  // {1.29, 2.31}
  EXPECT_THAT(BBMap.at(BB),
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));
}

TEST_F(IR2VecTestFixture, GetBBVector) {
  auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Result));
  auto Emb = std::move(*Result);

  const auto &BBVec = Emb->getBBVector(*BB);

  EXPECT_EQ(BBVec.size(), 2u);
  EXPECT_THAT(BBVec,
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));
}

TEST_F(IR2VecTestFixture, GetFunctionVector) {
  auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Result));
  auto Emb = std::move(*Result);

  const auto &FuncVec = Emb->getFunctionVector();

  EXPECT_EQ(FuncVec.size(), 2u);

  // Function vector should match BB vector (only one BB): {1.29, 2.31}
  EXPECT_THAT(FuncVec,
              ElementsAre(DoubleNear(1.29, 1e-6), DoubleNear(2.31, 1e-6)));
}

TEST_F(IR2VecTestFixture, GetFunctionVectorWithCustomWeights) {
  setWeights(1.0, 1.0, 1.0);

  auto Result = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Result));
  auto Emb = std::move(*Result);

  const auto &FuncVec = Emb->getFunctionVector();

  EXPECT_EQ(FuncVec.size(), 2u);

  // Expected: 1*([1.0 2.0] + [0.0 0.0]) + 1*([0.5 0.5] + [0.0 0.0]) + 1*([0.2
  // 0.3] + [0.0 0.0])
  EXPECT_THAT(FuncVec,
              ElementsAre(DoubleNear(1.7, 1e-6), DoubleNear(2.8, 1e-6)));
}

TEST(IR2VecTest, IR2VecVocabAnalysisWithPrepopulatedVocab) {
  Vocab InitialVocab = {{"key1", {1.1, 2.2}}, {"key2", {3.3, 4.4}}};
  Vocab ExpectedVocab = InitialVocab;
  unsigned ExpectedDim = InitialVocab.begin()->second.size();

  IR2VecVocabAnalysis VocabAnalysis(std::move(InitialVocab));

  LLVMContext TestCtx;
  Module TestMod("TestModuleForVocabAnalysis", TestCtx);
  ModuleAnalysisManager MAM;
  IR2VecVocabResult Result = VocabAnalysis.run(TestMod, MAM);

  EXPECT_TRUE(Result.isValid());
  ASSERT_FALSE(Result.getVocabulary().empty());
  EXPECT_EQ(Result.getDimension(), ExpectedDim);

  const auto &ResultVocab = Result.getVocabulary();
  EXPECT_EQ(ResultVocab.size(), ExpectedVocab.size());
  for (const auto &pair : ExpectedVocab) {
    EXPECT_TRUE(ResultVocab.count(pair.first));
    EXPECT_THAT(ResultVocab.at(pair.first), ElementsAreArray(pair.second));
  }
}

} // end anonymous namespace
