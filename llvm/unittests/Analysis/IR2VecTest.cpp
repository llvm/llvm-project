//===- IR2VecTest.cpp - Unit tests for IR2Vec -----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instruction.h"
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
  TestableEmbedder(const Function &F, const Vocabulary &V) : Embedder(F, V) {}
  void computeEmbeddings() const override {}
  void computeEmbeddings(const BasicBlock &BB) const override {}
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

TEST(EmbeddingTest, AddVectorsOutOfPlace) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {0.5, 1.5, -1.0};

  Embedding E3 = E1 + E2;
  EXPECT_THAT(E3, ElementsAre(1.5, 3.5, 2.0));

  // Check that E1 and E2 are unchanged
  EXPECT_THAT(E1, ElementsAre(1.0, 2.0, 3.0));
  EXPECT_THAT(E2, ElementsAre(0.5, 1.5, -1.0));
}

TEST(EmbeddingTest, AddVectors) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {0.5, 1.5, -1.0};

  E1 += E2;
  EXPECT_THAT(E1, ElementsAre(1.5, 3.5, 2.0));

  // Check that E2 is unchanged
  EXPECT_THAT(E2, ElementsAre(0.5, 1.5, -1.0));
}

TEST(EmbeddingTest, SubtractVectorsOutOfPlace) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = {0.5, 1.5, -1.0};

  Embedding E3 = E1 - E2;
  EXPECT_THAT(E3, ElementsAre(0.5, 0.5, 4.0));

  // Check that E1 and E2 are unchanged
  EXPECT_THAT(E1, ElementsAre(1.0, 2.0, 3.0));
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

TEST(EmbeddingTest, ScaleVector) {
  Embedding E1 = {1.0, 2.0, 3.0};
  E1 *= 0.5f;
  EXPECT_THAT(E1, ElementsAre(0.5, 1.0, 1.5));
}

TEST(EmbeddingTest, ScaleVectorOutOfPlace) {
  Embedding E1 = {1.0, 2.0, 3.0};
  Embedding E2 = E1 * 0.5f;
  EXPECT_THAT(E2, ElementsAre(0.5, 1.0, 1.5));

  // Check that E1 is unchanged
  EXPECT_THAT(E1, ElementsAre(1.0, 2.0, 3.0));
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
  EXPECT_FALSE(E1.approximatelyEquals(E3, 1e-6));
  EXPECT_TRUE(E1.approximatelyEquals(E3, 3e-5));

  Embedding E_clearly_within = {1.0000005, 2.0000005, 3.0000005}; // Diff = 5e-7
  EXPECT_TRUE(E1.approximatelyEquals(E_clearly_within));

  Embedding E_clearly_outside = {1.00001, 2.00001, 3.00001}; // Diff = 1e-5
  EXPECT_FALSE(E1.approximatelyEquals(E_clearly_outside, 1e-6));

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

TEST(EmbeddingTest, MismatchedDimensionsAddVectorsOutOfPlace) {
  Embedding E1 = {1.0, 2.0};
  Embedding E2 = {1.0};
  EXPECT_DEATH(E1 + E2, "Vectors must have the same dimension");
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
  Vocabulary V = Vocabulary(Vocabulary::createDummyVocabForTest());

  LLVMContext Ctx;
  Module M("M", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, V);
  EXPECT_NE(Emb, nullptr);
}

TEST(IR2VecTest, CreateInvalidMode) {
  Vocabulary V = Vocabulary(Vocabulary::createDummyVocabForTest());

  LLVMContext Ctx;
  Module M("M", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  // static_cast an invalid int to IR2VecKind
  auto Result = Embedder::create(static_cast<IR2VecKind>(-1), *F, V);
  EXPECT_FALSE(static_cast<bool>(Result));
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

// Fixture for IR2Vec tests requiring IR setup.
class IR2VecTestFixture : public ::testing::Test {
protected:
  Vocabulary V;
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  BasicBlock *BB = nullptr;
  Instruction *AddInst = nullptr;
  Instruction *RetInst = nullptr;

  void SetUp() override {
    V = Vocabulary(Vocabulary::createDummyVocabForTest(2));

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
};

TEST_F(IR2VecTestFixture, GetInstVecMap) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &InstMap = Emb->getInstVecMap();

  EXPECT_EQ(InstMap.size(), 2u);
  EXPECT_TRUE(InstMap.count(AddInst));
  EXPECT_TRUE(InstMap.count(RetInst));

  const auto &AddEmb = InstMap.at(AddInst);
  const auto &RetEmb = InstMap.at(RetInst);
  EXPECT_EQ(AddEmb.size(), 2u);
  EXPECT_EQ(RetEmb.size(), 2u);

  EXPECT_TRUE(AddEmb.approximatelyEquals(Embedding(2, 27.9)));
  EXPECT_TRUE(RetEmb.approximatelyEquals(Embedding(2, 17.0)));
}

TEST_F(IR2VecTestFixture, GetBBVecMap) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &BBMap = Emb->getBBVecMap();

  EXPECT_EQ(BBMap.size(), 1u);
  EXPECT_TRUE(BBMap.count(BB));
  EXPECT_EQ(BBMap.at(BB).size(), 2u);

  // BB vector should be sum of add and ret: {27.9, 27.9} + {17.0, 17.0} =
  // {44.9, 44.9}
  EXPECT_TRUE(BBMap.at(BB).approximatelyEquals(Embedding(2, 44.9)));
}

TEST_F(IR2VecTestFixture, GetBBVector) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &BBVec = Emb->getBBVector(*BB);

  EXPECT_EQ(BBVec.size(), 2u);
  EXPECT_TRUE(BBVec.approximatelyEquals(Embedding(2, 44.9)));
}

TEST_F(IR2VecTestFixture, GetFunctionVector) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &FuncVec = Emb->getFunctionVector();

  EXPECT_EQ(FuncVec.size(), 2u);

  // Function vector should match BB vector (only one BB): {44.9, 44.9}
  EXPECT_TRUE(FuncVec.approximatelyEquals(Embedding(2, 44.9)));
}

static constexpr unsigned MaxOpcodes = Vocabulary::MaxOpcodes;
static constexpr unsigned MaxTypeIDs = Vocabulary::MaxTypeIDs;
static constexpr unsigned MaxOperands = Vocabulary::MaxOperandKinds;

TEST(IR2VecVocabularyTest, DummyVocabTest) {
  for (unsigned Dim = 1; Dim <= 10; ++Dim) {
    auto VocabVec = Vocabulary::createDummyVocabForTest(Dim);

    // All embeddings should have the same dimension
    for (const auto &Emb : VocabVec)
      EXPECT_EQ(Emb.size(), Dim);

    // Should have the correct total number of embeddings
    EXPECT_EQ(VocabVec.size(), MaxOpcodes + MaxTypeIDs + MaxOperands);

    auto ExpectedVocab = VocabVec;

    IR2VecVocabAnalysis VocabAnalysis(std::move(VocabVec));
    LLVMContext TestCtx;
    Module TestMod("TestModuleForVocabAnalysis", TestCtx);
    ModuleAnalysisManager MAM;
    Vocabulary Result = VocabAnalysis.run(TestMod, MAM);
    EXPECT_TRUE(Result.isValid());
    EXPECT_EQ(Result.getDimension(), Dim);
    EXPECT_EQ(Result.size(), MaxOpcodes + MaxTypeIDs + MaxOperands);

    unsigned CurPos = 0;
    for (const auto &Entry : Result)
      EXPECT_TRUE(Entry.approximatelyEquals(ExpectedVocab[CurPos++], 0.01));
  }
}

TEST(IR2VecVocabularyTest, NumericIDMap) {
  // Test getNumericID for opcodes
  EXPECT_EQ(Vocabulary::getNumericID(1u), 0u);
  EXPECT_EQ(Vocabulary::getNumericID(13u), 12u);
  EXPECT_EQ(Vocabulary::getNumericID(MaxOpcodes), MaxOpcodes - 1);

  // Test getNumericID for Type IDs
  EXPECT_EQ(Vocabulary::getNumericID(Type::VoidTyID),
            MaxOpcodes + static_cast<unsigned>(Type::VoidTyID));
  EXPECT_EQ(Vocabulary::getNumericID(Type::HalfTyID),
            MaxOpcodes + static_cast<unsigned>(Type::HalfTyID));
  EXPECT_EQ(Vocabulary::getNumericID(Type::FloatTyID),
            MaxOpcodes + static_cast<unsigned>(Type::FloatTyID));
  EXPECT_EQ(Vocabulary::getNumericID(Type::IntegerTyID),
            MaxOpcodes + static_cast<unsigned>(Type::IntegerTyID));
  EXPECT_EQ(Vocabulary::getNumericID(Type::PointerTyID),
            MaxOpcodes + static_cast<unsigned>(Type::PointerTyID));

  // Test getNumericID for Value operands
  LLVMContext Ctx;
  Module M("TestM", Ctx);
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "testFunc", M);

  // Test Function operand
  EXPECT_EQ(Vocabulary::getNumericID(F),
            MaxOpcodes + MaxTypeIDs + 0u); // Function = 0

  // Test Constant operand
  Constant *C = ConstantInt::get(Type::getInt32Ty(Ctx), 42);
  EXPECT_EQ(Vocabulary::getNumericID(C),
            MaxOpcodes + MaxTypeIDs + 2u); // Constant = 2

  // Test Pointer operand
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  AllocaInst *PtrVal = new AllocaInst(Type::getInt32Ty(Ctx), 0, "ptr", BB);
  EXPECT_EQ(Vocabulary::getNumericID(PtrVal),
            MaxOpcodes + MaxTypeIDs + 1u); // Pointer = 1

  // Test Variable operand (function argument)
  Argument *Arg = F->getArg(0);
  EXPECT_EQ(Vocabulary::getNumericID(Arg),
            MaxOpcodes + MaxTypeIDs + 3u); // Variable = 3
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(IR2VecVocabularyTest, NumericIDMapInvalidInputs) {
  // Test invalid opcode IDs
  EXPECT_DEATH(Vocabulary::getNumericID(0u), "Invalid opcode");
  EXPECT_DEATH(Vocabulary::getNumericID(MaxOpcodes + 1), "Invalid opcode");

  // Test invalid type IDs
  EXPECT_DEATH(Vocabulary::getNumericID(static_cast<Type::TypeID>(MaxTypeIDs)),
               "Invalid type ID");
  EXPECT_DEATH(
      Vocabulary::getNumericID(static_cast<Type::TypeID>(MaxTypeIDs + 10)),
      "Invalid type ID");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

TEST(IR2VecVocabularyTest, StringKeyGeneration) {
  EXPECT_EQ(Vocabulary::getStringKey(0), "Ret");
  EXPECT_EQ(Vocabulary::getStringKey(12), "Add");

  StringRef HalfTypeKey = Vocabulary::getStringKey(MaxOpcodes + 0);
  StringRef FloatTypeKey = Vocabulary::getStringKey(MaxOpcodes + 2);
  StringRef VoidTypeKey = Vocabulary::getStringKey(MaxOpcodes + 7);
  StringRef IntTypeKey = Vocabulary::getStringKey(MaxOpcodes + 12);

  EXPECT_EQ(HalfTypeKey, "FloatTy");
  EXPECT_EQ(FloatTypeKey, "FloatTy");
  EXPECT_EQ(VoidTypeKey, "VoidTy");
  EXPECT_EQ(IntTypeKey, "IntegerTy");

  StringRef FuncArgKey = Vocabulary::getStringKey(MaxOpcodes + MaxTypeIDs + 0);
  StringRef PtrArgKey = Vocabulary::getStringKey(MaxOpcodes + MaxTypeIDs + 1);
  EXPECT_EQ(FuncArgKey, "Function");
  EXPECT_EQ(PtrArgKey, "Pointer");
}

TEST(IR2VecVocabularyTest, VocabularyDimensions) {
  {
    Vocabulary V(Vocabulary::createDummyVocabForTest(1));
    EXPECT_TRUE(V.isValid());
    EXPECT_EQ(V.getDimension(), 1u);
  }

  {
    Vocabulary V(Vocabulary::createDummyVocabForTest(5));
    EXPECT_TRUE(V.isValid());
    EXPECT_EQ(V.getDimension(), 5u);
  }

  {
    Vocabulary V(Vocabulary::createDummyVocabForTest(10));
    EXPECT_TRUE(V.isValid());
    EXPECT_EQ(V.getDimension(), 10u);
  }
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(IR2VecVocabularyTest, InvalidAccess) {
  Vocabulary V(Vocabulary::createDummyVocabForTest(2));

  EXPECT_DEATH(V[0u], "Invalid opcode");

  EXPECT_DEATH(V[100u], "Invalid opcode");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

TEST(IR2VecVocabularyTest, TypeIDStringKeyMapping) {
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::VoidTyID)),
            "VoidTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::IntegerTyID)),
            "IntegerTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::FloatTyID)),
            "FloatTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::PointerTyID)),
            "PointerTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::FunctionTyID)),
            "FunctionTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::StructTyID)),
            "StructTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::ArrayTyID)),
            "ArrayTy");
  EXPECT_EQ(Vocabulary::getStringKey(
                MaxOpcodes + static_cast<unsigned>(Type::FixedVectorTyID)),
            "VectorTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::LabelTyID)),
            "LabelTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::TokenTyID)),
            "TokenTy");
  EXPECT_EQ(Vocabulary::getStringKey(MaxOpcodes +
                                     static_cast<unsigned>(Type::MetadataTyID)),
            "MetadataTy");
}

TEST(IR2VecVocabularyTest, InvalidVocabularyConstruction) {
  std::vector<Embedding> InvalidVocab;
  InvalidVocab.push_back(Embedding(2, 1.0));
  InvalidVocab.push_back(Embedding(2, 2.0));

  Vocabulary V(std::move(InvalidVocab));
  EXPECT_FALSE(V.isValid());

  {
    Vocabulary InvalidResult;
    EXPECT_FALSE(InvalidResult.isValid());
#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
    EXPECT_DEATH(InvalidResult.getDimension(), "IR2Vec Vocabulary is invalid");
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST
  }
}

} // end anonymous namespace
