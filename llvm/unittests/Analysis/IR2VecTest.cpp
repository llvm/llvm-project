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

TEST(IR2VecTest, CreateFlowAwareEmbedder) {
  Vocabulary V = Vocabulary(Vocabulary::createDummyVocabForTest());

  LLVMContext Ctx;
  Module M("M", Ctx);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  auto Emb = Embedder::create(IR2VecKind::FlowAware, *F, V);
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
  std::unique_ptr<Vocabulary> V;
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F = nullptr;
  BasicBlock *BB = nullptr;
  Instruction *AddInst = nullptr;
  Instruction *RetInst = nullptr;

  void SetUp() override {
    V = std::make_unique<Vocabulary>(Vocabulary::createDummyVocabForTest(2));

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

TEST_F(IR2VecTestFixture, GetInstVecMap_Symbolic) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &InstMap = Emb->getInstVecMap();

  EXPECT_EQ(InstMap.size(), 2u);
  EXPECT_TRUE(InstMap.count(AddInst));
  EXPECT_TRUE(InstMap.count(RetInst));

  const auto &AddEmb = InstMap.at(AddInst);
  const auto &RetEmb = InstMap.at(RetInst);
  EXPECT_EQ(AddEmb.size(), 2u);
  EXPECT_EQ(RetEmb.size(), 2u);

  EXPECT_TRUE(AddEmb.approximatelyEquals(Embedding(2, 25.5)));
  EXPECT_TRUE(RetEmb.approximatelyEquals(Embedding(2, 15.5)));
}

TEST_F(IR2VecTestFixture, GetInstVecMap_FlowAware) {
  auto Emb = Embedder::create(IR2VecKind::FlowAware, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &InstMap = Emb->getInstVecMap();

  EXPECT_EQ(InstMap.size(), 2u);
  EXPECT_TRUE(InstMap.count(AddInst));
  EXPECT_TRUE(InstMap.count(RetInst));

  EXPECT_EQ(InstMap.at(AddInst).size(), 2u);
  EXPECT_EQ(InstMap.at(RetInst).size(), 2u);

  EXPECT_TRUE(InstMap.at(AddInst).approximatelyEquals(Embedding(2, 25.5)));
  EXPECT_TRUE(InstMap.at(RetInst).approximatelyEquals(Embedding(2, 32.6)));
}

TEST_F(IR2VecTestFixture, GetBBVecMap_Symbolic) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &BBMap = Emb->getBBVecMap();

  EXPECT_EQ(BBMap.size(), 1u);
  EXPECT_TRUE(BBMap.count(BB));
  EXPECT_EQ(BBMap.at(BB).size(), 2u);

  // BB vector should be sum of add and ret: {25.5, 25.5} + {15.5, 15.5} =
  // {41.0, 41.0}
  EXPECT_TRUE(BBMap.at(BB).approximatelyEquals(Embedding(2, 41.0)));
}

TEST_F(IR2VecTestFixture, GetBBVecMap_FlowAware) {
  auto Emb = Embedder::create(IR2VecKind::FlowAware, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &BBMap = Emb->getBBVecMap();

  EXPECT_EQ(BBMap.size(), 1u);
  EXPECT_TRUE(BBMap.count(BB));
  EXPECT_EQ(BBMap.at(BB).size(), 2u);

  // BB vector should be sum of add and ret: {25.5, 25.5} + {32.6, 32.6} =
  // {58.1, 58.1}
  EXPECT_TRUE(BBMap.at(BB).approximatelyEquals(Embedding(2, 58.1)));
}

TEST_F(IR2VecTestFixture, GetBBVector_Symbolic) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &BBVec = Emb->getBBVector(*BB);

  EXPECT_EQ(BBVec.size(), 2u);
  EXPECT_TRUE(BBVec.approximatelyEquals(Embedding(2, 41.0)));
}

TEST_F(IR2VecTestFixture, GetBBVector_FlowAware) {
  auto Emb = Embedder::create(IR2VecKind::FlowAware, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &BBVec = Emb->getBBVector(*BB);

  EXPECT_EQ(BBVec.size(), 2u);
  EXPECT_TRUE(BBVec.approximatelyEquals(Embedding(2, 58.1)));
}

TEST_F(IR2VecTestFixture, GetFunctionVector_Symbolic) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &FuncVec = Emb->getFunctionVector();

  EXPECT_EQ(FuncVec.size(), 2u);

  // Function vector should match BB vector (only one BB): {41.0, 41.0}
  EXPECT_TRUE(FuncVec.approximatelyEquals(Embedding(2, 41.0)));
}

TEST_F(IR2VecTestFixture, GetFunctionVector_FlowAware) {
  auto Emb = Embedder::create(IR2VecKind::FlowAware, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  const auto &FuncVec = Emb->getFunctionVector();

  EXPECT_EQ(FuncVec.size(), 2u);
  // Function vector should match BB vector (only one BB): {58.1, 58.1}
  EXPECT_TRUE(FuncVec.approximatelyEquals(Embedding(2, 58.1)));
}

TEST_F(IR2VecTestFixture, MultipleComputeEmbeddingsConsistency_Symbolic) {
  auto Emb = Embedder::create(IR2VecKind::Symbolic, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  // Get initial function vector
  const auto &FuncVec1 = Emb->getFunctionVector();

  // Compute embeddings again by calling getFunctionVector multiple times
  const auto &FuncVec2 = Emb->getFunctionVector();
  const auto &FuncVec3 = Emb->getFunctionVector();

  // All function vectors should be identical
  EXPECT_TRUE(FuncVec1.approximatelyEquals(FuncVec2));
  EXPECT_TRUE(FuncVec1.approximatelyEquals(FuncVec3));
  EXPECT_TRUE(FuncVec2.approximatelyEquals(FuncVec3));

  // Also check that instruction vectors remain consistent
  const auto &InstMap1 = Emb->getInstVecMap();
  const auto &InstMap2 = Emb->getInstVecMap();

  EXPECT_EQ(InstMap1.size(), InstMap2.size());
  for (const auto &[Inst, Vec1] : InstMap1) {
    ASSERT_TRUE(InstMap2.count(Inst));
    EXPECT_TRUE(Vec1.approximatelyEquals(InstMap2.at(Inst)));
  }
}

TEST_F(IR2VecTestFixture, MultipleComputeEmbeddingsConsistency_FlowAware) {
  auto Emb = Embedder::create(IR2VecKind::FlowAware, *F, *V);
  ASSERT_TRUE(static_cast<bool>(Emb));

  // Get initial function vector
  const auto &FuncVec1 = Emb->getFunctionVector();

  // Compute embeddings again by calling getFunctionVector multiple times
  const auto &FuncVec2 = Emb->getFunctionVector();
  const auto &FuncVec3 = Emb->getFunctionVector();

  // All function vectors should be identical
  EXPECT_TRUE(FuncVec1.approximatelyEquals(FuncVec2));
  EXPECT_TRUE(FuncVec1.approximatelyEquals(FuncVec3));
  EXPECT_TRUE(FuncVec2.approximatelyEquals(FuncVec3));

  // Also check that instruction vectors remain consistent
  const auto &InstMap1 = Emb->getInstVecMap();
  const auto &InstMap2 = Emb->getInstVecMap();

  EXPECT_EQ(InstMap1.size(), InstMap2.size());
  for (const auto &[Inst, Vec1] : InstMap1) {
    ASSERT_TRUE(InstMap2.count(Inst));
    EXPECT_TRUE(Vec1.approximatelyEquals(InstMap2.at(Inst)));
  }
}

static constexpr unsigned MaxOpcodes = Vocabulary::MaxOpcodes;
[[maybe_unused]]
static constexpr unsigned MaxTypeIDs = Vocabulary::MaxTypeIDs;
static constexpr unsigned MaxCanonicalTypeIDs = Vocabulary::MaxCanonicalTypeIDs;
static constexpr unsigned MaxOperands = Vocabulary::MaxOperandKinds;
static constexpr unsigned MaxPredicateKinds = Vocabulary::MaxPredicateKinds;

// Mapping between LLVM Type::TypeID tokens and Vocabulary::CanonicalTypeID
// names and their canonical string keys.
#define IR2VEC_HANDLE_TYPE_BIMAP(X)                                            \
  X(VoidTyID, VoidTy, "VoidTy")                                                \
  X(IntegerTyID, IntegerTy, "IntegerTy")                                       \
  X(FloatTyID, FloatTy, "FloatTy")                                             \
  X(PointerTyID, PointerTy, "PointerTy")                                       \
  X(FunctionTyID, FunctionTy, "FunctionTy")                                    \
  X(StructTyID, StructTy, "StructTy")                                          \
  X(ArrayTyID, ArrayTy, "ArrayTy")                                             \
  X(FixedVectorTyID, VectorTy, "VectorTy")                                     \
  X(LabelTyID, LabelTy, "LabelTy")                                             \
  X(TokenTyID, TokenTy, "TokenTy")                                             \
  X(MetadataTyID, MetadataTy, "MetadataTy")

TEST(IR2VecVocabularyTest, DummyVocabTest) {
  for (unsigned Dim = 1; Dim <= 10; ++Dim) {
    auto VocabVec = Vocabulary::createDummyVocabForTest(Dim);
    auto VocabVecSize = VocabVec.size();
    // All embeddings should have the same dimension
    for (const auto &Emb : VocabVec)
      EXPECT_EQ(Emb.size(), Dim);

    // Should have the correct total number of embeddings
    EXPECT_EQ(VocabVecSize, MaxOpcodes + MaxCanonicalTypeIDs + MaxOperands +
                                MaxPredicateKinds);

    // Collect embeddings for later comparison before moving VocabVec
    std::vector<Embedding> ExpectedVocab;
    for (const auto &Emb : VocabVec)
      ExpectedVocab.push_back(Emb);

    IR2VecVocabAnalysis VocabAnalysis(std::move(VocabVec));
    LLVMContext TestCtx;
    Module TestMod("TestModuleForVocabAnalysis", TestCtx);
    ModuleAnalysisManager MAM;
    Vocabulary Result = VocabAnalysis.run(TestMod, MAM);
    EXPECT_TRUE(Result.isValid());
    EXPECT_EQ(Result.getDimension(), Dim);
    EXPECT_EQ(Result.getCanonicalSize(), VocabVecSize);

    unsigned CurPos = 0;
    for (const auto &Entry : Result)
      EXPECT_TRUE(Entry.approximatelyEquals(ExpectedVocab[CurPos++], 0.01));
  }
}

TEST(IR2VecVocabularyTest, SlotIdxMapping) {
  // Test getIndex for Opcodes
#define EXPECT_OPCODE_SLOT(NUM, OPCODE, CLASS)                                 \
  EXPECT_EQ(Vocabulary::getIndex(NUM), static_cast<unsigned>(NUM - 1));
#define HANDLE_INST(NUM, OPCODE, CLASS) EXPECT_OPCODE_SLOT(NUM, OPCODE, CLASS)
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
#undef EXPECT_OPCODE_SLOT

  // Test getIndex for Types
#define EXPECT_TYPE_SLOT(TypeIDTok, CanonEnum, CanonStr)                       \
  EXPECT_EQ(Vocabulary::getIndex(Type::TypeIDTok),                             \
            MaxOpcodes + static_cast<unsigned>(                                \
                             Vocabulary::CanonicalTypeID::CanonEnum));

  IR2VEC_HANDLE_TYPE_BIMAP(EXPECT_TYPE_SLOT)

#undef EXPECT_TYPE_SLOT

  // Test getIndex for Value operands
  LLVMContext Ctx;
  Module M("TestM", Ctx);
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Ctx), {Type::getInt32Ty(Ctx)}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "testFunc", M);

#define EXPECTED_VOCAB_OPERAND_SLOT(X)                                         \
  MaxOpcodes + MaxCanonicalTypeIDs + static_cast<unsigned>(X)
  // Test Function operand
  EXPECT_EQ(Vocabulary::getIndex(*F),
            EXPECTED_VOCAB_OPERAND_SLOT(Vocabulary::OperandKind::FunctionID));

  // Test Constant operand
  Constant *C = ConstantInt::get(Type::getInt32Ty(Ctx), 42);
  EXPECT_EQ(Vocabulary::getIndex(*C),
            EXPECTED_VOCAB_OPERAND_SLOT(Vocabulary::OperandKind::ConstantID));

  // Test Pointer operand
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  AllocaInst *PtrVal = new AllocaInst(Type::getInt32Ty(Ctx), 0, "ptr", BB);
  EXPECT_EQ(Vocabulary::getIndex(*PtrVal),
            EXPECTED_VOCAB_OPERAND_SLOT(Vocabulary::OperandKind::PointerID));

  // Test Variable operand (function argument)
  Argument *Arg = F->getArg(0);
  EXPECT_EQ(Vocabulary::getIndex(*Arg),
            EXPECTED_VOCAB_OPERAND_SLOT(Vocabulary::OperandKind::VariableID));
#undef EXPECTED_VOCAB_OPERAND_SLOT

  // Test getIndex for predicates
#define EXPECTED_VOCAB_PREDICATE_SLOT(X)                                       \
  MaxOpcodes + MaxCanonicalTypeIDs + MaxOperands + static_cast<unsigned>(X)
  for (unsigned P = CmpInst::FIRST_FCMP_PREDICATE;
       P <= CmpInst::LAST_FCMP_PREDICATE; ++P) {
    CmpInst::Predicate Pred = static_cast<CmpInst::Predicate>(P);
    unsigned ExpectedIdx =
        EXPECTED_VOCAB_PREDICATE_SLOT((P - CmpInst::FIRST_FCMP_PREDICATE));
    EXPECT_EQ(Vocabulary::getIndex(Pred), ExpectedIdx);
  }
  auto ICMP_Start = CmpInst::LAST_FCMP_PREDICATE + 1;
  for (unsigned P = CmpInst::FIRST_ICMP_PREDICATE;
       P <= CmpInst::LAST_ICMP_PREDICATE; ++P) {
    CmpInst::Predicate Pred = static_cast<CmpInst::Predicate>(P);
    unsigned ExpectedIdx = EXPECTED_VOCAB_PREDICATE_SLOT(
        ICMP_Start + P - CmpInst::FIRST_ICMP_PREDICATE);
    EXPECT_EQ(Vocabulary::getIndex(Pred), ExpectedIdx);
  }
#undef EXPECTED_VOCAB_PREDICATE_SLOT
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(IR2VecVocabularyTest, NumericIDMapInvalidInputs) {
  // Test invalid opcode IDs
  EXPECT_DEATH(Vocabulary::getIndex(0u), "Invalid opcode");
  EXPECT_DEATH(Vocabulary::getIndex(MaxOpcodes + 1), "Invalid opcode");

  // Test invalid type IDs
  EXPECT_DEATH(Vocabulary::getIndex(static_cast<Type::TypeID>(MaxTypeIDs)),
               "Invalid type ID");
  EXPECT_DEATH(Vocabulary::getIndex(static_cast<Type::TypeID>(MaxTypeIDs + 10)),
               "Invalid type ID");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

TEST(IR2VecVocabularyTest, StringKeyGeneration) {
  EXPECT_EQ(Vocabulary::getStringKey(0), "Ret");
  EXPECT_EQ(Vocabulary::getStringKey(12), "Add");

#define EXPECT_OPCODE(NUM, OPCODE, CLASS)                                      \
  EXPECT_EQ(Vocabulary::getStringKey(Vocabulary::getIndex(NUM)),               \
            Vocabulary::getVocabKeyForOpcode(NUM));
#define HANDLE_INST(NUM, OPCODE, CLASS) EXPECT_OPCODE(NUM, OPCODE, CLASS)
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
#undef EXPECT_OPCODE

  // Verify CanonicalTypeID -> string mapping
#define EXPECT_CANONICAL_TYPE_NAME(TypeIDTok, CanonEnum, CanonStr)             \
  EXPECT_EQ(Vocabulary::getStringKey(                                          \
                MaxOpcodes + static_cast<unsigned>(                            \
                                 Vocabulary::CanonicalTypeID::CanonEnum)),     \
            CanonStr);

  IR2VEC_HANDLE_TYPE_BIMAP(EXPECT_CANONICAL_TYPE_NAME)

#undef EXPECT_CANONICAL_TYPE_NAME

  // Verify OperandKind -> string mapping
#define HANDLE_OPERAND_KINDS(X)                                                \
  X(FunctionID, "Function")                                                    \
  X(PointerID, "Pointer")                                                      \
  X(ConstantID, "Constant")                                                    \
  X(VariableID, "Variable")

#define EXPECT_OPERAND_KIND(EnumName, Str)                                     \
  EXPECT_EQ(Vocabulary::getStringKey(                                          \
                MaxOpcodes + MaxCanonicalTypeIDs +                             \
                static_cast<unsigned>(Vocabulary::OperandKind::EnumName)),     \
            Str);

  HANDLE_OPERAND_KINDS(EXPECT_OPERAND_KIND)

#undef EXPECT_OPERAND_KIND
#undef HANDLE_OPERAND_KINDS

  StringRef FuncArgKey =
      Vocabulary::getStringKey(MaxOpcodes + MaxCanonicalTypeIDs + 0);
  StringRef PtrArgKey =
      Vocabulary::getStringKey(MaxOpcodes + MaxCanonicalTypeIDs + 1);
  EXPECT_EQ(FuncArgKey, "Function");
  EXPECT_EQ(PtrArgKey, "Pointer");

// Verify PredicateKind -> string mapping
#define EXPECT_PREDICATE_KIND(PredNum, PredPos, PredKind)                      \
  do {                                                                         \
    std::string PredStr =                                                      \
        std::string(PredKind) + "_" +                                          \
        CmpInst::getPredicateName(static_cast<CmpInst::Predicate>(PredNum))    \
            .str();                                                            \
    unsigned Pos = MaxOpcodes + MaxCanonicalTypeIDs + MaxOperands + PredPos;   \
    EXPECT_EQ(Vocabulary::getStringKey(Pos), PredStr);                         \
  } while (0)

  for (unsigned P = CmpInst::FIRST_FCMP_PREDICATE;
       P <= CmpInst::LAST_FCMP_PREDICATE; ++P)
    EXPECT_PREDICATE_KIND(P, P - CmpInst::FIRST_FCMP_PREDICATE, "FCMP");

  auto ICMP_Pos = CmpInst::LAST_FCMP_PREDICATE + 1;
  for (unsigned P = CmpInst::FIRST_ICMP_PREDICATE;
       P <= CmpInst::LAST_ICMP_PREDICATE; ++P)
    EXPECT_PREDICATE_KIND(P, ICMP_Pos++, "ICMP");

#undef EXPECT_PREDICATE_KIND
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
  Vocabulary V = Vocabulary(Vocabulary::createDummyVocabForTest());
#define EXPECT_TYPE_TO_CANONICAL(TypeIDTok, CanonEnum, CanonStr)               \
  do {                                                                         \
    unsigned FlatIdx = V.getIndex(Type::TypeIDTok);                            \
    EXPECT_EQ(Vocabulary::getStringKey(FlatIdx), CanonStr);                    \
  } while (0);

  IR2VEC_HANDLE_TYPE_BIMAP(EXPECT_TYPE_TO_CANONICAL)

#undef EXPECT_TYPE_TO_CANONICAL
}

TEST(IR2VecVocabularyTest, InvalidVocabularyConstruction) {
  // Test 1: Create invalid VocabStorage with insufficient sections
  std::vector<std::vector<Embedding>> InvalidSectionData;
  // Only add one section with 2 embeddings, but the vocabulary needs 4 sections
  std::vector<Embedding> Section1;
  Section1.push_back(Embedding(2, 1.0));
  Section1.push_back(Embedding(2, 2.0));
  InvalidSectionData.push_back(std::move(Section1));

  VocabStorage InvalidStorage(std::move(InvalidSectionData));
  Vocabulary V(std::move(InvalidStorage));
  EXPECT_FALSE(V.isValid());

  {
    // Test 2: Default-constructed vocabulary should be invalid
    Vocabulary InvalidResult;
    EXPECT_FALSE(InvalidResult.isValid());
#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
    EXPECT_DEATH(InvalidResult.getDimension(), "IR2Vec Vocabulary is invalid");
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST
  }
}

TEST(VocabStorageTest, DefaultConstructor) {
  VocabStorage storage;

  EXPECT_EQ(storage.size(), 0u);
  EXPECT_EQ(storage.getNumSections(), 0u);
  EXPECT_EQ(storage.getDimension(), 0u);
  EXPECT_FALSE(storage.isValid());

  // Test iterators on empty storage
  EXPECT_EQ(storage.begin(), storage.end());
}

TEST(VocabStorageTest, BasicConstruction) {
  // Create test data with 3 sections
  std::vector<std::vector<Embedding>> sectionData;

  // Section 0: 2 embeddings of dimension 3
  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0, 2.0, 3.0});
  section0.emplace_back(std::vector<double>{4.0, 5.0, 6.0});
  sectionData.push_back(std::move(section0));

  // Section 1: 1 embedding of dimension 3
  std::vector<Embedding> section1;
  section1.emplace_back(std::vector<double>{7.0, 8.0, 9.0});
  sectionData.push_back(std::move(section1));

  // Section 2: 3 embeddings of dimension 3
  std::vector<Embedding> section2;
  section2.emplace_back(std::vector<double>{10.0, 11.0, 12.0});
  section2.emplace_back(std::vector<double>{13.0, 14.0, 15.0});
  section2.emplace_back(std::vector<double>{16.0, 17.0, 18.0});
  sectionData.push_back(std::move(section2));

  VocabStorage storage(std::move(sectionData));

  EXPECT_EQ(storage.size(), 6u); // Total: 2 + 1 + 3 = 6
  EXPECT_EQ(storage.getNumSections(), 3u);
  EXPECT_EQ(storage.getDimension(), 3u);
  EXPECT_TRUE(storage.isValid());
}

TEST(VocabStorageTest, SectionAccess) {
  // Create test data
  std::vector<std::vector<Embedding>> sectionData;

  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0, 2.0});
  section0.emplace_back(std::vector<double>{3.0, 4.0});
  sectionData.push_back(std::move(section0));

  std::vector<Embedding> section1;
  section1.emplace_back(std::vector<double>{5.0, 6.0});
  sectionData.push_back(std::move(section1));

  VocabStorage storage(std::move(sectionData));

  // Test section access
  EXPECT_EQ(storage[0].size(), 2u);
  EXPECT_EQ(storage[1].size(), 1u);

  // Test embedding values
  EXPECT_THAT(storage[0][0].getData(), ElementsAre(1.0, 2.0));
  EXPECT_THAT(storage[0][1].getData(), ElementsAre(3.0, 4.0));
  EXPECT_THAT(storage[1][0].getData(), ElementsAre(5.0, 6.0));
}

#if GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(VocabStorageTest, InvalidSectionAccess) {
  std::vector<std::vector<Embedding>> sectionData;
  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0, 2.0});
  sectionData.push_back(std::move(section0));

  VocabStorage storage(std::move(sectionData));

  EXPECT_DEATH(storage[1], "Invalid section ID");
  EXPECT_DEATH(storage[10], "Invalid section ID");
}

TEST(VocabStorageTest, EmptySection) {
  std::vector<std::vector<Embedding>> sectionData;
  std::vector<Embedding> emptySection; // Empty section
  sectionData.push_back(std::move(emptySection));

  std::vector<Embedding> validSection;
  validSection.emplace_back(std::vector<double>{1.0});
  sectionData.push_back(std::move(validSection));

  EXPECT_DEATH(VocabStorage(std::move(sectionData)),
               "Vocabulary section is empty");
}

TEST(VocabStorageTest, EmptyMiddleSection) {
  std::vector<std::vector<Embedding>> sectionData;

  // Valid first section
  std::vector<Embedding> validSection1;
  validSection1.emplace_back(std::vector<double>{1.0});
  sectionData.push_back(std::move(validSection1));

  // Empty middle section
  std::vector<Embedding> emptySection;
  sectionData.push_back(std::move(emptySection));

  // Valid last section
  std::vector<Embedding> validSection2;
  validSection2.emplace_back(std::vector<double>{2.0});
  sectionData.push_back(std::move(validSection2));

  EXPECT_DEATH(VocabStorage(std::move(sectionData)),
               "Vocabulary section is empty");
}

TEST(VocabStorageTest, NoSections) {
  std::vector<std::vector<Embedding>> sectionData; // No sections

  EXPECT_DEATH(VocabStorage(std::move(sectionData)),
               "Vocabulary has no sections");
}

TEST(VocabStorageTest, MismatchedDimensionsAcrossSections) {
  std::vector<std::vector<Embedding>> sectionData;

  // Section 0: embeddings with dimension 2
  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0, 2.0});
  section0.emplace_back(std::vector<double>{3.0, 4.0});
  sectionData.push_back(std::move(section0));

  // Section 1: embedding with dimension 3 (mismatch!)
  std::vector<Embedding> section1;
  section1.emplace_back(std::vector<double>{5.0, 6.0, 7.0});
  sectionData.push_back(std::move(section1));

  EXPECT_DEATH(VocabStorage(std::move(sectionData)),
               "All embeddings must have the same dimension");
}

TEST(VocabStorageTest, MismatchedDimensionsWithinSection) {
  std::vector<std::vector<Embedding>> sectionData;

  // Section 0: first embedding with dimension 2, second with dimension 3
  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0, 2.0});
  section0.emplace_back(std::vector<double>{3.0, 4.0, 5.0}); // Mismatch!
  sectionData.push_back(std::move(section0));

  EXPECT_DEATH(VocabStorage(std::move(sectionData)),
               "All embeddings must have the same dimension");
}
#endif // NDEBUG
#endif // GTEST_HAS_DEATH_TEST

TEST(VocabStorageTest, IteratorBasics) {
  std::vector<std::vector<Embedding>> sectionData;

  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0, 2.0});
  section0.emplace_back(std::vector<double>{3.0, 4.0});
  sectionData.push_back(std::move(section0));

  std::vector<Embedding> section1;
  section1.emplace_back(std::vector<double>{5.0, 6.0});
  sectionData.push_back(std::move(section1));

  VocabStorage storage(std::move(sectionData));

  // Test iterator basics
  auto it = storage.begin();
  auto end = storage.end();

  EXPECT_NE(it, end);

  // Check first embedding
  EXPECT_THAT((*it).getData(), ElementsAre(1.0, 2.0));

  // Advance to second embedding
  ++it;
  EXPECT_NE(it, end);
  EXPECT_THAT((*it).getData(), ElementsAre(3.0, 4.0));

  // Advance to third embedding (in section 1)
  ++it;
  EXPECT_NE(it, end);
  EXPECT_THAT((*it).getData(), ElementsAre(5.0, 6.0));

  // Advance past the end
  ++it;
  EXPECT_EQ(it, end);
}

TEST(VocabStorageTest, IteratorTraversal) {
  std::vector<std::vector<Embedding>> sectionData;

  // Section 0: 2 embeddings
  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{10.0});
  section0.emplace_back(std::vector<double>{20.0});
  sectionData.push_back(std::move(section0));

  // Section 1: 1 embedding
  std::vector<Embedding> section1;
  section1.emplace_back(std::vector<double>{25.0});
  sectionData.push_back(std::move(section1));

  // Section 2: 3 embeddings
  std::vector<Embedding> section2;
  section2.emplace_back(std::vector<double>{30.0});
  section2.emplace_back(std::vector<double>{40.0});
  section2.emplace_back(std::vector<double>{50.0});
  sectionData.push_back(std::move(section2));

  VocabStorage storage(std::move(sectionData));

  // Collect all values using iterator
  std::vector<double> values;
  for (const auto &emb : storage) {
    EXPECT_EQ(emb.size(), 1u);
    values.push_back(emb[0]);
  }

  // Should get all embeddings from all sections
  EXPECT_THAT(values, ElementsAre(10.0, 20.0, 25.0, 30.0, 40.0, 50.0));
}

TEST(VocabStorageTest, IteratorComparison) {
  std::vector<std::vector<Embedding>> sectionData;
  std::vector<Embedding> section0;
  section0.emplace_back(std::vector<double>{1.0});
  section0.emplace_back(std::vector<double>{2.0});
  sectionData.push_back(std::move(section0));

  VocabStorage storage(std::move(sectionData));

  auto it1 = storage.begin();
  auto it2 = storage.begin();
  auto end = storage.end();

  // Test equality
  EXPECT_EQ(it1, it2);
  EXPECT_NE(it1, end);

  // Advance one iterator
  ++it1;
  EXPECT_NE(it1, it2);
  EXPECT_NE(it1, end);

  // Advance second iterator to match
  ++it2;
  EXPECT_EQ(it1, it2);

  // Advance both to end
  ++it1;
  ++it2;
  EXPECT_EQ(it1, end);
  EXPECT_EQ(it2, end);
  EXPECT_EQ(it1, it2);
}

} // end anonymous namespace
