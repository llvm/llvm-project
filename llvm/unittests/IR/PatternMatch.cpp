//===---- llvm/unittest/IR/PatternMatch.cpp - PatternMatch unit tests ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "gtest/gtest.h"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

struct PatternMatchTest : ::testing::Test {
  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  IRBuilder<NoFolder> IRB;

  PatternMatchTest()
      : M(new Module("PatternMatchTestModule", Ctx)),
        F(Function::Create(
            FunctionType::get(Type::getVoidTy(Ctx), /* IsVarArg */ false),
            Function::ExternalLinkage, "f", M.get())),
        BB(BasicBlock::Create(Ctx, "entry", F)), IRB(BB) {}
};

TEST_F(PatternMatchTest, OneUse) {
  // Build up a little tree of values:
  //
  //   One  = (1 + 2) + 42
  //   Two  = One + 42
  //   Leaf = (Two + 8) + (Two + 13)
  Value *One = IRB.CreateAdd(IRB.CreateAdd(IRB.getInt32(1), IRB.getInt32(2)),
                             IRB.getInt32(42));
  Value *Two = IRB.CreateAdd(One, IRB.getInt32(42));
  Value *Leaf = IRB.CreateAdd(IRB.CreateAdd(Two, IRB.getInt32(8)),
                              IRB.CreateAdd(Two, IRB.getInt32(13)));
  Value *V;

  EXPECT_TRUE(m_OneUse(m_Value(V)).match(One));
  EXPECT_EQ(One, V);

  EXPECT_FALSE(m_OneUse(m_Value()).match(Two));
  EXPECT_FALSE(m_OneUse(m_Value()).match(Leaf));
}

TEST_F(PatternMatchTest, FloatingPointOrderedMin) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test OLT.
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test OLE.
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on OGE.
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), L, R)));

  // Test no match on OGT.
  EXPECT_FALSE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), L, R)));

  // Test match on OGE with inverted select.
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test match on OGT with inverted select.
  EXPECT_TRUE(m_OrdFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, FloatingPointOrderedMax) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test OGT.
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test OGE.
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOGE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on OLE.
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), L, R)));

  // Test no match on OLT.
  EXPECT_FALSE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), L, R)));

  // Test match on OLE with inverted select.
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test match on OLT with inverted select.
  EXPECT_TRUE(m_OrdFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpOLT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, FloatingPointUnorderedMin) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test ULT.
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test ULE.
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on UGE.
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), L, R)));

  // Test no match on UGT.
  EXPECT_FALSE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), L, R)));

  // Test match on UGE with inverted select.
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test match on UGT with inverted select.
  EXPECT_TRUE(m_UnordFMin(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, FloatingPointUnorderedMax) {
  Type *FltTy = IRB.getFloatTy();
  Value *L = ConstantFP::get(FltTy, 1.0);
  Value *R = ConstantFP::get(FltTy, 2.0);
  Value *MatchL, *MatchR;

  // Test UGT.
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGT(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test UGE.
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpUGE(L, R), L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test no match on ULE.
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), L, R)));

  // Test no match on ULT.
  EXPECT_FALSE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                   .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), L, R)));

  // Test match on ULE with inverted select.
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULE(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  // Test match on ULT with inverted select.
  EXPECT_TRUE(m_UnordFMax(m_Value(MatchL), m_Value(MatchR))
                  .match(IRB.CreateSelect(IRB.CreateFCmpULT(L, R), R, L)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

TEST_F(PatternMatchTest, OverflowingBinOps) {
  Value *L = IRB.getInt32(1);
  Value *R = IRB.getInt32(2);
  Value *MatchL, *MatchR;

  EXPECT_TRUE(
      m_NSWAdd(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NSWSub(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNSWSub(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NSWMul(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNSWMul(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(m_NSWShl(m_Value(MatchL), m_Value(MatchR)).match(
      IRB.CreateShl(L, R, "", /* NUW */ false, /* NSW */ true)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_TRUE(
      m_NUWAdd(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NUWSub(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNUWSub(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(
      m_NUWMul(m_Value(MatchL), m_Value(MatchR)).match(IRB.CreateNUWMul(L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
  MatchL = MatchR = nullptr;
  EXPECT_TRUE(m_NUWShl(m_Value(MatchL), m_Value(MatchR)).match(
      IRB.CreateShl(L, R, "", /* NUW */ true, /* NSW */ false)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);

  EXPECT_FALSE(m_NSWAdd(m_Value(), m_Value()).match(IRB.CreateAdd(L, R)));
  EXPECT_FALSE(m_NSWAdd(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_FALSE(m_NSWAdd(m_Value(), m_Value()).match(IRB.CreateNSWSub(L, R)));
  EXPECT_FALSE(m_NSWSub(m_Value(), m_Value()).match(IRB.CreateSub(L, R)));
  EXPECT_FALSE(m_NSWSub(m_Value(), m_Value()).match(IRB.CreateNUWSub(L, R)));
  EXPECT_FALSE(m_NSWSub(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_FALSE(m_NSWMul(m_Value(), m_Value()).match(IRB.CreateMul(L, R)));
  EXPECT_FALSE(m_NSWMul(m_Value(), m_Value()).match(IRB.CreateNUWMul(L, R)));
  EXPECT_FALSE(m_NSWMul(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_FALSE(m_NSWShl(m_Value(), m_Value()).match(IRB.CreateShl(L, R)));
  EXPECT_FALSE(m_NSWShl(m_Value(), m_Value()).match(
      IRB.CreateShl(L, R, "", /* NUW */ true, /* NSW */ false)));
  EXPECT_FALSE(m_NSWShl(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));

  EXPECT_FALSE(m_NUWAdd(m_Value(), m_Value()).match(IRB.CreateAdd(L, R)));
  EXPECT_FALSE(m_NUWAdd(m_Value(), m_Value()).match(IRB.CreateNSWAdd(L, R)));
  EXPECT_FALSE(m_NUWAdd(m_Value(), m_Value()).match(IRB.CreateNUWSub(L, R)));
  EXPECT_FALSE(m_NUWSub(m_Value(), m_Value()).match(IRB.CreateSub(L, R)));
  EXPECT_FALSE(m_NUWSub(m_Value(), m_Value()).match(IRB.CreateNSWSub(L, R)));
  EXPECT_FALSE(m_NUWSub(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_FALSE(m_NUWMul(m_Value(), m_Value()).match(IRB.CreateMul(L, R)));
  EXPECT_FALSE(m_NUWMul(m_Value(), m_Value()).match(IRB.CreateNSWMul(L, R)));
  EXPECT_FALSE(m_NUWMul(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
  EXPECT_FALSE(m_NUWShl(m_Value(), m_Value()).match(IRB.CreateShl(L, R)));
  EXPECT_FALSE(m_NUWShl(m_Value(), m_Value()).match(
      IRB.CreateShl(L, R, "", /* NUW */ false, /* NSW */ true)));
  EXPECT_FALSE(m_NUWShl(m_Value(), m_Value()).match(IRB.CreateNUWAdd(L, R)));
}

template <typename T> struct MutableConstTest : PatternMatchTest { };

typedef ::testing::Types<std::tuple<Value*, Instruction*>,
                         std::tuple<const Value*, const Instruction *>>
    MutableConstTestTypes;
TYPED_TEST_CASE(MutableConstTest, MutableConstTestTypes);

TYPED_TEST(MutableConstTest, ICmp) {
  auto &IRB = PatternMatchTest::IRB;

  typedef typename std::tuple_element<0, TypeParam>::type ValueType;
  typedef typename std::tuple_element<1, TypeParam>::type InstructionType;

  Value *L = IRB.getInt32(1);
  Value *R = IRB.getInt32(2);
  ICmpInst::Predicate Pred = ICmpInst::ICMP_UGT;

  ValueType MatchL;
  ValueType MatchR;
  ICmpInst::Predicate MatchPred;

  EXPECT_TRUE(m_ICmp(MatchPred, m_Value(MatchL), m_Value(MatchR))
              .match((InstructionType)IRB.CreateICmp(Pred, L, R)));
  EXPECT_EQ(L, MatchL);
  EXPECT_EQ(R, MatchR);
}

} // anonymous namespace.
