//===- OperationsTest.cpp - Tests for fuzzer operations -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/FuzzMutate/Operations.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/FuzzMutate/OpDescriptor.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <iostream>

// Define some pretty printers to help with debugging failures.
namespace llvm {
void PrintTo(Type *T, ::std::ostream *OS) {
  raw_os_ostream ROS(*OS);
  T->print(ROS);
}

void PrintTo(BasicBlock *BB, ::std::ostream *OS) {
  raw_os_ostream ROS(*OS);
  ROS << BB << " (" << BB->getName() << ")";
}

void PrintTo(Value *V, ::std::ostream *OS) {
  raw_os_ostream ROS(*OS);
  ROS << V << " (";
  V->print(ROS);
  ROS << ")";
}
void PrintTo(Constant *C, ::std::ostream *OS) { PrintTo(cast<Value>(C), OS); }

} // namespace llvm

using namespace llvm;

using testing::AllOf;
using testing::AnyOf;
using testing::Each;
using testing::ElementsAre;
using testing::Eq;
using testing::Ge;
using testing::PrintToString;
using testing::SizeIs;
using testing::Truly;

namespace {
std::unique_ptr<Module> parseAssembly(const char *Assembly,
                                      LLVMContext &Context) {

  SMDiagnostic Error;
  std::unique_ptr<Module> M = parseAssemblyString(Assembly, Error, Context);

  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  Error.print("", OS);

  assert(M && !verifyModule(*M, &errs()));
  return M;
}

MATCHER_P(TypesMatch, V, "has type " + PrintToString(V->getType())) {
  return arg->getType() == V->getType();
}

MATCHER_P(HasType, T, "") { return arg->getType() == T; }

TEST(OperationsTest, SourcePreds) {
  using namespace llvm::fuzzerop;

  LLVMContext Ctx;

  Constant *i1 = ConstantInt::getFalse(Ctx);
  Constant *i8 = ConstantInt::get(Type::getInt8Ty(Ctx), 3);
  Constant *i16 = ConstantInt::get(Type::getInt16Ty(Ctx), 1 << 15);
  Constant *i32 = ConstantInt::get(Type::getInt32Ty(Ctx), 0);
  Constant *i64 = ConstantInt::get(Type::getInt64Ty(Ctx),
                                   std::numeric_limits<uint64_t>::max());
  Constant *f16 = ConstantFP::getInfinity(Type::getHalfTy(Ctx));
  Constant *f32 = ConstantFP::get(Type::getFloatTy(Ctx), 0.0);
  Constant *f64 = ConstantFP::get(Type::getDoubleTy(Ctx), 123.45);
  Constant *s = ConstantStruct::get(StructType::create(Ctx, "OpaqueStruct"));
  Constant *a =
      ConstantArray::get(ArrayType::get(i32->getType(), 2), {i32, i32});
  Constant *v8i1 = ConstantVector::getSplat(ElementCount::getFixed(8), i1);
  Constant *v8i8 = ConstantVector::getSplat(ElementCount::getFixed(8), i8);
  Constant *v4f16 = ConstantVector::getSplat(ElementCount::getFixed(4), f16);
  Constant *p0i32 =
      ConstantPointerNull::get(PointerType::get(i32->getType(), 0));
  Constant *v8p0i32 =
      ConstantVector::getSplat(ElementCount::getFixed(8), p0i32);
  Constant *vni32 = ConstantVector::getSplat(ElementCount::getScalable(8), i32);
  Constant *vnf64 = ConstantVector::getSplat(ElementCount::getScalable(8), f64);
  Constant *vnp0i32 =
      ConstantVector::getSplat(ElementCount::getScalable(8), p0i32);

  auto OnlyI32 = onlyType(i32->getType());
  EXPECT_TRUE(OnlyI32.matches({}, i32));
  EXPECT_FALSE(OnlyI32.matches({}, i64));
  EXPECT_FALSE(OnlyI32.matches({}, p0i32));
  EXPECT_FALSE(OnlyI32.matches({}, a));

  EXPECT_THAT(OnlyI32.generate({}, {}),
              AllOf(SizeIs(Ge(1u)), Each(TypesMatch(i32))));

  auto AnyType = anyType();
  EXPECT_TRUE(AnyType.matches({}, i1));
  EXPECT_TRUE(AnyType.matches({}, f64));
  EXPECT_TRUE(AnyType.matches({}, s));
  EXPECT_TRUE(AnyType.matches({}, v8i8));
  EXPECT_TRUE(AnyType.matches({}, p0i32));

  EXPECT_THAT(
      AnyType.generate({}, {i32->getType(), f16->getType(), v8i8->getType()}),
      Each(AnyOf(TypesMatch(i32), TypesMatch(f16), TypesMatch(v8i8))));

  auto AnyInt = anyIntType();
  EXPECT_TRUE(AnyInt.matches({}, i1));
  EXPECT_TRUE(AnyInt.matches({}, i64));
  EXPECT_FALSE(AnyInt.matches({}, f32));
  EXPECT_FALSE(AnyInt.matches({}, v4f16));

  EXPECT_THAT(
      AnyInt.generate({}, {i32->getType(), f16->getType(), v8i8->getType()}),
      AllOf(SizeIs(Ge(1u)), Each(TypesMatch(i32))));

  auto AnyIntOrVecInt = anyIntOrVecIntType();
  EXPECT_TRUE(AnyIntOrVecInt.matches({}, i1));
  EXPECT_TRUE(AnyIntOrVecInt.matches({}, i64));
  EXPECT_FALSE(AnyIntOrVecInt.matches({}, f32));
  EXPECT_FALSE(AnyIntOrVecInt.matches({}, v4f16));
  EXPECT_TRUE(AnyIntOrVecInt.matches({}, v8i8));
  EXPECT_FALSE(AnyIntOrVecInt.matches({}, v4f16));
  EXPECT_FALSE(AnyIntOrVecInt.matches({}, v8p0i32));
  EXPECT_TRUE(AnyIntOrVecInt.matches({}, vni32));
  EXPECT_FALSE(AnyIntOrVecInt.matches({}, vnf64));
  EXPECT_FALSE(AnyIntOrVecInt.matches({}, vnp0i32));

  EXPECT_THAT(AnyIntOrVecInt.generate({}, {v8i8->getType()}),
              AllOf(Each(TypesMatch(v8i8))));

  auto BoolOrVecBool = boolOrVecBoolType();
  EXPECT_TRUE(BoolOrVecBool.matches({}, i1));
  EXPECT_FALSE(BoolOrVecBool.matches({}, i64));
  EXPECT_FALSE(BoolOrVecBool.matches({}, f32));
  EXPECT_FALSE(BoolOrVecBool.matches({}, v4f16));
  EXPECT_TRUE(BoolOrVecBool.matches({}, v8i1));
  EXPECT_FALSE(BoolOrVecBool.matches({}, v4f16));
  EXPECT_FALSE(BoolOrVecBool.matches({}, v8p0i32));
  EXPECT_FALSE(BoolOrVecBool.matches({}, vni32));
  EXPECT_FALSE(BoolOrVecBool.matches({}, vnf64));
  EXPECT_FALSE(BoolOrVecBool.matches({}, vnp0i32));

  EXPECT_THAT(BoolOrVecBool.generate({}, {v8i8->getType(), v8i1->getType()}),
              AllOf(Each(TypesMatch(v8i1))));

  auto AnyFP = anyFloatType();
  EXPECT_TRUE(AnyFP.matches({}, f16));
  EXPECT_TRUE(AnyFP.matches({}, f32));
  EXPECT_FALSE(AnyFP.matches({}, i16));
  EXPECT_FALSE(AnyFP.matches({}, p0i32));
  EXPECT_FALSE(AnyFP.matches({}, v4f16));

  EXPECT_THAT(
      AnyFP.generate({}, {i32->getType(), f16->getType(), v8i8->getType()}),
      AllOf(SizeIs(Ge(1u)), Each(TypesMatch(f16))));

  auto AnyFPOrVecFP = anyFloatOrVecFloatType();
  EXPECT_TRUE(AnyFPOrVecFP.matches({}, f16));
  EXPECT_TRUE(AnyFPOrVecFP.matches({}, f32));
  EXPECT_FALSE(AnyFPOrVecFP.matches({}, i16));
  EXPECT_FALSE(AnyFPOrVecFP.matches({}, p0i32));
  EXPECT_TRUE(AnyFPOrVecFP.matches({}, v4f16));
  EXPECT_FALSE(AnyFPOrVecFP.matches({}, v8p0i32));
  EXPECT_FALSE(AnyFPOrVecFP.matches({}, vni32));
  EXPECT_TRUE(AnyFPOrVecFP.matches({}, vnf64));
  EXPECT_FALSE(AnyFPOrVecFP.matches({}, vnp0i32));

  EXPECT_THAT(AnyFPOrVecFP.generate(
                  {}, {i32->getType(), f16->getType(), v8i8->getType()}),
              AllOf(SizeIs(Ge(1u)), Each(TypesMatch(f16))));
  EXPECT_THAT(AnyFPOrVecFP.generate({}, {v4f16->getType()}),
              AllOf(SizeIs(Ge(1u)), Each(TypesMatch(v4f16))));

  auto AnyPtr = anyPtrType();
  EXPECT_TRUE(AnyPtr.matches({}, p0i32));
  EXPECT_FALSE(AnyPtr.matches({}, i8));
  EXPECT_FALSE(AnyPtr.matches({}, a));
  EXPECT_FALSE(AnyPtr.matches({}, v8i8));
  EXPECT_FALSE(AnyPtr.matches({}, v8p0i32));
  EXPECT_FALSE(AnyPtr.matches({}, vni32));

  auto isPointer = [](Value *V) { return V->getType()->isPointerTy(); };
  EXPECT_THAT(
      AnyPtr.generate({}, {i32->getType(), f16->getType(), v8i8->getType()}),
      AllOf(SizeIs(Ge(3u)), Each(Truly(isPointer))));

  auto AnyVec = anyVectorType();
  EXPECT_TRUE(AnyVec.matches({}, v8i8));
  EXPECT_TRUE(AnyVec.matches({}, v4f16));
  EXPECT_FALSE(AnyVec.matches({}, i8));
  EXPECT_FALSE(AnyVec.matches({}, a));
  EXPECT_FALSE(AnyVec.matches({}, s));
  EXPECT_TRUE(AnyVec.matches({}, v8p0i32));
  EXPECT_TRUE(AnyVec.matches({}, vni32));
  EXPECT_TRUE(AnyVec.matches({}, vnf64));
  EXPECT_TRUE(AnyVec.matches({}, vnp0i32));

  EXPECT_THAT(AnyVec.generate({}, {v8i8->getType()}), Each(TypesMatch(v8i8)));

  auto First = matchFirstType();
  EXPECT_TRUE(First.matches({i8}, i8));
  EXPECT_TRUE(First.matches({s, a}, s));
  EXPECT_FALSE(First.matches({f16}, f32));
  EXPECT_FALSE(First.matches({v4f16, f64}, f64));

  EXPECT_THAT(First.generate({i8}, {}), Each(TypesMatch(i8)));
  EXPECT_THAT(First.generate({f16}, {i8->getType()}), Each(TypesMatch(f16)));
  EXPECT_THAT(First.generate({v8i8, i32}, {}), Each(TypesMatch(v8i8)));

  auto FirstLength = matchFirstLengthWAnyType();
  EXPECT_TRUE(FirstLength.matches({v8i8}, v8i1));

  EXPECT_THAT(FirstLength.generate({v8i1}, {i8->getType()}),
              Each(TypesMatch(v8i8)));

  auto Second = matchSecondType();
  EXPECT_TRUE(Second.matches({i32, i8}, i8));
  EXPECT_TRUE(Second.matches({i8, f16}, f16));

  EXPECT_THAT(Second.generate({v8i8, i32}, {}), Each(TypesMatch(i32)));
  EXPECT_THAT(Second.generate({f32, f16}, {f16->getType()}),
              Each(TypesMatch(f16)));

  auto FirstScalar = matchScalarOfFirstType();
  EXPECT_TRUE(FirstScalar.matches({v8i8}, i8));
  EXPECT_TRUE(FirstScalar.matches({i8}, i8));
  EXPECT_TRUE(FirstScalar.matches({v4f16}, f16));

  EXPECT_THAT(FirstScalar.generate({v8i8}, {i8->getType()}),
              Each(TypesMatch(i8)));
}

TEST(OperationsTest, SplitBlock) {
  LLVMContext Ctx;

  Module M("M", Ctx);
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), {},
                                                   /*isVarArg=*/false),
                                 GlobalValue::ExternalLinkage, "f", &M);
  auto SBOp = fuzzerop::splitBlockDescriptor(1);

  // Create a block with only a return and split it on the return.
  auto *BB = BasicBlock::Create(Ctx, "BB", F);
  auto *RI = ReturnInst::Create(Ctx, BB);
  SBOp.BuilderFunc({UndefValue::get(Type::getInt1Ty(Ctx))}, RI);

  // We should end up with an unconditional branch from BB to BB1, and the
  // return ends up in BB1.
  auto *UncondBr = cast<BranchInst>(BB->getTerminator());
  ASSERT_TRUE(UncondBr->isUnconditional());
  auto *BB1 = UncondBr->getSuccessor(0);
  ASSERT_THAT(RI->getParent(), Eq(BB1));

  // Now add an instruction to BB1 and split on that.
  auto *AI = new AllocaInst(Type::getInt8Ty(Ctx), 0, "a", RI);
  Value *Cond = ConstantInt::getFalse(Ctx);
  SBOp.BuilderFunc({Cond}, AI);

  // We should end up with a loop back on BB1 and the instruction we split on
  // moves to BB2.
  auto *CondBr = cast<BranchInst>(BB1->getTerminator());
  EXPECT_THAT(CondBr->getCondition(), Eq(Cond));
  ASSERT_THAT(CondBr->getNumSuccessors(), Eq(2u));
  ASSERT_THAT(CondBr->getSuccessor(0), Eq(BB1));
  auto *BB2 = CondBr->getSuccessor(1);
  EXPECT_THAT(AI->getParent(), Eq(BB2));
  EXPECT_THAT(RI->getParent(), Eq(BB2));

  EXPECT_FALSE(verifyModule(M, &errs()));
}

TEST(OperationsTest, SplitEHBlock) {
  // Check that we will not try to branch back to the landingpad block using
  // regular branch instruction

  LLVMContext Ctx;
  const char *SourceCode =
      "declare ptr @f()"
      "declare i32 @personality_function()"
      "define ptr @test() personality ptr @personality_function {\n"
      "entry:\n"
      "  %val = invoke ptr @f()\n"
      "          to label %normal unwind label %exceptional\n"
      "normal:\n"
      "  ret ptr %val\n"
      "exceptional:\n"
      "  %landing_pad4 = landingpad token cleanup\n"
      "  ret ptr undef\n"
      "}";
  auto M = parseAssembly(SourceCode, Ctx);

  // Get the landingpad block
  BasicBlock &BB = *std::next(M->getFunction("test")->begin(), 2);

  fuzzerop::OpDescriptor Descr = fuzzerop::splitBlockDescriptor(1);

  Descr.BuilderFunc({ConstantInt::getTrue(Ctx)}, &*BB.getFirstInsertionPt());
  ASSERT_TRUE(!verifyModule(*M, &errs()));
}

TEST(OperationsTest, SplitBlockWithPhis) {
  LLVMContext Ctx;

  Type *Int8Ty = Type::getInt8Ty(Ctx);

  Module M("M", Ctx);
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), {},
                                                   /*isVarArg=*/false),
                                 GlobalValue::ExternalLinkage, "f", &M);
  auto SBOp = fuzzerop::splitBlockDescriptor(1);

  // Create 3 blocks with an if-then branch.
  auto *BB1 = BasicBlock::Create(Ctx, "BB1", F);
  auto *BB2 = BasicBlock::Create(Ctx, "BB2", F);
  auto *BB3 = BasicBlock::Create(Ctx, "BB3", F);
  BranchInst::Create(BB2, BB3, ConstantInt::getFalse(Ctx), BB1);
  BranchInst::Create(BB3, BB2);

  // Set up phi nodes selecting values for the incoming edges.
  auto *PHI1 = PHINode::Create(Int8Ty, /*NumReservedValues=*/2, "p1", BB3);
  PHI1->addIncoming(ConstantInt::get(Int8Ty, 0), BB1);
  PHI1->addIncoming(ConstantInt::get(Int8Ty, 1), BB2);
  auto *PHI2 = PHINode::Create(Int8Ty, /*NumReservedValues=*/2, "p2", BB3);
  PHI2->addIncoming(ConstantInt::get(Int8Ty, 1), BB1);
  PHI2->addIncoming(ConstantInt::get(Int8Ty, 0), BB2);
  auto *RI = ReturnInst::Create(Ctx, BB3);

  // Now we split the block with PHI nodes, making sure they're all updated.
  Value *Cond = ConstantInt::getFalse(Ctx);
  SBOp.BuilderFunc({Cond}, RI);

  // Make sure the PHIs are updated with a value for the third incoming edge.
  EXPECT_THAT(PHI1->getNumIncomingValues(), Eq(3u));
  EXPECT_THAT(PHI2->getNumIncomingValues(), Eq(3u));
  EXPECT_FALSE(verifyModule(M, &errs()));
}

TEST(OperationsTest, GEP) {
  LLVMContext Ctx;

  Type *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);

  Module M("M", Ctx);
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), {},
                                                   /*isVarArg=*/false),
                                 GlobalValue::ExternalLinkage, "f", &M);
  auto *BB = BasicBlock::Create(Ctx, "BB", F);
  auto *RI = ReturnInst::Create(Ctx, BB);

  auto GEPOp = fuzzerop::gepDescriptor(1);
  EXPECT_TRUE(GEPOp.SourcePreds[0].matches({}, UndefValue::get(Int8PtrTy)));
  EXPECT_TRUE(GEPOp.SourcePreds[1].matches({UndefValue::get(Int8PtrTy)},
                                           ConstantInt::get(Int32Ty, 0)));

  GEPOp.BuilderFunc({UndefValue::get(Int8PtrTy), ConstantInt::get(Int32Ty, 0)},
                    RI);
  EXPECT_FALSE(verifyModule(M, &errs()));
}

TEST(OperationsTest, GEPPointerOperand) {
  // Check that we only pick sized pointers for the GEP instructions

  LLVMContext Ctx;
  const char *SourceCode = "%opaque = type opaque\n"
                           "declare void @f()\n"
                           "define void @test(%opaque %o) {\n"
                           "  %a = alloca i64, i32 10\n"
                           "  ret void\n"
                           "}";
  auto M = parseAssembly(SourceCode, Ctx);

  fuzzerop::OpDescriptor Descr = fuzzerop::gepDescriptor(1);

  // Get first basic block of the test function
  Function &F = *M->getFunction("test");
  BasicBlock &BB = *F.begin();

  // Don't match %o
  ASSERT_FALSE(Descr.SourcePreds[0].matches({}, &*F.arg_begin()));

  // Match %a
  ASSERT_TRUE(Descr.SourcePreds[0].matches({}, &*BB.begin()));
}

TEST(OperationsTest, ExtractAndInsertValue) {
  LLVMContext Ctx;

  Type *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  Type *Int32Ty = Type::getInt32Ty(Ctx);
  Type *Int64Ty = Type::getInt64Ty(Ctx);

  Type *StructTy = StructType::create(Ctx, {Int8PtrTy, Int32Ty});
  Type *OpaqueTy = StructType::create(Ctx, "OpaqueStruct");
  Type *ZeroSizedArrayTy = ArrayType::get(Int64Ty, 0);
  Type *ArrayTy = ArrayType::get(Int64Ty, 4);
  Type *VectorTy = FixedVectorType::get(Int32Ty, 2);

  auto EVOp = fuzzerop::extractValueDescriptor(1);
  auto IVOp = fuzzerop::insertValueDescriptor(1);

  // Sanity check the source preds.
  Constant *SVal = UndefValue::get(StructTy);
  Constant *OVal = UndefValue::get(OpaqueTy);
  Constant *AVal = UndefValue::get(ArrayTy);
  Constant *ZAVal = UndefValue::get(ZeroSizedArrayTy);
  Constant *VVal = UndefValue::get(VectorTy);

  EXPECT_TRUE(EVOp.SourcePreds[0].matches({}, SVal));
  EXPECT_FALSE(EVOp.SourcePreds[0].matches({}, OVal));
  EXPECT_TRUE(EVOp.SourcePreds[0].matches({}, AVal));
  EXPECT_FALSE(EVOp.SourcePreds[0].matches({}, VVal));
  EXPECT_TRUE(IVOp.SourcePreds[0].matches({}, SVal));
  EXPECT_FALSE(IVOp.SourcePreds[0].matches({}, OVal));
  EXPECT_TRUE(IVOp.SourcePreds[0].matches({}, AVal));
  EXPECT_FALSE(IVOp.SourcePreds[0].matches({}, VVal));

  // Don't consider zero sized arrays as viable sources
  EXPECT_FALSE(EVOp.SourcePreds[0].matches({}, ZAVal));
  EXPECT_FALSE(IVOp.SourcePreds[0].matches({}, ZAVal));

  // Make sure we're range checking appropriately.
  EXPECT_TRUE(
      EVOp.SourcePreds[1].matches({SVal}, ConstantInt::get(Int32Ty, 0)));
  EXPECT_TRUE(
      EVOp.SourcePreds[1].matches({SVal}, ConstantInt::get(Int32Ty, 1)));
  EXPECT_FALSE(
      EVOp.SourcePreds[1].matches({SVal}, ConstantInt::get(Int32Ty, 2)));
  EXPECT_FALSE(
      EVOp.SourcePreds[1].matches({OVal}, ConstantInt::get(Int32Ty, 0)));
  EXPECT_FALSE(
      EVOp.SourcePreds[1].matches({OVal}, ConstantInt::get(Int32Ty, 65536)));
  EXPECT_TRUE(
      EVOp.SourcePreds[1].matches({AVal}, ConstantInt::get(Int32Ty, 0)));
  EXPECT_TRUE(
      EVOp.SourcePreds[1].matches({AVal}, ConstantInt::get(Int32Ty, 3)));
  EXPECT_FALSE(
      EVOp.SourcePreds[1].matches({AVal}, ConstantInt::get(Int32Ty, 4)));

  EXPECT_THAT(
      EVOp.SourcePreds[1].generate({SVal}, {}),
      ElementsAre(ConstantInt::get(Int32Ty, 0), ConstantInt::get(Int32Ty, 1)));

  // InsertValue should accept any type in the struct, but only in positions
  // where it makes sense.
  EXPECT_TRUE(IVOp.SourcePreds[1].matches({SVal}, UndefValue::get(Int8PtrTy)));
  EXPECT_TRUE(IVOp.SourcePreds[1].matches({SVal}, UndefValue::get(Int32Ty)));
  EXPECT_FALSE(IVOp.SourcePreds[1].matches({SVal}, UndefValue::get(Int64Ty)));
  EXPECT_FALSE(IVOp.SourcePreds[2].matches({SVal, UndefValue::get(Int32Ty)},
                                           ConstantInt::get(Int32Ty, 0)));
  EXPECT_TRUE(IVOp.SourcePreds[2].matches({SVal, UndefValue::get(Int32Ty)},
                                          ConstantInt::get(Int32Ty, 1)));

  EXPECT_THAT(IVOp.SourcePreds[1].generate({SVal}, {}),
              Each(AnyOf(HasType(Int32Ty), HasType(Int8PtrTy))));
  EXPECT_THAT(
      IVOp.SourcePreds[2].generate({SVal, ConstantInt::get(Int32Ty, 0)}, {}),
      ElementsAre(ConstantInt::get(Int32Ty, 1)));
}

} // namespace
