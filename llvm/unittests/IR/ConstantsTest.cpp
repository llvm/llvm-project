//===- llvm/unittest/IR/ConstantsTest.cpp - Constants unit tests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "llvm-c/Core.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/ConstantFold.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(ConstantsTest, Integer_i1) {
  LLVMContext Context;
  IntegerType *Int1 = IntegerType::get(Context, 1);
  Constant *One = ConstantInt::get(Int1, 1, true);
  Constant *Zero = ConstantInt::get(Int1, 0);
  Constant *NegOne = ConstantInt::get(Int1, static_cast<uint64_t>(-1), true);
  EXPECT_EQ(NegOne, ConstantInt::getSigned(Int1, -1));
  Constant *Poison = PoisonValue::get(Int1);

  // Input:  @b = constant i1 add(i1 1 , i1 1)
  // Output: @b = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getAdd(One, One));

  // @c = constant i1 add(i1 -1, i1 1)
  // @c = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getAdd(NegOne, One));

  // @d = constant i1 add(i1 -1, i1 -1)
  // @d = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getAdd(NegOne, NegOne));

  // @e = constant i1 sub(i1 -1, i1 1)
  // @e = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getSub(NegOne, One));

  // @f = constant i1 sub(i1 1 , i1 -1)
  // @f = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getSub(One, NegOne));

  // @g = constant i1 sub(i1 1 , i1 1)
  // @g = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getSub(One, One));

  // @h = constant i1 shl(i1 1 , i1 1)  ; poison
  // @h = constant i1 poison
  EXPECT_EQ(Poison, ConstantExpr::getShl(One, One));

  // @i = constant i1 shl(i1 1 , i1 0)
  // @i = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getShl(One, Zero));

  // @j = constant i1 lshr(i1 1, i1 1)  ; poison
  // @j = constant i1 poison
  EXPECT_EQ(Poison, ConstantExpr::getLShr(One, One));

  // @m = constant i1 ashr(i1 1, i1 1)  ; poison
  // @m = constant i1 poison
  EXPECT_EQ(Poison, ConstantExpr::getAShr(One, One));

  // @n = constant i1 mul(i1 -1, i1 1)
  // @n = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getMul(NegOne, One));

  // @o = constant i1 sdiv(i1 -1, i1 1) ; overflow
  // @o = constant i1 true
  EXPECT_EQ(One, ConstantFoldBinaryInstruction(Instruction::SDiv, NegOne, One));

  // @p = constant i1 sdiv(i1 1 , i1 -1); overflow
  // @p = constant i1 true
  EXPECT_EQ(One, ConstantFoldBinaryInstruction(Instruction::SDiv, One, NegOne));

  // @q = constant i1 udiv(i1 -1, i1 1)
  // @q = constant i1 true
  EXPECT_EQ(One, ConstantFoldBinaryInstruction(Instruction::UDiv, NegOne, One));

  // @r = constant i1 udiv(i1 1, i1 -1)
  // @r = constant i1 true
  EXPECT_EQ(One, ConstantFoldBinaryInstruction(Instruction::UDiv, One, NegOne));

  // @s = constant i1 srem(i1 -1, i1 1) ; overflow
  // @s = constant i1 false
  EXPECT_EQ(Zero,
            ConstantFoldBinaryInstruction(Instruction::SRem, NegOne, One));

  // @u = constant i1 srem(i1  1, i1 -1) ; overflow
  // @u = constant i1 false
  EXPECT_EQ(Zero,
            ConstantFoldBinaryInstruction(Instruction::SRem, One, NegOne));
}

TEST(ConstantsTest, IntSigns) {
  LLVMContext Context;
  IntegerType *Int8Ty = Type::getInt8Ty(Context);
  EXPECT_EQ(100, ConstantInt::get(Int8Ty, 100, false)->getSExtValue());
  EXPECT_EQ(100, ConstantInt::get(Int8Ty, 100, true)->getSExtValue());
  EXPECT_EQ(100, ConstantInt::getSigned(Int8Ty, 100)->getSExtValue());
  EXPECT_EQ(-50, ConstantInt::get(Int8Ty, 206)->getSExtValue());
  EXPECT_EQ(-50, ConstantInt::getSigned(Int8Ty, -50)->getSExtValue());
  EXPECT_EQ(206U, ConstantInt::getSigned(Int8Ty, -50)->getZExtValue());

  // Overflow is handled by truncation.
  EXPECT_EQ(0x3b, ConstantInt::get(Int8Ty, 0x13b)->getSExtValue());
}

TEST(ConstantsTest, FP128Test) {
  LLVMContext Context;
  Type *FP128Ty = Type::getFP128Ty(Context);

  IntegerType *Int128Ty = Type::getIntNTy(Context, 128);
  Constant *Zero128 = Constant::getNullValue(Int128Ty);
  Constant *X = ConstantExpr::getUIToFP(Zero128, FP128Ty);
  EXPECT_TRUE(isa<ConstantFP>(X));
}

TEST(ConstantsTest, PointerCast) {
  LLVMContext C;
  Type *Int8PtrTy = Type::getInt8PtrTy(C);
  Type *Int32PtrTy = Type::getInt32PtrTy(C);
  Type *Int64Ty = Type::getInt64Ty(C);
  VectorType *Int8PtrVecTy = FixedVectorType::get(Int8PtrTy, 4);
  VectorType *Int32PtrVecTy = FixedVectorType::get(Int32PtrTy, 4);
  VectorType *Int64VecTy = FixedVectorType::get(Int64Ty, 4);
  VectorType *Int8PtrScalableVecTy = ScalableVectorType::get(Int8PtrTy, 4);
  VectorType *Int32PtrScalableVecTy = ScalableVectorType::get(Int32PtrTy, 4);
  VectorType *Int64ScalableVecTy = ScalableVectorType::get(Int64Ty, 4);

  // ptrtoint i8* to i64
  EXPECT_EQ(
      Constant::getNullValue(Int64Ty),
      ConstantExpr::getPointerCast(Constant::getNullValue(Int8PtrTy), Int64Ty));

  // bitcast i8* to i32*
  EXPECT_EQ(Constant::getNullValue(Int32PtrTy),
            ConstantExpr::getPointerCast(Constant::getNullValue(Int8PtrTy),
                                         Int32PtrTy));

  // ptrtoint <4 x i8*> to <4 x i64>
  EXPECT_EQ(Constant::getNullValue(Int64VecTy),
            ConstantExpr::getPointerCast(Constant::getNullValue(Int8PtrVecTy),
                                         Int64VecTy));

  // ptrtoint <vscale x 4 x i8*> to <vscale x 4 x i64>
  EXPECT_EQ(
      Constant::getNullValue(Int64ScalableVecTy),
      ConstantExpr::getPointerCast(Constant::getNullValue(Int8PtrScalableVecTy),
                                   Int64ScalableVecTy));

  // bitcast <4 x i8*> to <4 x i32*>
  EXPECT_EQ(Constant::getNullValue(Int32PtrVecTy),
            ConstantExpr::getPointerCast(Constant::getNullValue(Int8PtrVecTy),
                                         Int32PtrVecTy));

  // bitcast <vscale x 4 x i8*> to <vscale x 4 x i32*>
  EXPECT_EQ(
      Constant::getNullValue(Int32PtrScalableVecTy),
      ConstantExpr::getPointerCast(Constant::getNullValue(Int8PtrScalableVecTy),
                                   Int32PtrScalableVecTy));

  Type *Int32Ptr1Ty = Type::getInt32PtrTy(C, 1);
  ConstantInt *K = ConstantInt::get(Type::getInt64Ty(C), 1234);

  // Make sure that addrspacecast of inttoptr is not folded away.
  EXPECT_NE(K, ConstantExpr::getAddrSpaceCast(
                   ConstantExpr::getIntToPtr(K, Int32PtrTy), Int32Ptr1Ty));
  EXPECT_NE(K, ConstantExpr::getAddrSpaceCast(
                   ConstantExpr::getIntToPtr(K, Int32Ptr1Ty), Int32PtrTy));

  Constant *NullInt32Ptr0 = Constant::getNullValue(Int32PtrTy);
  Constant *NullInt32Ptr1 = Constant::getNullValue(Int32Ptr1Ty);

  // Make sure that addrspacecast of null is not folded away.
  EXPECT_NE(Constant::getNullValue(Int32PtrTy),
            ConstantExpr::getAddrSpaceCast(NullInt32Ptr0, Int32Ptr1Ty));

  EXPECT_NE(Constant::getNullValue(Int32Ptr1Ty),
            ConstantExpr::getAddrSpaceCast(NullInt32Ptr1, Int32PtrTy));
}

#define CHECK(x, y)                                                            \
  {                                                                            \
    std::string __s;                                                           \
    raw_string_ostream __o(__s);                                               \
    Instruction *__I = cast<ConstantExpr>(x)->getAsInstruction();              \
    __I->print(__o);                                                           \
    __I->deleteValue();                                                        \
    __o.flush();                                                               \
    EXPECT_EQ(std::string("  <badref> = " y), __s);                            \
  }

TEST(ConstantsTest, AsInstructionsTest) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  Type *Int64Ty = Type::getInt64Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int16Ty = Type::getInt16Ty(Context);
  Type *Int1Ty = Type::getInt1Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  Type *DoubleTy = Type::getDoubleTy(Context);

  Constant *Global =
      M->getOrInsertGlobal("dummy", PointerType::getUnqual(Int32Ty));
  Constant *Global2 =
      M->getOrInsertGlobal("dummy2", PointerType::getUnqual(Int32Ty));

  Constant *P0 = ConstantExpr::getPtrToInt(Global, Int32Ty);
  Constant *P1 = ConstantExpr::getUIToFP(P0, FloatTy);
  Constant *P2 = ConstantExpr::getUIToFP(P0, DoubleTy);
  Constant *P3 = ConstantExpr::getTrunc(P0, Int1Ty);
  Constant *P4 = ConstantExpr::getPtrToInt(Global2, Int32Ty);
  Constant *P5 = ConstantExpr::getUIToFP(P4, FloatTy);
  Constant *P6 = ConstantExpr::getBitCast(P4, FixedVectorType::get(Int16Ty, 2));

  Constant *One = ConstantInt::get(Int32Ty, 1);
  Constant *Two = ConstantInt::get(Int64Ty, 2);
  Constant *Big = ConstantInt::get(Context, APInt{256, uint64_t(-1), true});
  Constant *Elt = ConstantInt::get(Int16Ty, 2015);
  Constant *Poison16 = PoisonValue::get(Int16Ty);
  Constant *Undef64 = UndefValue::get(Int64Ty);
  Constant *PoisonV16 = PoisonValue::get(P6->getType());

#define P0STR "ptrtoint (i32** @dummy to i32)"
#define P1STR "uitofp (i32 ptrtoint (i32** @dummy to i32) to float)"
#define P2STR "uitofp (i32 ptrtoint (i32** @dummy to i32) to double)"
#define P3STR "ptrtoint (i32** @dummy to i1)"
#define P4STR "ptrtoint (i32** @dummy2 to i32)"
#define P5STR "uitofp (i32 ptrtoint (i32** @dummy2 to i32) to float)"
#define P6STR "bitcast (i32 ptrtoint (i32** @dummy2 to i32) to <2 x i16>)"

  CHECK(ConstantExpr::getNeg(P0), "sub i32 0, " P0STR);
  CHECK(ConstantExpr::getFNeg(P1), "fneg float " P1STR);
  CHECK(ConstantExpr::getNot(P0), "xor i32 " P0STR ", -1");
  CHECK(ConstantExpr::getAdd(P0, P0), "add i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAdd(P0, P0, false, true),
        "add nsw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAdd(P0, P0, true, true),
        "add nuw nsw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getSub(P0, P0), "sub i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getMul(P0, P0), "mul i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAnd(P0, P0), "and i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getOr(P0, P0), "or i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getXor(P0, P0), "xor i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getShl(P0, P0), "shl i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getShl(P0, P0, true), "shl nuw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getShl(P0, P0, false, true),
        "shl nsw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getLShr(P0, P0, false), "lshr i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getLShr(P0, P0, true),
        "lshr exact i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAShr(P0, P0, false), "ashr i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAShr(P0, P0, true),
        "ashr exact i32 " P0STR ", " P0STR);

  CHECK(ConstantExpr::getSExt(P0, Int64Ty), "sext i32 " P0STR " to i64");
  CHECK(ConstantExpr::getZExt(P0, Int64Ty), "zext i32 " P0STR " to i64");
  CHECK(ConstantExpr::getFPTrunc(P2, FloatTy),
        "fptrunc double " P2STR " to float");
  CHECK(ConstantExpr::getFPExtend(P1, DoubleTy),
        "fpext float " P1STR " to double");

  CHECK(ConstantExpr::getSelect(P3, P0, P4),
        "select i1 " P3STR ", i32 " P0STR ", i32 " P4STR);
  CHECK(ConstantExpr::getICmp(CmpInst::ICMP_EQ, P0, P4),
        "icmp eq i32 " P0STR ", " P4STR);
  CHECK(ConstantExpr::getFCmp(CmpInst::FCMP_ULT, P1, P5),
        "fcmp ult float " P1STR ", " P5STR);

  std::vector<Constant *> V;
  V.push_back(One);
  // FIXME: getGetElementPtr() actually creates an inbounds ConstantGEP,
  //        not a normal one!
  // CHECK(ConstantExpr::getGetElementPtr(Global, V, false),
  //      "getelementptr i32*, i32** @dummy, i32 1");
  CHECK(ConstantExpr::getInBoundsGetElementPtr(PointerType::getUnqual(Int32Ty),
                                               Global, V),
        "getelementptr inbounds i32*, i32** @dummy, i32 1");

  CHECK(ConstantExpr::getExtractElement(P6, One),
        "extractelement <2 x i16> " P6STR ", i32 1");

  EXPECT_EQ(Poison16, ConstantExpr::getExtractElement(P6, Two));
  EXPECT_EQ(Poison16, ConstantExpr::getExtractElement(P6, Big));
  EXPECT_EQ(Poison16, ConstantExpr::getExtractElement(P6, Undef64));

  EXPECT_EQ(Elt, ConstantExpr::getExtractElement(
                 ConstantExpr::getInsertElement(P6, Elt, One), One));
  EXPECT_EQ(PoisonV16, ConstantExpr::getInsertElement(P6, Elt, Two));
  EXPECT_EQ(PoisonV16, ConstantExpr::getInsertElement(P6, Elt, Big));
  EXPECT_EQ(PoisonV16, ConstantExpr::getInsertElement(P6, Elt, Undef64));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(ConstantsTest, ReplaceWithConstantTest) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  Type *Int32Ty = Type::getInt32Ty(Context);
  Constant *One = ConstantInt::get(Int32Ty, 1);

  Constant *Global =
      M->getOrInsertGlobal("dummy", PointerType::getUnqual(Int32Ty));
  Constant *GEP = ConstantExpr::getGetElementPtr(
      PointerType::getUnqual(Int32Ty), Global, One);
  EXPECT_DEATH(Global->replaceAllUsesWith(GEP),
               "this->replaceAllUsesWith\\(expr\\(this\\)\\) is NOT valid!");
}

#endif
#endif

#undef CHECK

TEST(ConstantsTest, ConstantArrayReplaceWithConstant) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  Type *IntTy = Type::getInt8Ty(Context);
  ArrayType *ArrayTy = ArrayType::get(IntTy, 2);
  Constant *A01Vals[2] = {ConstantInt::get(IntTy, 0),
                          ConstantInt::get(IntTy, 1)};
  Constant *A01 = ConstantArray::get(ArrayTy, A01Vals);

  Constant *Global = new GlobalVariable(*M, IntTy, false,
                                        GlobalValue::ExternalLinkage, nullptr);
  Constant *GlobalInt = ConstantExpr::getPtrToInt(Global, IntTy);
  Constant *A0GVals[2] = {ConstantInt::get(IntTy, 0), GlobalInt};
  Constant *A0G = ConstantArray::get(ArrayTy, A0GVals);
  ASSERT_NE(A01, A0G);

  GlobalVariable *RefArray =
      new GlobalVariable(*M, ArrayTy, false, GlobalValue::ExternalLinkage, A0G);
  ASSERT_EQ(A0G, RefArray->getInitializer());

  GlobalInt->replaceAllUsesWith(ConstantInt::get(IntTy, 1));
  ASSERT_EQ(A01, RefArray->getInitializer());
}

TEST(ConstantsTest, ConstantExprReplaceWithConstant) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  Type *IntTy = Type::getInt8Ty(Context);
  Constant *G1 = new GlobalVariable(*M, IntTy, false,
                                    GlobalValue::ExternalLinkage, nullptr);
  Constant *G2 = new GlobalVariable(*M, IntTy, false,
                                    GlobalValue::ExternalLinkage, nullptr);
  ASSERT_NE(G1, G2);

  Constant *Int1 = ConstantExpr::getPtrToInt(G1, IntTy);
  Constant *Int2 = ConstantExpr::getPtrToInt(G2, IntTy);
  ASSERT_NE(Int1, Int2);

  GlobalVariable *Ref =
      new GlobalVariable(*M, IntTy, false, GlobalValue::ExternalLinkage, Int1);
  ASSERT_EQ(Int1, Ref->getInitializer());

  G1->replaceAllUsesWith(G2);
  ASSERT_EQ(Int2, Ref->getInitializer());
}

TEST(ConstantsTest, GEPReplaceWithConstant) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  Type *IntTy = Type::getInt32Ty(Context);
  Type *PtrTy = PointerType::get(IntTy, 0);
  auto *C1 = ConstantInt::get(IntTy, 1);
  auto *Placeholder = new GlobalVariable(
      *M, IntTy, false, GlobalValue::ExternalWeakLinkage, nullptr);
  auto *GEP = ConstantExpr::getGetElementPtr(IntTy, Placeholder, C1);
  ASSERT_EQ(GEP->getOperand(0), Placeholder);

  auto *Ref =
      new GlobalVariable(*M, PtrTy, false, GlobalValue::ExternalLinkage, GEP);
  ASSERT_EQ(GEP, Ref->getInitializer());

  auto *Global = new GlobalVariable(*M, IntTy, false,
                                    GlobalValue::ExternalLinkage, nullptr);
  auto *Alias = GlobalAlias::create(IntTy, 0, GlobalValue::ExternalLinkage,
                                    "alias", Global, M.get());
  Placeholder->replaceAllUsesWith(Alias);
  ASSERT_EQ(GEP, Ref->getInitializer());
  ASSERT_EQ(GEP->getOperand(0), Alias);
}

TEST(ConstantsTest, AliasCAPI) {
  LLVMContext Context;
  SMDiagnostic Error;
  std::unique_ptr<Module> M =
      parseAssemblyString("@g = global i32 42", Error, Context);
  GlobalVariable *G = M->getGlobalVariable("g");
  Type *I16Ty = Type::getInt16Ty(Context);
  Type *I16PTy = PointerType::get(I16Ty, 0);
  Constant *Aliasee = ConstantExpr::getBitCast(G, I16PTy);
  LLVMValueRef AliasRef =
      LLVMAddAlias2(wrap(M.get()), wrap(I16Ty), 0, wrap(Aliasee), "a");
  ASSERT_EQ(unwrap<GlobalAlias>(AliasRef)->getAliasee(), Aliasee);
}

static std::string getNameOfType(Type *T) {
  std::string S;
  raw_string_ostream RSOS(S);
  T->print(RSOS);
  return S;
}

TEST(ConstantsTest, BuildConstantDataArrays) {
  LLVMContext Context;

  for (Type *T : {Type::getInt8Ty(Context), Type::getInt16Ty(Context),
                  Type::getInt32Ty(Context), Type::getInt64Ty(Context)}) {
    ArrayType *ArrayTy = ArrayType::get(T, 2);
    Constant *Vals[] = {ConstantInt::get(T, 0), ConstantInt::get(T, 1)};
    Constant *CA = ConstantArray::get(ArrayTy, Vals);
    ASSERT_TRUE(isa<ConstantDataArray>(CA)) << " T = " << getNameOfType(T);
    auto *CDA = cast<ConstantDataArray>(CA);
    Constant *CA2 = ConstantDataArray::getRaw(
        CDA->getRawDataValues(), CDA->getNumElements(), CDA->getElementType());
    ASSERT_TRUE(CA == CA2) << " T = " << getNameOfType(T);
  }

  for (Type *T : {Type::getHalfTy(Context), Type::getBFloatTy(Context),
                  Type::getFloatTy(Context), Type::getDoubleTy(Context)}) {
    ArrayType *ArrayTy = ArrayType::get(T, 2);
    Constant *Vals[] = {ConstantFP::get(T, 0), ConstantFP::get(T, 1)};
    Constant *CA = ConstantArray::get(ArrayTy, Vals);
    ASSERT_TRUE(isa<ConstantDataArray>(CA)) << " T = " << getNameOfType(T);
    auto *CDA = cast<ConstantDataArray>(CA);
    Constant *CA2 = ConstantDataArray::getRaw(
        CDA->getRawDataValues(), CDA->getNumElements(), CDA->getElementType());
    ASSERT_TRUE(CA == CA2) << " T = " << getNameOfType(T);
  }
}

TEST(ConstantsTest, BuildConstantDataVectors) {
  LLVMContext Context;

  for (Type *T : {Type::getInt8Ty(Context), Type::getInt16Ty(Context),
                  Type::getInt32Ty(Context), Type::getInt64Ty(Context)}) {
    Constant *Vals[] = {ConstantInt::get(T, 0), ConstantInt::get(T, 1)};
    Constant *CV = ConstantVector::get(Vals);
    ASSERT_TRUE(isa<ConstantDataVector>(CV)) << " T = " << getNameOfType(T);
    auto *CDV = cast<ConstantDataVector>(CV);
    Constant *CV2 = ConstantDataVector::getRaw(
        CDV->getRawDataValues(), CDV->getNumElements(), CDV->getElementType());
    ASSERT_TRUE(CV == CV2) << " T = " << getNameOfType(T);
  }

  for (Type *T : {Type::getHalfTy(Context), Type::getBFloatTy(Context),
                  Type::getFloatTy(Context), Type::getDoubleTy(Context)}) {
    Constant *Vals[] = {ConstantFP::get(T, 0), ConstantFP::get(T, 1)};
    Constant *CV = ConstantVector::get(Vals);
    ASSERT_TRUE(isa<ConstantDataVector>(CV)) << " T = " << getNameOfType(T);
    auto *CDV = cast<ConstantDataVector>(CV);
    Constant *CV2 = ConstantDataVector::getRaw(
        CDV->getRawDataValues(), CDV->getNumElements(), CDV->getElementType());
    ASSERT_TRUE(CV == CV2) << " T = " << getNameOfType(T);
  }
}

void bitcastToGEPHelper(bool useOpaquePointers) {
  LLVMContext Context;
  Context.setOpaquePointers(useOpaquePointers);
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  auto *i32 = Type::getInt32Ty(Context);
  auto *U = StructType::create(Context, "Unsized");
  Type *EltTys[] = {i32, U};
  auto *S = StructType::create(EltTys);

  auto *G =
      new GlobalVariable(*M, S, false, GlobalValue::ExternalLinkage, nullptr);
  auto *PtrTy = PointerType::get(i32, 0);
  auto *C = ConstantExpr::getBitCast(G, PtrTy);
  if (Context.supportsTypedPointers()) {
    EXPECT_EQ(cast<ConstantExpr>(C)->getOpcode(), Instruction::BitCast);
  } else {
    /* With opaque pointers, no cast is necessary. */
    EXPECT_EQ(C, G);
  }
}

TEST(ConstantsTest, BitcastToGEP) {
  bitcastToGEPHelper(true);
  bitcastToGEPHelper(false);
}

bool foldFuncPtrAndConstToNull(LLVMContext &Context, Module *TheModule,
                               uint64_t AndValue,
                               MaybeAlign FunctionAlign = llvm::None) {
  Type *VoidType(Type::getVoidTy(Context));
  FunctionType *FuncType(FunctionType::get(VoidType, false));
  Function *Func(
      Function::Create(FuncType, GlobalValue::ExternalLinkage, "", TheModule));

  if (FunctionAlign)
    Func->setAlignment(*FunctionAlign);

  IntegerType *ConstantIntType(Type::getInt32Ty(Context));
  ConstantInt *TheConstant(ConstantInt::get(ConstantIntType, AndValue));

  Constant *TheConstantExpr(ConstantExpr::getPtrToInt(Func, ConstantIntType));

  bool Result =
      ConstantExpr::get(Instruction::And, TheConstantExpr, TheConstant)
          ->isNullValue();

  if (!TheModule) {
    // If the Module exists then it will delete the Function.
    delete Func;
  }

  return Result;
}

TEST(ConstantsTest, FoldFunctionPtrAlignUnknownAnd2) {
  LLVMContext Context;
  Module TheModule("TestModule", Context);
  // When the DataLayout doesn't specify a function pointer alignment we
  // assume in this case that it is 4 byte aligned. This is a bug but we can't
  // fix it directly because it causes a code size regression on X86.
  // FIXME: This test should be changed once existing targets have
  // appropriate defaults. See associated FIXME in ConstantFoldBinaryInstruction
  ASSERT_TRUE(foldFuncPtrAndConstToNull(Context, &TheModule, 2));
}

TEST(ConstantsTest, DontFoldFunctionPtrAlignUnknownAnd4) {
  LLVMContext Context;
  Module TheModule("TestModule", Context);
  ASSERT_FALSE(foldFuncPtrAndConstToNull(Context, &TheModule, 4));
}

TEST(ConstantsTest, FoldFunctionPtrAlign4) {
  LLVMContext Context;
  Module TheModule("TestModule", Context);
  const char *AlignmentStrings[] = {"Fi32", "Fn32"};

  for (unsigned AndValue = 1; AndValue <= 2; ++AndValue) {
    for (const char *AlignmentString : AlignmentStrings) {
      TheModule.setDataLayout(AlignmentString);
      ASSERT_TRUE(foldFuncPtrAndConstToNull(Context, &TheModule, AndValue));
    }
  }
}

TEST(ConstantsTest, DontFoldFunctionPtrAlign1) {
  LLVMContext Context;
  Module TheModule("TestModule", Context);
  const char *AlignmentStrings[] = {"Fi8", "Fn8"};

  for (const char *AlignmentString : AlignmentStrings) {
    TheModule.setDataLayout(AlignmentString);
    ASSERT_FALSE(foldFuncPtrAndConstToNull(Context, &TheModule, 2));
  }
}

TEST(ConstantsTest, FoldFunctionAlign4PtrAlignMultiple) {
  LLVMContext Context;
  Module TheModule("TestModule", Context);
  TheModule.setDataLayout("Fn8");
  ASSERT_TRUE(foldFuncPtrAndConstToNull(Context, &TheModule, 2, Align(4)));
}

TEST(ConstantsTest, DontFoldFunctionAlign4PtrAlignIndependent) {
  LLVMContext Context;
  Module TheModule("TestModule", Context);
  TheModule.setDataLayout("Fi8");
  ASSERT_FALSE(foldFuncPtrAndConstToNull(Context, &TheModule, 2, Align(4)));
}

TEST(ConstantsTest, DontFoldFunctionPtrIfNoModule) {
  LLVMContext Context;
  // Even though the function is explicitly 4 byte aligned, in the absence of a
  // DataLayout we can't assume that the function pointer is aligned.
  ASSERT_FALSE(foldFuncPtrAndConstToNull(Context, nullptr, 2, Align(4)));
}

TEST(ConstantsTest, FoldGlobalVariablePtr) {
  LLVMContext Context;

  IntegerType *IntType(Type::getInt32Ty(Context));

  std::unique_ptr<GlobalVariable> Global(
      new GlobalVariable(IntType, true, GlobalValue::ExternalLinkage));

  Global->setAlignment(Align(4));

  ConstantInt *TheConstant(ConstantInt::get(IntType, 2));

  Constant *TheConstantExpr(ConstantExpr::getPtrToInt(Global.get(), IntType));

  ASSERT_TRUE(ConstantExpr::get(Instruction::And, TheConstantExpr, TheConstant)
                  ->isNullValue());
}

// Check that containsUndefOrPoisonElement and containsPoisonElement is working
// great

TEST(ConstantsTest, containsUndefElemTest) {
  LLVMContext Context;

  Type *Int32Ty = Type::getInt32Ty(Context);
  Constant *CU = UndefValue::get(Int32Ty);
  Constant *CP = PoisonValue::get(Int32Ty);
  Constant *C1 = ConstantInt::get(Int32Ty, 1);
  Constant *C2 = ConstantInt::get(Int32Ty, 2);

  {
    Constant *V1 = ConstantVector::get({C1, C2});
    EXPECT_FALSE(V1->containsUndefOrPoisonElement());
    EXPECT_FALSE(V1->containsPoisonElement());
  }

  {
    Constant *V2 = ConstantVector::get({C1, CU});
    EXPECT_TRUE(V2->containsUndefOrPoisonElement());
    EXPECT_FALSE(V2->containsPoisonElement());
  }

  {
    Constant *V3 = ConstantVector::get({C1, CP});
    EXPECT_TRUE(V3->containsUndefOrPoisonElement());
    EXPECT_TRUE(V3->containsPoisonElement());
  }

  {
    Constant *V4 = ConstantVector::get({CU, CP});
    EXPECT_TRUE(V4->containsUndefOrPoisonElement());
    EXPECT_TRUE(V4->containsPoisonElement());
  }
}

// Check that undefined elements in vector constants are matched
// correctly for both integer and floating-point types. Just don't
// crash on vectors of pointers (could be handled?).

TEST(ConstantsTest, isElementWiseEqual) {
  LLVMContext Context;

  Type *Int32Ty = Type::getInt32Ty(Context);
  Constant *CU = UndefValue::get(Int32Ty);
  Constant *C1 = ConstantInt::get(Int32Ty, 1);
  Constant *C2 = ConstantInt::get(Int32Ty, 2);

  Constant *C1211 = ConstantVector::get({C1, C2, C1, C1});
  Constant *C12U1 = ConstantVector::get({C1, C2, CU, C1});
  Constant *C12U2 = ConstantVector::get({C1, C2, CU, C2});
  Constant *C12U21 = ConstantVector::get({C1, C2, CU, C2, C1});

  EXPECT_TRUE(C1211->isElementWiseEqual(C12U1));
  EXPECT_TRUE(C12U1->isElementWiseEqual(C1211));
  EXPECT_FALSE(C12U2->isElementWiseEqual(C12U1));
  EXPECT_FALSE(C12U1->isElementWiseEqual(C12U2));
  EXPECT_FALSE(C12U21->isElementWiseEqual(C12U2));

  Type *FltTy = Type::getFloatTy(Context);
  Constant *CFU = UndefValue::get(FltTy);
  Constant *CF1 = ConstantFP::get(FltTy, 1.0);
  Constant *CF2 = ConstantFP::get(FltTy, 2.0);

  Constant *CF1211 = ConstantVector::get({CF1, CF2, CF1, CF1});
  Constant *CF12U1 = ConstantVector::get({CF1, CF2, CFU, CF1});
  Constant *CF12U2 = ConstantVector::get({CF1, CF2, CFU, CF2});
  Constant *CFUU1U = ConstantVector::get({CFU, CFU, CF1, CFU});

  EXPECT_TRUE(CF1211->isElementWiseEqual(CF12U1));
  EXPECT_TRUE(CF12U1->isElementWiseEqual(CF1211));
  EXPECT_TRUE(CFUU1U->isElementWiseEqual(CF12U1));
  EXPECT_FALSE(CF12U2->isElementWiseEqual(CF12U1));
  EXPECT_FALSE(CF12U1->isElementWiseEqual(CF12U2));

  PointerType *PtrTy = Type::getInt8PtrTy(Context);
  Constant *CPU = UndefValue::get(PtrTy);
  Constant *CP0 = ConstantPointerNull::get(PtrTy);

  Constant *CP0000 = ConstantVector::get({CP0, CP0, CP0, CP0});
  Constant *CP00U0 = ConstantVector::get({CP0, CP0, CPU, CP0});
  Constant *CP00U = ConstantVector::get({CP0, CP0, CPU});

  EXPECT_FALSE(CP0000->isElementWiseEqual(CP00U0));
  EXPECT_FALSE(CP00U0->isElementWiseEqual(CP0000));
  EXPECT_FALSE(CP0000->isElementWiseEqual(CP00U));
  EXPECT_FALSE(CP00U->isElementWiseEqual(CP00U0));
}

// Check that vector/aggregate constants correctly store undef and poison
// elements.

TEST(ConstantsTest, CheckElementWiseUndefPoison) {
  LLVMContext Context;

  Type *Int32Ty = Type::getInt32Ty(Context);
  StructType *STy = StructType::get(Int32Ty, Int32Ty);
  ArrayType *ATy = ArrayType::get(Int32Ty, 2);
  Constant *CU = UndefValue::get(Int32Ty);
  Constant *CP = PoisonValue::get(Int32Ty);

  {
    Constant *CUU = ConstantVector::get({CU, CU});
    Constant *CPP = ConstantVector::get({CP, CP});
    Constant *CUP = ConstantVector::get({CU, CP});
    Constant *CPU = ConstantVector::get({CP, CU});
    EXPECT_EQ(CUU, UndefValue::get(CUU->getType()));
    EXPECT_EQ(CPP, PoisonValue::get(CPP->getType()));
    EXPECT_NE(CUP, UndefValue::get(CUP->getType()));
    EXPECT_NE(CPU, UndefValue::get(CPU->getType()));
  }

  {
    Constant *CUU = ConstantStruct::get(STy, {CU, CU});
    Constant *CPP = ConstantStruct::get(STy, {CP, CP});
    Constant *CUP = ConstantStruct::get(STy, {CU, CP});
    Constant *CPU = ConstantStruct::get(STy, {CP, CU});
    EXPECT_EQ(CUU, UndefValue::get(CUU->getType()));
    EXPECT_EQ(CPP, PoisonValue::get(CPP->getType()));
    EXPECT_NE(CUP, UndefValue::get(CUP->getType()));
    EXPECT_NE(CPU, UndefValue::get(CPU->getType()));
  }

  {
    Constant *CUU = ConstantArray::get(ATy, {CU, CU});
    Constant *CPP = ConstantArray::get(ATy, {CP, CP});
    Constant *CUP = ConstantArray::get(ATy, {CU, CP});
    Constant *CPU = ConstantArray::get(ATy, {CP, CU});
    EXPECT_EQ(CUU, UndefValue::get(CUU->getType()));
    EXPECT_EQ(CPP, PoisonValue::get(CPP->getType()));
    EXPECT_NE(CUP, UndefValue::get(CUP->getType()));
    EXPECT_NE(CPU, UndefValue::get(CPU->getType()));
  }
}

TEST(ConstantsTest, GetSplatValueRoundTrip) {
  LLVMContext Context;

  Type *FloatTy = Type::getFloatTy(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int8Ty = Type::getInt8Ty(Context);

  for (unsigned Min : {1, 2, 8}) {
    auto ScalableEC = ElementCount::getScalable(Min);
    auto FixedEC = ElementCount::getFixed(Min);

    for (auto EC : {ScalableEC, FixedEC}) {
      for (auto *Ty : {FloatTy, Int32Ty, Int8Ty}) {
        Constant *Zero = Constant::getNullValue(Ty);
        Constant *One = Constant::getAllOnesValue(Ty);

        for (auto *C : {Zero, One}) {
          Constant *Splat = ConstantVector::getSplat(EC, C);
          ASSERT_NE(nullptr, Splat);

          Constant *SplatVal = Splat->getSplatValue();
          EXPECT_NE(nullptr, SplatVal);
          EXPECT_EQ(SplatVal, C);
        }
      }
    }
  }
}

TEST(ConstantsTest, ComdatUserTracking) {
  LLVMContext Context;
  Module M("MyModule", Context);

  Comdat *C = M.getOrInsertComdat("comdat");
  const SmallPtrSetImpl<GlobalObject *> &Users = C->getUsers();
  EXPECT_TRUE(Users.size() == 0);

  Type *Ty = Type::getInt8Ty(Context);
  GlobalVariable *GV1 = cast<GlobalVariable>(M.getOrInsertGlobal("gv1", Ty));
  GV1->setComdat(C);
  EXPECT_TRUE(Users.size() == 1);
  EXPECT_TRUE(Users.contains(GV1));

  GlobalVariable *GV2 = cast<GlobalVariable>(M.getOrInsertGlobal("gv2", Ty));
  GV2->setComdat(C);
  EXPECT_TRUE(Users.size() == 2);
  EXPECT_TRUE(Users.contains(GV2));

  GV1->eraseFromParent();
  EXPECT_TRUE(Users.size() == 1);
  EXPECT_TRUE(Users.contains(GV2));

  GV2->eraseFromParent();
  EXPECT_TRUE(Users.size() == 0);
}

} // end anonymous namespace
} // end namespace llvm
