//===- llvm/unittest/IR/ConstantsTest.cpp - Constants unit tests ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm-c/Core.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

TEST(ConstantsTest, Integer_i1) {
  IntegerType* Int1 = IntegerType::get(getGlobalContext(), 1);
  Constant* One = ConstantInt::get(Int1, 1, true);
  Constant* Zero = ConstantInt::get(Int1, 0);
  Constant* NegOne = ConstantInt::get(Int1, static_cast<uint64_t>(-1), true);
  EXPECT_EQ(NegOne, ConstantInt::getSigned(Int1, -1));
  Constant* Undef = UndefValue::get(Int1);

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

  // @h = constant i1 shl(i1 1 , i1 1)  ; undefined
  // @h = constant i1 undef
  EXPECT_EQ(Undef, ConstantExpr::getShl(One, One));

  // @i = constant i1 shl(i1 1 , i1 0)
  // @i = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getShl(One, Zero));

  // @j = constant i1 lshr(i1 1, i1 1)  ; undefined
  // @j = constant i1 undef
  EXPECT_EQ(Undef, ConstantExpr::getLShr(One, One));

  // @m = constant i1 ashr(i1 1, i1 1)  ; undefined
  // @m = constant i1 undef
  EXPECT_EQ(Undef, ConstantExpr::getAShr(One, One));

  // @n = constant i1 mul(i1 -1, i1 1)
  // @n = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getMul(NegOne, One));

  // @o = constant i1 sdiv(i1 -1, i1 1) ; overflow
  // @o = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getSDiv(NegOne, One));

  // @p = constant i1 sdiv(i1 1 , i1 -1); overflow
  // @p = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getSDiv(One, NegOne));

  // @q = constant i1 udiv(i1 -1, i1 1)
  // @q = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getUDiv(NegOne, One));

  // @r = constant i1 udiv(i1 1, i1 -1)
  // @r = constant i1 true
  EXPECT_EQ(One, ConstantExpr::getUDiv(One, NegOne));

  // @s = constant i1 srem(i1 -1, i1 1) ; overflow
  // @s = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getSRem(NegOne, One));

  // @t = constant i1 urem(i1 -1, i1 1)
  // @t = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getURem(NegOne, One));

  // @u = constant i1 srem(i1  1, i1 -1) ; overflow
  // @u = constant i1 false
  EXPECT_EQ(Zero, ConstantExpr::getSRem(One, NegOne));
}

TEST(ConstantsTest, IntSigns) {
  IntegerType* Int8Ty = Type::getInt8Ty(getGlobalContext());
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
  Type *FP128Ty = Type::getFP128Ty(getGlobalContext());

  IntegerType *Int128Ty = Type::getIntNTy(getGlobalContext(), 128);
  Constant *Zero128 = Constant::getNullValue(Int128Ty);
  Constant *X = ConstantExpr::getUIToFP(Zero128, FP128Ty);
  EXPECT_TRUE(isa<ConstantFP>(X));
}

TEST(ConstantsTest, PointerCast) {
  LLVMContext &C(getGlobalContext());
  Type *Int8PtrTy = Type::getInt8PtrTy(C);
  Type *Int32PtrTy = Type::getInt32PtrTy(C);
  Type *Int64Ty = Type::getInt64Ty(C);
  VectorType *Int8PtrVecTy = VectorType::get(Int8PtrTy, 4);
  VectorType *Int32PtrVecTy = VectorType::get(Int32PtrTy, 4);
  VectorType *Int64VecTy = VectorType::get(Int64Ty, 4);

  // ptrtoint i8* to i64
  EXPECT_EQ(Constant::getNullValue(Int64Ty),
            ConstantExpr::getPointerCast(
              Constant::getNullValue(Int8PtrTy), Int64Ty));

  // bitcast i8* to i32*
  EXPECT_EQ(Constant::getNullValue(Int32PtrTy),
            ConstantExpr::getPointerCast(
              Constant::getNullValue(Int8PtrTy), Int32PtrTy));

  // ptrtoint <4 x i8*> to <4 x i64>
  EXPECT_EQ(Constant::getNullValue(Int64VecTy),
            ConstantExpr::getPointerCast(
              Constant::getNullValue(Int8PtrVecTy), Int64VecTy));

  // bitcast <4 x i8*> to <4 x i32*>
  EXPECT_EQ(Constant::getNullValue(Int32PtrVecTy),
            ConstantExpr::getPointerCast(
              Constant::getNullValue(Int8PtrVecTy), Int32PtrVecTy));
}

#define CHECK(x, y) {                                         		\
    std::string __s;                                            	\
    raw_string_ostream __o(__s);                                	\
    Instruction *__I = cast<ConstantExpr>(x)->getAsInstruction();	\
    __I->print(__o);      						\
    delete __I; 							\
    __o.flush();                                                	\
    EXPECT_EQ(std::string("  <badref> = " y), __s);             	\
  }

TEST(ConstantsTest, AsInstructionsTest) {
  std::unique_ptr<Module> M(new Module("MyModule", getGlobalContext()));

  Type *Int64Ty = Type::getInt64Ty(getGlobalContext());
  Type *Int32Ty = Type::getInt32Ty(getGlobalContext());
  Type *Int16Ty = Type::getInt16Ty(getGlobalContext());
  Type *Int1Ty = Type::getInt1Ty(getGlobalContext());
  Type *FloatTy = Type::getFloatTy(getGlobalContext());
  Type *DoubleTy = Type::getDoubleTy(getGlobalContext());

  Constant *Global = M->getOrInsertGlobal("dummy",
                                         PointerType::getUnqual(Int32Ty));
  Constant *Global2 = M->getOrInsertGlobal("dummy2",
                                         PointerType::getUnqual(Int32Ty));

  Constant *P0 = ConstantExpr::getPtrToInt(Global, Int32Ty);
  Constant *P1 = ConstantExpr::getUIToFP(P0, FloatTy);
  Constant *P2 = ConstantExpr::getUIToFP(P0, DoubleTy);
  Constant *P3 = ConstantExpr::getTrunc(P0, Int1Ty);
  Constant *P4 = ConstantExpr::getPtrToInt(Global2, Int32Ty);
  Constant *P5 = ConstantExpr::getUIToFP(P4, FloatTy);
  Constant *P6 = ConstantExpr::getBitCast(P4, VectorType::get(Int16Ty, 2));

  Constant *One = ConstantInt::get(Int32Ty, 1);
  Constant *Two = ConstantInt::get(Int64Ty, 2);
  Constant *Big = ConstantInt::get(getGlobalContext(),
                                   APInt{256, uint64_t(-1), true});
  Constant *Elt = ConstantInt::get(Int16Ty, 2015);
  Constant *Undef16  = UndefValue::get(Int16Ty);
  Constant *Undef64  = UndefValue::get(Int64Ty);
  Constant *UndefV16 = UndefValue::get(P6->getType());

  #define P0STR "ptrtoint (i32** @dummy to i32)"
  #define P1STR "uitofp (i32 ptrtoint (i32** @dummy to i32) to float)"
  #define P2STR "uitofp (i32 ptrtoint (i32** @dummy to i32) to double)"
  #define P3STR "ptrtoint (i32** @dummy to i1)"
  #define P4STR "ptrtoint (i32** @dummy2 to i32)"
  #define P5STR "uitofp (i32 ptrtoint (i32** @dummy2 to i32) to float)"
  #define P6STR "bitcast (i32 ptrtoint (i32** @dummy2 to i32) to <2 x i16>)"

  CHECK(ConstantExpr::getNeg(P0), "sub i32 0, " P0STR);
  CHECK(ConstantExpr::getFNeg(P1), "fsub float -0.000000e+00, " P1STR);
  CHECK(ConstantExpr::getNot(P0), "xor i32 " P0STR ", -1");
  CHECK(ConstantExpr::getAdd(P0, P0), "add i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAdd(P0, P0, false, true), "add nsw i32 " P0STR ", "
        P0STR);
  CHECK(ConstantExpr::getAdd(P0, P0, true, true), "add nuw nsw i32 " P0STR ", "
        P0STR);
  CHECK(ConstantExpr::getFAdd(P1, P1), "fadd float " P1STR ", " P1STR);
  CHECK(ConstantExpr::getSub(P0, P0), "sub i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getFSub(P1, P1), "fsub float " P1STR ", " P1STR);
  CHECK(ConstantExpr::getMul(P0, P0), "mul i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getFMul(P1, P1), "fmul float " P1STR ", " P1STR);
  CHECK(ConstantExpr::getUDiv(P0, P0), "udiv i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getSDiv(P0, P0), "sdiv i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getFDiv(P1, P1), "fdiv float " P1STR ", " P1STR);
  CHECK(ConstantExpr::getURem(P0, P0), "urem i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getSRem(P0, P0), "srem i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getFRem(P1, P1), "frem float " P1STR ", " P1STR);
  CHECK(ConstantExpr::getAnd(P0, P0), "and i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getOr(P0, P0), "or i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getXor(P0, P0), "xor i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getShl(P0, P0), "shl i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getShl(P0, P0, true), "shl nuw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getShl(P0, P0, false, true), "shl nsw i32 " P0STR ", "
        P0STR);
  CHECK(ConstantExpr::getLShr(P0, P0, false), "lshr i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getLShr(P0, P0, true), "lshr exact i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAShr(P0, P0, false), "ashr i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAShr(P0, P0, true), "ashr exact i32 " P0STR ", " P0STR);

  CHECK(ConstantExpr::getSExt(P0, Int64Ty), "sext i32 " P0STR " to i64");
  CHECK(ConstantExpr::getZExt(P0, Int64Ty), "zext i32 " P0STR " to i64");
  CHECK(ConstantExpr::getFPTrunc(P2, FloatTy), "fptrunc double " P2STR
        " to float");
  CHECK(ConstantExpr::getFPExtend(P1, DoubleTy), "fpext float " P1STR
        " to double");

  CHECK(ConstantExpr::getExactUDiv(P0, P0), "udiv exact i32 " P0STR ", " P0STR);

  CHECK(ConstantExpr::getSelect(P3, P0, P4), "select i1 " P3STR ", i32 " P0STR
        ", i32 " P4STR);
  CHECK(ConstantExpr::getICmp(CmpInst::ICMP_EQ, P0, P4), "icmp eq i32 " P0STR
        ", " P4STR);
  CHECK(ConstantExpr::getFCmp(CmpInst::FCMP_ULT, P1, P5), "fcmp ult float "
        P1STR ", " P5STR);

  std::vector<Constant*> V;
  V.push_back(One);
  // FIXME: getGetElementPtr() actually creates an inbounds ConstantGEP,
  //        not a normal one!
  //CHECK(ConstantExpr::getGetElementPtr(Global, V, false),
  //      "getelementptr i32*, i32** @dummy, i32 1");
  CHECK(ConstantExpr::getInBoundsGetElementPtr(PointerType::getUnqual(Int32Ty),
                                               Global, V),
        "getelementptr inbounds i32*, i32** @dummy, i32 1");

  CHECK(ConstantExpr::getExtractElement(P6, One), "extractelement <2 x i16> "
        P6STR ", i32 1");

  EXPECT_EQ(Undef16, ConstantExpr::getExtractElement(P6, Two));
  EXPECT_EQ(Undef16, ConstantExpr::getExtractElement(P6, Big));
  EXPECT_EQ(Undef16, ConstantExpr::getExtractElement(P6, Undef64));

  EXPECT_EQ(Elt, ConstantExpr::getExtractElement(
                 ConstantExpr::getInsertElement(P6, Elt, One), One));
  EXPECT_EQ(UndefV16, ConstantExpr::getInsertElement(P6, Elt, Two));
  EXPECT_EQ(UndefV16, ConstantExpr::getInsertElement(P6, Elt, Big));
  EXPECT_EQ(UndefV16, ConstantExpr::getInsertElement(P6, Elt, Undef64));
}

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
TEST(ConstantsTest, ReplaceWithConstantTest) {
  std::unique_ptr<Module> M(new Module("MyModule", getGlobalContext()));

  Type *Int32Ty = Type::getInt32Ty(getGlobalContext());
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

  auto *Global = new GlobalVariable(*M, PtrTy, false,
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
      LLVMAddAlias(wrap(M.get()), wrap(I16PTy), wrap(Aliasee), "a");
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
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  for (Type *T : {Type::getInt8Ty(Context), Type::getInt16Ty(Context),
                  Type::getInt32Ty(Context), Type::getInt64Ty(Context)}) {
    ArrayType *ArrayTy = ArrayType::get(T, 2);
    Constant *Vals[] = {ConstantInt::get(T, 0), ConstantInt::get(T, 1)};
    Constant *CDV = ConstantArray::get(ArrayTy, Vals);
    ASSERT_TRUE(dyn_cast<ConstantDataArray>(CDV) != nullptr)
        << " T = " << getNameOfType(T);
  }

  for (Type *T : {Type::getHalfTy(Context), Type::getFloatTy(Context),
                  Type::getDoubleTy(Context)}) {
    ArrayType *ArrayTy = ArrayType::get(T, 2);
    Constant *Vals[] = {ConstantFP::get(T, 0), ConstantFP::get(T, 1)};
    Constant *CDV = ConstantArray::get(ArrayTy, Vals);
    ASSERT_TRUE(dyn_cast<ConstantDataArray>(CDV) != nullptr)
        << " T = " << getNameOfType(T);
  }
}

TEST(ConstantsTest, BuildConstantDataVectors) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  for (Type *T : {Type::getInt8Ty(Context), Type::getInt16Ty(Context),
                  Type::getInt32Ty(Context), Type::getInt64Ty(Context)}) {
    Constant *Vals[] = {ConstantInt::get(T, 0), ConstantInt::get(T, 1)};
    Constant *CDV = ConstantVector::get(Vals);
    ASSERT_TRUE(dyn_cast<ConstantDataVector>(CDV) != nullptr)
        << " T = " << getNameOfType(T);
  }

  for (Type *T : {Type::getHalfTy(Context), Type::getFloatTy(Context),
                  Type::getDoubleTy(Context)}) {
    Constant *Vals[] = {ConstantFP::get(T, 0), ConstantFP::get(T, 1)};
    Constant *CDV = ConstantVector::get(Vals);
    ASSERT_TRUE(dyn_cast<ConstantDataVector>(CDV) != nullptr)
        << " T = " << getNameOfType(T);
  }
}

TEST(ConstantsTest, BitcastToGEP) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  auto *i32 = Type::getInt32Ty(Context);
  auto *U = StructType::create(Context, "Unsized");
  Type *EltTys[] = {i32, U};
  auto *S = StructType::create(EltTys);

  auto *G = new GlobalVariable(*M, S, false,
                               GlobalValue::ExternalLinkage, nullptr);
  auto *PtrTy = PointerType::get(i32, 0);
  auto *C = ConstantExpr::getBitCast(G, PtrTy);
  ASSERT_EQ(dyn_cast<ConstantExpr>(C)->getOpcode(),
            Instruction::BitCast);
}

}  // end anonymous namespace
}  // end namespace llvm
