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
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

// Check that use count checks treat ConstantData like they have no uses.
TEST(ConstantsTest, UseCounts) {
  LLVMContext Context;
  Type *Int32Ty = Type::getInt32Ty(Context);
  Constant *Zero = ConstantInt::get(Int32Ty, 0);

  EXPECT_TRUE(Zero->use_empty());
  EXPECT_EQ(Zero->getNumUses(), 0u);
  EXPECT_TRUE(Zero->hasNUses(0));
  EXPECT_FALSE(Zero->hasOneUse());
  EXPECT_FALSE(Zero->hasOneUser());
  EXPECT_FALSE(Zero->hasNUses(1));
  EXPECT_FALSE(Zero->hasNUsesOrMore(1));
  EXPECT_FALSE(Zero->hasNUses(2));
  EXPECT_FALSE(Zero->hasNUsesOrMore(2));

  std::unique_ptr<Module> M(new Module("MyModule", Context));

  // Introduce some uses
  new GlobalVariable(*M, Int32Ty, /*isConstant=*/false,
                     GlobalValue::ExternalLinkage, /*Initializer=*/Zero,
                     "gv_user0");
  new GlobalVariable(*M, Int32Ty, /*isConstant=*/false,
                     GlobalValue::ExternalLinkage, /*Initializer=*/Zero,
                     "gv_user1");

  // Still looks like use_empty with uses.
  EXPECT_TRUE(Zero->use_empty());
  EXPECT_EQ(Zero->getNumUses(), 0u);
  EXPECT_TRUE(Zero->hasNUses(0));
  EXPECT_FALSE(Zero->hasOneUse());
  EXPECT_FALSE(Zero->hasOneUser());
  EXPECT_FALSE(Zero->hasNUses(1));
  EXPECT_FALSE(Zero->hasNUsesOrMore(1));
  EXPECT_FALSE(Zero->hasNUses(2));
  EXPECT_FALSE(Zero->hasNUsesOrMore(2));
}

TEST(ConstantsTest, Integer_i1) {
  LLVMContext Context;
  IntegerType *Int1 = IntegerType::get(Context, 1);
  Constant *One = ConstantInt::get(Int1, 1);
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
  EXPECT_EQ(Poison, ConstantFoldBinaryInstruction(Instruction::Shl, One, One));

  // @i = constant i1 shl(i1 1 , i1 0)
  // @i = constant i1 true
  EXPECT_EQ(One, ConstantFoldBinaryInstruction(Instruction::Shl, One, Zero));

  // @n = constant i1 mul(i1 -1, i1 1)
  // @n = constant i1 true
  EXPECT_EQ(One, ConstantFoldBinaryInstruction(Instruction::Mul, NegOne, One));

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
  EXPECT_EQ(0x3b, ConstantInt::get(Int8Ty, 0x13b, /*IsSigned=*/false,
                                   /*ImplicitTrunc=*/true)
                      ->getSExtValue());
}

TEST(ConstantsTest, PointerCast) {
  LLVMContext C;
  Type *PtrTy = PointerType::get(C, 0);
  Type *Int64Ty = Type::getInt64Ty(C);
  VectorType *PtrVecTy = FixedVectorType::get(PtrTy, 4);
  VectorType *Int64VecTy = FixedVectorType::get(Int64Ty, 4);
  VectorType *PtrScalableVecTy = ScalableVectorType::get(PtrTy, 4);
  VectorType *Int64ScalableVecTy = ScalableVectorType::get(Int64Ty, 4);

  // ptrtoint ptr to i64
  EXPECT_EQ(
      Constant::getNullValue(Int64Ty),
      ConstantExpr::getPointerCast(Constant::getNullValue(PtrTy), Int64Ty));

  // bitcast ptr to ptr
  EXPECT_EQ(Constant::getNullValue(PtrTy),
            ConstantExpr::getPointerCast(Constant::getNullValue(PtrTy), PtrTy));

  // ptrtoint <4 x ptr> to <4 x i64>
  EXPECT_EQ(Constant::getNullValue(Int64VecTy),
            ConstantExpr::getPointerCast(Constant::getNullValue(PtrVecTy),
                                         Int64VecTy));

  // ptrtoint <vscale x 4 x ptr> to <vscale x 4 x i64>
  EXPECT_EQ(Constant::getNullValue(Int64ScalableVecTy),
            ConstantExpr::getPointerCast(
                Constant::getNullValue(PtrScalableVecTy), Int64ScalableVecTy));

  // bitcast <4 x ptr> to <4 x ptr>
  EXPECT_EQ(
      Constant::getNullValue(PtrVecTy),
      ConstantExpr::getPointerCast(Constant::getNullValue(PtrVecTy), PtrVecTy));

  // bitcast <vscale x 4 x ptr> to <vscale x 4 x ptr>
  EXPECT_EQ(Constant::getNullValue(PtrScalableVecTy),
            ConstantExpr::getPointerCast(
                Constant::getNullValue(PtrScalableVecTy), PtrScalableVecTy));

  Type *Ptr1Ty = PointerType::get(C, 1);
  ConstantInt *K = ConstantInt::get(Type::getInt64Ty(C), 1234);

  // Make sure that addrspacecast of inttoptr is not folded away.
  EXPECT_NE(K, ConstantExpr::getAddrSpaceCast(
                   ConstantExpr::getIntToPtr(K, PtrTy), Ptr1Ty));
  EXPECT_NE(K, ConstantExpr::getAddrSpaceCast(
                   ConstantExpr::getIntToPtr(K, Ptr1Ty), PtrTy));

  Constant *NullPtr0 = Constant::getNullValue(PtrTy);
  Constant *NullPtr1 = Constant::getNullValue(Ptr1Ty);

  // Make sure that addrspacecast of null is not folded away.
  EXPECT_NE(Constant::getNullValue(PtrTy),
            ConstantExpr::getAddrSpaceCast(NullPtr0, Ptr1Ty));

  EXPECT_NE(Constant::getNullValue(Ptr1Ty),
            ConstantExpr::getAddrSpaceCast(NullPtr1, PtrTy));
}

#define CHECK(x, y)                                                            \
  {                                                                            \
    std::string __s;                                                           \
    raw_string_ostream __o(__s);                                               \
    Instruction *__I = cast<ConstantExpr>(x)->getAsInstruction();              \
    __I->print(__o);                                                           \
    __I->deleteValue();                                                        \
    EXPECT_EQ(std::string("  <badref> = " y), __s);                            \
  }

TEST(ConstantsTest, AsInstructionsTest) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  Type *Int64Ty = Type::getInt64Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *Int16Ty = Type::getInt16Ty(Context);

  Constant *Global =
      M->getOrInsertGlobal("dummy", PointerType::getUnqual(Context));
  Constant *Global2 =
      M->getOrInsertGlobal("dummy2", PointerType::getUnqual(Context));

  Constant *P0 = ConstantExpr::getPtrToInt(Global, Int32Ty);
  Constant *P4 = ConstantExpr::getPtrToInt(Global2, Int32Ty);
  Constant *P6 = ConstantExpr::getBitCast(P4, FixedVectorType::get(Int16Ty, 2));

  Constant *One = ConstantInt::get(Int32Ty, 1);
  Constant *Two = ConstantInt::get(Int64Ty, 2);
  Constant *Big = ConstantInt::get(Context, APInt{256, uint64_t(-1), true});
  Constant *Elt = ConstantInt::get(Int16Ty, 2015);
  Constant *Poison16 = PoisonValue::get(Int16Ty);
  Constant *Undef64 = UndefValue::get(Int64Ty);
  Constant *PoisonV16 = PoisonValue::get(P6->getType());

#define P0STR "ptrtoint (ptr @dummy to i32)"
#define P3STR "ptrtoint (ptr @dummy to i1)"
#define P4STR "ptrtoint (ptr @dummy2 to i32)"
#define P6STR "bitcast (i32 ptrtoint (ptr @dummy2 to i32) to <2 x i16>)"

  CHECK(ConstantExpr::getNeg(P0), "sub i32 0, " P0STR);
  CHECK(ConstantExpr::getNot(P0), "xor i32 " P0STR ", -1");
  CHECK(ConstantExpr::getAdd(P0, P0), "add i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAdd(P0, P0, false, true),
        "add nsw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getAdd(P0, P0, true, true),
        "add nuw nsw i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getSub(P0, P0), "sub i32 " P0STR ", " P0STR);
  CHECK(ConstantExpr::getXor(P0, P0), "xor i32 " P0STR ", " P0STR);

  std::vector<Constant *> V;
  V.push_back(One);
  // FIXME: getGetElementPtr() actually creates an inbounds ConstantGEP,
  //        not a normal one!
  // CHECK(ConstantExpr::getGetElementPtr(Global, V, false),
  //      "getelementptr ptr, ptr @dummy, i32 1");
  CHECK(ConstantExpr::getInBoundsGetElementPtr(PointerType::getUnqual(Context),
                                               Global, V),
        "getelementptr inbounds ptr, ptr @dummy, i32 1");

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
      M->getOrInsertGlobal("dummy", PointerType::getUnqual(Context));
  Constant *GEP = ConstantExpr::getGetElementPtr(
      PointerType::getUnqual(Context), Global, One);
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
  Type *PtrTy = PointerType::get(Context, 0);
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
  Type *I16PTy = PointerType::get(Context, 0);
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

TEST(ConstantsTest, BitcastToGEP) {
  LLVMContext Context;
  std::unique_ptr<Module> M(new Module("MyModule", Context));

  auto *i32 = Type::getInt32Ty(Context);
  auto *U = StructType::create(Context, "Unsized");
  Type *EltTys[] = {i32, U};
  auto *S = StructType::create(EltTys);

  auto *G =
      new GlobalVariable(*M, S, false, GlobalValue::ExternalLinkage, nullptr);
  auto *PtrTy = PointerType::get(Context, 0);
  auto *C = ConstantExpr::getBitCast(G, PtrTy);
  /* With opaque pointers, no cast is necessary. */
  EXPECT_EQ(C, G);
}

bool foldFuncPtrAndConstToNull(LLVMContext &Context, Module *TheModule,
                               uint64_t AndValue,
                               MaybeAlign FunctionAlign = std::nullopt) {
  Type *VoidType(Type::getVoidTy(Context));
  FunctionType *FuncType(FunctionType::get(VoidType, false));
  Function *Func(
      Function::Create(FuncType, GlobalValue::ExternalLinkage, "", TheModule));

  if (FunctionAlign)
    Func->setAlignment(*FunctionAlign);

  IntegerType *ConstantIntType(Type::getInt32Ty(Context));
  ConstantInt *TheConstant(ConstantInt::get(ConstantIntType, AndValue));

  Constant *TheConstantExpr(ConstantExpr::getPtrToInt(Func, ConstantIntType));

  Constant *C = ConstantFoldBinaryInstruction(Instruction::And, TheConstantExpr,
                                              TheConstant);
  bool Result = C && C->isNullValue();

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

  ConstantInt *TheConstant = ConstantInt::get(IntType, 2);

  Constant *PtrToInt = ConstantExpr::getPtrToInt(Global.get(), IntType);
  ASSERT_TRUE(
      ConstantFoldBinaryInstruction(Instruction::And, PtrToInt, TheConstant)
          ->isNullValue());

  Constant *PtrToAddr = ConstantExpr::getPtrToAddr(Global.get(), IntType);
  ASSERT_TRUE(
      ConstantFoldBinaryInstruction(Instruction::And, PtrToAddr, TheConstant)
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

// Check that poison elements in vector constants are matched
// correctly for both integer and floating-point types. Just don't
// crash on vectors of pointers (could be handled?).

TEST(ConstantsTest, isElementWiseEqual) {
  LLVMContext Context;

  Type *Int32Ty = Type::getInt32Ty(Context);
  Constant *CU = UndefValue::get(Int32Ty);
  Constant *CP = PoisonValue::get(Int32Ty);
  Constant *C1 = ConstantInt::get(Int32Ty, 1);
  Constant *C2 = ConstantInt::get(Int32Ty, 2);

  Constant *C1211 = ConstantVector::get({C1, C2, C1, C1});
  Constant *C12U1 = ConstantVector::get({C1, C2, CU, C1});
  Constant *C12U2 = ConstantVector::get({C1, C2, CU, C2});
  Constant *C12U21 = ConstantVector::get({C1, C2, CU, C2, C1});
  Constant *C12P1 = ConstantVector::get({C1, C2, CP, C1});
  Constant *C12P2 = ConstantVector::get({C1, C2, CP, C2});
  Constant *C12P21 = ConstantVector::get({C1, C2, CP, C2, C1});

  EXPECT_FALSE(C1211->isElementWiseEqual(C12U1));
  EXPECT_FALSE(C12U1->isElementWiseEqual(C1211));
  EXPECT_FALSE(C12U2->isElementWiseEqual(C12U1));
  EXPECT_FALSE(C12U1->isElementWiseEqual(C12U2));
  EXPECT_FALSE(C12U21->isElementWiseEqual(C12U2));

  EXPECT_TRUE(C1211->isElementWiseEqual(C12P1));
  EXPECT_TRUE(C12P1->isElementWiseEqual(C1211));
  EXPECT_FALSE(C12P2->isElementWiseEqual(C12P1));
  EXPECT_FALSE(C12P1->isElementWiseEqual(C12P2));
  EXPECT_FALSE(C12P21->isElementWiseEqual(C12P2));

  Type *FltTy = Type::getFloatTy(Context);
  Constant *CFU = UndefValue::get(FltTy);
  Constant *CFP = PoisonValue::get(FltTy);
  Constant *CF1 = ConstantFP::get(FltTy, 1.0);
  Constant *CF2 = ConstantFP::get(FltTy, 2.0);

  Constant *CF1211 = ConstantVector::get({CF1, CF2, CF1, CF1});
  Constant *CF12U1 = ConstantVector::get({CF1, CF2, CFU, CF1});
  Constant *CF12U2 = ConstantVector::get({CF1, CF2, CFU, CF2});
  Constant *CFUU1U = ConstantVector::get({CFU, CFU, CF1, CFU});
  Constant *CF12P1 = ConstantVector::get({CF1, CF2, CFP, CF1});
  Constant *CF12P2 = ConstantVector::get({CF1, CF2, CFP, CF2});
  Constant *CFPP1P = ConstantVector::get({CFP, CFP, CF1, CFP});

  EXPECT_FALSE(CF1211->isElementWiseEqual(CF12U1));
  EXPECT_FALSE(CF12U1->isElementWiseEqual(CF1211));
  EXPECT_FALSE(CFUU1U->isElementWiseEqual(CF12U1));
  EXPECT_FALSE(CF12U2->isElementWiseEqual(CF12U1));
  EXPECT_FALSE(CF12U1->isElementWiseEqual(CF12U2));

  EXPECT_TRUE(CF1211->isElementWiseEqual(CF12P1));
  EXPECT_TRUE(CF12P1->isElementWiseEqual(CF1211));
  EXPECT_TRUE(CFPP1P->isElementWiseEqual(CF12P1));
  EXPECT_FALSE(CF12P2->isElementWiseEqual(CF12P1));
  EXPECT_FALSE(CF12P1->isElementWiseEqual(CF12P2));

  PointerType *PtrTy = PointerType::get(Context, 0);
  Constant *CPU = UndefValue::get(PtrTy);
  Constant *CPP = PoisonValue::get(PtrTy);
  Constant *CP0 = ConstantPointerNull::get(PtrTy);

  Constant *CP0000 = ConstantVector::get({CP0, CP0, CP0, CP0});
  Constant *CP00U0 = ConstantVector::get({CP0, CP0, CPU, CP0});
  Constant *CP00U = ConstantVector::get({CP0, CP0, CPU});
  Constant *CP00P0 = ConstantVector::get({CP0, CP0, CPP, CP0});
  Constant *CP00P = ConstantVector::get({CP0, CP0, CPP});

  EXPECT_FALSE(CP0000->isElementWiseEqual(CP00U0));
  EXPECT_FALSE(CP00U0->isElementWiseEqual(CP0000));
  EXPECT_FALSE(CP0000->isElementWiseEqual(CP00U));
  EXPECT_FALSE(CP00U->isElementWiseEqual(CP00U0));
  EXPECT_FALSE(CP0000->isElementWiseEqual(CP00P0));
  EXPECT_FALSE(CP00P0->isElementWiseEqual(CP0000));
  EXPECT_FALSE(CP0000->isElementWiseEqual(CP00P));
  EXPECT_FALSE(CP00P->isElementWiseEqual(CP00P0));
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
  GlobalVariable *GV1 = M.getOrInsertGlobal("gv1", Ty);
  GV1->setComdat(C);
  EXPECT_TRUE(Users.size() == 1);
  EXPECT_TRUE(Users.contains(GV1));

  GlobalVariable *GV2 = M.getOrInsertGlobal("gv2", Ty);
  GV2->setComdat(C);
  EXPECT_TRUE(Users.size() == 2);
  EXPECT_TRUE(Users.contains(GV2));

  GV1->eraseFromParent();
  EXPECT_TRUE(Users.size() == 1);
  EXPECT_TRUE(Users.contains(GV2));

  GV2->eraseFromParent();
  EXPECT_TRUE(Users.size() == 0);
}

// Verify that the C API getters for BlockAddress work
TEST(ConstantsTest, BlockAddressCAPITest) {
  const char *BlockAddressIR = R"(
    define void @test_block_address_func() {
    entry:
      br label %block_bb_0
    block_bb_0:
      ret void
    }
  )";

  LLVMContext Context;
  SMDiagnostic Error;
  std::unique_ptr<Module> M =
      parseAssemblyString(BlockAddressIR, Error, Context);

  EXPECT_TRUE(M.get() != nullptr);

  // Get the function
  auto *Func = M->getFunction("test_block_address_func");
  EXPECT_TRUE(Func != nullptr);

  // Get the second basic block, since we can't use the entry one
  const BasicBlock &BB = *(++Func->begin());
  EXPECT_EQ(BB.getName(), "block_bb_0");

  // Construct the C API values
  LLVMValueRef BlockAddr = LLVMBlockAddress(wrap(Func), wrap(&BB));
  EXPECT_TRUE(LLVMIsABlockAddress(BlockAddr));

  // Get the Function/BasicBlock values back out
  auto *OutFunc = unwrap(LLVMGetBlockAddressFunction(BlockAddr));
  auto *OutBB = unwrap(LLVMGetBlockAddressBasicBlock(BlockAddr));

  // Verify that they round-tripped properly
  EXPECT_EQ(Func, OutFunc);
  EXPECT_EQ(&BB, OutBB);
}

TEST(ConstantsTest, Float128Test) {
  LLVMContextRef C = LLVMContextCreate();
  LLVMTypeRef Ty128 = LLVMFP128TypeInContext(C);
  LLVMTypeRef TyPPC128 = LLVMPPCFP128TypeInContext(C);
  LLVMTypeRef TyFloat = LLVMFloatTypeInContext(C);
  LLVMTypeRef TyDouble = LLVMDoubleTypeInContext(C);
  LLVMTypeRef TyHalf = LLVMHalfTypeInContext(C);
  LLVMBuilderRef Builder = LLVMCreateBuilderInContext(C);
  uint64_t n[2] = {0x4000000000000000, 0x0}; //+2
  uint64_t m[2] = {0xC000000000000000, 0x0}; //-2
  LLVMValueRef val1 = LLVMConstFPFromBits(Ty128, n);
  EXPECT_TRUE(val1 != nullptr);
  LLVMValueRef val2 = LLVMConstFPFromBits(Ty128, m);
  EXPECT_TRUE(val2 != nullptr);
  LLVMValueRef val3 = LLVMBuildFAdd(Builder, val1, val2, "test");
  EXPECT_TRUE(val3 != nullptr);
  LLVMValueRef val4 = LLVMConstFPFromBits(TyPPC128, n);
  EXPECT_TRUE(val4 != nullptr);
  uint64_t p[1] = {0x0000000040000000}; //+2
  LLVMValueRef val5 = LLVMConstFPFromBits(TyFloat, p);
  EXPECT_EQ(APFloat(2.0f), unwrap<ConstantFP>(val5)->getValue());
  uint64_t q[1] = {0x4000000000000000}; //+2
  LLVMValueRef val6 = LLVMConstFPFromBits(TyDouble, q);
  EXPECT_EQ(APFloat(2.0), unwrap<ConstantFP>(val6)->getValue());
  uint64_t r[1] = {0x0000000000003c00}; //+1
  LLVMValueRef val7 = LLVMConstFPFromBits(TyHalf, r);
  EXPECT_TRUE(val7 != nullptr);
  LLVMDisposeBuilder(Builder);
  LLVMContextDispose(C);
}

TEST(ConstantsTest, ZeroValueAPIs) {
  LLVMContext Context;

  // Basic types.
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  Type *PtrTy = PointerType::get(Context, 0);
  Type *Ptr1Ty = PointerType::get(Context, 1);

  // --- getZeroValue: currently returns same as getNullValue ---
  EXPECT_EQ(Constant::getZeroValue(Int32Ty), Constant::getNullValue(Int32Ty));
  EXPECT_EQ(Constant::getZeroValue(FloatTy), Constant::getNullValue(FloatTy));
  EXPECT_EQ(Constant::getZeroValue(PtrTy), Constant::getNullValue(PtrTy));
  EXPECT_EQ(Constant::getZeroValue(Ptr1Ty), Constant::getNullValue(Ptr1Ty));

  // Aggregate types.
  StructType *StructTy = StructType::get(Int32Ty, PtrTy);
  ArrayType *ArrayTy = ArrayType::get(Int32Ty, 4);
  EXPECT_EQ(Constant::getZeroValue(StructTy), Constant::getNullValue(StructTy));
  EXPECT_EQ(Constant::getZeroValue(ArrayTy), Constant::getNullValue(ArrayTy));

  // --- isZeroValue(nullptr): identity check against getZeroValue ---
  Constant *IntZero = ConstantInt::get(Int32Ty, 0);
  Constant *IntOne = ConstantInt::get(Int32Ty, 1);
  Constant *FPZero = ConstantFP::get(FloatTy, 0.0);
  Constant *FPNegZero = ConstantFP::get(FloatTy, -0.0);
  Constant *FPOne = ConstantFP::get(FloatTy, 1.0);
  Constant *PtrNull0 = ConstantPointerNull::get(cast<PointerType>(PtrTy));
  Constant *PtrNull1 = ConstantPointerNull::get(cast<PointerType>(Ptr1Ty));
  Constant *CAZ = ConstantAggregateZero::get(StructTy);

  EXPECT_TRUE(IntZero->isZeroValue());
  EXPECT_FALSE(IntOne->isZeroValue());
  EXPECT_TRUE(FPZero->isZeroValue());
  // -0.0 has a non-zero bit pattern (sign bit set), so it is NOT a zero value.
  EXPECT_FALSE(FPNegZero->isZeroValue());
  EXPECT_FALSE(FPOne->isZeroValue());
  EXPECT_TRUE(PtrNull0->isZeroValue());
  EXPECT_TRUE(PtrNull1->isZeroValue());
  EXPECT_TRUE(CAZ->isZeroValue());

  // --- isZeroValue: FP corner cases ---
  // -0.0 is NOT zero (sign bit set = non-zero bit pattern).
  // Verify consistency with isNullValue: both agree +0.0 is zero, -0.0 is not.
  EXPECT_TRUE(FPZero->isNullValue());
  EXPECT_FALSE(FPNegZero->isNullValue());
  EXPECT_TRUE(FPZero->isZeroValue());
  EXPECT_FALSE(FPNegZero->isZeroValue());

  // Double precision: same behavior.
  Type *DoubleTy = Type::getDoubleTy(Context);
  Constant *DblZero = ConstantFP::get(DoubleTy, 0.0);
  Constant *DblNegZero = ConstantFP::get(DoubleTy, -0.0);
  EXPECT_TRUE(DblZero->isZeroValue());
  EXPECT_FALSE(DblNegZero->isZeroValue());

  // Vector splats of FP zeros.
  Constant *VecPosZero =
      ConstantVector::getSplat(ElementCount::getFixed(2), FPZero);
  Constant *VecNegZero =
      ConstantVector::getSplat(ElementCount::getFixed(2), FPNegZero);
  // Splat of +0.0 collapses to CAZ, which is zero.
  EXPECT_TRUE(isa<ConstantAggregateZero>(VecPosZero));
  EXPECT_TRUE(VecPosZero->isZeroValue());
  // Splat of -0.0 does NOT collapse to CAZ and is NOT zero.
  EXPECT_FALSE(isa<ConstantAggregateZero>(VecNegZero));
  EXPECT_FALSE(VecNegZero->isZeroValue());

  // --- isZeroValue(&DL) with default DataLayout (all AS have zero null) ---
  DataLayout DefaultDL("");
  EXPECT_TRUE(IntZero->isZeroValue(&DefaultDL));
  EXPECT_FALSE(IntOne->isZeroValue(&DefaultDL));
  EXPECT_TRUE(FPZero->isZeroValue(&DefaultDL));
  EXPECT_FALSE(FPNegZero->isZeroValue(&DefaultDL));
  EXPECT_FALSE(FPOne->isZeroValue(&DefaultDL));
  EXPECT_TRUE(PtrNull0->isZeroValue(&DefaultDL));
  EXPECT_TRUE(PtrNull1->isZeroValue(&DefaultDL));
  EXPECT_TRUE(CAZ->isZeroValue(&DefaultDL));

  // --- isZeroValue(&DL) with all-ones-null AS 1 ---
  // Format: p<flags><as>:<size>:<abi> -- flags before AS number.
  DataLayout AllOnesDL("po1:64:64");
  // AS 0 still has zero null, so CPN for AS 0 is still a zero value.
  EXPECT_TRUE(PtrNull0->isZeroValue(&AllOnesDL));
  // AS 1 has all-ones null, so CPN for AS 1 is NOT a zero value.
  EXPECT_FALSE(PtrNull1->isZeroValue(&AllOnesDL));
  // Non-pointer constants are unaffected by DataLayout.
  EXPECT_TRUE(IntZero->isZeroValue(&AllOnesDL));
  EXPECT_TRUE(FPZero->isZeroValue(&AllOnesDL));
  EXPECT_TRUE(CAZ->isZeroValue(&AllOnesDL));

  // --- getNullValue(Ty, nullptr): same as getNullValue(Ty) ---
  EXPECT_EQ(Constant::getNullValue(Int32Ty, nullptr),
            Constant::getNullValue(Int32Ty));
  EXPECT_EQ(Constant::getNullValue(PtrTy, nullptr),
            Constant::getNullValue(PtrTy));
  EXPECT_EQ(Constant::getNullValue(StructTy, nullptr),
            Constant::getNullValue(StructTy));

  // --- getNullValue(Ty, &DL) fast path: no non-zero-null pointers ---
  EXPECT_EQ(Constant::getNullValue(Int32Ty, &DefaultDL),
            Constant::getNullValue(Int32Ty));
  EXPECT_EQ(Constant::getNullValue(PtrTy, &DefaultDL),
            Constant::getNullValue(PtrTy));
  EXPECT_EQ(Constant::getNullValue(StructTy, &DefaultDL),
            Constant::getNullValue(StructTy));
  EXPECT_EQ(Constant::getNullValue(ArrayTy, &DefaultDL),
            Constant::getNullValue(ArrayTy));

  // With AllOnesDL, types that don't contain AS 1 pointers still take fast
  // path.
  EXPECT_EQ(Constant::getNullValue(Int32Ty, &AllOnesDL),
            Constant::getNullValue(Int32Ty));
  EXPECT_EQ(Constant::getNullValue(PtrTy, &AllOnesDL),
            Constant::getNullValue(PtrTy));
  // Struct containing AS 0 pointer -- fast path (AS 0 is zero null).
  EXPECT_EQ(Constant::getNullValue(StructTy, &AllOnesDL),
            Constant::getNullValue(StructTy));

  // TODO: getNullValue slow path for aggregates with non-zero-null pointers is
  // deferred to PR 4 testing (requires ConstantPointerNull semantic change).
}

TEST(ConstantsTest, AggregateCollapseAndCAZExtraction) {
  LLVMContext Context;
  Type *Int32Ty = Type::getInt32Ty(Context);
  Type *FloatTy = Type::getFloatTy(Context);
  PointerType *PtrTy = PointerType::get(Context, 0);

  // --- ConstantAggregateZero element extraction returns getZeroValue ---
  auto *ArrTy = ArrayType::get(Int32Ty, 3);
  auto *CAZ = ConstantAggregateZero::get(ArrTy);
  Constant *Elt = CAZ->getSequentialElement();
  EXPECT_EQ(Elt, Constant::getZeroValue(Int32Ty));
  // For pointer element types.
  auto *PtrArrTy = ArrayType::get(PtrTy, 2);
  auto *PtrCAZ = ConstantAggregateZero::get(PtrArrTy);
  Constant *PtrElt = PtrCAZ->getSequentialElement();
  EXPECT_EQ(PtrElt, Constant::getZeroValue(PtrTy));

  // Struct element extraction.
  auto *StructTy = StructType::get(Int32Ty, PtrTy, FloatTy);
  auto *StructCAZ = ConstantAggregateZero::get(StructTy);
  EXPECT_EQ(StructCAZ->getStructElement(0), Constant::getZeroValue(Int32Ty));
  EXPECT_EQ(StructCAZ->getStructElement(1), Constant::getZeroValue(PtrTy));
  EXPECT_EQ(StructCAZ->getStructElement(2), Constant::getZeroValue(FloatTy));

  // --- Zero-valued aggregates collapse to ConstantAggregateZero ---
  Constant *ZeroI32 = Constant::getZeroValue(Int32Ty);
  Constant *ZeroFloat = Constant::getZeroValue(FloatTy);
  Constant *ZeroPtr = Constant::getZeroValue(PtrTy);

  // Array of zero ints collapses.
  Constant *ZeroArr = ConstantArray::get(ArrTy, {ZeroI32, ZeroI32, ZeroI32});
  EXPECT_TRUE(isa<ConstantAggregateZero>(ZeroArr));

  // Vector of zero ints collapses.
  Constant *ZeroVec = ConstantVector::get({ZeroI32, ZeroI32, ZeroI32, ZeroI32});
  EXPECT_TRUE(isa<ConstantAggregateZero>(ZeroVec));

  // Struct of zeros collapses.
  Constant *ZeroStruct =
      ConstantStruct::get(StructTy, {ZeroI32, ZeroPtr, ZeroFloat});
  EXPECT_TRUE(isa<ConstantAggregateZero>(ZeroStruct));

  // Splat of zero collapses.
  Constant *SplatZero =
      ConstantVector::getSplat(ElementCount::getFixed(4), ZeroI32);
  EXPECT_TRUE(isa<ConstantAggregateZero>(SplatZero));

  // --- FP -0.0 does NOT collapse to ConstantAggregateZero ---
  // -0.0 has a non-zero bit pattern (sign bit set), so it must not collapse.
  Constant *NegZeroFP = ConstantFP::get(
      FloatTy, APFloat::getZero(APFloat::IEEEsingle(), /*Negative=*/true));
  EXPECT_NE(NegZeroFP, Constant::getZeroValue(FloatTy));

  auto *FloatArrTy = ArrayType::get(FloatTy, 2);
  Constant *NegZeroArr = ConstantArray::get(FloatArrTy, {NegZeroFP, NegZeroFP});
  EXPECT_FALSE(isa<ConstantAggregateZero>(NegZeroArr));

  Constant *NegZeroVec = ConstantVector::get({NegZeroFP, NegZeroFP});
  EXPECT_FALSE(isa<ConstantAggregateZero>(NegZeroVec));

  auto *FloatStructTy = StructType::get(FloatTy, FloatTy);
  Constant *NegZeroStruct =
      ConstantStruct::get(FloatStructTy, {NegZeroFP, NegZeroFP});
  EXPECT_FALSE(isa<ConstantAggregateZero>(NegZeroStruct));

  Constant *NegZeroSplat =
      ConstantVector::getSplat(ElementCount::getFixed(4), NegZeroFP);
  EXPECT_FALSE(isa<ConstantAggregateZero>(NegZeroSplat));

  // --- getSplatValue for CAZ returns getZeroValue ---
  auto *IntVecTy = FixedVectorType::get(Int32Ty, 4);
  auto *IntVecCAZ = ConstantAggregateZero::get(IntVecTy);
  Constant *SplatVal = IntVecCAZ->getSplatValue();
  EXPECT_EQ(SplatVal, Constant::getZeroValue(Int32Ty));

  auto *PtrVecTy = FixedVectorType::get(PtrTy, 2);
  auto *PtrVecCAZ = ConstantAggregateZero::get(PtrVecTy);
  Constant *PtrSplatVal = PtrVecCAZ->getSplatValue();
  EXPECT_EQ(PtrSplatVal, Constant::getZeroValue(PtrTy));
}

TEST(ConstantsTest, ConstantFoldCastWithDL) {
  LLVMContext Context;
  // A DataLayout where AS 1 has all-ones null pointer.
  DataLayout AllOnesDL("e-po1:64:64");
  // A DataLayout where all address spaces have zero null (the default).
  DataLayout DefaultDL("e-p:64:64");

  Type *Int64Ty = Type::getInt64Ty(Context);
  PointerType *PtrTy0 = PointerType::get(Context, 0);
  PointerType *PtrTy1 = PointerType::get(Context, 1);

  // --- Without DL, null pointer casts fold normally ---
  Constant *NullPtr0 = ConstantPointerNull::get(PtrTy0);
  Constant *NullPtr1 = ConstantPointerNull::get(PtrTy1);

  // ptrtoint(null AS0) -> 0 (no DL)
  Constant *Result =
      ConstantFoldCastInstruction(Instruction::PtrToInt, NullPtr0, Int64Ty);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isNullValue());

  // ptrtoint(null AS1) -> 0 (no DL, backward compat)
  Result =
      ConstantFoldCastInstruction(Instruction::PtrToInt, NullPtr1, Int64Ty);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isNullValue());

  // --- With DefaultDL, null pointer casts still fold (AS 0 is zero null) ---
  Result = ConstantFoldCastInstruction(Instruction::PtrToInt, NullPtr0, Int64Ty,
                                       &DefaultDL);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isNullValue());

  // --- With AllOnesDL, AS 1 null cast is deferred ---
  // ptrtoint(null AS1) should return nullptr (defer to DL-aware folder).
  Result = ConstantFoldCastInstruction(Instruction::PtrToInt, NullPtr1, Int64Ty,
                                       &AllOnesDL);
  EXPECT_EQ(Result, nullptr);

  // inttoptr(0, AS1) should also be deferred.
  Constant *ZeroI64 = ConstantInt::get(Int64Ty, 0);
  Result = ConstantFoldCastInstruction(Instruction::IntToPtr, ZeroI64, PtrTy1,
                                       &AllOnesDL);
  EXPECT_EQ(Result, nullptr);

  // But AS 0 with AllOnesDL still folds fine.
  Result = ConstantFoldCastInstruction(Instruction::PtrToInt, NullPtr0, Int64Ty,
                                       &AllOnesDL);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isNullValue());
}

TEST(ConstantsTest, ConstantFoldCompareWithDL) {
  LLVMContext Context;
  DataLayout AllOnesDL("e-po1:64:64");
  DataLayout DefaultDL("e-p:64:64");

  PointerType *PtrTy0 = PointerType::get(Context, 0);
  PointerType *PtrTy1 = PointerType::get(Context, 1);

  Constant *NullPtr0 = ConstantPointerNull::get(PtrTy0);
  Constant *NullPtr1 = ConstantPointerNull::get(PtrTy1);

  // Create a non-null pointer constant expression for comparison.
  Type *Int64Ty = Type::getInt64Ty(Context);
  Constant *One = ConstantInt::get(Int64Ty, 1);
  Constant *NonNullPtr0 = ConstantExpr::getIntToPtr(One, PtrTy0);
  Constant *NonNullPtr1 = ConstantExpr::getIntToPtr(One, PtrTy1);

  // --- Without DL, unsigned null comparisons fold ---
  // ptr >= null -> true (always, since null is the unsigned minimum)
  Constant *Result =
      ConstantFoldCompareInstruction(CmpInst::ICMP_UGE, NonNullPtr0, NullPtr0);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isAllOnesValue());

  // ptr < null -> false
  Result =
      ConstantFoldCompareInstruction(CmpInst::ICMP_ULT, NonNullPtr0, NullPtr0);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isNullValue());

  // --- With AllOnesDL, AS 1 unsigned null comparisons are deferred ---
  Result = ConstantFoldCompareInstruction(CmpInst::ICMP_UGE, NonNullPtr1,
                                          NullPtr1, &AllOnesDL);
  EXPECT_EQ(Result, nullptr);

  Result = ConstantFoldCompareInstruction(CmpInst::ICMP_ULT, NonNullPtr1,
                                          NullPtr1, &AllOnesDL);
  EXPECT_EQ(Result, nullptr);

  // --- With AllOnesDL, AS 0 still folds (zero null) ---
  Result = ConstantFoldCompareInstruction(CmpInst::ICMP_UGE, NonNullPtr0,
                                          NullPtr0, &AllOnesDL);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isAllOnesValue());

  // --- With DefaultDL, everything folds normally ---
  Result = ConstantFoldCompareInstruction(CmpInst::ICMP_UGE, NonNullPtr0,
                                          NullPtr0, &DefaultDL);
  ASSERT_NE(Result, nullptr);
  EXPECT_TRUE(Result->isAllOnesValue());
}

} // end anonymous namespace
} // end namespace llvm
