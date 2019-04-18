//===- llvm/unittest/IR/IRBuilderTest.cpp - IRBuilder tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Verifier.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

class IRBuilderTest : public testing::Test {
protected:
  void SetUp() override {
    M.reset(new Module("MyModule", Ctx));
    FunctionType *FTy = FunctionType::get(Type::getVoidTy(Ctx),
                                          /*isVarArg=*/false);
    F = Function::Create(FTy, Function::ExternalLinkage, "", M.get());
    BB = BasicBlock::Create(Ctx, "", F);
    GV = new GlobalVariable(*M, Type::getFloatTy(Ctx), true,
                            GlobalValue::ExternalLinkage, nullptr);
  }

  void TearDown() override {
    BB = nullptr;
    M.reset();
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *F;
  BasicBlock *BB;
  GlobalVariable *GV;
};

TEST_F(IRBuilderTest, Intrinsics) {
  IRBuilder<> Builder(BB);
  Value *V;
  Instruction *I;
  CallInst *Call;
  IntrinsicInst *II;

  V = Builder.CreateLoad(GV->getValueType(), GV);
  I = cast<Instruction>(Builder.CreateFAdd(V, V));
  I->setHasNoInfs(true);
  I->setHasNoNaNs(false);

  Call = Builder.CreateMinNum(V, V);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::minnum);

  Call = Builder.CreateMaxNum(V, V);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::maxnum);

  Call = Builder.CreateMinimum(V, V);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::minimum);

  Call = Builder.CreateMaximum(V, V);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::maximum);

  Call = Builder.CreateIntrinsic(Intrinsic::readcyclecounter, {}, {});
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::readcyclecounter);

  Call = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, V);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fabs);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Call = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, V, I);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fabs);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Call = Builder.CreateBinaryIntrinsic(Intrinsic::pow, V, V);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::pow);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Call = Builder.CreateBinaryIntrinsic(Intrinsic::pow, V, V, I);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::pow);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Call = Builder.CreateIntrinsic(Intrinsic::fma, {V->getType()}, {V, V, V});
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fma);
  EXPECT_FALSE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Call = Builder.CreateIntrinsic(Intrinsic::fma, {V->getType()}, {V, V, V}, I);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fma);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());

  Call = Builder.CreateIntrinsic(Intrinsic::fma, {V->getType()}, {V, V, V}, I);
  II = cast<IntrinsicInst>(Call);
  EXPECT_EQ(II->getIntrinsicID(), Intrinsic::fma);
  EXPECT_TRUE(II->hasNoInfs());
  EXPECT_FALSE(II->hasNoNaNs());
}

TEST_F(IRBuilderTest, Lifetime) {
  IRBuilder<> Builder(BB);
  AllocaInst *Var1 = Builder.CreateAlloca(Builder.getInt8Ty());
  AllocaInst *Var2 = Builder.CreateAlloca(Builder.getInt32Ty());
  AllocaInst *Var3 = Builder.CreateAlloca(Builder.getInt8Ty(),
                                          Builder.getInt32(123));

  CallInst *Start1 = Builder.CreateLifetimeStart(Var1);
  CallInst *Start2 = Builder.CreateLifetimeStart(Var2);
  CallInst *Start3 = Builder.CreateLifetimeStart(Var3, Builder.getInt64(100));

  EXPECT_EQ(Start1->getArgOperand(0), Builder.getInt64(-1));
  EXPECT_EQ(Start2->getArgOperand(0), Builder.getInt64(-1));
  EXPECT_EQ(Start3->getArgOperand(0), Builder.getInt64(100));

  EXPECT_EQ(Start1->getArgOperand(1), Var1);
  EXPECT_NE(Start2->getArgOperand(1), Var2);
  EXPECT_EQ(Start3->getArgOperand(1), Var3);

  Value *End1 = Builder.CreateLifetimeEnd(Var1);
  Builder.CreateLifetimeEnd(Var2);
  Builder.CreateLifetimeEnd(Var3);

  IntrinsicInst *II_Start1 = dyn_cast<IntrinsicInst>(Start1);
  IntrinsicInst *II_End1 = dyn_cast<IntrinsicInst>(End1);
  ASSERT_TRUE(II_Start1 != nullptr);
  EXPECT_EQ(II_Start1->getIntrinsicID(), Intrinsic::lifetime_start);
  ASSERT_TRUE(II_End1 != nullptr);
  EXPECT_EQ(II_End1->getIntrinsicID(), Intrinsic::lifetime_end);
}

TEST_F(IRBuilderTest, CreateCondBr) {
  IRBuilder<> Builder(BB);
  BasicBlock *TBB = BasicBlock::Create(Ctx, "", F);
  BasicBlock *FBB = BasicBlock::Create(Ctx, "", F);

  BranchInst *BI = Builder.CreateCondBr(Builder.getTrue(), TBB, FBB);
  Instruction *TI = BB->getTerminator();
  EXPECT_EQ(BI, TI);
  EXPECT_EQ(2u, TI->getNumSuccessors());
  EXPECT_EQ(TBB, TI->getSuccessor(0));
  EXPECT_EQ(FBB, TI->getSuccessor(1));

  BI->eraseFromParent();
  MDNode *Weights = MDBuilder(Ctx).createBranchWeights(42, 13);
  BI = Builder.CreateCondBr(Builder.getTrue(), TBB, FBB, Weights);
  TI = BB->getTerminator();
  EXPECT_EQ(BI, TI);
  EXPECT_EQ(2u, TI->getNumSuccessors());
  EXPECT_EQ(TBB, TI->getSuccessor(0));
  EXPECT_EQ(FBB, TI->getSuccessor(1));
  EXPECT_EQ(Weights, TI->getMetadata(LLVMContext::MD_prof));
}

TEST_F(IRBuilderTest, LandingPadName) {
  IRBuilder<> Builder(BB);
  LandingPadInst *LP = Builder.CreateLandingPad(Builder.getInt32Ty(), 0, "LP");
  EXPECT_EQ(LP->getName(), "LP");
}

TEST_F(IRBuilderTest, DataLayout) {
  std::unique_ptr<Module> M(new Module("test", Ctx));
  M->setDataLayout("e-n32");
  EXPECT_TRUE(M->getDataLayout().isLegalInteger(32));
  M->setDataLayout("e");
  EXPECT_FALSE(M->getDataLayout().isLegalInteger(32));
}

TEST_F(IRBuilderTest, GetIntTy) {
  IRBuilder<> Builder(BB);
  IntegerType *Ty1 = Builder.getInt1Ty();
  EXPECT_EQ(Ty1, IntegerType::get(Ctx, 1));

  DataLayout* DL = new DataLayout(M.get());
  IntegerType *IntPtrTy = Builder.getIntPtrTy(*DL);
  unsigned IntPtrBitSize =  DL->getPointerSizeInBits(0);
  EXPECT_EQ(IntPtrTy, IntegerType::get(Ctx, IntPtrBitSize));
  delete DL;
}

TEST_F(IRBuilderTest, FastMathFlags) {
  IRBuilder<> Builder(BB);
  Value *F, *FC;
  Instruction *FDiv, *FAdd, *FCmp, *FCall;

  F = Builder.CreateLoad(GV->getValueType(), GV);
  F = Builder.CreateFAdd(F, F);

  EXPECT_FALSE(Builder.getFastMathFlags().any());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_FALSE(FAdd->hasNoNaNs());

  FastMathFlags FMF;
  Builder.setFastMathFlags(FMF);

  // By default, no flags are set.
  F = Builder.CreateFAdd(F, F);
  EXPECT_FALSE(Builder.getFastMathFlags().any());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_FALSE(FAdd->hasNoNaNs());
  EXPECT_FALSE(FAdd->hasNoInfs());
  EXPECT_FALSE(FAdd->hasNoSignedZeros());
  EXPECT_FALSE(FAdd->hasAllowReciprocal());
  EXPECT_FALSE(FAdd->hasAllowContract());
  EXPECT_FALSE(FAdd->hasAllowReassoc());
  EXPECT_FALSE(FAdd->hasApproxFunc());

  // Set all flags in the instruction.
  FAdd->setFast(true);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  EXPECT_TRUE(FAdd->hasNoInfs());
  EXPECT_TRUE(FAdd->hasNoSignedZeros());
  EXPECT_TRUE(FAdd->hasAllowReciprocal());
  EXPECT_TRUE(FAdd->hasAllowContract());
  EXPECT_TRUE(FAdd->hasAllowReassoc());
  EXPECT_TRUE(FAdd->hasApproxFunc());

  // All flags are set in the builder.
  FMF.setFast();
  Builder.setFastMathFlags(FMF);

  F = Builder.CreateFAdd(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().all());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  EXPECT_TRUE(FAdd->isFast());

  // Now, try it with CreateBinOp
  F = Builder.CreateBinOp(Instruction::FAdd, F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  ASSERT_TRUE(isa<Instruction>(F));
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasNoNaNs());
  EXPECT_TRUE(FAdd->isFast());

  F = Builder.CreateFDiv(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().all());
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_TRUE(FDiv->hasAllowReciprocal());

  // Clear all FMF in the builder.
  Builder.clearFastMathFlags();

  F = Builder.CreateFDiv(F, F);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_FALSE(FDiv->hasAllowReciprocal());
 
  // Try individual flags.
  FMF.clear();
  FMF.setAllowReciprocal();
  Builder.setFastMathFlags(FMF);

  F = Builder.CreateFDiv(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().AllowReciprocal);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_TRUE(FDiv->hasAllowReciprocal());

  Builder.clearFastMathFlags();

  FC = Builder.CreateFCmpOEQ(F, F);
  ASSERT_TRUE(isa<Instruction>(FC));
  FCmp = cast<Instruction>(FC);
  EXPECT_FALSE(FCmp->hasAllowReciprocal());

  FMF.clear();
  FMF.setAllowReciprocal();
  Builder.setFastMathFlags(FMF);

  FC = Builder.CreateFCmpOEQ(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().AllowReciprocal);
  ASSERT_TRUE(isa<Instruction>(FC));
  FCmp = cast<Instruction>(FC);
  EXPECT_TRUE(FCmp->hasAllowReciprocal());

  Builder.clearFastMathFlags();

  // Test FP-contract
  FC = Builder.CreateFAdd(F, F);
  ASSERT_TRUE(isa<Instruction>(FC));
  FAdd = cast<Instruction>(FC);
  EXPECT_FALSE(FAdd->hasAllowContract());

  FMF.clear();
  FMF.setAllowContract(true);
  Builder.setFastMathFlags(FMF);

  FC = Builder.CreateFAdd(F, F);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().AllowContract);
  ASSERT_TRUE(isa<Instruction>(FC));
  FAdd = cast<Instruction>(FC);
  EXPECT_TRUE(FAdd->hasAllowContract());

  FMF.setApproxFunc();
  Builder.clearFastMathFlags();
  Builder.setFastMathFlags(FMF);
  // Now 'aml' and 'contract' are set.
  F = Builder.CreateFMul(F, F);
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasApproxFunc());
  EXPECT_TRUE(FAdd->hasAllowContract());
  EXPECT_FALSE(FAdd->hasAllowReassoc());
  
  FMF.setAllowReassoc();
  Builder.clearFastMathFlags();
  Builder.setFastMathFlags(FMF);
  // Now 'aml' and 'contract' and 'reassoc' are set.
  F = Builder.CreateFMul(F, F);
  FAdd = cast<Instruction>(F);
  EXPECT_TRUE(FAdd->hasApproxFunc());
  EXPECT_TRUE(FAdd->hasAllowContract());
  EXPECT_TRUE(FAdd->hasAllowReassoc());

  // Test a call with FMF.
  auto CalleeTy = FunctionType::get(Type::getFloatTy(Ctx),
                                    /*isVarArg=*/false);
  auto Callee =
      Function::Create(CalleeTy, Function::ExternalLinkage, "", M.get());

  FCall = Builder.CreateCall(Callee, None);
  EXPECT_FALSE(FCall->hasNoNaNs());

  Function *V =
      Function::Create(CalleeTy, Function::ExternalLinkage, "", M.get());
  FCall = Builder.CreateCall(V, None);
  EXPECT_FALSE(FCall->hasNoNaNs());

  FMF.clear();
  FMF.setNoNaNs();
  Builder.setFastMathFlags(FMF);

  FCall = Builder.CreateCall(Callee, None);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().NoNaNs);
  EXPECT_TRUE(FCall->hasNoNaNs());

  FCall = Builder.CreateCall(V, None);
  EXPECT_TRUE(Builder.getFastMathFlags().any());
  EXPECT_TRUE(Builder.getFastMathFlags().NoNaNs);
  EXPECT_TRUE(FCall->hasNoNaNs());

  Builder.clearFastMathFlags();

  // To test a copy, make sure that a '0' and a '1' change state.
  F = Builder.CreateFDiv(F, F);
  ASSERT_TRUE(isa<Instruction>(F));
  FDiv = cast<Instruction>(F);
  EXPECT_FALSE(FDiv->getFastMathFlags().any());
  FDiv->setHasAllowReciprocal(true);
  FAdd->setHasAllowReciprocal(false);
  FAdd->setHasNoNaNs(true);
  FDiv->copyFastMathFlags(FAdd);
  EXPECT_TRUE(FDiv->hasNoNaNs());
  EXPECT_FALSE(FDiv->hasAllowReciprocal());

}

TEST_F(IRBuilderTest, WrapFlags) {
  IRBuilder<NoFolder> Builder(BB);

  // Test instructions.
  GlobalVariable *G = new GlobalVariable(*M, Builder.getInt32Ty(), true,
                                         GlobalValue::ExternalLinkage, nullptr);
  Value *V = Builder.CreateLoad(G->getValueType(), G);
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNSWAdd(V, V))->hasNoSignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNSWMul(V, V))->hasNoSignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNSWSub(V, V))->hasNoSignedWrap());
  EXPECT_TRUE(cast<BinaryOperator>(
                  Builder.CreateShl(V, V, "", /* NUW */ false, /* NSW */ true))
                  ->hasNoSignedWrap());

  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNUWAdd(V, V))->hasNoUnsignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNUWMul(V, V))->hasNoUnsignedWrap());
  EXPECT_TRUE(
      cast<BinaryOperator>(Builder.CreateNUWSub(V, V))->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<BinaryOperator>(
                  Builder.CreateShl(V, V, "", /* NUW */ true, /* NSW */ false))
                  ->hasNoUnsignedWrap());

  // Test operators created with constants.
  Constant *C = Builder.getInt32(42);
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNSWAdd(C, C))
                  ->hasNoSignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNSWSub(C, C))
                  ->hasNoSignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNSWMul(C, C))
                  ->hasNoSignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(
                  Builder.CreateShl(C, C, "", /* NUW */ false, /* NSW */ true))
                  ->hasNoSignedWrap());

  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNUWAdd(C, C))
                  ->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNUWSub(C, C))
                  ->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(Builder.CreateNUWMul(C, C))
                  ->hasNoUnsignedWrap());
  EXPECT_TRUE(cast<OverflowingBinaryOperator>(
                  Builder.CreateShl(C, C, "", /* NUW */ true, /* NSW */ false))
                  ->hasNoUnsignedWrap());
}

TEST_F(IRBuilderTest, RAIIHelpersTest) {
  IRBuilder<> Builder(BB);
  EXPECT_FALSE(Builder.getFastMathFlags().allowReciprocal());
  MDBuilder MDB(M->getContext());

  MDNode *FPMathA = MDB.createFPMath(0.01f);
  MDNode *FPMathB = MDB.createFPMath(0.1f);

  Builder.setDefaultFPMathTag(FPMathA);

  {
    IRBuilder<>::FastMathFlagGuard Guard(Builder);
    FastMathFlags FMF;
    FMF.setAllowReciprocal();
    Builder.setFastMathFlags(FMF);
    Builder.setDefaultFPMathTag(FPMathB);
    EXPECT_TRUE(Builder.getFastMathFlags().allowReciprocal());
    EXPECT_EQ(FPMathB, Builder.getDefaultFPMathTag());
  }

  EXPECT_FALSE(Builder.getFastMathFlags().allowReciprocal());
  EXPECT_EQ(FPMathA, Builder.getDefaultFPMathTag());

  Value *F = Builder.CreateLoad(GV->getValueType(), GV);

  {
    IRBuilder<>::InsertPointGuard Guard(Builder);
    Builder.SetInsertPoint(cast<Instruction>(F));
    EXPECT_EQ(F, &*Builder.GetInsertPoint());
  }

  EXPECT_EQ(BB->end(), Builder.GetInsertPoint());
  EXPECT_EQ(BB, Builder.GetInsertBlock());
}

TEST_F(IRBuilderTest, createFunction) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File = DIB.createFile("error.swift", "/");
  auto CU =
      DIB.createCompileUnit(dwarf::DW_LANG_Swift, File, "swiftc", true, "", 0);
  auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
  auto NoErr = DIB.createFunction(
      CU, "noerr", "", File, 1, Type, 1, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
  EXPECT_TRUE(!NoErr->getThrownTypes());
  auto Int = DIB.createBasicType("Int", 64, dwarf::DW_ATE_signed);
  auto Error = DIB.getOrCreateArray({Int});
  auto Err = DIB.createFunction(
      CU, "err", "", File, 1, Type, 1, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized, nullptr,
      nullptr, Error.get());
  EXPECT_TRUE(Err->getThrownTypes().get() == Error.get());
  DIB.finalize();
}

TEST_F(IRBuilderTest, DIBuilder) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File = DIB.createFile("F.CBL", "/");
  auto CU = DIB.createCompileUnit(dwarf::DW_LANG_Cobol74,
                                  DIB.createFile("F.CBL", "/"), "llvm-cobol74",
                                  true, "", 0);
  auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
  auto SP = DIB.createFunction(
      CU, "foo", "", File, 1, Type, 1, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
  F->setSubprogram(SP);
  AllocaInst *I = Builder.CreateAlloca(Builder.getInt8Ty());
  auto BarSP = DIB.createFunction(
      CU, "bar", "", File, 1, Type, 1, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
  auto BadScope = DIB.createLexicalBlockFile(BarSP, File, 0);
  I->setDebugLoc(DebugLoc::get(2, 0, BadScope));
  DIB.finalize();
  EXPECT_TRUE(verifyModule(*M));
}

TEST_F(IRBuilderTest, createArtificialSubprogram) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File = DIB.createFile("main.c", "/");
  auto CU = DIB.createCompileUnit(dwarf::DW_LANG_C, File, "clang",
                                  /*isOptimized=*/true, /*Flags=*/"",
                                  /*Runtime Version=*/0);
  auto Type = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
  auto SP = DIB.createFunction(
      CU, "foo", /*LinkageName=*/"", File,
      /*LineNo=*/1, Type, /*ScopeLine=*/2, DINode::FlagZero,
      DISubprogram::SPFlagDefinition | DISubprogram::SPFlagOptimized);
  EXPECT_TRUE(SP->isDistinct());

  F->setSubprogram(SP);
  AllocaInst *I = Builder.CreateAlloca(Builder.getInt8Ty());
  ReturnInst *R = Builder.CreateRetVoid();
  I->setDebugLoc(DebugLoc::get(3, 2, SP));
  R->setDebugLoc(DebugLoc::get(4, 2, SP));
  DIB.finalize();
  EXPECT_FALSE(verifyModule(*M));

  Function *G = Function::Create(F->getFunctionType(),
                                 Function::ExternalLinkage, "", M.get());
  BasicBlock *GBB = BasicBlock::Create(Ctx, "", G);
  Builder.SetInsertPoint(GBB);
  I->removeFromParent();
  Builder.Insert(I);
  Builder.CreateRetVoid();
  EXPECT_FALSE(verifyModule(*M));

  DISubprogram *GSP = DIBuilder::createArtificialSubprogram(F->getSubprogram());
  EXPECT_EQ(SP->getFile(), GSP->getFile());
  EXPECT_EQ(SP->getType(), GSP->getType());
  EXPECT_EQ(SP->getLine(), GSP->getLine());
  EXPECT_EQ(SP->getScopeLine(), GSP->getScopeLine());
  EXPECT_TRUE(GSP->isDistinct());

  G->setSubprogram(GSP);
  EXPECT_TRUE(verifyModule(*M));

  auto *InlinedAtNode =
      DILocation::getDistinct(Ctx, GSP->getScopeLine(), 0, GSP);
  DebugLoc DL = I->getDebugLoc();
  DenseMap<const MDNode *, MDNode *> IANodes;
  auto IA = DebugLoc::appendInlinedAt(DL, InlinedAtNode, Ctx, IANodes);
  auto NewDL = DebugLoc::get(DL.getLine(), DL.getCol(), DL.getScope(), IA);
  I->setDebugLoc(NewDL);
  EXPECT_FALSE(verifyModule(*M));

  EXPECT_EQ("foo", SP->getName());
  EXPECT_EQ("foo", GSP->getName());
  EXPECT_FALSE(SP->isArtificial());
  EXPECT_TRUE(GSP->isArtificial());
}

TEST_F(IRBuilderTest, InsertExtractElement) {
  IRBuilder<> Builder(BB);

  auto VecTy = VectorType::get(Builder.getInt64Ty(), 4);
  auto Elt1 = Builder.getInt64(-1);
  auto Elt2 = Builder.getInt64(-2);
  Value *Vec = UndefValue::get(VecTy);
  Vec = Builder.CreateInsertElement(Vec, Elt1, Builder.getInt8(1));
  Vec = Builder.CreateInsertElement(Vec, Elt2, 2);
  auto X1 = Builder.CreateExtractElement(Vec, 1);
  auto X2 = Builder.CreateExtractElement(Vec, Builder.getInt32(2));
  EXPECT_EQ(Elt1, X1);
  EXPECT_EQ(Elt2, X2);
}

TEST_F(IRBuilderTest, CreateGlobalStringPtr) {
  IRBuilder<> Builder(BB);

  auto String1a = Builder.CreateGlobalStringPtr("TestString", "String1a");
  auto String1b = Builder.CreateGlobalStringPtr("TestString", "String1b", 0);
  auto String2 = Builder.CreateGlobalStringPtr("TestString", "String2", 1);
  auto String3 = Builder.CreateGlobalString("TestString", "String3", 2);

  EXPECT_TRUE(String1a->getType()->getPointerAddressSpace() == 0);
  EXPECT_TRUE(String1b->getType()->getPointerAddressSpace() == 0);
  EXPECT_TRUE(String2->getType()->getPointerAddressSpace() == 1);
  EXPECT_TRUE(String3->getType()->getPointerAddressSpace() == 2);
}

TEST_F(IRBuilderTest, DebugLoc) {
  auto CalleeTy = FunctionType::get(Type::getVoidTy(Ctx),
                                    /*isVarArg=*/false);
  auto Callee =
      Function::Create(CalleeTy, Function::ExternalLinkage, "", M.get());

  DIBuilder DIB(*M);
  auto File = DIB.createFile("tmp.cpp", "/");
  auto CU = DIB.createCompileUnit(dwarf::DW_LANG_C_plus_plus_11,
                                  DIB.createFile("tmp.cpp", "/"), "", true, "",
                                  0);
  auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
  auto SP =
      DIB.createFunction(CU, "foo", "foo", File, 1, SPType, 1, DINode::FlagZero,
                         DISubprogram::SPFlagDefinition);
  DebugLoc DL1 = DILocation::get(Ctx, 2, 0, SP);
  DebugLoc DL2 = DILocation::get(Ctx, 3, 0, SP);

  auto BB2 = BasicBlock::Create(Ctx, "bb2", F);
  auto Br = BranchInst::Create(BB2, BB);
  Br->setDebugLoc(DL1);

  IRBuilder<> Builder(Ctx);
  Builder.SetInsertPoint(Br);
  EXPECT_EQ(DL1, Builder.getCurrentDebugLocation());
  auto Call1 = Builder.CreateCall(Callee, None);
  EXPECT_EQ(DL1, Call1->getDebugLoc());

  Call1->setDebugLoc(DL2);
  Builder.SetInsertPoint(Call1->getParent(), Call1->getIterator());
  EXPECT_EQ(DL2, Builder.getCurrentDebugLocation());
  auto Call2 = Builder.CreateCall(Callee, None);
  EXPECT_EQ(DL2, Call2->getDebugLoc());

  DIB.finalize();
}

TEST_F(IRBuilderTest, DIImportedEntity) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto F = DIB.createFile("F.CBL", "/");
  auto CU = DIB.createCompileUnit(dwarf::DW_LANG_Cobol74,
                                  F, "llvm-cobol74",
                                  true, "", 0);
  DIB.createImportedDeclaration(CU, nullptr, F, 1);
  DIB.createImportedDeclaration(CU, nullptr, F, 1);
  DIB.createImportedModule(CU, (DIImportedEntity *)nullptr, F, 2);
  DIB.createImportedModule(CU, (DIImportedEntity *)nullptr, F, 2);
  DIB.finalize();
  EXPECT_TRUE(verifyModule(*M));
  EXPECT_TRUE(CU->getImportedEntities().size() == 2);
}

//  0: #define M0 V0          <-- command line definition
//  0: main.c                 <-- main file
//     3:   #define M1 V1     <-- M1 definition in main.c
//     5:   #include "file.h" <-- inclusion of file.h from main.c
//          1: #define M2     <-- M2 definition in file.h with no value
//     7:   #undef M1 V1      <-- M1 un-definition in main.c
TEST_F(IRBuilderTest, DIBuilderMacro) {
  IRBuilder<> Builder(BB);
  DIBuilder DIB(*M);
  auto File1 = DIB.createFile("main.c", "/");
  auto File2 = DIB.createFile("file.h", "/");
  auto CU = DIB.createCompileUnit(
      dwarf::DW_LANG_C, DIB.createFile("main.c", "/"), "llvm-c", true, "", 0);
  auto MDef0 =
      DIB.createMacro(nullptr, 0, dwarf::DW_MACINFO_define, "M0", "V0");
  auto TMF1 = DIB.createTempMacroFile(nullptr, 0, File1);
  auto MDef1 = DIB.createMacro(TMF1, 3, dwarf::DW_MACINFO_define, "M1", "V1");
  auto TMF2 = DIB.createTempMacroFile(TMF1, 5, File2);
  auto MDef2 = DIB.createMacro(TMF2, 1, dwarf::DW_MACINFO_define, "M2");
  auto MUndef1 = DIB.createMacro(TMF1, 7, dwarf::DW_MACINFO_undef, "M1");

  EXPECT_EQ(dwarf::DW_MACINFO_define, MDef1->getMacinfoType());
  EXPECT_EQ(3u, MDef1->getLine());
  EXPECT_EQ("M1", MDef1->getName());
  EXPECT_EQ("V1", MDef1->getValue());

  EXPECT_EQ(dwarf::DW_MACINFO_undef, MUndef1->getMacinfoType());
  EXPECT_EQ(7u, MUndef1->getLine());
  EXPECT_EQ("M1", MUndef1->getName());
  EXPECT_EQ("", MUndef1->getValue());

  EXPECT_EQ(dwarf::DW_MACINFO_start_file, TMF2->getMacinfoType());
  EXPECT_EQ(5u, TMF2->getLine());
  EXPECT_EQ(File2, TMF2->getFile());

  DIB.finalize();

  SmallVector<Metadata *, 4> Elements;
  Elements.push_back(MDef2);
  auto MF2 = DIMacroFile::get(Ctx, dwarf::DW_MACINFO_start_file, 5, File2,
                              DIB.getOrCreateMacroArray(Elements));

  Elements.clear();
  Elements.push_back(MDef1);
  Elements.push_back(MF2);
  Elements.push_back(MUndef1);
  auto MF1 = DIMacroFile::get(Ctx, dwarf::DW_MACINFO_start_file, 0, File1,
                              DIB.getOrCreateMacroArray(Elements));

  Elements.clear();
  Elements.push_back(MDef0);
  Elements.push_back(MF1);
  auto MN0 = MDTuple::get(Ctx, Elements);
  EXPECT_EQ(MN0, CU->getRawMacros());

  Elements.clear();
  Elements.push_back(MDef1);
  Elements.push_back(MF2);
  Elements.push_back(MUndef1);
  auto MN1 = MDTuple::get(Ctx, Elements);
  EXPECT_EQ(MN1, MF1->getRawElements());

  Elements.clear();
  Elements.push_back(MDef2);
  auto MN2 = MDTuple::get(Ctx, Elements);
  EXPECT_EQ(MN2, MF2->getRawElements());
  EXPECT_TRUE(verifyModule(*M));
}
}
