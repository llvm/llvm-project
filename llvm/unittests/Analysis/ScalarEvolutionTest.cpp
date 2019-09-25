//===- ScalarEvolutionsTest.cpp - ScalarEvolution unit tests --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {

// We use this fixture to ensure that we clean up ScalarEvolution before
// deleting the PassManager.
class ScalarEvolutionsTest : public testing::Test {
protected:
  LLVMContext Context;
  Module M;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;

  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  ScalarEvolutionsTest() : M("", Context), TLII(), TLI(TLII) {}

  ScalarEvolution buildSE(Function &F) {
    AC.reset(new AssumptionCache(F));
    DT.reset(new DominatorTree(F));
    LI.reset(new LoopInfo(*DT));
    return ScalarEvolution(F, TLI, *AC, *DT, *LI);
  }

  void runWithSE(
      Module &M, StringRef FuncName,
      function_ref<void(Function &F, LoopInfo &LI, ScalarEvolution &SE)> Test) {
    auto *F = M.getFunction(FuncName);
    ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
    ScalarEvolution SE = buildSE(*F);
    Test(*F, *LI, SE);
  }

  static Optional<APInt> computeConstantDifference(ScalarEvolution &SE,
                                                   const SCEV *LHS,
                                                   const SCEV *RHS) {
    return SE.computeConstantDifference(LHS, RHS);
  }
};

TEST_F(ScalarEvolutionsTest, SCEVUnknownRAUW) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context),
                                              std::vector<Type *>(), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  ReturnInst::Create(Context, nullptr, BB);

  Type *Ty = Type::getInt1Ty(Context);
  Constant *Init = Constant::getNullValue(Ty);
  Value *V0 = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, Init, "V0");
  Value *V1 = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, Init, "V1");
  Value *V2 = new GlobalVariable(M, Ty, false, GlobalValue::ExternalLinkage, Init, "V2");

  ScalarEvolution SE = buildSE(*F);

  const SCEV *S0 = SE.getSCEV(V0);
  const SCEV *S1 = SE.getSCEV(V1);
  const SCEV *S2 = SE.getSCEV(V2);

  const SCEV *P0 = SE.getAddExpr(S0, S0);
  const SCEV *P1 = SE.getAddExpr(S1, S1);
  const SCEV *P2 = SE.getAddExpr(S2, S2);

  const SCEVMulExpr *M0 = cast<SCEVMulExpr>(P0);
  const SCEVMulExpr *M1 = cast<SCEVMulExpr>(P1);
  const SCEVMulExpr *M2 = cast<SCEVMulExpr>(P2);

  EXPECT_EQ(cast<SCEVConstant>(M0->getOperand(0))->getValue()->getZExtValue(),
            2u);
  EXPECT_EQ(cast<SCEVConstant>(M1->getOperand(0))->getValue()->getZExtValue(),
            2u);
  EXPECT_EQ(cast<SCEVConstant>(M2->getOperand(0))->getValue()->getZExtValue(),
            2u);

  // Before the RAUWs, these are all pointing to separate values.
  EXPECT_EQ(cast<SCEVUnknown>(M0->getOperand(1))->getValue(), V0);
  EXPECT_EQ(cast<SCEVUnknown>(M1->getOperand(1))->getValue(), V1);
  EXPECT_EQ(cast<SCEVUnknown>(M2->getOperand(1))->getValue(), V2);

  // Do some RAUWs.
  V2->replaceAllUsesWith(V1);
  V1->replaceAllUsesWith(V0);

  // After the RAUWs, these should all be pointing to V0.
  EXPECT_EQ(cast<SCEVUnknown>(M0->getOperand(1))->getValue(), V0);
  EXPECT_EQ(cast<SCEVUnknown>(M1->getOperand(1))->getValue(), V0);
  EXPECT_EQ(cast<SCEVUnknown>(M2->getOperand(1))->getValue(), V0);
}

TEST_F(ScalarEvolutionsTest, SimplifiedPHI) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context),
                                              std::vector<Type *>(), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(Context, "loop", F);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "exit", F);
  BranchInst::Create(LoopBB, EntryBB);
  BranchInst::Create(LoopBB, ExitBB, UndefValue::get(Type::getInt1Ty(Context)),
                     LoopBB);
  ReturnInst::Create(Context, nullptr, ExitBB);
  auto *Ty = Type::getInt32Ty(Context);
  auto *PN = PHINode::Create(Ty, 2, "", &*LoopBB->begin());
  PN->addIncoming(Constant::getNullValue(Ty), EntryBB);
  PN->addIncoming(UndefValue::get(Ty), LoopBB);
  ScalarEvolution SE = buildSE(*F);
  auto *S1 = SE.getSCEV(PN);
  auto *S2 = SE.getSCEV(PN);
  auto *ZeroConst = SE.getConstant(Ty, 0);

  // At some point, only the first call to getSCEV returned the simplified
  // SCEVConstant and later calls just returned a SCEVUnknown referencing the
  // PHI node.
  EXPECT_EQ(S1, ZeroConst);
  EXPECT_EQ(S1, S2);
}

TEST_F(ScalarEvolutionsTest, ExpandPtrTypeSCEV) {
  // It is to test the fix for PR30213. It exercises the branch in scev
  // expansion when the value in ValueOffsetPair is a ptr and the offset
  // is not divisible by the elem type size of value.
  auto *I8Ty = Type::getInt8Ty(Context);
  auto *I8PtrTy = Type::getInt8PtrTy(Context);
  auto *I32Ty = Type::getInt32Ty(Context);
  auto *I32PtrTy = Type::getInt32PtrTy(Context);
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), std::vector<Type *>(), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(Context, "loop", F);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "exit", F);
  BranchInst::Create(LoopBB, EntryBB);
  ReturnInst::Create(Context, nullptr, ExitBB);

  // loop:                            ; preds = %loop, %entry
  //   %alloca = alloca i32
  //   %gep0 = getelementptr i32, i32* %alloca, i32 1
  //   %bitcast1 = bitcast i32* %gep0 to i8*
  //   %gep1 = getelementptr i8, i8* %bitcast1, i32 1
  //   %gep2 = getelementptr i8, i8* undef, i32 1
  //   %cmp = icmp ult i8* undef, %bitcast1
  //   %select = select i1 %cmp, i8* %gep1, i8* %gep2
  //   %bitcast2 = bitcast i8* %select to i32*
  //   br i1 undef, label %loop, label %exit

  const DataLayout &DL = F->getParent()->getDataLayout();
  BranchInst *Br = BranchInst::Create(
      LoopBB, ExitBB, UndefValue::get(Type::getInt1Ty(Context)), LoopBB);
  AllocaInst *Alloca = new AllocaInst(I32Ty, DL.getAllocaAddrSpace(),
                                      "alloca", Br);
  ConstantInt *Ci32 = ConstantInt::get(Context, APInt(32, 1));
  GetElementPtrInst *Gep0 =
      GetElementPtrInst::Create(I32Ty, Alloca, Ci32, "gep0", Br);
  CastInst *CastA =
      CastInst::CreateBitOrPointerCast(Gep0, I8PtrTy, "bitcast1", Br);
  GetElementPtrInst *Gep1 =
      GetElementPtrInst::Create(I8Ty, CastA, Ci32, "gep1", Br);
  GetElementPtrInst *Gep2 = GetElementPtrInst::Create(
      I8Ty, UndefValue::get(I8PtrTy), Ci32, "gep2", Br);
  CmpInst *Cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULT,
                                 UndefValue::get(I8PtrTy), CastA, "cmp", Br);
  SelectInst *Sel = SelectInst::Create(Cmp, Gep1, Gep2, "select", Br);
  CastInst *CastB =
      CastInst::CreateBitOrPointerCast(Sel, I32PtrTy, "bitcast2", Br);

  ScalarEvolution SE = buildSE(*F);
  auto *S = SE.getSCEV(CastB);
  SCEVExpander Exp(SE, M.getDataLayout(), "expander");
  Value *V =
      Exp.expandCodeFor(cast<SCEVAddExpr>(S)->getOperand(1), nullptr, Br);

  // Expect the expansion code contains:
  //   %0 = bitcast i32* %bitcast2 to i8*
  //   %uglygep = getelementptr i8, i8* %0, i64 -1
  //   %1 = bitcast i8* %uglygep to i32*
  EXPECT_TRUE(isa<BitCastInst>(V));
  Instruction *Gep = cast<Instruction>(V)->getPrevNode();
  EXPECT_TRUE(isa<GetElementPtrInst>(Gep));
  EXPECT_TRUE(isa<ConstantInt>(Gep->getOperand(1)));
  EXPECT_EQ(cast<ConstantInt>(Gep->getOperand(1))->getSExtValue(), -1);
  EXPECT_TRUE(isa<BitCastInst>(Gep->getPrevNode()));
}

static Instruction *getInstructionByName(Function &F, StringRef Name) {
  for (auto &I : instructions(F))
    if (I.getName() == Name)
      return &I;
  llvm_unreachable("Expected to find instruction!");
}

TEST_F(ScalarEvolutionsTest, CommutativeExprOperandOrder) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128\" "
      " "
      "@var_0 = external global i32, align 4"
      "@var_1 = external global i32, align 4"
      "@var_2 = external global i32, align 4"
      " "
      "declare i32 @unknown(i32, i32, i32)"
      " "
      "define void @f_1(i8* nocapture %arr, i32 %n, i32* %A, i32* %B) "
      "    local_unnamed_addr { "
      "entry: "
      "  %entrycond = icmp sgt i32 %n, 0 "
      "  br i1 %entrycond, label %loop.ph, label %for.end "
      " "
      "loop.ph: "
      "  %a = load i32, i32* %A, align 4 "
      "  %b = load i32, i32* %B, align 4 "
      "  %mul = mul nsw i32 %b, %a "
      "  %iv0.init = getelementptr inbounds i8, i8* %arr, i32 %mul "
      "  br label %loop "
      " "
      "loop: "
      "  %iv0 = phi i8* [ %iv0.inc, %loop ], [ %iv0.init, %loop.ph ] "
      "  %iv1 = phi i32 [ %iv1.inc, %loop ], [ 0, %loop.ph ] "
      "  %conv = trunc i32 %iv1 to i8 "
      "  store i8 %conv, i8* %iv0, align 1 "
      "  %iv0.inc = getelementptr inbounds i8, i8* %iv0, i32 %b "
      "  %iv1.inc = add nuw nsw i32 %iv1, 1 "
      "  %exitcond = icmp eq i32 %iv1.inc, %n "
      "  br i1 %exitcond, label %for.end.loopexit, label %loop "
      " "
      "for.end.loopexit: "
      "  br label %for.end "
      " "
      "for.end: "
      "  ret void "
      "} "
      " "
      "define void @f_2(i32* %X, i32* %Y, i32* %Z) { "
      "  %x = load i32, i32* %X "
      "  %y = load i32, i32* %Y "
      "  %z = load i32, i32* %Z "
      "  ret void "
      "} "
      " "
      "define void @f_3() { "
      "  %x = load i32, i32* @var_0"
      "  %y = load i32, i32* @var_1"
      "  %z = load i32, i32* @var_2"
      "  ret void"
      "} "
      " "
      "define void @f_4(i32 %a, i32 %b, i32 %c) { "
      "  %x = call i32 @unknown(i32 %a, i32 %b, i32 %c)"
      "  %y = call i32 @unknown(i32 %b, i32 %c, i32 %a)"
      "  %z = call i32 @unknown(i32 %c, i32 %a, i32 %b)"
      "  ret void"
      "} "
      ,
      Err, C);

  assert(M && "Could not parse module?");
  assert(!verifyModule(*M) && "Must have been well formed!");

  runWithSE(*M, "f_1", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    auto *IV0 = getInstructionByName(F, "iv0");
    auto *IV0Inc = getInstructionByName(F, "iv0.inc");

    auto *FirstExprForIV0 = SE.getSCEV(IV0);
    auto *FirstExprForIV0Inc = SE.getSCEV(IV0Inc);
    auto *SecondExprForIV0 = SE.getSCEV(IV0);

    EXPECT_TRUE(isa<SCEVAddRecExpr>(FirstExprForIV0));
    EXPECT_TRUE(isa<SCEVAddRecExpr>(FirstExprForIV0Inc));
    EXPECT_TRUE(isa<SCEVAddRecExpr>(SecondExprForIV0));
  });

  auto CheckCommutativeMulExprs = [&](ScalarEvolution &SE, const SCEV *A,
                                      const SCEV *B, const SCEV *C) {
    EXPECT_EQ(SE.getMulExpr(A, B), SE.getMulExpr(B, A));
    EXPECT_EQ(SE.getMulExpr(B, C), SE.getMulExpr(C, B));
    EXPECT_EQ(SE.getMulExpr(A, C), SE.getMulExpr(C, A));

    SmallVector<const SCEV *, 3> Ops0 = {A, B, C};
    SmallVector<const SCEV *, 3> Ops1 = {A, C, B};
    SmallVector<const SCEV *, 3> Ops2 = {B, A, C};
    SmallVector<const SCEV *, 3> Ops3 = {B, C, A};
    SmallVector<const SCEV *, 3> Ops4 = {C, B, A};
    SmallVector<const SCEV *, 3> Ops5 = {C, A, B};

    auto *Mul0 = SE.getMulExpr(Ops0);
    auto *Mul1 = SE.getMulExpr(Ops1);
    auto *Mul2 = SE.getMulExpr(Ops2);
    auto *Mul3 = SE.getMulExpr(Ops3);
    auto *Mul4 = SE.getMulExpr(Ops4);
    auto *Mul5 = SE.getMulExpr(Ops5);

    EXPECT_EQ(Mul0, Mul1) << "Expected " << *Mul0 << " == " << *Mul1;
    EXPECT_EQ(Mul1, Mul2) << "Expected " << *Mul1 << " == " << *Mul2;
    EXPECT_EQ(Mul2, Mul3) << "Expected " << *Mul2 << " == " << *Mul3;
    EXPECT_EQ(Mul3, Mul4) << "Expected " << *Mul3 << " == " << *Mul4;
    EXPECT_EQ(Mul4, Mul5) << "Expected " << *Mul4 << " == " << *Mul5;
  };

  for (StringRef FuncName : {"f_2", "f_3", "f_4"})
    runWithSE(
        *M, FuncName, [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
          CheckCommutativeMulExprs(SE, SE.getSCEV(getInstructionByName(F, "x")),
                                   SE.getSCEV(getInstructionByName(F, "y")),
                                   SE.getSCEV(getInstructionByName(F, "z")));
        });
}

TEST_F(ScalarEvolutionsTest, CompareSCEVComplexity) {
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), std::vector<Type *>(), false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(Context, "bb1", F);
  BranchInst::Create(LoopBB, EntryBB);

  auto *Ty = Type::getInt32Ty(Context);
  SmallVector<Instruction*, 8> Muls(8), Acc(8), NextAcc(8);

  Acc[0] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[1] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[2] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[3] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[4] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[5] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[6] = PHINode::Create(Ty, 2, "", LoopBB);
  Acc[7] = PHINode::Create(Ty, 2, "", LoopBB);

  for (int i = 0; i < 20; i++) {
    Muls[0] = BinaryOperator::CreateMul(Acc[0], Acc[0], "", LoopBB);
    NextAcc[0] = BinaryOperator::CreateAdd(Muls[0], Acc[4], "", LoopBB);
    Muls[1] = BinaryOperator::CreateMul(Acc[1], Acc[1], "", LoopBB);
    NextAcc[1] = BinaryOperator::CreateAdd(Muls[1], Acc[5], "", LoopBB);
    Muls[2] = BinaryOperator::CreateMul(Acc[2], Acc[2], "", LoopBB);
    NextAcc[2] = BinaryOperator::CreateAdd(Muls[2], Acc[6], "", LoopBB);
    Muls[3] = BinaryOperator::CreateMul(Acc[3], Acc[3], "", LoopBB);
    NextAcc[3] = BinaryOperator::CreateAdd(Muls[3], Acc[7], "", LoopBB);

    Muls[4] = BinaryOperator::CreateMul(Acc[4], Acc[4], "", LoopBB);
    NextAcc[4] = BinaryOperator::CreateAdd(Muls[4], Acc[0], "", LoopBB);
    Muls[5] = BinaryOperator::CreateMul(Acc[5], Acc[5], "", LoopBB);
    NextAcc[5] = BinaryOperator::CreateAdd(Muls[5], Acc[1], "", LoopBB);
    Muls[6] = BinaryOperator::CreateMul(Acc[6], Acc[6], "", LoopBB);
    NextAcc[6] = BinaryOperator::CreateAdd(Muls[6], Acc[2], "", LoopBB);
    Muls[7] = BinaryOperator::CreateMul(Acc[7], Acc[7], "", LoopBB);
    NextAcc[7] = BinaryOperator::CreateAdd(Muls[7], Acc[3], "", LoopBB);
    Acc = NextAcc;
  }

  auto II = LoopBB->begin();
  for (int i = 0; i < 8; i++) {
    PHINode *Phi = cast<PHINode>(&*II++);
    Phi->addIncoming(Acc[i], LoopBB);
    Phi->addIncoming(UndefValue::get(Ty), EntryBB);
  }

  BasicBlock *ExitBB = BasicBlock::Create(Context, "bb2", F);
  BranchInst::Create(LoopBB, ExitBB, UndefValue::get(Type::getInt1Ty(Context)),
                     LoopBB);

  Acc[0] = BinaryOperator::CreateAdd(Acc[0], Acc[1], "", ExitBB);
  Acc[1] = BinaryOperator::CreateAdd(Acc[2], Acc[3], "", ExitBB);
  Acc[2] = BinaryOperator::CreateAdd(Acc[4], Acc[5], "", ExitBB);
  Acc[3] = BinaryOperator::CreateAdd(Acc[6], Acc[7], "", ExitBB);
  Acc[0] = BinaryOperator::CreateAdd(Acc[0], Acc[1], "", ExitBB);
  Acc[1] = BinaryOperator::CreateAdd(Acc[2], Acc[3], "", ExitBB);
  Acc[0] = BinaryOperator::CreateAdd(Acc[0], Acc[1], "", ExitBB);

  ReturnInst::Create(Context, nullptr, ExitBB);

  ScalarEvolution SE = buildSE(*F);

  EXPECT_NE(nullptr, SE.getSCEV(Acc[0]));
}

TEST_F(ScalarEvolutionsTest, CompareValueComplexity) {
  IntegerType *IntPtrTy = M.getDataLayout().getIntPtrType(Context);
  PointerType *IntPtrPtrTy = IntPtrTy->getPointerTo();

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), {IntPtrTy, IntPtrTy}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);

  Value *X = &*F->arg_begin();
  Value *Y = &*std::next(F->arg_begin());

  const int ValueDepth = 10;
  for (int i = 0; i < ValueDepth; i++) {
    X = new LoadInst(IntPtrTy, new IntToPtrInst(X, IntPtrPtrTy, "", EntryBB),
                     "",
                     /*isVolatile*/ false, EntryBB);
    Y = new LoadInst(IntPtrTy, new IntToPtrInst(Y, IntPtrPtrTy, "", EntryBB),
                     "",
                     /*isVolatile*/ false, EntryBB);
  }

  auto *MulA = BinaryOperator::CreateMul(X, Y, "", EntryBB);
  auto *MulB = BinaryOperator::CreateMul(Y, X, "", EntryBB);
  ReturnInst::Create(Context, nullptr, EntryBB);

  // This test isn't checking for correctness.  Today making A and B resolve to
  // the same SCEV would require deeper searching in CompareValueComplexity,
  // which will slow down compilation.  However, this test can fail (with LLVM's
  // behavior still being correct) if we ever have a smarter
  // CompareValueComplexity that is both fast and more accurate.

  ScalarEvolution SE = buildSE(*F);
  auto *A = SE.getSCEV(MulA);
  auto *B = SE.getSCEV(MulB);
  EXPECT_NE(A, B);
}

TEST_F(ScalarEvolutionsTest, SCEVAddExpr) {
  Type *Ty32 = Type::getInt32Ty(Context);
  Type *ArgTys[] = {Type::getInt64Ty(Context), Ty32};

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), ArgTys, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);

  Argument *A1 = &*F->arg_begin();
  Argument *A2 = &*(std::next(F->arg_begin()));
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);

  Instruction *Trunc = CastInst::CreateTruncOrBitCast(A1, Ty32, "", EntryBB);
  Instruction *Mul1 = BinaryOperator::CreateMul(Trunc, A2, "", EntryBB);
  Instruction *Add1 = BinaryOperator::CreateAdd(Mul1, Trunc, "", EntryBB);
  Mul1 = BinaryOperator::CreateMul(Add1, Trunc, "", EntryBB);
  Instruction *Add2 = BinaryOperator::CreateAdd(Mul1, Add1, "", EntryBB);
  // FIXME: The size of this is arbitrary and doesn't seem to change the
  // result, but SCEV will do quadratic work for these so a large number here
  // will be extremely slow. We should revisit what and how this is testing
  // SCEV.
  for (int i = 0; i < 10; i++) {
    Mul1 = BinaryOperator::CreateMul(Add2, Add1, "", EntryBB);
    Add1 = Add2;
    Add2 = BinaryOperator::CreateAdd(Mul1, Add1, "", EntryBB);
  }

  ReturnInst::Create(Context, nullptr, EntryBB);
  ScalarEvolution SE = buildSE(*F);
  EXPECT_NE(nullptr, SE.getSCEV(Mul1));
}

static Instruction &GetInstByName(Function &F, StringRef Name) {
  for (auto &I : instructions(F))
    if (I.getName() == Name)
      return I;
  llvm_unreachable("Could not find instructions!");
}

TEST_F(ScalarEvolutionsTest, SCEVNormalization) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(
      "target datalayout = \"e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128\" "
      " "
      "@var_0 = external global i32, align 4"
      "@var_1 = external global i32, align 4"
      "@var_2 = external global i32, align 4"
      " "
      "declare i32 @unknown(i32, i32, i32)"
      " "
      "define void @f_1(i8* nocapture %arr, i32 %n, i32* %A, i32* %B) "
      "    local_unnamed_addr { "
      "entry: "
      "  br label %loop.ph "
      " "
      "loop.ph: "
      "  br label %loop "
      " "
      "loop: "
      "  %iv0 = phi i32 [ %iv0.inc, %loop ], [ 0, %loop.ph ] "
      "  %iv1 = phi i32 [ %iv1.inc, %loop ], [ -2147483648, %loop.ph ] "
      "  %iv0.inc = add i32 %iv0, 1 "
      "  %iv1.inc = add i32 %iv1, 3 "
      "  br i1 undef, label %for.end.loopexit, label %loop "
      " "
      "for.end.loopexit: "
      "  ret void "
      "} "
      " "
      "define void @f_2(i32 %a, i32 %b, i32 %c, i32 %d) "
      "    local_unnamed_addr { "
      "entry: "
      "  br label %loop_0 "
      " "
      "loop_0: "
      "  br i1 undef, label %loop_0, label %loop_1 "
      " "
      "loop_1: "
      "  br i1 undef, label %loop_2, label %loop_1 "
      " "
      " "
      "loop_2: "
      "  br i1 undef, label %end, label %loop_2 "
      " "
      "end: "
      "  ret void "
      "} "
      ,
      Err, C);

  assert(M && "Could not parse module?");
  assert(!verifyModule(*M) && "Must have been well formed!");

  runWithSE(*M, "f_1", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    auto &I0 = GetInstByName(F, "iv0");
    auto &I1 = *I0.getNextNode();

    auto *S0 = cast<SCEVAddRecExpr>(SE.getSCEV(&I0));
    PostIncLoopSet Loops;
    Loops.insert(S0->getLoop());
    auto *N0 = normalizeForPostIncUse(S0, Loops, SE);
    auto *D0 = denormalizeForPostIncUse(N0, Loops, SE);
    EXPECT_EQ(S0, D0) << *S0 << " " << *D0;

    auto *S1 = cast<SCEVAddRecExpr>(SE.getSCEV(&I1));
    Loops.clear();
    Loops.insert(S1->getLoop());
    auto *N1 = normalizeForPostIncUse(S1, Loops, SE);
    auto *D1 = denormalizeForPostIncUse(N1, Loops, SE);
    EXPECT_EQ(S1, D1) << *S1 << " " << *D1;
  });

  runWithSE(*M, "f_2", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    auto *L2 = *LI.begin();
    auto *L1 = *std::next(LI.begin());
    auto *L0 = *std::next(LI.begin(), 2);

    auto GetAddRec = [&SE](const Loop *L, std::initializer_list<const SCEV *> Ops) {
      SmallVector<const SCEV *, 4> OpsCopy(Ops);
      return SE.getAddRecExpr(OpsCopy, L, SCEV::FlagAnyWrap);
    };

    auto GetAdd = [&SE](std::initializer_list<const SCEV *> Ops) {
      SmallVector<const SCEV *, 4> OpsCopy(Ops);
      return SE.getAddExpr(OpsCopy, SCEV::FlagAnyWrap);
    };

    // We first populate the AddRecs vector with a few "interesting" SCEV
    // expressions, and then we go through the list and assert that each
    // expression in it has an invertible normalization.

    std::vector<const SCEV *> Exprs;
    {
      const SCEV *V0 = SE.getSCEV(&*F.arg_begin());
      const SCEV *V1 = SE.getSCEV(&*std::next(F.arg_begin(), 1));
      const SCEV *V2 = SE.getSCEV(&*std::next(F.arg_begin(), 2));
      const SCEV *V3 = SE.getSCEV(&*std::next(F.arg_begin(), 3));

      Exprs.push_back(GetAddRec(L0, {V0}));             // 0
      Exprs.push_back(GetAddRec(L0, {V0, V1}));         // 1
      Exprs.push_back(GetAddRec(L0, {V0, V1, V2}));     // 2
      Exprs.push_back(GetAddRec(L0, {V0, V1, V2, V3})); // 3

      Exprs.push_back(
          GetAddRec(L1, {Exprs[1], Exprs[2], Exprs[3], Exprs[0]})); // 4
      Exprs.push_back(
          GetAddRec(L1, {Exprs[1], Exprs[2], Exprs[0], Exprs[3]})); // 5
      Exprs.push_back(
          GetAddRec(L1, {Exprs[1], Exprs[3], Exprs[3], Exprs[1]})); // 6

      Exprs.push_back(GetAdd({Exprs[6], Exprs[3], V2})); // 7

      Exprs.push_back(
          GetAddRec(L2, {Exprs[4], Exprs[3], Exprs[3], Exprs[5]})); // 8

      Exprs.push_back(
          GetAddRec(L2, {Exprs[4], Exprs[6], Exprs[7], Exprs[3], V0})); // 9
    }

    std::vector<PostIncLoopSet> LoopSets;
    for (int i = 0; i < 8; i++) {
      LoopSets.emplace_back();
      if (i & 1)
        LoopSets.back().insert(L0);
      if (i & 2)
        LoopSets.back().insert(L1);
      if (i & 4)
        LoopSets.back().insert(L2);
    }

    for (const auto &LoopSet : LoopSets)
      for (auto *S : Exprs) {
        {
          auto *N = llvm::normalizeForPostIncUse(S, LoopSet, SE);
          auto *D = llvm::denormalizeForPostIncUse(N, LoopSet, SE);

          // Normalization and then denormalizing better give us back the same
          // value.
          EXPECT_EQ(S, D) << "S = " << *S << "  D = " << *D << " N = " << *N;
        }
        {
          auto *D = llvm::denormalizeForPostIncUse(S, LoopSet, SE);
          auto *N = llvm::normalizeForPostIncUse(D, LoopSet, SE);

          // Denormalization and then normalizing better give us back the same
          // value.
          EXPECT_EQ(S, N) << "S = " << *S << "  N = " << *N;
        }
      }
  });
}

// Expect the call of getZeroExtendExpr will not cost exponential time.
TEST_F(ScalarEvolutionsTest, SCEVZeroExtendExpr) {
  LLVMContext C;
  SMDiagnostic Err;

  // Generate a function like below:
  // define void @foo() {
  // entry:
  //   br label %for.cond
  //
  // for.cond:
  //   %0 = phi i64 [ 100, %entry ], [ %dec, %for.inc ]
  //   %cmp = icmp sgt i64 %0, 90
  //   br i1 %cmp, label %for.inc, label %for.cond1
  //
  // for.inc:
  //   %dec = add nsw i64 %0, -1
  //   br label %for.cond
  //
  // for.cond1:
  //   %1 = phi i64 [ 100, %for.cond ], [ %dec5, %for.inc2 ]
  //   %cmp3 = icmp sgt i64 %1, 90
  //   br i1 %cmp3, label %for.inc2, label %for.cond4
  //
  // for.inc2:
  //   %dec5 = add nsw i64 %1, -1
  //   br label %for.cond1
  //
  // ......
  //
  // for.cond89:
  //   %19 = phi i64 [ 100, %for.cond84 ], [ %dec94, %for.inc92 ]
  //   %cmp93 = icmp sgt i64 %19, 90
  //   br i1 %cmp93, label %for.inc92, label %for.end
  //
  // for.inc92:
  //   %dec94 = add nsw i64 %19, -1
  //   br label %for.cond89
  //
  // for.end:
  //   %gep = getelementptr i8, i8* null, i64 %dec
  //   %gep6 = getelementptr i8, i8* %gep, i64 %dec5
  //   ......
  //   %gep95 = getelementptr i8, i8* %gep91, i64 %dec94
  //   ret void
  // }
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), {}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", M);

  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *CondBB = BasicBlock::Create(Context, "for.cond", F);
  BasicBlock *EndBB = BasicBlock::Create(Context, "for.end", F);
  BranchInst::Create(CondBB, EntryBB);
  BasicBlock *PrevBB = EntryBB;

  Type *I64Ty = Type::getInt64Ty(Context);
  Type *I8Ty = Type::getInt8Ty(Context);
  Type *I8PtrTy = Type::getInt8PtrTy(Context);
  Value *Accum = Constant::getNullValue(I8PtrTy);
  int Iters = 20;
  for (int i = 0; i < Iters; i++) {
    BasicBlock *IncBB = BasicBlock::Create(Context, "for.inc", F, EndBB);
    auto *PN = PHINode::Create(I64Ty, 2, "", CondBB);
    PN->addIncoming(ConstantInt::get(Context, APInt(64, 100)), PrevBB);
    auto *Cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_SGT, PN,
                                ConstantInt::get(Context, APInt(64, 90)), "cmp",
                                CondBB);
    BasicBlock *NextBB;
    if (i != Iters - 1)
      NextBB = BasicBlock::Create(Context, "for.cond", F, EndBB);
    else
      NextBB = EndBB;
    BranchInst::Create(IncBB, NextBB, Cmp, CondBB);
    auto *Dec = BinaryOperator::CreateNSWAdd(
        PN, ConstantInt::get(Context, APInt(64, -1)), "dec", IncBB);
    PN->addIncoming(Dec, IncBB);
    BranchInst::Create(CondBB, IncBB);

    Accum = GetElementPtrInst::Create(I8Ty, Accum, PN, "gep", EndBB);

    PrevBB = CondBB;
    CondBB = NextBB;
  }
  ReturnInst::Create(Context, nullptr, EndBB);
  ScalarEvolution SE = buildSE(*F);
  const SCEV *S = SE.getSCEV(Accum);
  Type *I128Ty = Type::getInt128Ty(Context);
  SE.getZeroExtendExpr(S, I128Ty);
}

// Make sure that SCEV doesn't introduce illegal ptrtoint/inttoptr instructions
TEST_F(ScalarEvolutionsTest, SCEVZeroExtendExprNonIntegral) {
  /*
   * Create the following code:
   * func(i64 addrspace(10)* %arg)
   * top:
   *  br label %L.ph
   * L.ph:
   *  br label %L
   * L:
   *  %phi = phi i64 [i64 0, %L.ph], [ %add, %L2 ]
   *  %add = add i64 %phi2, 1
   *  br i1 undef, label %post, label %L2
   * post:
   *  %gepbase = getelementptr i64 addrspace(10)* %arg, i64 1
   *  #= %gep = getelementptr i64 addrspace(10)* %gepbase, i64 %add =#
   *  ret void
   *
   * We will create the appropriate SCEV expression for %gep and expand it,
   * then check that no inttoptr/ptrtoint instructions got inserted.
   */

  // Create a module with non-integral pointers in it's datalayout
  Module NIM("nonintegral", Context);
  std::string DataLayout = M.getDataLayoutStr();
  if (!DataLayout.empty())
    DataLayout += "-";
  DataLayout += "ni:10";
  NIM.setDataLayout(DataLayout);

  Type *T_int1 = Type::getInt1Ty(Context);
  Type *T_int64 = Type::getInt64Ty(Context);
  Type *T_pint64 = T_int64->getPointerTo(10);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), {T_pint64}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", NIM);

  Argument *Arg = &*F->arg_begin();

  BasicBlock *Top = BasicBlock::Create(Context, "top", F);
  BasicBlock *LPh = BasicBlock::Create(Context, "L.ph", F);
  BasicBlock *L = BasicBlock::Create(Context, "L", F);
  BasicBlock *Post = BasicBlock::Create(Context, "post", F);

  IRBuilder<> Builder(Top);
  Builder.CreateBr(LPh);

  Builder.SetInsertPoint(LPh);
  Builder.CreateBr(L);

  Builder.SetInsertPoint(L);
  PHINode *Phi = Builder.CreatePHI(T_int64, 2);
  Value *Add = Builder.CreateAdd(Phi, ConstantInt::get(T_int64, 1), "add");
  Builder.CreateCondBr(UndefValue::get(T_int1), L, Post);
  Phi->addIncoming(ConstantInt::get(T_int64, 0), LPh);
  Phi->addIncoming(Add, L);

  Builder.SetInsertPoint(Post);
  Value *GepBase =
      Builder.CreateGEP(T_int64, Arg, ConstantInt::get(T_int64, 1));
  Instruction *Ret = Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  auto *AddRec =
      SE.getAddRecExpr(SE.getUnknown(GepBase), SE.getConstant(T_int64, 1),
                       LI->getLoopFor(L), SCEV::FlagNUW);

  SCEVExpander Exp(SE, NIM.getDataLayout(), "expander");
  Exp.disableCanonicalMode();
  Exp.expandCodeFor(AddRec, T_pint64, Ret);

  // Make sure none of the instructions inserted were inttoptr/ptrtoint.
  // The verifier will check this.
  EXPECT_FALSE(verifyFunction(*F, &errs()));
}

// Make sure that SCEV invalidates exit limits after invalidating the values it
// depends on when we forget a loop.
TEST_F(ScalarEvolutionsTest, SCEVExitLimitForgetLoop) {
  /*
   * Create the following code:
   * func(i64 addrspace(10)* %arg)
   * top:
   *  br label %L.ph
   * L.ph:
   *  br label %L
   * L:
   *  %phi = phi i64 [i64 0, %L.ph], [ %add, %L2 ]
   *  %add = add i64 %phi2, 1
   *  %cond = icmp slt i64 %add, 1000; then becomes 2000.
   *  br i1 %cond, label %post, label %L2
   * post:
   *  ret void
   *
   */

  // Create a module with non-integral pointers in it's datalayout
  Module NIM("nonintegral", Context);
  std::string DataLayout = M.getDataLayoutStr();
  if (!DataLayout.empty())
    DataLayout += "-";
  DataLayout += "ni:10";
  NIM.setDataLayout(DataLayout);

  Type *T_int64 = Type::getInt64Ty(Context);
  Type *T_pint64 = T_int64->getPointerTo(10);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), {T_pint64}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", NIM);

  BasicBlock *Top = BasicBlock::Create(Context, "top", F);
  BasicBlock *LPh = BasicBlock::Create(Context, "L.ph", F);
  BasicBlock *L = BasicBlock::Create(Context, "L", F);
  BasicBlock *Post = BasicBlock::Create(Context, "post", F);

  IRBuilder<> Builder(Top);
  Builder.CreateBr(LPh);

  Builder.SetInsertPoint(LPh);
  Builder.CreateBr(L);

  Builder.SetInsertPoint(L);
  PHINode *Phi = Builder.CreatePHI(T_int64, 2);
  auto *Add = cast<Instruction>(
      Builder.CreateAdd(Phi, ConstantInt::get(T_int64, 1), "add"));
  auto *Limit = ConstantInt::get(T_int64, 1000);
  auto *Cond = cast<Instruction>(
      Builder.CreateICmp(ICmpInst::ICMP_SLT, Add, Limit, "cond"));
  auto *Br = cast<Instruction>(Builder.CreateCondBr(Cond, L, Post));
  Phi->addIncoming(ConstantInt::get(T_int64, 0), LPh);
  Phi->addIncoming(Add, L);

  Builder.SetInsertPoint(Post);
  Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  auto *Loop = LI->getLoopFor(L);
  const SCEV *EC = SE.getBackedgeTakenCount(Loop);
  EXPECT_FALSE(isa<SCEVCouldNotCompute>(EC));
  EXPECT_TRUE(isa<SCEVConstant>(EC));
  EXPECT_EQ(cast<SCEVConstant>(EC)->getAPInt().getLimitedValue(), 999u);

  // The add recurrence {5,+,1} does not correspond to any PHI in the IR, and
  // that is relevant to this test.
  auto *Five = SE.getConstant(APInt(/*numBits=*/64, 5));
  auto *AR =
      SE.getAddRecExpr(Five, SE.getOne(T_int64), Loop, SCEV::FlagAnyWrap);
  const SCEV *ARAtLoopExit = SE.getSCEVAtScope(AR, nullptr);
  EXPECT_FALSE(isa<SCEVCouldNotCompute>(ARAtLoopExit));
  EXPECT_TRUE(isa<SCEVConstant>(ARAtLoopExit));
  EXPECT_EQ(cast<SCEVConstant>(ARAtLoopExit)->getAPInt().getLimitedValue(),
            1004u);

  SE.forgetLoop(Loop);
  Br->eraseFromParent();
  Cond->eraseFromParent();

  Builder.SetInsertPoint(L);
  auto *NewCond = Builder.CreateICmp(
      ICmpInst::ICMP_SLT, Add, ConstantInt::get(T_int64, 2000), "new.cond");
  Builder.CreateCondBr(NewCond, L, Post);
  const SCEV *NewEC = SE.getBackedgeTakenCount(Loop);
  EXPECT_FALSE(isa<SCEVCouldNotCompute>(NewEC));
  EXPECT_TRUE(isa<SCEVConstant>(NewEC));
  EXPECT_EQ(cast<SCEVConstant>(NewEC)->getAPInt().getLimitedValue(), 1999u);
  const SCEV *NewARAtLoopExit = SE.getSCEVAtScope(AR, nullptr);
  EXPECT_FALSE(isa<SCEVCouldNotCompute>(NewARAtLoopExit));
  EXPECT_TRUE(isa<SCEVConstant>(NewARAtLoopExit));
  EXPECT_EQ(cast<SCEVConstant>(NewARAtLoopExit)->getAPInt().getLimitedValue(),
            2004u);
}

// Make sure that SCEV invalidates exit limits after invalidating the values it
// depends on when we forget a value.
TEST_F(ScalarEvolutionsTest, SCEVExitLimitForgetValue) {
  /*
   * Create the following code:
   * func(i64 addrspace(10)* %arg)
   * top:
   *  br label %L.ph
   * L.ph:
   *  %load = load i64 addrspace(10)* %arg
   *  br label %L
   * L:
   *  %phi = phi i64 [i64 0, %L.ph], [ %add, %L2 ]
   *  %add = add i64 %phi2, 1
   *  %cond = icmp slt i64 %add, %load ; then becomes 2000.
   *  br i1 %cond, label %post, label %L2
   * post:
   *  ret void
   *
   */

  // Create a module with non-integral pointers in it's datalayout
  Module NIM("nonintegral", Context);
  std::string DataLayout = M.getDataLayoutStr();
  if (!DataLayout.empty())
    DataLayout += "-";
  DataLayout += "ni:10";
  NIM.setDataLayout(DataLayout);

  Type *T_int64 = Type::getInt64Ty(Context);
  Type *T_pint64 = T_int64->getPointerTo(10);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), {T_pint64}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", NIM);

  Argument *Arg = &*F->arg_begin();

  BasicBlock *Top = BasicBlock::Create(Context, "top", F);
  BasicBlock *LPh = BasicBlock::Create(Context, "L.ph", F);
  BasicBlock *L = BasicBlock::Create(Context, "L", F);
  BasicBlock *Post = BasicBlock::Create(Context, "post", F);

  IRBuilder<> Builder(Top);
  Builder.CreateBr(LPh);

  Builder.SetInsertPoint(LPh);
  auto *Load = cast<Instruction>(Builder.CreateLoad(T_int64, Arg, "load"));
  Builder.CreateBr(L);

  Builder.SetInsertPoint(L);
  PHINode *Phi = Builder.CreatePHI(T_int64, 2);
  auto *Add = cast<Instruction>(
      Builder.CreateAdd(Phi, ConstantInt::get(T_int64, 1), "add"));
  auto *Cond = cast<Instruction>(
      Builder.CreateICmp(ICmpInst::ICMP_SLT, Add, Load, "cond"));
  auto *Br = cast<Instruction>(Builder.CreateCondBr(Cond, L, Post));
  Phi->addIncoming(ConstantInt::get(T_int64, 0), LPh);
  Phi->addIncoming(Add, L);

  Builder.SetInsertPoint(Post);
  Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  auto *Loop = LI->getLoopFor(L);
  const SCEV *EC = SE.getBackedgeTakenCount(Loop);
  EXPECT_FALSE(isa<SCEVCouldNotCompute>(EC));
  EXPECT_FALSE(isa<SCEVConstant>(EC));

  SE.forgetValue(Load);
  Br->eraseFromParent();
  Cond->eraseFromParent();
  Load->eraseFromParent();

  Builder.SetInsertPoint(L);
  auto *NewCond = Builder.CreateICmp(
      ICmpInst::ICMP_SLT, Add, ConstantInt::get(T_int64, 2000), "new.cond");
  Builder.CreateCondBr(NewCond, L, Post);
  const SCEV *NewEC = SE.getBackedgeTakenCount(Loop);
  EXPECT_FALSE(isa<SCEVCouldNotCompute>(NewEC));
  EXPECT_TRUE(isa<SCEVConstant>(NewEC));
  EXPECT_EQ(cast<SCEVConstant>(NewEC)->getAPInt().getLimitedValue(), 1999u);
}

TEST_F(ScalarEvolutionsTest, SCEVAddRecFromPHIwithLargeConstants) {
  // Reference: https://reviews.llvm.org/D37265
  // Make sure that SCEV does not blow up when constructing an AddRec
  // with predicates for a phi with the update pattern:
  //  (SExt/ZExt ix (Trunc iy (%SymbolicPHI) to ix) to iy) + InvariantAccum
  // when either the initial value of the Phi or the InvariantAccum are
  // constants that are too large to fit in an ix but are zero when truncated to
  // ix.
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), std::vector<Type *>(), false);
  Function *F =
      Function::Create(FTy, Function::ExternalLinkage, "addrecphitest", M);

  /*
    Create IR:
    entry:
     br label %loop
    loop:
     %0 = phi i64 [-9223372036854775808, %entry], [%3, %loop]
     %1 = shl i64 %0, 32
     %2 = ashr exact i64 %1, 32
     %3 = add i64 %2, -9223372036854775808
     br i1 undef, label %exit, label %loop
    exit:
     ret void
   */
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(Context, "loop", F);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "exit", F);

  // entry:
  BranchInst::Create(LoopBB, EntryBB);
  // loop:
  auto *MinInt64 =
      ConstantInt::get(Context, APInt(64, 0x8000000000000000U, true));
  auto *Int64_32 = ConstantInt::get(Context, APInt(64, 32));
  auto *Br = BranchInst::Create(
      LoopBB, ExitBB, UndefValue::get(Type::getInt1Ty(Context)), LoopBB);
  auto *Phi = PHINode::Create(Type::getInt64Ty(Context), 2, "", Br);
  auto *Shl = BinaryOperator::CreateShl(Phi, Int64_32, "", Br);
  auto *AShr = BinaryOperator::CreateExactAShr(Shl, Int64_32, "", Br);
  auto *Add = BinaryOperator::CreateAdd(AShr, MinInt64, "", Br);
  Phi->addIncoming(MinInt64, EntryBB);
  Phi->addIncoming(Add, LoopBB);
  // exit:
  ReturnInst::Create(Context, nullptr, ExitBB);

  // Make sure that SCEV doesn't blow up
  ScalarEvolution SE = buildSE(*F);
  SCEVUnionPredicate Preds;
  const SCEV *Expr = SE.getSCEV(Phi);
  EXPECT_NE(nullptr, Expr);
  EXPECT_TRUE(isa<SCEVUnknown>(Expr));
  auto Result = SE.createAddRecFromPHIWithCasts(cast<SCEVUnknown>(Expr));
}

TEST_F(ScalarEvolutionsTest, SCEVAddRecFromPHIwithLargeConstantAccum) {
  // Make sure that SCEV does not blow up when constructing an AddRec
  // with predicates for a phi with the update pattern:
  //  (SExt/ZExt ix (Trunc iy (%SymbolicPHI) to ix) to iy) + InvariantAccum
  // when the InvariantAccum is a constant that is too large to fit in an
  // ix but are zero when truncated to ix, and the initial value of the
  // phi is not a constant.
  Type *Int32Ty = Type::getInt32Ty(Context);
  SmallVector<Type *, 1> Types;
  Types.push_back(Int32Ty);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), Types, false);
  Function *F =
      Function::Create(FTy, Function::ExternalLinkage, "addrecphitest", M);

  /*
    Create IR:
    define @addrecphitest(i32)
    entry:
     br label %loop
    loop:
     %1 = phi i32 [%0, %entry], [%4, %loop]
     %2 = shl i32 %1, 16
     %3 = ashr exact i32 %2, 16
     %4 = add i32 %3, -2147483648
     br i1 undef, label %exit, label %loop
    exit:
     ret void
   */
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);
  BasicBlock *LoopBB = BasicBlock::Create(Context, "loop", F);
  BasicBlock *ExitBB = BasicBlock::Create(Context, "exit", F);

  // entry:
  BranchInst::Create(LoopBB, EntryBB);
  // loop:
  auto *MinInt32 = ConstantInt::get(Context, APInt(32, 0x80000000U, true));
  auto *Int32_16 = ConstantInt::get(Context, APInt(32, 16));
  auto *Br = BranchInst::Create(
      LoopBB, ExitBB, UndefValue::get(Type::getInt1Ty(Context)), LoopBB);
  auto *Phi = PHINode::Create(Int32Ty, 2, "", Br);
  auto *Shl = BinaryOperator::CreateShl(Phi, Int32_16, "", Br);
  auto *AShr = BinaryOperator::CreateExactAShr(Shl, Int32_16, "", Br);
  auto *Add = BinaryOperator::CreateAdd(AShr, MinInt32, "", Br);
  auto *Arg = &*(F->arg_begin());
  Phi->addIncoming(Arg, EntryBB);
  Phi->addIncoming(Add, LoopBB);
  // exit:
  ReturnInst::Create(Context, nullptr, ExitBB);

  // Make sure that SCEV doesn't blow up
  ScalarEvolution SE = buildSE(*F);
  SCEVUnionPredicate Preds;
  const SCEV *Expr = SE.getSCEV(Phi);
  EXPECT_NE(nullptr, Expr);
  EXPECT_TRUE(isa<SCEVUnknown>(Expr));
  auto Result = SE.createAddRecFromPHIWithCasts(cast<SCEVUnknown>(Expr));
}

TEST_F(ScalarEvolutionsTest, SCEVFoldSumOfTruncs) {
  // Verify that the following SCEV gets folded to a zero:
  //  (-1 * (trunc i64 (-1 * %0) to i32)) + (-1 * (trunc i64 %0 to i32)
  Type *ArgTy = Type::getInt64Ty(Context);
  Type *Int32Ty = Type::getInt32Ty(Context);
  SmallVector<Type *, 1> Types;
  Types.push_back(ArgTy);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), Types, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", M);
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  ReturnInst::Create(Context, nullptr, BB);

  ScalarEvolution SE = buildSE(*F);

  auto *Arg = &*(F->arg_begin());
  const auto *ArgSCEV = SE.getSCEV(Arg);

  // Build the SCEV
  const auto *A0 = SE.getNegativeSCEV(ArgSCEV);
  const auto *A1 = SE.getTruncateExpr(A0, Int32Ty);
  const auto *A = SE.getNegativeSCEV(A1);

  const auto *B0 = SE.getTruncateExpr(ArgSCEV, Int32Ty);
  const auto *B = SE.getNegativeSCEV(B0);

  const auto *Expr = SE.getAddExpr(A, B);
  // Verify that the SCEV was folded to 0
  const auto *ZeroConst = SE.getConstant(Int32Ty, 0);
  EXPECT_EQ(Expr, ZeroConst);
}

// Check that we can correctly identify the points at which the SCEV of the
// AddRec can be expanded.
TEST_F(ScalarEvolutionsTest, SCEVExpanderIsSafeToExpandAt) {
  /*
   * Create the following code:
   * func(i64 addrspace(10)* %arg)
   * top:
   *  br label %L.ph
   * L.ph:
   *  br label %L
   * L:
   *  %phi = phi i64 [i64 0, %L.ph], [ %add, %L2 ]
   *  %add = add i64 %phi2, 1
   *  %cond = icmp slt i64 %add, 1000; then becomes 2000.
   *  br i1 %cond, label %post, label %L2
   * post:
   *  ret void
   *
   */

  // Create a module with non-integral pointers in it's datalayout
  Module NIM("nonintegral", Context);
  std::string DataLayout = M.getDataLayoutStr();
  if (!DataLayout.empty())
    DataLayout += "-";
  DataLayout += "ni:10";
  NIM.setDataLayout(DataLayout);

  Type *T_int64 = Type::getInt64Ty(Context);
  Type *T_pint64 = T_int64->getPointerTo(10);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), {T_pint64}, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "foo", NIM);

  BasicBlock *Top = BasicBlock::Create(Context, "top", F);
  BasicBlock *LPh = BasicBlock::Create(Context, "L.ph", F);
  BasicBlock *L = BasicBlock::Create(Context, "L", F);
  BasicBlock *Post = BasicBlock::Create(Context, "post", F);

  IRBuilder<> Builder(Top);
  Builder.CreateBr(LPh);

  Builder.SetInsertPoint(LPh);
  Builder.CreateBr(L);

  Builder.SetInsertPoint(L);
  PHINode *Phi = Builder.CreatePHI(T_int64, 2);
  auto *Add = cast<Instruction>(
      Builder.CreateAdd(Phi, ConstantInt::get(T_int64, 1), "add"));
  auto *Limit = ConstantInt::get(T_int64, 1000);
  auto *Cond = cast<Instruction>(
      Builder.CreateICmp(ICmpInst::ICMP_SLT, Add, Limit, "cond"));
  Builder.CreateCondBr(Cond, L, Post);
  Phi->addIncoming(ConstantInt::get(T_int64, 0), LPh);
  Phi->addIncoming(Add, L);

  Builder.SetInsertPoint(Post);
  Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  const SCEV *S = SE.getSCEV(Phi);
  EXPECT_TRUE(isa<SCEVAddRecExpr>(S));
  const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(S);
  EXPECT_TRUE(AR->isAffine());
  EXPECT_FALSE(isSafeToExpandAt(AR, Top->getTerminator(), SE));
  EXPECT_FALSE(isSafeToExpandAt(AR, LPh->getTerminator(), SE));
  EXPECT_TRUE(isSafeToExpandAt(AR, L->getTerminator(), SE));
  EXPECT_TRUE(isSafeToExpandAt(AR, Post->getTerminator(), SE));
}

// Check that SCEV expander does not use the nuw instruction
// for expansion.
TEST_F(ScalarEvolutionsTest, SCEVExpanderNUW) {
  /*
   * Create the following code:
   * func(i64 %a)
   * entry:
   *   br false, label %exit, label %body
   * body:
   *  %s1 = add i64 %a, -1
   *  br label %exit
   * exit:
   *  %s = add nuw i64 %a, -1
   *  ret %s
   */

  // Create a module.
  Module M("SCEVExpanderNUW", Context);

  Type *T_int64 = Type::getInt64Ty(Context);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), { T_int64 }, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "func", M);
  Argument *Arg = &*F->arg_begin();
  ConstantInt *C = ConstantInt::get(Context, APInt(64, -1));

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  BasicBlock *Body = BasicBlock::Create(Context, "body", F);
  BasicBlock *Exit = BasicBlock::Create(Context, "exit", F);

  IRBuilder<> Builder(Entry);
  ConstantInt *Cond = ConstantInt::get(Context, APInt(1, 0));
  Builder.CreateCondBr(Cond, Exit, Body);

  Builder.SetInsertPoint(Body);
  auto *S1 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  Builder.CreateBr(Exit);

  Builder.SetInsertPoint(Exit);
  auto *S2 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  S2->setHasNoUnsignedWrap(true);
  auto *R = cast<Instruction>(Builder.CreateRetVoid());

  ScalarEvolution SE = buildSE(*F);
  const SCEV *S = SE.getSCEV(S1);
  EXPECT_TRUE(isa<SCEVAddExpr>(S));
  SCEVExpander Exp(SE, M.getDataLayout(), "expander");
  auto *I = cast<Instruction>(Exp.expandCodeFor(S, nullptr, R));
  EXPECT_FALSE(I->hasNoUnsignedWrap());
}

// Check that SCEV expander does not use the nsw instruction
// for expansion.
TEST_F(ScalarEvolutionsTest, SCEVExpanderNSW) {
  /*
   * Create the following code:
   * func(i64 %a)
   * entry:
   *   br false, label %exit, label %body
   * body:
   *  %s1 = add i64 %a, -1
   *  br label %exit
   * exit:
   *  %s = add nsw i64 %a, -1
   *  ret %s
   */

  // Create a module.
  Module M("SCEVExpanderNSW", Context);

  Type *T_int64 = Type::getInt64Ty(Context);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), { T_int64 }, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "func", M);
  Argument *Arg = &*F->arg_begin();
  ConstantInt *C = ConstantInt::get(Context, APInt(64, -1));

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  BasicBlock *Body = BasicBlock::Create(Context, "body", F);
  BasicBlock *Exit = BasicBlock::Create(Context, "exit", F);

  IRBuilder<> Builder(Entry);
  ConstantInt *Cond = ConstantInt::get(Context, APInt(1, 0));
  Builder.CreateCondBr(Cond, Exit, Body);

  Builder.SetInsertPoint(Body);
  auto *S1 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  Builder.CreateBr(Exit);

  Builder.SetInsertPoint(Exit);
  auto *S2 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  S2->setHasNoSignedWrap(true);
  auto *R = cast<Instruction>(Builder.CreateRetVoid());

  ScalarEvolution SE = buildSE(*F);
  const SCEV *S = SE.getSCEV(S1);
  EXPECT_TRUE(isa<SCEVAddExpr>(S));
  SCEVExpander Exp(SE, M.getDataLayout(), "expander");
  auto *I = cast<Instruction>(Exp.expandCodeFor(S, nullptr, R));
  EXPECT_FALSE(I->hasNoSignedWrap());
}

// Check that SCEV does not save the SCEV -> V
// mapping of SCEV differ from V in NUW flag.
TEST_F(ScalarEvolutionsTest, SCEVCacheNUW) {
  /*
   * Create the following code:
   * func(i64 %a)
   * entry:
   *  %s1 = add i64 %a, -1
   *  %s2 = add nuw i64 %a, -1
   *  br label %exit
   * exit:
   *  ret %s
   */

  // Create a module.
  Module M("SCEVCacheNUW", Context);

  Type *T_int64 = Type::getInt64Ty(Context);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), { T_int64 }, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "func", M);
  Argument *Arg = &*F->arg_begin();
  ConstantInt *C = ConstantInt::get(Context, APInt(64, -1));

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  BasicBlock *Exit = BasicBlock::Create(Context, "exit", F);

  IRBuilder<> Builder(Entry);
  auto *S1 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  auto *S2 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  S2->setHasNoUnsignedWrap(true);
  Builder.CreateBr(Exit);

  Builder.SetInsertPoint(Exit);
  auto *R = cast<Instruction>(Builder.CreateRetVoid());

  ScalarEvolution SE = buildSE(*F);
  // Get S2 first to move it to cache.
  const SCEV *SC2 = SE.getSCEV(S2);
  EXPECT_TRUE(isa<SCEVAddExpr>(SC2));
  // Now get S1.
  const SCEV *SC1 = SE.getSCEV(S1);
  EXPECT_TRUE(isa<SCEVAddExpr>(SC1));
  // Expand for S1, it should use S1 not S2 in spite S2
  // first in the cache.
  SCEVExpander Exp(SE, M.getDataLayout(), "expander");
  auto *I = cast<Instruction>(Exp.expandCodeFor(SC1, nullptr, R));
  EXPECT_FALSE(I->hasNoUnsignedWrap());
}

// Check that SCEV does not save the SCEV -> V
// mapping of SCEV differ from V in NSW flag.
TEST_F(ScalarEvolutionsTest, SCEVCacheNSW) {
  /*
   * Create the following code:
   * func(i64 %a)
   * entry:
   *  %s1 = add i64 %a, -1
   *  %s2 = add nsw i64 %a, -1
   *  br label %exit
   * exit:
   *  ret %s
   */

  // Create a module.
  Module M("SCEVCacheNUW", Context);

  Type *T_int64 = Type::getInt64Ty(Context);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), { T_int64 }, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "func", M);
  Argument *Arg = &*F->arg_begin();
  ConstantInt *C = ConstantInt::get(Context, APInt(64, -1));

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  BasicBlock *Exit = BasicBlock::Create(Context, "exit", F);

  IRBuilder<> Builder(Entry);
  auto *S1 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  auto *S2 = cast<Instruction>(Builder.CreateAdd(Arg, C, "add"));
  S2->setHasNoSignedWrap(true);
  Builder.CreateBr(Exit);

  Builder.SetInsertPoint(Exit);
  auto *R = cast<Instruction>(Builder.CreateRetVoid());

  ScalarEvolution SE = buildSE(*F);
  // Get S2 first to move it to cache.
  const SCEV *SC2 = SE.getSCEV(S2);
  EXPECT_TRUE(isa<SCEVAddExpr>(SC2));
  // Now get S1.
  const SCEV *SC1 = SE.getSCEV(S1);
  EXPECT_TRUE(isa<SCEVAddExpr>(SC1));
  // Expand for S1, it should use S1 not S2 in spite S2
  // first in the cache.
  SCEVExpander Exp(SE, M.getDataLayout(), "expander");
  auto *I = cast<Instruction>(Exp.expandCodeFor(SC1, nullptr, R));
  EXPECT_FALSE(I->hasNoSignedWrap());
}

// Check logic of SCEV expression size computation.
TEST_F(ScalarEvolutionsTest, SCEVComputeExpressionSize) {
  /*
   * Create the following code:
   * void func(i64 %a, i64 %b)
   * entry:
   *  %s1 = add i64 %a, 1
   *  %s2 = udiv i64 %s1, %b
   *  br label %exit
   * exit:
   *  ret
   */

  // Create a module.
  Module M("SCEVComputeExpressionSize", Context);

  Type *T_int64 = Type::getInt64Ty(Context);

  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), { T_int64, T_int64 }, false);
  Function *F = Function::Create(FTy, Function::ExternalLinkage, "func", M);
  Argument *A = &*F->arg_begin();
  Argument *B = &*std::next(F->arg_begin());
  ConstantInt *C = ConstantInt::get(Context, APInt(64, 1));

  BasicBlock *Entry = BasicBlock::Create(Context, "entry", F);
  BasicBlock *Exit = BasicBlock::Create(Context, "exit", F);

  IRBuilder<> Builder(Entry);
  auto *S1 = cast<Instruction>(Builder.CreateAdd(A, C, "s1"));
  auto *S2 = cast<Instruction>(Builder.CreateUDiv(S1, B, "s2"));
  Builder.CreateBr(Exit);

  Builder.SetInsertPoint(Exit);
  Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  // Get S2 first to move it to cache.
  const SCEV *AS = SE.getSCEV(A);
  const SCEV *BS = SE.getSCEV(B);
  const SCEV *CS = SE.getSCEV(C);
  const SCEV *S1S = SE.getSCEV(S1);
  const SCEV *S2S = SE.getSCEV(S2);
  EXPECT_EQ(AS->getExpressionSize(), 1u);
  EXPECT_EQ(BS->getExpressionSize(), 1u);
  EXPECT_EQ(CS->getExpressionSize(), 1u);
  EXPECT_EQ(S1S->getExpressionSize(), 3u);
  EXPECT_EQ(S2S->getExpressionSize(), 5u);
}

TEST_F(ScalarEvolutionsTest, SCEVExpandInsertCanonicalIV) {
  LLVMContext C;
  SMDiagnostic Err;

  // Expand the addrec produced by GetAddRec into a loop without a canonical IV.
  // SCEVExpander will insert one.
  auto TestNoCanonicalIV = [&](
      std::function<const SCEV *(ScalarEvolution & SE, Loop * L)> GetAddRec) {
    std::unique_ptr<Module> M =
        parseAssemblyString("define i32 @test(i32 %limit) { "
                            "entry: "
                            "  br label %loop "
                            "loop: "
                            "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
                            "  %i.inc = add nsw i32 %i, 1 "
                            "  %cont = icmp slt i32 %i.inc, %limit "
                            "  br i1 %cont, label %loop, label %exit "
                            "exit: "
                            "  ret i32 %i.inc "
                            "}",
                            Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
      auto &I = GetInstByName(F, "i");
      auto *Loop = LI.getLoopFor(I.getParent());
      EXPECT_FALSE(Loop->getCanonicalInductionVariable());

      auto *AR = GetAddRec(SE, Loop);
      unsigned ExpectedCanonicalIVWidth = SE.getTypeSizeInBits(AR->getType());

      SCEVExpander Exp(SE, M->getDataLayout(), "expander");
      auto *InsertAt = I.getNextNode();
      Exp.expandCodeFor(AR, nullptr, InsertAt);
      PHINode *CanonicalIV = Loop->getCanonicalInductionVariable();
      unsigned CanonicalIVBitWidth =
          cast<IntegerType>(CanonicalIV->getType())->getBitWidth();
      EXPECT_EQ(CanonicalIVBitWidth, ExpectedCanonicalIVWidth);
    });
  };

  // Expand the addrec produced by GetAddRec into a loop with a canonical IV
  // which is narrower than addrec type.
  // SCEVExpander will insert a canonical IV of a wider type to expand the
  // addrec.
  auto TestNarrowCanonicalIV = [&](
      std::function<const SCEV *(ScalarEvolution & SE, Loop * L)> GetAddRec) {
    std::unique_ptr<Module> M = parseAssemblyString(
        "define i32 @test(i32 %limit) { "
        "entry: "
        "  br label %loop "
        "loop: "
        "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
        "  %canonical.iv = phi i8 [ 0, %entry ], [ %canonical.iv.inc, %loop ] "
        "  %i.inc = add nsw i32 %i, 1 "
        "  %canonical.iv.inc = add i8 %canonical.iv, 1 "
        "  %cont = icmp slt i32 %i.inc, %limit "
        "  br i1 %cont, label %loop, label %exit "
        "exit: "
        "  ret i32 %i.inc "
        "}",
        Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
      auto &I = GetInstByName(F, "i");

      auto *LoopHeaderBB = I.getParent();
      auto *Loop = LI.getLoopFor(LoopHeaderBB);
      PHINode *CanonicalIV = Loop->getCanonicalInductionVariable();
      EXPECT_EQ(CanonicalIV, &GetInstByName(F, "canonical.iv"));

      auto *AR = GetAddRec(SE, Loop);

      unsigned ExpectedCanonicalIVWidth = SE.getTypeSizeInBits(AR->getType());
      unsigned CanonicalIVBitWidth =
          cast<IntegerType>(CanonicalIV->getType())->getBitWidth();
      EXPECT_LT(CanonicalIVBitWidth, ExpectedCanonicalIVWidth);

      SCEVExpander Exp(SE, M->getDataLayout(), "expander");
      auto *InsertAt = I.getNextNode();
      Exp.expandCodeFor(AR, nullptr, InsertAt);

      // Loop over all of the PHI nodes, looking for the new canonical indvar.
      PHINode *NewCanonicalIV = nullptr;
      for (BasicBlock::iterator i = LoopHeaderBB->begin(); isa<PHINode>(i);
           ++i) {
        PHINode *PN = cast<PHINode>(i);
        if (PN == &I || PN == CanonicalIV)
          continue;
        // We expect that the only PHI added is the new canonical IV
        EXPECT_FALSE(NewCanonicalIV);
        NewCanonicalIV = PN;
      }

      // Check that NewCanonicalIV is a canonical IV, i.e {0,+,1}
      BasicBlock *Incoming = nullptr, *Backedge = nullptr;
      EXPECT_TRUE(Loop->getIncomingAndBackEdge(Incoming, Backedge));
      auto *Start = NewCanonicalIV->getIncomingValueForBlock(Incoming);
      EXPECT_TRUE(isa<ConstantInt>(Start));
      EXPECT_TRUE(dyn_cast<ConstantInt>(Start)->isZero());
      auto *Next = NewCanonicalIV->getIncomingValueForBlock(Backedge);
      EXPECT_TRUE(isa<BinaryOperator>(Next));
      auto *NextBinOp = dyn_cast<BinaryOperator>(Next);
      EXPECT_EQ(NextBinOp->getOpcode(), Instruction::Add);
      EXPECT_EQ(NextBinOp->getOperand(0), NewCanonicalIV);
      auto *Step = NextBinOp->getOperand(1);
      EXPECT_TRUE(isa<ConstantInt>(Step));
      EXPECT_TRUE(dyn_cast<ConstantInt>(Step)->isOne());

      unsigned NewCanonicalIVBitWidth =
          cast<IntegerType>(NewCanonicalIV->getType())->getBitWidth();
      EXPECT_EQ(NewCanonicalIVBitWidth, ExpectedCanonicalIVWidth);
    });
  };

  // Expand the addrec produced by GetAddRec into a loop with a canonical IV
  // of addrec width.
  // To expand the addrec SCEVExpander should use the existing canonical IV.
  auto TestMatchingCanonicalIV = [&](
      std::function<const SCEV *(ScalarEvolution & SE, Loop * L)> GetAddRec,
      unsigned ARBitWidth) {
    auto ARBitWidthTypeStr = "i" + std::to_string(ARBitWidth);
    std::unique_ptr<Module> M = parseAssemblyString(
        "define i32 @test(i32 %limit) { "
        "entry: "
        "  br label %loop "
        "loop: "
        "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
        "  %canonical.iv = phi " + ARBitWidthTypeStr +
            " [ 0, %entry ], [ %canonical.iv.inc, %loop ] "
        "  %i.inc = add nsw i32 %i, 1 "
        "  %canonical.iv.inc = add " + ARBitWidthTypeStr +
            " %canonical.iv, 1 "
        "  %cont = icmp slt i32 %i.inc, %limit "
        "  br i1 %cont, label %loop, label %exit "
        "exit: "
        "  ret i32 %i.inc "
        "}",
        Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
      auto &I = GetInstByName(F, "i");
      auto &CanonicalIV = GetInstByName(F, "canonical.iv");

      auto *LoopHeaderBB = I.getParent();
      auto *Loop = LI.getLoopFor(LoopHeaderBB);
      EXPECT_EQ(&CanonicalIV, Loop->getCanonicalInductionVariable());
      unsigned CanonicalIVBitWidth =
          cast<IntegerType>(CanonicalIV.getType())->getBitWidth();

      auto *AR = GetAddRec(SE, Loop);
      EXPECT_EQ(ARBitWidth, SE.getTypeSizeInBits(AR->getType()));
      EXPECT_EQ(CanonicalIVBitWidth, ARBitWidth);

      SCEVExpander Exp(SE, M->getDataLayout(), "expander");
      auto *InsertAt = I.getNextNode();
      Exp.expandCodeFor(AR, nullptr, InsertAt);

      // Loop over all of the PHI nodes, looking if a new canonical indvar was
      // introduced.
      PHINode *NewCanonicalIV = nullptr;
      for (BasicBlock::iterator i = LoopHeaderBB->begin(); isa<PHINode>(i);
           ++i) {
        PHINode *PN = cast<PHINode>(i);
        if (PN == &I || PN == &CanonicalIV)
          continue;
        NewCanonicalIV = PN;
      }
      EXPECT_FALSE(NewCanonicalIV);
    });
  };

  unsigned ARBitWidth = 16;
  Type *ARType = IntegerType::get(C, ARBitWidth);

  // Expand {5,+,1}
  auto GetAR2 = [&](ScalarEvolution &SE, Loop *L) -> const SCEV * {
    return SE.getAddRecExpr(SE.getConstant(APInt(ARBitWidth, 5)),
                            SE.getOne(ARType), L, SCEV::FlagAnyWrap);
  };
  TestNoCanonicalIV(GetAR2);
  TestNarrowCanonicalIV(GetAR2);
  TestMatchingCanonicalIV(GetAR2, ARBitWidth);
}

TEST_F(ScalarEvolutionsTest, SCEVExpanderShlNSW) {

  auto checkOneCase = [this](std::string &&str) {
    LLVMContext C;
    SMDiagnostic Err;
    std::unique_ptr<Module> M = parseAssemblyString(str, Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    Function *F = M->getFunction("f");
    ASSERT_NE(F, nullptr) << "Could not find function 'f'";

    BasicBlock &Entry = F->getEntryBlock();
    LoadInst *Load = cast<LoadInst>(&Entry.front());
    BinaryOperator *And = cast<BinaryOperator>(*Load->user_begin());

    ScalarEvolution SE = buildSE(*F);
    const SCEV *AndSCEV = SE.getSCEV(And);
    EXPECT_TRUE(isa<SCEVMulExpr>(AndSCEV));
    EXPECT_TRUE(cast<SCEVMulExpr>(AndSCEV)->hasNoSignedWrap());

    SCEVExpander Exp(SE, M->getDataLayout(), "expander");
    auto *I = cast<Instruction>(Exp.expandCodeFor(AndSCEV, nullptr, And));
    EXPECT_EQ(I->getOpcode(), Instruction::Shl);
    EXPECT_FALSE(I->hasNoSignedWrap());
  };

  checkOneCase("define void @f(i16* %arrayidx) { "
               "  %1 = load i16, i16* %arrayidx "
               "  %2 = and i16 %1, -32768 "
               "  ret void "
               "} ");

  checkOneCase("define void @f(i8* %arrayidx) { "
               "  %1 = load i8, i8* %arrayidx "
               "  %2 = and i8 %1, -128 "
               "  ret void "
               "} ");
}

TEST_F(ScalarEvolutionsTest, SCEVComputeConstantDifference) {
  LLVMContext C;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseAssemblyString(
      "define void @foo(i32 %sz, i32 %pp) { "
      "entry: "
      "  %v0 = add i32 %pp, 0 "
      "  %v3 = add i32 %pp, 3 "
      "  br label %loop.body "
      "loop.body: "
      "  %iv = phi i32 [ %iv.next, %loop.body ], [ 0, %entry ] "
      "  %xa = add nsw i32 %iv, %v0 "
      "  %yy = add nsw i32 %iv, %v3 "
      "  %xb = sub nsw i32 %yy, 3 "
      "  %iv.next = add nsw i32 %iv, 1 "
      "  %cmp = icmp sle i32 %iv.next, %sz "
      "  br i1 %cmp, label %loop.body, label %exit "
      "exit: "
      "  ret void "
      "} ",
      Err, C);

  ASSERT_TRUE(M && "Could not parse module?");
  ASSERT_TRUE(!verifyModule(*M) && "Must have been well formed!");

  runWithSE(*M, "foo", [](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    auto *ScevV0 = SE.getSCEV(getInstructionByName(F, "v0")); // %pp
    auto *ScevV3 = SE.getSCEV(getInstructionByName(F, "v3")); // (3 + %pp)
    auto *ScevIV = SE.getSCEV(getInstructionByName(F, "iv")); // {0,+,1}
    auto *ScevXA = SE.getSCEV(getInstructionByName(F, "xa")); // {%pp,+,1}
    auto *ScevYY = SE.getSCEV(getInstructionByName(F, "yy")); // {(3 + %pp),+,1}
    auto *ScevXB = SE.getSCEV(getInstructionByName(F, "xb")); // {%pp,+,1}
    auto *ScevIVNext = SE.getSCEV(getInstructionByName(F, "iv.next")); // {1,+,1}

    auto diff = [&SE](const SCEV *LHS, const SCEV *RHS) -> Optional<int> {
      auto ConstantDiffOrNone = computeConstantDifference(SE, LHS, RHS);
      if (!ConstantDiffOrNone)
        return None;

      auto ExtDiff = ConstantDiffOrNone->getSExtValue();
      int Diff = ExtDiff;
      assert(Diff == ExtDiff && "Integer overflow");
      return Diff;
    };

    EXPECT_EQ(diff(ScevV3, ScevV0), 3);
    EXPECT_EQ(diff(ScevV0, ScevV3), -3);
    EXPECT_EQ(diff(ScevV0, ScevV0), 0);
    EXPECT_EQ(diff(ScevV3, ScevV3), 0);
    EXPECT_EQ(diff(ScevIV, ScevIV), 0);
    EXPECT_EQ(diff(ScevXA, ScevXB), 0);
    EXPECT_EQ(diff(ScevXA, ScevYY), -3);
    EXPECT_EQ(diff(ScevYY, ScevXB), 3);
    EXPECT_EQ(diff(ScevIV, ScevIVNext), -1);
    EXPECT_EQ(diff(ScevIVNext, ScevIV), 1);
    EXPECT_EQ(diff(ScevIVNext, ScevIVNext), 0);
    EXPECT_EQ(diff(ScevV0, ScevIV), None);
    EXPECT_EQ(diff(ScevIVNext, ScevV3), None);
    EXPECT_EQ(diff(ScevYY, ScevV3), None);
  });
}

// Test expansion of nested addrecs in CanonicalMode.
// Expanding nested addrecs in canonical mode requiers a canonical IV of a
// type wider than the type of the addrec itself. Currently, SCEVExpander
// just falls back to literal mode for nested addrecs.
TEST_F(ScalarEvolutionsTest, SCEVExpandNonAffineAddRec) {
  LLVMContext C;
  SMDiagnostic Err;

  // Expand the addrec produced by GetAddRec into a loop without a canonical IV.
  auto TestNoCanonicalIV = [&](std::function<const SCEVAddRecExpr *(
                                   ScalarEvolution & SE, Loop * L)> GetAddRec) {
    std::unique_ptr<Module> M =
        parseAssemblyString("define i32 @test(i32 %limit) { "
                            "entry: "
                            "  br label %loop "
                            "loop: "
                            "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
                            "  %i.inc = add nsw i32 %i, 1 "
                            "  %cont = icmp slt i32 %i.inc, %limit "
                            "  br i1 %cont, label %loop, label %exit "
                            "exit: "
                            "  ret i32 %i.inc "
                            "}",
                            Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
      auto &I = GetInstByName(F, "i");
      auto *Loop = LI.getLoopFor(I.getParent());
      EXPECT_FALSE(Loop->getCanonicalInductionVariable());

      auto *AR = GetAddRec(SE, Loop);
      EXPECT_FALSE(AR->isAffine());

      SCEVExpander Exp(SE, M->getDataLayout(), "expander");
      auto *InsertAt = I.getNextNode();
      Value *V = Exp.expandCodeFor(AR, nullptr, InsertAt);
      auto *ExpandedAR = SE.getSCEV(V);
      // Check that the expansion happened literally.
      EXPECT_EQ(AR, ExpandedAR);
    });
  };

  // Expand the addrec produced by GetAddRec into a loop with a canonical IV
  // which is narrower than addrec type.
  auto TestNarrowCanonicalIV = [&](
      std::function<const SCEVAddRecExpr *(ScalarEvolution & SE, Loop * L)>
          GetAddRec) {
    std::unique_ptr<Module> M = parseAssemblyString(
        "define i32 @test(i32 %limit) { "
        "entry: "
        "  br label %loop "
        "loop: "
        "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
        "  %canonical.iv = phi i8 [ 0, %entry ], [ %canonical.iv.inc, %loop ] "
        "  %i.inc = add nsw i32 %i, 1 "
        "  %canonical.iv.inc = add i8 %canonical.iv, 1 "
        "  %cont = icmp slt i32 %i.inc, %limit "
        "  br i1 %cont, label %loop, label %exit "
        "exit: "
        "  ret i32 %i.inc "
        "}",
        Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
      auto &I = GetInstByName(F, "i");

      auto *LoopHeaderBB = I.getParent();
      auto *Loop = LI.getLoopFor(LoopHeaderBB);
      PHINode *CanonicalIV = Loop->getCanonicalInductionVariable();
      EXPECT_EQ(CanonicalIV, &GetInstByName(F, "canonical.iv"));

      auto *AR = GetAddRec(SE, Loop);
      EXPECT_FALSE(AR->isAffine());

      unsigned ExpectedCanonicalIVWidth = SE.getTypeSizeInBits(AR->getType());
      unsigned CanonicalIVBitWidth =
          cast<IntegerType>(CanonicalIV->getType())->getBitWidth();
      EXPECT_LT(CanonicalIVBitWidth, ExpectedCanonicalIVWidth);

      SCEVExpander Exp(SE, M->getDataLayout(), "expander");
      auto *InsertAt = I.getNextNode();
      Value *V = Exp.expandCodeFor(AR, nullptr, InsertAt);
      auto *ExpandedAR = SE.getSCEV(V);
      // Check that the expansion happened literally.
      EXPECT_EQ(AR, ExpandedAR);
    });
  };

  // Expand the addrec produced by GetAddRec into a loop with a canonical IV
  // of addrec width.
  auto TestMatchingCanonicalIV = [&](
      std::function<const SCEVAddRecExpr *(ScalarEvolution & SE, Loop * L)>
          GetAddRec,
      unsigned ARBitWidth) {
    auto ARBitWidthTypeStr = "i" + std::to_string(ARBitWidth);
    std::unique_ptr<Module> M = parseAssemblyString(
        "define i32 @test(i32 %limit) { "
        "entry: "
        "  br label %loop "
        "loop: "
        "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
        "  %canonical.iv = phi " + ARBitWidthTypeStr +
            " [ 0, %entry ], [ %canonical.iv.inc, %loop ] "
        "  %i.inc = add nsw i32 %i, 1 "
        "  %canonical.iv.inc = add " + ARBitWidthTypeStr +
            " %canonical.iv, 1 "
        "  %cont = icmp slt i32 %i.inc, %limit "
        "  br i1 %cont, label %loop, label %exit "
        "exit: "
        "  ret i32 %i.inc "
        "}",
        Err, C);

    assert(M && "Could not parse module?");
    assert(!verifyModule(*M) && "Must have been well formed!");

    runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
      auto &I = GetInstByName(F, "i");
      auto &CanonicalIV = GetInstByName(F, "canonical.iv");

      auto *LoopHeaderBB = I.getParent();
      auto *Loop = LI.getLoopFor(LoopHeaderBB);
      EXPECT_EQ(&CanonicalIV, Loop->getCanonicalInductionVariable());
      unsigned CanonicalIVBitWidth =
          cast<IntegerType>(CanonicalIV.getType())->getBitWidth();

      auto *AR = GetAddRec(SE, Loop);
      EXPECT_FALSE(AR->isAffine());
      EXPECT_EQ(ARBitWidth, SE.getTypeSizeInBits(AR->getType()));
      EXPECT_EQ(CanonicalIVBitWidth, ARBitWidth);

      SCEVExpander Exp(SE, M->getDataLayout(), "expander");
      auto *InsertAt = I.getNextNode();
      Value *V = Exp.expandCodeFor(AR, nullptr, InsertAt);
      auto *ExpandedAR = SE.getSCEV(V);
      // Check that the expansion happened literally.
      EXPECT_EQ(AR, ExpandedAR);
    });
  };

  unsigned ARBitWidth = 16;
  Type *ARType = IntegerType::get(C, ARBitWidth);

  // Expand {5,+,1,+,1}
  auto GetAR3 = [&](ScalarEvolution &SE, Loop *L) -> const SCEVAddRecExpr * {
    SmallVector<const SCEV *, 3> Ops = {SE.getConstant(APInt(ARBitWidth, 5)),
                                        SE.getOne(ARType), SE.getOne(ARType)};
    return cast<SCEVAddRecExpr>(SE.getAddRecExpr(Ops, L, SCEV::FlagAnyWrap));
  };
  TestNoCanonicalIV(GetAR3);
  TestNarrowCanonicalIV(GetAR3);
  TestMatchingCanonicalIV(GetAR3, ARBitWidth);

  // Expand {5,+,1,+,1,+,1}
  auto GetAR4 = [&](ScalarEvolution &SE, Loop *L) -> const SCEVAddRecExpr * {
    SmallVector<const SCEV *, 4> Ops = {SE.getConstant(APInt(ARBitWidth, 5)),
                                        SE.getOne(ARType), SE.getOne(ARType),
                                        SE.getOne(ARType)};
    return cast<SCEVAddRecExpr>(SE.getAddRecExpr(Ops, L, SCEV::FlagAnyWrap));
  };
  TestNoCanonicalIV(GetAR4);
  TestNarrowCanonicalIV(GetAR4);
  TestMatchingCanonicalIV(GetAR4, ARBitWidth);

  // Expand {5,+,1,+,1,+,1,+,1}
  auto GetAR5 = [&](ScalarEvolution &SE, Loop *L) -> const SCEVAddRecExpr * {
    SmallVector<const SCEV *, 5> Ops = {SE.getConstant(APInt(ARBitWidth, 5)),
                                        SE.getOne(ARType), SE.getOne(ARType),
                                        SE.getOne(ARType), SE.getOne(ARType)};
    return cast<SCEVAddRecExpr>(SE.getAddRecExpr(Ops, L, SCEV::FlagAnyWrap));
  };
  TestNoCanonicalIV(GetAR5);
  TestNarrowCanonicalIV(GetAR5);
  TestMatchingCanonicalIV(GetAR5, ARBitWidth);
}

}  // end namespace llvm
