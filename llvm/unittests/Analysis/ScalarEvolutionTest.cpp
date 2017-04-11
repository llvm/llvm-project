//===- ScalarEvolutionsTest.cpp - ScalarEvolution unit tests --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {
namespace {

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

  void runWithFunctionAndSE(
      Module &M, StringRef FuncName,
      function_ref<void(Function &F, ScalarEvolution &SE)> Test) {
    auto *F = M.getFunction(FuncName);
    ASSERT_NE(F, nullptr) << "Could not find " << FuncName;
    ScalarEvolution SE = buildSE(*F);
    Test(*F, SE);
  }
};

TEST_F(ScalarEvolutionsTest, SCEVUnknownRAUW) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context),
                                              std::vector<Type *>(), false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
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

TEST_F(ScalarEvolutionsTest, SCEVMultiplyAddRecs) {
  Type *Ty = Type::getInt32Ty(Context);
  SmallVector<Type *, 10> Types;
  Types.append(10, Ty);
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context), Types, false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  BasicBlock *BB = BasicBlock::Create(Context, "entry", F);
  ReturnInst::Create(Context, nullptr, BB);

  ScalarEvolution SE = buildSE(*F);

  // It's possible to produce an empty loop through the default constructor,
  // but you can't add any blocks to it without a LoopInfo pass.
  Loop L;
  const_cast<std::vector<BasicBlock*>&>(L.getBlocks()).push_back(BB);

  Function::arg_iterator AI = F->arg_begin();
  SmallVector<const SCEV *, 5> A;
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  A.push_back(SE.getSCEV(&*AI++));
  const SCEV *A_rec = SE.getAddRecExpr(A, &L, SCEV::FlagAnyWrap);

  SmallVector<const SCEV *, 5> B;
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  B.push_back(SE.getSCEV(&*AI++));
  const SCEV *B_rec = SE.getAddRecExpr(B, &L, SCEV::FlagAnyWrap);

  /* Spot check that we perform this transformation:
     {A0,+,A1,+,A2,+,A3,+,A4} * {B0,+,B1,+,B2,+,B3,+,B4} =
     {A0*B0,+,
      A1*B0 + A0*B1 + A1*B1,+,
      A2*B0 + 2A1*B1 + A0*B2 + 2A2*B1 + 2A1*B2 + A2*B2,+,
      A3*B0 + 3A2*B1 + 3A1*B2 + A0*B3 + 3A3*B1 + 6A2*B2 + 3A1*B3 + 3A3*B2 +
        3A2*B3 + A3*B3,+,
      A4*B0 + 4A3*B1 + 6A2*B2 + 4A1*B3 + A0*B4 + 4A4*B1 + 12A3*B2 + 12A2*B3 +
        4A1*B4 + 6A4*B2 + 12A3*B3 + 6A2*B4 + 4A4*B3 + 4A3*B4 + A4*B4,+,
      5A4*B1 + 10A3*B2 + 10A2*B3 + 5A1*B4 + 20A4*B2 + 30A3*B3 + 20A2*B4 +
        30A4*B3 + 30A3*B4 + 20A4*B4,+,
      15A4*B2 + 20A3*B3 + 15A2*B4 + 60A4*B3 + 60A3*B4 + 90A4*B4,+,
      35A4*B3 + 35A3*B4 + 140A4*B4,+,
      70A4*B4}
  */

  const SCEVAddRecExpr *Product =
      dyn_cast<SCEVAddRecExpr>(SE.getMulExpr(A_rec, B_rec));
  ASSERT_TRUE(Product);
  ASSERT_EQ(Product->getNumOperands(), 9u);

  SmallVector<const SCEV *, 16> Sum;
  Sum.push_back(SE.getMulExpr(A[0], B[0]));
  EXPECT_EQ(Product->getOperand(0), SE.getAddExpr(Sum));
  Sum.clear();

  // SCEV produces different an equal but different expression for these.
  // Re-enable when PR11052 is fixed.
#if 0
  Sum.push_back(SE.getMulExpr(A[1], B[0]));
  Sum.push_back(SE.getMulExpr(A[0], B[1]));
  Sum.push_back(SE.getMulExpr(A[1], B[1]));
  EXPECT_EQ(Product->getOperand(1), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(A[2], B[0]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 2), A[1], B[1]));
  Sum.push_back(SE.getMulExpr(A[0], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 2), A[2], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 2), A[1], B[2]));
  Sum.push_back(SE.getMulExpr(A[2], B[2]));
  EXPECT_EQ(Product->getOperand(2), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(A[3], B[0]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[2], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[1], B[2]));
  Sum.push_back(SE.getMulExpr(A[0], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[3], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[2], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[1], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[3], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 3), A[2], B[3]));
  Sum.push_back(SE.getMulExpr(A[3], B[3]));
  EXPECT_EQ(Product->getOperand(3), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(A[4], B[0]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[3], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[2], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[1], B[3]));
  Sum.push_back(SE.getMulExpr(A[0], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[4], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 12), A[3], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 12), A[2], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[1], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[4], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 12), A[3], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 6), A[2], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 4), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(A[4], B[4]));
  EXPECT_EQ(Product->getOperand(4), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 5), A[4], B[1]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 10), A[3], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 10), A[2], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 5), A[1], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[4], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 30), A[3], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[2], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 30), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 30), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(5), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 15), A[4], B[2]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 20), A[3], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 15), A[2], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 60), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 60), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 90), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(6), SE.getAddExpr(Sum));
  Sum.clear();

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 35), A[4], B[3]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 35), A[3], B[4]));
  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 140), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(7), SE.getAddExpr(Sum));
  Sum.clear();
#endif

  Sum.push_back(SE.getMulExpr(SE.getConstant(Ty, 70), A[4], B[4]));
  EXPECT_EQ(Product->getOperand(8), SE.getAddExpr(Sum));
}

TEST_F(ScalarEvolutionsTest, SimplifiedPHI) {
  FunctionType *FTy = FunctionType::get(Type::getVoidTy(Context),
                                              std::vector<Type *>(), false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
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
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
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

  runWithFunctionAndSE(*M, "f_1", [&](Function &F, ScalarEvolution &SE) {
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
    runWithFunctionAndSE(*M, FuncName, [&](Function &F, ScalarEvolution &SE) {
      CheckCommutativeMulExprs(SE, SE.getSCEV(getInstructionByName(F, "x")),
                               SE.getSCEV(getInstructionByName(F, "y")),
                               SE.getSCEV(getInstructionByName(F, "z")));
    });
}

TEST_F(ScalarEvolutionsTest, CompareSCEVComplexity) {
  FunctionType *FTy =
      FunctionType::get(Type::getVoidTy(Context), std::vector<Type *>(), false);
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
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
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", F);

  Value *X = &*F->arg_begin();
  Value *Y = &*std::next(F->arg_begin());

  const int ValueDepth = 10;
  for (int i = 0; i < ValueDepth; i++) {
    X = new LoadInst(new IntToPtrInst(X, IntPtrPtrTy, "", EntryBB), "",
                     /*isVolatile*/ false, EntryBB);
    Y = new LoadInst(new IntToPtrInst(Y, IntPtrPtrTy, "", EntryBB), "",
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
  Function *F = cast<Function>(M.getOrInsertFunction("f", FTy));

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

}  // end anonymous namespace
}  // end namespace llvm
