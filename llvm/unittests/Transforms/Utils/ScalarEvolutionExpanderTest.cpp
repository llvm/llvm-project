//=== ScalarEvolutionExpanderTest.cpp - ScalarEvolutionExpander unit tests ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

namespace llvm {

using namespace PatternMatch;

// We use this fixture to ensure that we clean up ScalarEvolution before
// deleting the PassManager.
class ScalarEvolutionExpanderTest : public testing::Test {
protected:
  LLVMContext Context;
  Module M;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;

  std::unique_ptr<AssumptionCache> AC;
  std::unique_ptr<DominatorTree> DT;
  std::unique_ptr<LoopInfo> LI;

  ScalarEvolutionExpanderTest() : M("", Context), TLII(), TLI(TLII) {}

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
};

static Instruction &GetInstByName(Function &F, StringRef Name) {
  for (auto &I : instructions(F))
    if (I.getName() == Name)
      return I;
  llvm_unreachable("Could not find instructions!");
}

TEST_F(ScalarEvolutionExpanderTest, ExpandPtrTypeSCEV) {
  // It is to test the fix for PR30213. It exercises the branch in scev
  // expansion when the value in ValueOffsetPair is a ptr and the offset
  // is not divisible by the elem type size of value.
  auto *I8Ty = Type::getInt8Ty(Context);
  auto *I8PtrTy = PointerType::get(Context, 0);
  auto *I32Ty = Type::getInt32Ty(Context);
  auto *I32PtrTy = PointerType::get(Context, 0);
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

  const DataLayout &DL = F->getDataLayout();
  BranchInst *Br = BranchInst::Create(
      LoopBB, ExitBB, UndefValue::get(Type::getInt1Ty(Context)), LoopBB);
  AllocaInst *Alloca = new AllocaInst(I32Ty, DL.getAllocaAddrSpace(), "alloca",
                                      Br->getIterator());
  ConstantInt *Ci32 = ConstantInt::get(Context, APInt(32, 1));
  GetElementPtrInst *Gep0 =
      GetElementPtrInst::Create(I32Ty, Alloca, Ci32, "gep0", Br->getIterator());
  CastInst *CastA = CastInst::CreateBitOrPointerCast(Gep0, I8PtrTy, "bitcast1",
                                                     Br->getIterator());
  GetElementPtrInst *Gep1 =
      GetElementPtrInst::Create(I8Ty, CastA, Ci32, "gep1", Br->getIterator());
  GetElementPtrInst *Gep2 = GetElementPtrInst::Create(
      I8Ty, UndefValue::get(I8PtrTy), Ci32, "gep2", Br->getIterator());
  CmpInst *Cmp = CmpInst::Create(Instruction::ICmp, CmpInst::ICMP_ULT,
                                 UndefValue::get(I8PtrTy), CastA, "cmp",
                                 Br->getIterator());
  SelectInst *Sel =
      SelectInst::Create(Cmp, Gep1, Gep2, "select", Br->getIterator());
  CastInst *CastB = CastInst::CreateBitOrPointerCast(Sel, I32PtrTy, "bitcast2",
                                                     Br->getIterator());

  ScalarEvolution SE = buildSE(*F);
  const SCEV *S = SE.getSCEV(CastB);
  EXPECT_TRUE(isa<SCEVUnknown>(S));
}

// Make sure that SCEV doesn't introduce illegal ptrtoint/inttoptr instructions
TEST_F(ScalarEvolutionExpanderTest, SCEVZeroExtendExprNonIntegral) {
  /*
   * Create the following code:
   * func(i64 addrspace(10)* %arg)
   * top:
   *  br label %L.ph
   * L.ph:
   *  %gepbase = getelementptr i64 addrspace(10)* %arg, i64 1
   *  br label %L
   * L:
   *  %phi = phi i64 [i64 0, %L.ph], [ %add, %L2 ]
   *  %add = add i64 %phi2, 1
   *  br i1 undef, label %post, label %L2
   * post:
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
  Type *T_pint64 = PointerType::get(Context, 10);

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
  Value *GepBase =
      Builder.CreateGEP(T_int64, Arg, ConstantInt::get(T_int64, 1));
  Builder.CreateBr(L);

  Builder.SetInsertPoint(L);
  PHINode *Phi = Builder.CreatePHI(T_int64, 2);
  Value *Add = Builder.CreateAdd(Phi, ConstantInt::get(T_int64, 1), "add");
  Builder.CreateCondBr(UndefValue::get(T_int1), L, Post);
  Phi->addIncoming(ConstantInt::get(T_int64, 0), LPh);
  Phi->addIncoming(Add, L);

  Builder.SetInsertPoint(Post);
  Instruction *Ret = Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  const SCEV *AddRec =
      SE.getAddRecExpr(SE.getUnknown(GepBase), SE.getConstant(T_int64, 1),
                       LI->getLoopFor(L), SCEV::FlagNUW);

  SCEVExpander Exp(SE, NIM.getDataLayout(), "expander");
  Exp.disableCanonicalMode();
  Exp.expandCodeFor(AddRec, T_pint64, Ret);

  // Make sure none of the instructions inserted were inttoptr/ptrtoint.
  // The verifier will check this.
  EXPECT_FALSE(verifyFunction(*F, &errs()));
}

// Check that we can correctly identify the points at which the SCEV of the
// AddRec can be expanded.
TEST_F(ScalarEvolutionExpanderTest, SCEVExpanderIsSafeToExpandAt) {
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
  Type *T_pint64 = PointerType::get(Context, 10);

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
  Instruction *Ret = Builder.CreateRetVoid();

  ScalarEvolution SE = buildSE(*F);
  SCEVExpander Exp(SE, M.getDataLayout(), "expander");
  const SCEV *S = SE.getSCEV(Phi);
  EXPECT_TRUE(isa<SCEVAddRecExpr>(S));
  const SCEVAddRecExpr *AR = cast<SCEVAddRecExpr>(S);
  EXPECT_TRUE(AR->isAffine());
  EXPECT_FALSE(Exp.isSafeToExpandAt(AR, Top->getTerminator()));
  EXPECT_FALSE(Exp.isSafeToExpandAt(AR, LPh->getTerminator()));
  EXPECT_TRUE(Exp.isSafeToExpandAt(AR, L->getTerminator()));
  EXPECT_TRUE(Exp.isSafeToExpandAt(AR, Post->getTerminator()));

  EXPECT_TRUE(LI->getLoopFor(L)->isLCSSAForm(*DT));
  Exp.expandCodeFor(SE.getSCEV(Add), nullptr, Ret);
  EXPECT_TRUE(LI->getLoopFor(L)->isLCSSAForm(*DT));
}

// Check that SCEV expander does not use the nuw instruction
// for expansion.
TEST_F(ScalarEvolutionExpanderTest, SCEVExpanderNUW) {
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
      FunctionType::get(Type::getVoidTy(Context), {T_int64}, false);
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
TEST_F(ScalarEvolutionExpanderTest, SCEVExpanderNSW) {
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
      FunctionType::get(Type::getVoidTy(Context), {T_int64}, false);
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
TEST_F(ScalarEvolutionExpanderTest, SCEVCacheNUW) {
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
      FunctionType::get(Type::getVoidTy(Context), {T_int64}, false);
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
TEST_F(ScalarEvolutionExpanderTest, SCEVCacheNSW) {
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
      FunctionType::get(Type::getVoidTy(Context), {T_int64}, false);
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

TEST_F(ScalarEvolutionExpanderTest, SCEVExpandInsertCanonicalIV) {
  LLVMContext C;
  SMDiagnostic Err;

  // Expand the addrec produced by GetAddRec into a loop without a canonical IV.
  // SCEVExpander will insert one.
  auto TestNoCanonicalIV =
      [&](std::function<const SCEV *(ScalarEvolution & SE, Loop * L)>
              GetAddRec) {
        std::unique_ptr<Module> M = parseAssemblyString(
            "define i32 @test(i32 %limit) { "
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

        runWithSE(
            *M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
              auto &I = GetInstByName(F, "i");
              auto *Loop = LI.getLoopFor(I.getParent());
              EXPECT_FALSE(Loop->getCanonicalInductionVariable());

              auto *AR = GetAddRec(SE, Loop);
              unsigned ExpectedCanonicalIVWidth =
                  SE.getTypeSizeInBits(AR->getType());

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
  auto TestNarrowCanonicalIV = [&](std::function<const SCEV *(
                                       ScalarEvolution & SE, Loop * L)>
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
  auto TestMatchingCanonicalIV =
      [&](std::function<const SCEV *(ScalarEvolution & SE, Loop * L)> GetAddRec,
          unsigned ARBitWidth) {
        auto ARBitWidthTypeStr = "i" + std::to_string(ARBitWidth);
        std::unique_ptr<Module> M = parseAssemblyString(
            "define i32 @test(i32 %limit) { "
            "entry: "
            "  br label %loop "
            "loop: "
            "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
            "  %canonical.iv = phi " +
                ARBitWidthTypeStr +
                " [ 0, %entry ], [ %canonical.iv.inc, %loop ] "
                "  %i.inc = add nsw i32 %i, 1 "
                "  %canonical.iv.inc = add " +
                ARBitWidthTypeStr +
                " %canonical.iv, 1 "
                "  %cont = icmp slt i32 %i.inc, %limit "
                "  br i1 %cont, label %loop, label %exit "
                "exit: "
                "  ret i32 %i.inc "
                "}",
            Err, C);

        assert(M && "Could not parse module?");
        assert(!verifyModule(*M) && "Must have been well formed!");

        runWithSE(
            *M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
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

              // Loop over all of the PHI nodes, looking if a new canonical
              // indvar was introduced.
              PHINode *NewCanonicalIV = nullptr;
              for (BasicBlock::iterator i = LoopHeaderBB->begin();
                   isa<PHINode>(i); ++i) {
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

TEST_F(ScalarEvolutionExpanderTest, SCEVExpanderShlNSW) {

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

// Test expansion of nested addrecs in CanonicalMode.
// Expanding nested addrecs in canonical mode requiers a canonical IV of a
// type wider than the type of the addrec itself. Currently, SCEVExpander
// just falls back to literal mode for nested addrecs.
TEST_F(ScalarEvolutionExpanderTest, SCEVExpandNonAffineAddRec) {
  LLVMContext C;
  SMDiagnostic Err;

  // Expand the addrec produced by GetAddRec into a loop without a canonical IV.
  auto TestNoCanonicalIV =
      [&](std::function<const SCEVAddRecExpr *(ScalarEvolution & SE, Loop * L)>
              GetAddRec) {
        std::unique_ptr<Module> M = parseAssemblyString(
            "define i32 @test(i32 %limit) { "
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

        runWithSE(*M, "test",
                  [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
                    auto &I = GetInstByName(F, "i");
                    auto *Loop = LI.getLoopFor(I.getParent());
                    EXPECT_FALSE(Loop->getCanonicalInductionVariable());

                    auto *AR = GetAddRec(SE, Loop);
                    EXPECT_FALSE(AR->isAffine());

                    SCEVExpander Exp(SE, M->getDataLayout(), "expander");
                    auto *InsertAt = I.getNextNode();
                    Value *V = Exp.expandCodeFor(AR, nullptr, InsertAt);
                    const SCEV *ExpandedAR = SE.getSCEV(V);
                    // Check that the expansion happened literally.
                    EXPECT_EQ(AR, ExpandedAR);
                  });
      };

  // Expand the addrec produced by GetAddRec into a loop with a canonical IV
  // which is narrower than addrec type.
  auto TestNarrowCanonicalIV = [&](std::function<const SCEVAddRecExpr *(
                                       ScalarEvolution & SE, Loop * L)>
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
      const SCEV *ExpandedAR = SE.getSCEV(V);
      // Check that the expansion happened literally.
      EXPECT_EQ(AR, ExpandedAR);
    });
  };

  // Expand the addrec produced by GetAddRec into a loop with a canonical IV
  // of addrec width.
  auto TestMatchingCanonicalIV =
      [&](std::function<const SCEVAddRecExpr *(ScalarEvolution & SE, Loop * L)>
              GetAddRec,
          unsigned ARBitWidth) {
        auto ARBitWidthTypeStr = "i" + std::to_string(ARBitWidth);
        std::unique_ptr<Module> M = parseAssemblyString(
            "define i32 @test(i32 %limit) { "
            "entry: "
            "  br label %loop "
            "loop: "
            "  %i = phi i32 [ 1, %entry ], [ %i.inc, %loop ] "
            "  %canonical.iv = phi " +
                ARBitWidthTypeStr +
                " [ 0, %entry ], [ %canonical.iv.inc, %loop ] "
                "  %i.inc = add nsw i32 %i, 1 "
                "  %canonical.iv.inc = add " +
                ARBitWidthTypeStr +
                " %canonical.iv, 1 "
                "  %cont = icmp slt i32 %i.inc, %limit "
                "  br i1 %cont, label %loop, label %exit "
                "exit: "
                "  ret i32 %i.inc "
                "}",
            Err, C);

        assert(M && "Could not parse module?");
        assert(!verifyModule(*M) && "Must have been well formed!");

        runWithSE(
            *M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
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
              const SCEV *ExpandedAR = SE.getSCEV(V);
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

TEST_F(ScalarEvolutionExpanderTest, ExpandNonIntegralPtrWithNullBase) {
  LLVMContext C;
  SMDiagnostic Err;

  std::unique_ptr<Module> M =
      parseAssemblyString("target datalayout = "
                          "\"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:"
                          "128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2\""
                          "define ptr addrspace(1) @test(i64 %offset) { "
                          "  %ptr = getelementptr inbounds float, ptr "
                          "addrspace(1) null, i64 %offset"
                          "  ret ptr addrspace(1) %ptr"
                          "}",
                          Err, C);

  assert(M && "Could not parse module?");
  assert(!verifyModule(*M) && "Must have been well formed!");

  runWithSE(*M, "test", [&](Function &F, LoopInfo &LI, ScalarEvolution &SE) {
    auto &I = GetInstByName(F, "ptr");
    auto PtrPlus1 =
        SE.getAddExpr(SE.getSCEV(&I), SE.getConstant(I.getType(), 1));
    SCEVExpander Exp(SE, M->getDataLayout(), "expander");

    Value *V = Exp.expandCodeFor(PtrPlus1, I.getType(), &I);
    I.replaceAllUsesWith(V);

    // Check that the expander created:
    // define ptr addrspace(1) @test(i64 %off) {
    //   %1 = shl i64 %offset, 2
    //   %2 = add nuw nsw i64 %1, 1
    //   %uglygep = getelementptr i8, ptr addrspace(1) null, i64 %2
    //   %ptr = getelementptr inbounds float, ptr addrspace(1) null, i64 %off
    //   ret ptr addrspace(1) %uglygep
    // }

    Value *Offset = &*F.arg_begin();
    auto *GEP = dyn_cast<GetElementPtrInst>(V);
    EXPECT_TRUE(GEP);
    EXPECT_TRUE(cast<Constant>(GEP->getPointerOperand())->isNullValue());
    EXPECT_EQ(GEP->getNumOperands(), 2U);
    EXPECT_TRUE(match(
        GEP->getOperand(1),
        m_Add(m_Shl(m_Specific(Offset), m_SpecificInt(2)), m_SpecificInt(1))));
    EXPECT_EQ(cast<PointerType>(GEP->getPointerOperand()->getType())
                  ->getAddressSpace(),
              cast<PointerType>(I.getType())->getAddressSpace());
    EXPECT_FALSE(verifyFunction(F, &errs()));
  });
}

} // end namespace llvm
