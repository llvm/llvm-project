//===- InjectorIRStrategyTest.cpp - Tests for injector strategy -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/AsmParser/SlotMapping.h"
#include "llvm/FuzzMutate/IRMutator.h"
#include "llvm/FuzzMutate/Operations.h"
#include "llvm/FuzzMutate/RandomIRBuilder.h"
#include "llvm/IR/FMF.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include <random>

#include "gtest/gtest.h"

using namespace llvm;

static constexpr int Seed = 5;

namespace {

std::unique_ptr<IRMutator> createInjectorMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  // Add vector 1, 2, 3, 4, and 8.
  int VectorLength[] = {1, 2, 3, 4, 8};
  std::vector<TypeGetter> BasicTypeGetters(Types);
  for (auto typeGetter : BasicTypeGetters) {
    for (int length : VectorLength) {
      Types.push_back([typeGetter, length](LLVMContext &C) {
        return VectorType::get(typeGetter(C), length, false);
      });
    }
  }

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(std::make_unique<InjectorIRStrategy>(
      InjectorIRStrategy::getDefaultOps()));

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

template <class Strategy> std::unique_ptr<IRMutator> createMutator() {
  std::vector<TypeGetter> Types{
      Type::getInt1Ty,  Type::getInt8Ty,  Type::getInt16Ty, Type::getInt32Ty,
      Type::getInt64Ty, Type::getFloatTy, Type::getDoubleTy};

  std::vector<std::unique_ptr<IRMutationStrategy>> Strategies;
  Strategies.push_back(std::make_unique<Strategy>());

  return std::make_unique<IRMutator>(std::move(Types), std::move(Strategies));
}

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

void IterateOnSource(StringRef Source, IRMutator &Mutator) {
  LLVMContext Ctx;

  for (int i = 0; i < 10; ++i) {
    auto M = parseAssembly(Source.data(), Ctx);
    ASSERT_TRUE(M && !verifyModule(*M, &errs()));

    Mutator.mutateModule(*M, Seed, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
  }
}

static void mutateAndVerifyModule(StringRef Source,
                                  std::unique_ptr<IRMutator> &Mutator,
                                  int repeat = 100) {
  LLVMContext Ctx;
  auto M = parseAssembly(Source.data(), Ctx);
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);
  for (int i = 0; i < repeat; i++) {
    Mutator->mutateModule(*M, RandInt(mt), Source.size(), Source.size() + 1024);
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
}
template <class Strategy>
static void mutateAndVerifyModule(StringRef Source, int repeat = 100) {
  auto Mutator = createMutator<Strategy>();
  ASSERT_TRUE(Mutator);
  mutateAndVerifyModule(Source, Mutator, repeat);
}

TEST(InjectorIRStrategyTest, EmptyModule) {
  // Test that we can inject into empty module

  LLVMContext Ctx;
  auto M = std::make_unique<Module>("M", Ctx);
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));

  auto Mutator = createInjectorMutator();
  ASSERT_TRUE(Mutator);

  Mutator->mutateModule(*M, Seed, 1, 1);
  EXPECT_TRUE(!verifyModule(*M, &errs()));
}

TEST(InjectorIRStrategyTest, LargeInsertion) {
  StringRef Source = "";
  auto Mutator = createInjectorMutator();
  ASSERT_TRUE(Mutator);
  mutateAndVerifyModule(Source, Mutator, 100);
}

TEST(InstDeleterIRStrategyTest, EmptyFunction) {
  // Test that we don't crash even if we can't remove from one of the functions.

  StringRef Source = ""
                     "define <8 x i32> @func1() {\n"
                     "ret <8 x i32> undef\n"
                     "}\n"
                     "\n"
                     "define i32 @func2() {\n"
                     "%A9 = alloca i32\n"
                     "%L6 = load i32, i32* %A9\n"
                     "ret i32 %L6\n"
                     "}\n";

  auto Mutator = createMutator<InstDeleterIRStrategy>();
  ASSERT_TRUE(Mutator);

  IterateOnSource(Source, *Mutator);
}

TEST(InstDeleterIRStrategyTest, PhiNodes) {
  // Test that inst deleter works correctly with the phi nodes.

  LLVMContext Ctx;
  StringRef Source = "\n\
      define i32 @earlyreturncrash(i32 %x) {\n\
      entry:\n\
        switch i32 %x, label %sw.epilog [\n\
          i32 1, label %sw.bb1\n\
        ]\n\
      \n\
      sw.bb1:\n\
        br label %sw.epilog\n\
      \n\
      sw.epilog:\n\
        %a.0 = phi i32 [ 7, %entry ],  [ 9, %sw.bb1 ]\n\
        %b.0 = phi i32 [ 10, %entry ], [ 4, %sw.bb1 ]\n\
        ret i32 %a.0\n\
      }";

  auto Mutator = createMutator<InstDeleterIRStrategy>();
  ASSERT_TRUE(Mutator);

  IterateOnSource(Source, *Mutator);
}

static void checkModifyNoUnsignedAndNoSignedWrap(StringRef Opc) {
  LLVMContext Ctx;
  std::string Source = std::string("\n\
      define i32 @test(i32 %x) {\n\
        %a = ") + Opc.str() +
                       std::string(" i32 %x, 10\n\
        ret i32 %a\n\
      }");

  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  auto *AddI = &*F.begin()->begin();
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  bool FoundNUW = false;
  bool FoundNSW = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, Seed + i, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
    FoundNUW |= AddI->hasNoUnsignedWrap();
    FoundNSW |= AddI->hasNoSignedWrap();
  }

  // The mutator should have added nuw and nsw during some mutations.
  EXPECT_TRUE(FoundNUW);
  EXPECT_TRUE(FoundNSW);
}
TEST(InstModificationIRStrategyTest, Add) {
  checkModifyNoUnsignedAndNoSignedWrap("add");
}

TEST(InstModificationIRStrategyTest, Sub) {
  checkModifyNoUnsignedAndNoSignedWrap("sub");
}

TEST(InstModificationIRStrategyTest, Mul) {
  checkModifyNoUnsignedAndNoSignedWrap("mul");
}

TEST(InstModificationIRStrategyTest, Shl) {
  checkModifyNoUnsignedAndNoSignedWrap("shl");
}

TEST(InstModificationIRStrategyTest, ICmp) {
  LLVMContext Ctx;
  StringRef Source = "\n\
      define i1 @test(i32 %x) {\n\
        %a = icmp eq i32 %x, 10\n\
        ret i1 %a\n\
      }";

  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  CmpInst *CI = cast<CmpInst>(&*F.begin()->begin());
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  bool FoundNE = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, Seed + i, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
    FoundNE |= CI->getPredicate() == CmpInst::ICMP_NE;
  }

  EXPECT_TRUE(FoundNE);
}

TEST(InstModificationIRStrategyTest, GEP) {
  LLVMContext Ctx;
  StringRef Source = "\n\
      define i32* @test(i32* %ptr) {\n\
        %gep = getelementptr i32, i32* %ptr, i32 10\n\
        ret i32* %gep\n\
      }";

  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  GetElementPtrInst *GEP = cast<GetElementPtrInst>(&*F.begin()->begin());
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  bool FoundInbounds = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, Seed + i, Source.size(), Source.size() + 100);
    EXPECT_TRUE(!verifyModule(*M, &errs()));
    FoundInbounds |= GEP->isInBounds();
  }

  EXPECT_TRUE(FoundInbounds);
}

/// The caller has to guarantee that function argument are used in the SAME
/// place as the operand.
void VerfyOperandShuffled(StringRef Source, std::pair<int, int> ShuffleItems) {
  LLVMContext Ctx;
  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  Instruction *Inst = &*F.begin()->begin();
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  ASSERT_TRUE(Inst->getOperand(ShuffleItems.first) ==
              dyn_cast<Value>(F.getArg(ShuffleItems.first)));
  ASSERT_TRUE(Inst->getOperand(ShuffleItems.second) ==
              dyn_cast<Value>(F.getArg(ShuffleItems.second)));

  Mutator->mutateModule(*M, 0, Source.size(), Source.size() + 100);
  ASSERT_TRUE(!verifyModule(*M, &errs()));

  ASSERT_TRUE(Inst->getOperand(ShuffleItems.first) ==
              dyn_cast<Value>(F.getArg(ShuffleItems.second)));
  ASSERT_TRUE(Inst->getOperand(ShuffleItems.second) ==
              dyn_cast<Value>(F.getArg(ShuffleItems.first)));
}

TEST(InstModificationIRStrategyTest, ShuffleAnd) {
  StringRef Source = "\n\
      define i32 @test(i32 %0, i32 %1) {\n\
        %add = and i32 %0, %1\n\
        ret i32 %add\n\
      }";
  VerfyOperandShuffled(Source, {0, 1});
}
TEST(InstModificationIRStrategyTest, ShuffleSelect) {
  StringRef Source = "\n\
      define i32 @test(i1 %0, i32 %1, i32 %2) {\n\
        %select = select i1 %0, i32 %1, i32 %2\n\
        ret i32 %select\n\
      }";
  VerfyOperandShuffled(Source, {1, 2});
}

void VerfyDivDidntShuffle(StringRef Source) {
  LLVMContext Ctx;
  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  auto M = parseAssembly(Source.data(), Ctx);
  auto &F = *M->begin();
  Instruction *Inst = &*F.begin()->begin();
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));

  EXPECT_TRUE(isa<Constant>(Inst->getOperand(0)));
  EXPECT_TRUE(Inst->getOperand(1) == dyn_cast<Value>(F.getArg(0)));

  Mutator->mutateModule(*M, Seed, Source.size(), Source.size() + 100);
  EXPECT_TRUE(!verifyModule(*M, &errs()));

  // Didn't shuffle.
  EXPECT_TRUE(isa<Constant>(Inst->getOperand(0)));
  EXPECT_TRUE(Inst->getOperand(1) == dyn_cast<Value>(F.getArg(0)));
}
TEST(InstModificationIRStrategyTest, DidntShuffleSDiv) {
  StringRef Source = "\n\
      define i32 @test(i32 %0) {\n\
        %div = sdiv i32 0, %0\n\
        ret i32 %div\n\
      }";
  VerfyDivDidntShuffle(Source);
}
TEST(InstModificationIRStrategyTest, DidntShuffleFRem) {
  StringRef Source = "\n\
      define <2 x double> @test(<2 x double> %0) {\n\
        %div = frem <2 x double> <double 0.0, double 0.0>, %0\n\
        ret <2 x double> %div\n\
      }";
  VerfyDivDidntShuffle(Source);
}
TEST(InstModificationIRStrategy, Exact) {
  LLVMContext Ctx;
  StringRef Source = "\n\
      define i32 @test(i32 %a, i32 %b) {\n\
        %c = ashr i32 %a, %b \n\
        ret i32 %c\n\
      }";

  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  std::unique_ptr<Module> M = parseAssembly(Source.data(), Ctx);
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);
  auto &F = *M->begin();
  BinaryOperator *AShr = cast<BinaryOperator>(&*F.begin()->begin());
  bool FoundExact = false;
  for (int i = 0; i < 100; ++i) {
    Mutator->mutateModule(*M, RandInt(mt), Source.size(), Source.size() + 100);
    ASSERT_FALSE(verifyModule(*M, &errs()));
    FoundExact |= AShr->isExact();
  }

  EXPECT_TRUE(FoundExact);
}
TEST(InstModificationIRStrategy, FastMath) {
  LLVMContext Ctx;
  StringRef Source = "\n\
      declare [4 x <4 x double>] @vecdouble(double)  \n\
      define double @test(i1 %C, double %a, double %b) {  \n\
      Entry:  \n\
        br i1 %C, label %True, label %False  \n\
      True:  \n\
        br label %Exit  \n\
      False:  \n\
        br label %Exit  \n\
      Exit:  \n\
        %PHIi32 = phi i32 [1, %True], [2, %False]  \n\
        %PHIdouble = phi double [%a, %True], [%b, %False]  \n\
        %Call = call [4 x <4 x double>] @vecdouble(double %PHIdouble)  \n\
        %c = fneg double %PHIdouble  \n\
        %s = select i1 %C, double %a, double %b  \n\
        %d = fadd double %s, %c  \n\
        ret double %d \n\
      }";

  auto Mutator = createMutator<InstModificationIRStrategy>();
  ASSERT_TRUE(Mutator);

  std::unique_ptr<Module> M = parseAssembly(Source.data(), Ctx);
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);
  DenseMap<Instruction *, bool> FPOpsHasFastMath;
  for (auto &F : *M) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        Type *Ty = I.getType();
        if (Ty->isFPOrFPVectorTy() || Ty->isArrayTy()) {
          FPOpsHasFastMath[&I] = false;
        }
      }
    }
  }
  ASSERT_TRUE(M && !verifyModule(*M, &errs()));
  for (int i = 0; i < 300; ++i) {
    Mutator->mutateModule(*M, RandInt(mt), Source.size(), Source.size() + 100);
    for (auto p : FPOpsHasFastMath)
      FPOpsHasFastMath[p.first] |= p.first->getFastMathFlags().any();
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
  for (auto p : FPOpsHasFastMath)
    ASSERT_TRUE(p.second);
}

TEST(InsertCFGStrategy, CFG) {
  StringRef Source = "\n\
      define i32 @test(i1 %C1, i1 %C2, i1 %C3, i16 %S1, i16 %S2, i32 %I1) { \n\
        Entry:  \n\
         %I2 = add i32 %I1, 1 \n\
          %C = and i1 %C1, %C2 \n\
        br label %Body \n\
        Body: \n\
         %IB = add i32 %I1, %I2 \n\
          %CB = and i1 %C1, %C \n\
        br label %Exit \n\
        Exit: \n\
          %IE = add i32 %IB, %I2 \n\
          %CE = and i1 %CB, %C \n\
         ret i32 %IE \n\
      }";
  mutateAndVerifyModule<InsertCFGStrategy>(Source);
}

TEST(InsertPHIStrategy, PHI) {
  StringRef Source = "\n\
        define void @test(i1 %C1, i1 %C2, i32 %I, double %FP) { \n\
        Entry:  \n\
          %C = and i1 %C1, %C2 \n\
          br i1 %C, label %LoopHead, label %Exit \n\
        LoopHead: ; pred Entry, LoopBody \n\
          switch i32 %I, label %Default [ \n\
              i32 1, label %OnOne  \n\
              i32 2, label %OnTwo \n\
              i32 3, label %OnThree \n\
          ] \n\
        Default:  \n\
          br label %LoopBody \n\
        OnOne: ; pred LoopHead \n\
          %DFP = fmul double %FP, 2.0 \n\
          %OnOneCond = fcmp ogt double %DFP, %FP \n\
          br i1 %OnOneCond, label %LoopBody, label %Exit \n\
        OnTwo: ; pred Entry \n\
          br i1 %C1, label %OnThree, label %LoopBody \n\
        OnThree: ; pred Entry, OnTwo, OnThree \n\
          br i1 %C2, label %OnThree, label %LoopBody \n\
        LoopBody: ; pred Default, OnOne, OnTwo, OnThree \n\
          br label %LoopHead \n\
        Exit: ; pred Entry, OnOne \n\
          ret void \n\
        }";
  mutateAndVerifyModule<InsertPHIStrategy>(Source);
}

TEST(InsertPHIStrategy, PHIWithSameIncomingBlock) {
  LLVMContext Ctx;
  StringRef Source = "\n\
        define void @test(i32 %I) { \n\
        Entry:  \n\
          switch i32 %I, label %Exit [ \n\
              i32 1, label %IdentCase  \n\
              i32 2, label %IdentCase \n\
              i32 3, label %IdentCase \n\
              i32 4, label %IdentCase \n\
          ] \n\
        IdentCase: \n\
          br label %Exit \n\
        Exit: \n\
          ret void \n\
        }";
  auto IPS = std::make_unique<InsertPHIStrategy>();
  RandomIRBuilder IB(Seed, {IntegerType::getInt32Ty(Ctx)});
  auto M = parseAssembly(Source.data(), Ctx);
  Function &F = *M->begin();
  for (auto &BB : F) {
    IPS->mutate(BB, IB);
    ASSERT_FALSE(verifyModule(*M, &errs()));
  }
}

TEST(SinkInstructionStrategy, Operand) {
  StringRef Source = "\n\
      define i32 @test(i1 %C1, i1 %C2, i1 %C3, i32 %I, i32 %J) { \n\
        Entry:  \n\
          %I100 = add i32 %I, 100  \n\
          switch i32 %I100, label %BB0 [ \n\
            i32 42, label %BB1  \n\
          ] \n\
        BB0:  \n\
          %IAJ = add i32 %I, %J  \n\
          %ISJ = sub i32 %I, %J  \n\
          br label %Exit  \n\
        BB1:  \n\
          %IJ = mul i32 %I, %J  \n\
          %C = and i1 %C2, %C3  \n\
          br i1 %C, label %BB0, label %Exit  \n\
        Exit:  \n\
          ret i32 %I  \n\
      }";
  mutateAndVerifyModule<SinkInstructionStrategy>(Source);
}

static void VerifyBlockShuffle(StringRef Source) {
  LLVMContext Ctx;
  auto Mutator = createMutator<ShuffleBlockStrategy>();
  ASSERT_TRUE(Mutator);

  std::unique_ptr<Module> M = parseAssembly(Source.data(), Ctx);
  Function *F = &*M->begin();
  DenseMap<BasicBlock *, int> PreShuffleInstCnt;
  for (BasicBlock &BB : *F) {
    PreShuffleInstCnt.insert({&BB, BB.size()});
  }
  std::mt19937 mt(Seed);
  std::uniform_int_distribution<int> RandInt(INT_MIN, INT_MAX);
  for (int i = 0; i < 100; i++) {
    Mutator->mutateModule(*M, RandInt(mt), Source.size(), Source.size() + 1024);
    for (BasicBlock &BB : *F) {
      int PostShuffleIntCnt = BB.size();
      EXPECT_EQ(PostShuffleIntCnt, PreShuffleInstCnt[&BB]);
    }
    EXPECT_FALSE(verifyModule(*M, &errs()));
  }
}

TEST(ShuffleBlockStrategy, ShuffleBlocks) {
  StringRef Source = "\n\
        define i64 @test(i1 %0, i1 %1, i1 %2, i32 %3, i32 %4) { \n\
        Entry:  \n\
          %A = alloca i32, i32 8, align 4 \n\
          %E.1 = and i32 %3, %4 \n\
          %E.2 = add i32 %4 , 1 \n\
          %A.GEP.1 = getelementptr i32, ptr %A, i32 0 \n\
          %A.GEP.2 = getelementptr i32, ptr %A.GEP.1, i32 1 \n\
          %L.2 = load i32, ptr %A.GEP.2 \n\
          %L.1 = load i32, ptr %A.GEP.1 \n\
          %E.3 = sub i32 %E.2, %L.1 \n\
          %Cond.1 = icmp eq i32 %E.3, %E.2 \n\
          %Cond.2 = and i1 %0, %1 \n\
          %Cond = or i1 %Cond.1, %Cond.2 \n\
          br i1 %Cond, label %BB0, label %BB1  \n\
        BB0:  \n\
          %Add = add i32 %L.1, %L.2 \n\
          %Sub = sub i32 %L.1, %L.2 \n\
          %Sub.1 = sub i32 %Sub, 12 \n\
          %Cast.1 = bitcast i32 %4 to float \n\
          %Add.2 = add i32 %3, 1 \n\
          %Cast.2 = bitcast i32 %Add.2 to float \n\
          %FAdd = fadd float %Cast.1, %Cast.2 \n\
          %Add.3 = add i32 %L.2, %L.1 \n\
          %Cast.3 = bitcast float %FAdd to i32 \n\
          %Sub.2 = sub i32 %Cast.3, %Sub.1 \n\
          %SExt = sext i32 %Cast.3 to i64 \n\
          %A.GEP.3 = getelementptr i64, ptr %A, i32 1 \n\
          store i64 %SExt, ptr %A.GEP.3 \n\
          br label %Exit  \n\
        BB1:  \n\
          %PHI.1 = phi i32 [0, %Entry] \n\
          %SExt.1 = sext i1 %Cond.2 to i32 \n\
          %SExt.2 = sext i1 %Cond.1 to i32 \n\
          %E.164 = zext i32 %E.1 to i64 \n\
          %E.264 = zext i32 %E.2 to i64 \n\
          %E.1264 = mul i64 %E.164, %E.264 \n\
          %E.12 = trunc i64 %E.1264 to i32 \n\
          %A.GEP.4 = getelementptr i32, ptr %A, i32 2 \n\
          %A.GEP.5 = getelementptr i32, ptr %A.GEP.4, i32 2 \n\
          store i32 %E.12, ptr %A.GEP.5 \n\
          br label %Exit  \n\
        Exit:  \n\
          %PHI.2 = phi i32 [%Add, %BB0], [%E.3, %BB1] \n\
          %PHI.3 = phi i64 [%SExt, %BB0], [%E.1264, %BB1] \n\
          %ZExt = zext i32 %PHI.2 to i64 \n\
          %Add.5 = add i64 %PHI.3, 3 \n\
          ret i64 %Add.5  \n\
      }";
  VerifyBlockShuffle(Source);
}

TEST(ShuffleBlockStrategy, ShuffleLoop) {
  StringRef Source = "\n\
    define i32 @foo(i32 %Left, i32 %Right) { \n\
    Entry: \n\
      %LPtr = alloca i32, align 4 \n\
      %RPtr = alloca i32, align 4 \n\
      %RetValPtr = alloca i32, align 4 \n\
      store i32 %Left, ptr %LPtr, align 4 \n\
      store i32 %Right, ptr %RPtr, align 4 \n\
      store i32 0, ptr %RetValPtr, align 4 \n\
      br label %LoopHead \n\
    LoopHead: \n\
      %L = load i32, ptr %LPtr, align 4 \n\
      %R = load i32, ptr %RPtr, align 4 \n\
      %C = icmp slt i32 %L, %R \n\
      br i1 %C, label %LoopBody, label %Exit \n\
    LoopBody: \n\
      %OldL = load i32, ptr %LPtr, align 4 \n\
      %NewL = add nsw i32 %OldL, 1 \n\
      store i32 %NewL, ptr %LPtr, align 4 \n\
      %OldRetVal = load i32, ptr %RetValPtr, align 4 \n\
      %NewRetVal = add nsw i32 %OldRetVal, 1 \n\
      store i32 %NewRetVal, ptr %RetValPtr, align 4 \n\
      br label %LoopHead \n\
    Exit: \n\
      %RetVal = load i32, ptr %RetValPtr, align 4 \n\
      ret i32 %RetVal \n\
    }";
  VerifyBlockShuffle(Source);
}
} // namespace
