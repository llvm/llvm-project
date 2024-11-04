//===- llvm/unittest/IR/BasicBlockTest.cpp - BasicBlock unit tests --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/BasicBlock.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"
#include <memory>

namespace llvm {
namespace {

TEST(BasicBlockTest, PhiRange) {
  LLVMContext Context;

  // Create the main block.
  std::unique_ptr<BasicBlock> BB(BasicBlock::Create(Context));

  // Create some predecessors of it.
  std::unique_ptr<BasicBlock> BB1(BasicBlock::Create(Context));
  BranchInst::Create(BB.get(), BB1.get());
  std::unique_ptr<BasicBlock> BB2(BasicBlock::Create(Context));
  BranchInst::Create(BB.get(), BB2.get());

  // Make sure this doesn't crash if there are no phis.
  int PhiCount = 0;
  for (auto &PN : BB->phis()) {
    (void)PN;
    PhiCount++;
  }
  ASSERT_EQ(PhiCount, 0) << "empty block should have no phis";

  // Make it a cycle.
  auto *BI = BranchInst::Create(BB.get(), BB.get());

  // Now insert some PHI nodes.
  auto *Int32Ty = Type::getInt32Ty(Context);
  auto *P1 = PHINode::Create(Int32Ty, /*NumReservedValues*/ 3, "phi.1",
                             BI->getIterator());
  auto *P2 = PHINode::Create(Int32Ty, /*NumReservedValues*/ 3, "phi.2",
                             BI->getIterator());
  auto *P3 = PHINode::Create(Int32Ty, /*NumReservedValues*/ 3, "phi.3",
                             BI->getIterator());

  // Some non-PHI nodes.
  auto *Sum = BinaryOperator::CreateAdd(P1, P2, "sum", BI->getIterator());

  // Now wire up the incoming values that are interesting.
  P1->addIncoming(P2, BB.get());
  P2->addIncoming(P1, BB.get());
  P3->addIncoming(Sum, BB.get());

  // Finally, let's iterate them, which is the thing we're trying to test.
  // We'll use this to wire up the rest of the incoming values.
  for (auto &PN : BB->phis()) {
    PN.addIncoming(UndefValue::get(Int32Ty), BB1.get());
    PN.addIncoming(UndefValue::get(Int32Ty), BB2.get());
  }

  // Test that we can use const iterators and generally that the iterators
  // behave like iterators.
  BasicBlock::const_phi_iterator CI;
  CI = BB->phis().begin();
  EXPECT_NE(CI, BB->phis().end());

  // Test that filtering iterators work with basic blocks.
  auto isPhi = [](Instruction &I) { return isa<PHINode>(&I); };
  auto Phis = make_filter_range(*BB, isPhi);
  auto ReversedPhis = reverse(make_filter_range(*BB, isPhi));
  EXPECT_EQ(std::distance(Phis.begin(), Phis.end()), 3);
  EXPECT_EQ(&*Phis.begin(), P1);
  EXPECT_EQ(std::distance(ReversedPhis.begin(), ReversedPhis.end()), 3);
  EXPECT_EQ(&*ReversedPhis.begin(), P3);

  // And iterate a const range.
  for (const auto &PN : const_cast<const BasicBlock *>(BB.get())->phis()) {
    EXPECT_EQ(BB.get(), PN.getIncomingBlock(0));
    EXPECT_EQ(BB1.get(), PN.getIncomingBlock(1));
    EXPECT_EQ(BB2.get(), PN.getIncomingBlock(2));
  }
}

#define CHECK_ITERATORS(Range1, Range2)                                        \
  EXPECT_EQ(std::distance(Range1.begin(), Range1.end()),                       \
            std::distance(Range2.begin(), Range2.end()));                      \
  for (auto Pair : zip(Range1, Range2))                                        \
    EXPECT_EQ(&std::get<0>(Pair), std::get<1>(Pair));

TEST(BasicBlockTest, TestInstructionsWithoutDebug) {
  LLVMContext Ctx;

  Module *M = new Module("MyModule", Ctx);
  Type *ArgTy1[] = {PointerType::getUnqual(Ctx)};
  FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), ArgTy1, false);
  Argument *V = new Argument(Type::getInt32Ty(Ctx));
  Function *F = Function::Create(FT, Function::ExternalLinkage, "", M);

  Function *DbgDeclare = Intrinsic::getDeclaration(M, Intrinsic::dbg_declare);
  Function *DbgValue = Intrinsic::getDeclaration(M, Intrinsic::dbg_value);
  Value *DIV = MetadataAsValue::get(Ctx, (Metadata *)nullptr);
  SmallVector<Value *, 3> Args = {DIV, DIV, DIV};

  BasicBlock *BB1 = BasicBlock::Create(Ctx, "", F);
  const BasicBlock *BBConst = BB1;
  IRBuilder<> Builder1(BB1);

  AllocaInst *Var = Builder1.CreateAlloca(Builder1.getInt8Ty());
  Builder1.CreateCall(DbgValue, Args);
  Instruction *AddInst = cast<Instruction>(Builder1.CreateAdd(V, V));
  Instruction *MulInst = cast<Instruction>(Builder1.CreateMul(AddInst, V));
  Builder1.CreateCall(DbgDeclare, Args);
  Instruction *SubInst = cast<Instruction>(Builder1.CreateSub(MulInst, V));

  SmallVector<Instruction *, 4> Exp = {Var, AddInst, MulInst, SubInst};
  CHECK_ITERATORS(BB1->instructionsWithoutDebug(), Exp);
  CHECK_ITERATORS(BBConst->instructionsWithoutDebug(), Exp);

  EXPECT_EQ(static_cast<size_t>(BB1->sizeWithoutDebug()), Exp.size());
  EXPECT_EQ(static_cast<size_t>(BBConst->sizeWithoutDebug()), Exp.size());

  delete M;
  delete V;
}

TEST(BasicBlockTest, ComesBefore) {
  const char *ModuleString = R"(define i32 @f(i32 %x) {
                                  %add = add i32 %x, 42
                                  ret i32 %add
                                })";
  LLVMContext Ctx;
  SMDiagnostic Err;
  auto M = parseAssemblyString(ModuleString, Err, Ctx);
  ASSERT_TRUE(M.get());

  Function *F = M->getFunction("f");
  BasicBlock &BB = F->front();
  BasicBlock::iterator I = BB.begin();
  Instruction *Add = &*I++;
  Instruction *Ret = &*I++;

  // Intentionally duplicated to verify cached and uncached are the same.
  EXPECT_FALSE(BB.isInstrOrderValid());
  EXPECT_FALSE(Add->comesBefore(Add));
  EXPECT_TRUE(BB.isInstrOrderValid());
  EXPECT_FALSE(Add->comesBefore(Add));
  BB.invalidateOrders();
  EXPECT_FALSE(BB.isInstrOrderValid());
  EXPECT_TRUE(Add->comesBefore(Ret));
  EXPECT_TRUE(BB.isInstrOrderValid());
  EXPECT_TRUE(Add->comesBefore(Ret));
  BB.invalidateOrders();
  EXPECT_FALSE(Ret->comesBefore(Add));
  EXPECT_FALSE(Ret->comesBefore(Add));
  BB.invalidateOrders();
  EXPECT_FALSE(Ret->comesBefore(Ret));
  EXPECT_FALSE(Ret->comesBefore(Ret));
}

class InstrOrderInvalidationTest : public ::testing::Test {
protected:
  void SetUp() override {
    M.reset(new Module("MyModule", Ctx));
    Nop = Intrinsic::getDeclaration(M.get(), Intrinsic::donothing);
    FunctionType *FT = FunctionType::get(Type::getVoidTy(Ctx), {}, false);
    Function *F = Function::Create(FT, Function::ExternalLinkage, "foo", *M);
    BB = BasicBlock::Create(Ctx, "entry", F);

    IRBuilder<> Builder(BB);
    I1 = Builder.CreateCall(Nop);
    I2 = Builder.CreateCall(Nop);
    I3 = Builder.CreateCall(Nop);
    Ret = Builder.CreateRetVoid();
  }

  LLVMContext Ctx;
  std::unique_ptr<Module> M;
  Function *Nop = nullptr;
  BasicBlock *BB = nullptr;
  Instruction *I1 = nullptr;
  Instruction *I2 = nullptr;
  Instruction *I3 = nullptr;
  Instruction *Ret = nullptr;
};

TEST_F(InstrOrderInvalidationTest, InsertInvalidation) {
  EXPECT_FALSE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1->comesBefore(I2));
  EXPECT_TRUE(BB->isInstrOrderValid());
  EXPECT_TRUE(I2->comesBefore(I3));
  EXPECT_TRUE(I3->comesBefore(Ret));
  EXPECT_TRUE(BB->isInstrOrderValid());

  // Invalidate orders.
  IRBuilder<> Builder(BB, I2->getIterator());
  Instruction *I1a = Builder.CreateCall(Nop);
  EXPECT_FALSE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1->comesBefore(I1a));
  EXPECT_TRUE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1a->comesBefore(I2));
  EXPECT_TRUE(I2->comesBefore(I3));
  EXPECT_TRUE(I3->comesBefore(Ret));
  EXPECT_TRUE(BB->isInstrOrderValid());
}

TEST_F(InstrOrderInvalidationTest, SpliceInvalidation) {
  EXPECT_TRUE(I1->comesBefore(I2));
  EXPECT_TRUE(I2->comesBefore(I3));
  EXPECT_TRUE(I3->comesBefore(Ret));
  EXPECT_TRUE(BB->isInstrOrderValid());

  // Use Instruction::moveBefore, which uses splice.
  I2->moveBefore(I1);
  EXPECT_FALSE(BB->isInstrOrderValid());

  EXPECT_TRUE(I2->comesBefore(I1));
  EXPECT_TRUE(I1->comesBefore(I3));
  EXPECT_TRUE(I3->comesBefore(Ret));
  EXPECT_TRUE(BB->isInstrOrderValid());
}

TEST_F(InstrOrderInvalidationTest, RemoveNoInvalidation) {
  // Cache the instruction order.
  EXPECT_FALSE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1->comesBefore(I2));
  EXPECT_TRUE(BB->isInstrOrderValid());

  // Removing does not invalidate instruction order.
  I2->removeFromParent();
  I2->deleteValue();
  I2 = nullptr;
  EXPECT_TRUE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1->comesBefore(I3));
  EXPECT_EQ(std::next(I1->getIterator()), I3->getIterator());
}

TEST_F(InstrOrderInvalidationTest, EraseNoInvalidation) {
  // Cache the instruction order.
  EXPECT_FALSE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1->comesBefore(I2));
  EXPECT_TRUE(BB->isInstrOrderValid());

  // Removing does not invalidate instruction order.
  I2->eraseFromParent();
  I2 = nullptr;
  EXPECT_TRUE(BB->isInstrOrderValid());
  EXPECT_TRUE(I1->comesBefore(I3));
  EXPECT_EQ(std::next(I1->getIterator()), I3->getIterator());
}

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print(__FILE__, errs());
  return Mod;
}

TEST(BasicBlockTest, SpliceFromBB) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     from:
       %fromInstr1 = add i32 %a, %a
       %fromInstr2 = sub i32 %a, %a
       br label %to

     to:
       %toInstr1 = mul i32 %a, %a
       %toInstr2 = sdiv i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();
  auto BBIt = F->begin();
  BasicBlock *FromBB = &*BBIt++;
  BasicBlock *ToBB = &*BBIt++;

  auto FromBBIt = FromBB->begin();
  Instruction *FromI1 = &*FromBBIt++;
  Instruction *FromI2 = &*FromBBIt++;
  Instruction *FromBr = &*FromBBIt++;

  auto ToBBIt = ToBB->begin();
  Instruction *ToI1 = &*ToBBIt++;
  Instruction *ToI2 = &*ToBBIt++;
  Instruction *ToRet = &*ToBBIt++;
  ToBB->splice(ToI1->getIterator(), FromBB);

  EXPECT_TRUE(FromBB->empty());

  auto It = ToBB->begin();
  EXPECT_EQ(&*It++, FromI1);
  EXPECT_EQ(&*It++, FromI2);
  EXPECT_EQ(&*It++, FromBr);
  EXPECT_EQ(&*It++, ToI1);
  EXPECT_EQ(&*It++, ToI2);
  EXPECT_EQ(&*It++, ToRet);
}

TEST(BasicBlockTest, SpliceOneInstr) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     from:
       %fromInstr1 = add i32 %a, %a
       %fromInstr2 = sub i32 %a, %a
       br label %to

     to:
       %toInstr1 = mul i32 %a, %a
       %toInstr2 = sdiv i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();
  auto BBIt = F->begin();
  BasicBlock *FromBB = &*BBIt++;
  BasicBlock *ToBB = &*BBIt++;

  auto FromBBIt = FromBB->begin();
  Instruction *FromI1 = &*FromBBIt++;
  Instruction *FromI2 = &*FromBBIt++;
  Instruction *FromBr = &*FromBBIt++;

  auto ToBBIt = ToBB->begin();
  Instruction *ToI1 = &*ToBBIt++;
  Instruction *ToI2 = &*ToBBIt++;
  Instruction *ToRet = &*ToBBIt++;
  ToBB->splice(ToI1->getIterator(), FromBB, FromI2->getIterator());

  EXPECT_EQ(FromBB->size(), 2u);
  EXPECT_EQ(ToBB->size(), 4u);

  auto It = FromBB->begin();
  EXPECT_EQ(&*It++, FromI1);
  EXPECT_EQ(&*It++, FromBr);

  It = ToBB->begin();
  EXPECT_EQ(&*It++, FromI2);
  EXPECT_EQ(&*It++, ToI1);
  EXPECT_EQ(&*It++, ToI2);
  EXPECT_EQ(&*It++, ToRet);
}

TEST(BasicBlockTest, SpliceOneInstrWhenFromIsSameAsTo) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     bb:
       %instr1 = add i32 %a, %a
       %instr2 = sub i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();
  auto BBIt = F->begin();
  BasicBlock *BB = &*BBIt++;

  auto It = BB->begin();
  Instruction *Instr1 = &*It++;
  Instruction *Instr2 = &*It++;
  Instruction *Ret = &*It++;

  // According to ilist's splice() a single-element splice where dst == src
  // should be a noop.
  BB->splice(Instr1->getIterator(), BB, Instr1->getIterator());

  It = BB->begin();
  EXPECT_EQ(&*It++, Instr1);
  EXPECT_EQ(&*It++, Instr2);
  EXPECT_EQ(&*It++, Ret);
  EXPECT_EQ(BB->size(), 3u);
}

TEST(BasicBlockTest, SpliceLastInstr) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     from:
       %fromInstr1 = add i32 %a, %a
       %fromInstr2 = sub i32 %a, %a
       br label %to

     to:
       %toInstr1 = mul i32 %a, %a
       %toInstr2 = sdiv i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();
  auto BBIt = F->begin();
  BasicBlock *FromBB = &*BBIt++;
  BasicBlock *ToBB = &*BBIt++;

  auto FromBBIt = FromBB->begin();
  Instruction *FromI1 = &*FromBBIt++;
  Instruction *FromI2 = &*FromBBIt++;
  Instruction *FromBr = &*FromBBIt++;

  auto ToBBIt = ToBB->begin();
  Instruction *ToI1 = &*ToBBIt++;
  Instruction *ToI2 = &*ToBBIt++;
  Instruction *ToRet = &*ToBBIt++;
  ToBB->splice(ToI1->getIterator(), FromBB, FromI2->getIterator(),
               FromBr->getIterator());

  EXPECT_EQ(FromBB->size(), 2u);
  auto It = FromBB->begin();
  EXPECT_EQ(&*It++, FromI1);
  EXPECT_EQ(&*It++, FromBr);

  EXPECT_EQ(ToBB->size(), 4u);
  It = ToBB->begin();
  EXPECT_EQ(&*It++, FromI2);
  EXPECT_EQ(&*It++, ToI1);
  EXPECT_EQ(&*It++, ToI2);
  EXPECT_EQ(&*It++, ToRet);
}

TEST(BasicBlockTest, SpliceInstrRange) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     from:
       %fromInstr1 = add i32 %a, %a
       %fromInstr2 = sub i32 %a, %a
       br label %to

     to:
       %toInstr1 = mul i32 %a, %a
       %toInstr2 = sdiv i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();
  auto BBIt = F->begin();
  BasicBlock *FromBB = &*BBIt++;
  BasicBlock *ToBB = &*BBIt++;

  auto FromBBIt = FromBB->begin();
  Instruction *FromI1 = &*FromBBIt++;
  Instruction *FromI2 = &*FromBBIt++;
  Instruction *FromBr = &*FromBBIt++;

  auto ToBBIt = ToBB->begin();
  Instruction *ToI1 = &*ToBBIt++;
  Instruction *ToI2 = &*ToBBIt++;
  Instruction *ToRet = &*ToBBIt++;
  ToBB->splice(ToI2->getIterator(), FromBB, FromBB->begin(), FromBB->end());

  EXPECT_EQ(FromBB->size(), 0u);

  EXPECT_EQ(ToBB->size(), 6u);
  auto It = ToBB->begin();
  EXPECT_EQ(&*It++, ToI1);
  EXPECT_EQ(&*It++, FromI1);
  EXPECT_EQ(&*It++, FromI2);
  EXPECT_EQ(&*It++, FromBr);
  EXPECT_EQ(&*It++, ToI2);
  EXPECT_EQ(&*It++, ToRet);
}

#ifdef EXPENSIVE_CHECKS
TEST(BasicBlockTest, SpliceEndBeforeBegin) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     from:
       %fromInstr1 = add i32 %a, %a
       %fromInstr2 = sub i32 %a, %a
       br label %to

     to:
       %toInstr1 = mul i32 %a, %a
       %toInstr2 = sdiv i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();
  auto BBIt = F->begin();
  BasicBlock *FromBB = &*BBIt++;
  BasicBlock *ToBB = &*BBIt++;

  auto FromBBIt = FromBB->begin();
  Instruction *FromI1 = &*FromBBIt++;
  Instruction *FromI2 = &*FromBBIt++;

  auto ToBBIt = ToBB->begin();
  Instruction *ToI2 = &*ToBBIt++;

  EXPECT_DEATH(ToBB->splice(ToI2->getIterator(), FromBB, FromI2->getIterator(),
                            FromI1->getIterator()),
               "FromBeginIt not before FromEndIt!");
}
#endif //EXPENSIVE_CHECKS

TEST(BasicBlockTest, EraseRange) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = parseIR(Ctx, R"(
    define void @f(i32 %a) {
     bb0:
       %instr1 = add i32 %a, %a
       %instr2 = sub i32 %a, %a
       ret void
    }
)");
  Function *F = &*M->begin();

  auto BB0It = F->begin();
  BasicBlock *BB0 = &*BB0It;

  auto It = BB0->begin();
  Instruction *Instr1 = &*It++;
  Instruction *Instr2 = &*It++;

  EXPECT_EQ(BB0->size(), 3u);

  // Erase no instruction
  BB0->erase(Instr1->getIterator(), Instr1->getIterator());
  EXPECT_EQ(BB0->size(), 3u);

  // Erase %instr1
  BB0->erase(Instr1->getIterator(), Instr2->getIterator());
  EXPECT_EQ(BB0->size(), 2u);
  EXPECT_EQ(&*BB0->begin(), Instr2);

  // Erase all instructions
  BB0->erase(BB0->begin(), BB0->end());
  EXPECT_TRUE(BB0->empty());
}

TEST(BasicBlockTest, DiscardValueNames) {
  const char *ModuleString = "declare void @f(i32 %dangling)";
  SMDiagnostic Err;
  LLVMContext Ctx;
  { // Scope of M.
    auto M = parseAssemblyString(ModuleString, Err, Ctx);
    ASSERT_TRUE(M.get());
    EXPECT_FALSE(Ctx.shouldDiscardValueNames());
  }
  { // Scope of M.
    auto M = parseAssemblyString(ModuleString, Err, Ctx);
    ASSERT_TRUE(M.get());
    Ctx.setDiscardValueNames(true);
  }
}

TEST(BasicBlockTest, DiscardValueNames2) {
  SMDiagnostic Err;
  LLVMContext Ctx;
  Module M("Mod", Ctx);
  auto FTy = FunctionType::get(Type::getVoidTy(M.getContext()),
                               {Type::getInt32Ty(Ctx)}, /*isVarArg=*/false);
  { // Scope of F.
    Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", &M);
    F->getArg(0)->setName("dangling");
    F->removeFromParent();
    EXPECT_FALSE(Ctx.shouldDiscardValueNames());
    delete F;
  }
  { // Scope of F.
    Function *F = Function::Create(FTy, Function::ExternalLinkage, "f", &M);
    F->getArg(0)->setName("dangling");
    F->removeFromParent();
    Ctx.setDiscardValueNames(true);
    delete F;
  }
}

} // End anonymous namespace.
} // End llvm namespace.
