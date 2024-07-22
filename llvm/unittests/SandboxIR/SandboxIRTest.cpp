//===- SandboxIRTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(SandboxIRTest, ClassID) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add = add i32 %v1, 42
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  llvm::BasicBlock *LLVMBB = &*LLVMF->begin();
  llvm::Instruction *LLVMAdd = &*LLVMBB->begin();
  auto *LLVMC = cast<llvm::Constant>(LLVMAdd->getOperand(1));

  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  sandboxir::Argument *Arg0 = F->getArg(0);
  sandboxir::BasicBlock *BB = &*F->begin();
  sandboxir::Instruction *AddI = &*BB->begin();
  sandboxir::OpaqueInst *OpaqueI = cast<sandboxir::OpaqueInst>(AddI);
  sandboxir::Constant *Const0 = cast<sandboxir::Constant>(Ctx.getValue(LLVMC));

  EXPECT_TRUE(isa<sandboxir::Function>(F));
  EXPECT_FALSE(isa<sandboxir::Function>(Arg0));
  EXPECT_FALSE(isa<sandboxir::Function>(BB));
  EXPECT_FALSE(isa<sandboxir::Function>(AddI));
  EXPECT_FALSE(isa<sandboxir::Function>(Const0));
  EXPECT_FALSE(isa<sandboxir::Function>(OpaqueI));

  EXPECT_FALSE(isa<sandboxir::Argument>(F));
  EXPECT_TRUE(isa<sandboxir::Argument>(Arg0));
  EXPECT_FALSE(isa<sandboxir::Argument>(BB));
  EXPECT_FALSE(isa<sandboxir::Argument>(AddI));
  EXPECT_FALSE(isa<sandboxir::Argument>(Const0));
  EXPECT_FALSE(isa<sandboxir::Argument>(OpaqueI));

  EXPECT_TRUE(isa<sandboxir::Constant>(F));
  EXPECT_FALSE(isa<sandboxir::Constant>(Arg0));
  EXPECT_FALSE(isa<sandboxir::Constant>(BB));
  EXPECT_FALSE(isa<sandboxir::Constant>(AddI));
  EXPECT_TRUE(isa<sandboxir::Constant>(Const0));
  EXPECT_FALSE(isa<sandboxir::Constant>(OpaqueI));

  EXPECT_FALSE(isa<sandboxir::OpaqueInst>(F));
  EXPECT_FALSE(isa<sandboxir::OpaqueInst>(Arg0));
  EXPECT_FALSE(isa<sandboxir::OpaqueInst>(BB));
  EXPECT_TRUE(isa<sandboxir::OpaqueInst>(AddI));
  EXPECT_FALSE(isa<sandboxir::OpaqueInst>(Const0));
  EXPECT_TRUE(isa<sandboxir::OpaqueInst>(OpaqueI));

  EXPECT_FALSE(isa<sandboxir::Instruction>(F));
  EXPECT_FALSE(isa<sandboxir::Instruction>(Arg0));
  EXPECT_FALSE(isa<sandboxir::Instruction>(BB));
  EXPECT_TRUE(isa<sandboxir::Instruction>(AddI));
  EXPECT_FALSE(isa<sandboxir::Instruction>(Const0));
  EXPECT_TRUE(isa<sandboxir::Instruction>(OpaqueI));

  EXPECT_FALSE(isa<sandboxir::User>(F));
  EXPECT_FALSE(isa<sandboxir::User>(Arg0));
  EXPECT_FALSE(isa<sandboxir::User>(BB));
  EXPECT_TRUE(isa<sandboxir::User>(AddI));
  EXPECT_TRUE(isa<sandboxir::User>(Const0));
  EXPECT_TRUE(isa<sandboxir::User>(OpaqueI));

#ifndef NDEBUG
  std::string Buff;
  raw_string_ostream BS(Buff);
  F->dump(BS);
  Arg0->dump(BS);
  BB->dump(BS);
  AddI->dump(BS);
  Const0->dump(BS);
  OpaqueI->dump(BS);
#endif
}

TEST_F(SandboxIRTest, Use) {
  parseIR(C, R"IR(
define i32 @foo(i32 %v0, i32 %v1) {
  %add0 = add i32 %v0, %v1
  ret i32 %add0
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  BasicBlock *LLVMBB = &*LLVMF.begin();
  auto LLVMBBIt = LLVMBB->begin();
  Instruction *LLVMI0 = &*LLVMBBIt++;
  Instruction *LLVMRet = &*LLVMBBIt++;
  Argument *LLVMArg0 = LLVMF.getArg(0);
  Argument *LLVMArg1 = LLVMF.getArg(1);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *F.begin();
  auto *Arg0 = F.getArg(0);
  auto *Arg1 = F.getArg(1);
  auto It = BB.begin();
  auto *I0 = &*It++;
  auto *Ret = &*It++;

  SmallVector<sandboxir::Argument *> Args{Arg0, Arg1};
  unsigned OpIdx = 0;
  for (sandboxir::Use Use : I0->operands()) {
    // Check Use.getOperandNo().
    EXPECT_EQ(Use.getOperandNo(), OpIdx);
    // Check Use.getUser().
    EXPECT_EQ(Use.getUser(), I0);
    // Check Use.getContext().
    EXPECT_EQ(Use.getContext(), &Ctx);
    // Check Use.get().
    sandboxir::Value *Op = Use.get();
    EXPECT_EQ(Op, Ctx.getValue(LLVMI0->getOperand(OpIdx)));
    // Check Use.getUser().
    EXPECT_EQ(Use.getUser(), I0);
    // Check implicit cast to Value.
    sandboxir::Value *Cast = Use;
    EXPECT_EQ(Cast, Op);
    // Check that Use points to the correct operand.
    EXPECT_EQ(Op, Args[OpIdx]);
    // Check getOperand().
    EXPECT_EQ(Op, I0->getOperand(OpIdx));
    // Check getOperandUse().
    EXPECT_EQ(Use, I0->getOperandUse(OpIdx));
    ++OpIdx;
  }
  EXPECT_EQ(OpIdx, 2u);

  // Check Use.operator==() and Use.operator!=().
  sandboxir::Use UseA = I0->getOperandUse(0);
  sandboxir::Use UseB = I0->getOperandUse(0);
  EXPECT_TRUE(UseA == UseB);
  EXPECT_FALSE(UseA != UseB);

  // Check getNumOperands().
  EXPECT_EQ(I0->getNumOperands(), 2u);
  EXPECT_EQ(Ret->getNumOperands(), 1u);

  EXPECT_EQ(Ret->getOperand(0), I0);

#ifndef NDEBUG
  // Check Use.dump()
  std::string Buff;
  raw_string_ostream BS(Buff);
  BS << "\n";
  I0->getOperandUse(0).dump(BS);
  EXPECT_EQ(Buff, R"IR(
Def:  i32 %v0 ; SB1. (Argument)
User:   %add0 = add i32 %v0, %v1 ; SB4. (Opaque)
OperandNo: 0
)IR");
#endif // NDEBUG

  // Check Value.user_begin().
  sandboxir::Value::user_iterator UIt = I0->user_begin();
  sandboxir::User *U = *UIt;
  EXPECT_EQ(U, Ret);
  // Check Value.uses().
  EXPECT_EQ(range_size(I0->uses()), 1u);
  EXPECT_EQ((*I0->uses().begin()).getUser(), Ret);
  // Check Value.users().
  EXPECT_EQ(range_size(I0->users()), 1u);
  EXPECT_EQ(*I0->users().begin(), Ret);
  // Check Value.getNumUses().
  EXPECT_EQ(I0->getNumUses(), 1u);
  // Check Value.hasNUsesOrMore().
  EXPECT_TRUE(I0->hasNUsesOrMore(0u));
  EXPECT_TRUE(I0->hasNUsesOrMore(1u));
  EXPECT_FALSE(I0->hasNUsesOrMore(2u));
  // Check Value.hasNUses().
  EXPECT_FALSE(I0->hasNUses(0u));
  EXPECT_TRUE(I0->hasNUses(1u));
  EXPECT_FALSE(I0->hasNUses(2u));

  // Check User.setOperand().
  Ret->setOperand(0, Arg0);
  EXPECT_EQ(Ret->getOperand(0), Arg0);
  EXPECT_EQ(Ret->getOperandUse(0).get(), Arg0);
  EXPECT_EQ(LLVMRet->getOperand(0), LLVMArg0);

  Ret->setOperand(0, Arg1);
  EXPECT_EQ(Ret->getOperand(0), Arg1);
  EXPECT_EQ(Ret->getOperandUse(0).get(), Arg1);
  EXPECT_EQ(LLVMRet->getOperand(0), LLVMArg1);
}

TEST_F(SandboxIRTest, RUOW) {
  parseIR(C, R"IR(
declare void @bar0()
declare void @bar1()

@glob0 = global ptr @bar0
@glob1 = global ptr @bar1

define i32 @foo(i32 %arg0, i32 %arg1) {
  %add0 = add i32 %arg0, %arg1
  %gep1 = getelementptr i8, ptr @glob0, i32 1
  %gep2 = getelementptr i8, ptr @glob1, i32 1
  ret i32 %add0
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);

  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *F.begin();
  auto *Arg0 = F.getArg(0);
  auto *Arg1 = F.getArg(1);
  auto It = BB.begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  auto *Ret = &*It++;

  bool Replaced;
  // Try to replace an operand that doesn't match.
  Replaced = I0->replaceUsesOfWith(Ret, Arg1);
  EXPECT_FALSE(Replaced);
  EXPECT_EQ(I0->getOperand(0), Arg0);
  EXPECT_EQ(I0->getOperand(1), Arg1);

  // Replace I0 operands when operands differ.
  Replaced = I0->replaceUsesOfWith(Arg0, Arg1);
  EXPECT_TRUE(Replaced);
  EXPECT_EQ(I0->getOperand(0), Arg1);
  EXPECT_EQ(I0->getOperand(1), Arg1);

  // Replace I0 operands when operands are the same.
  Replaced = I0->replaceUsesOfWith(Arg1, Arg0);
  EXPECT_TRUE(Replaced);
  EXPECT_EQ(I0->getOperand(0), Arg0);
  EXPECT_EQ(I0->getOperand(1), Arg0);

  // Replace Ret operand.
  Replaced = Ret->replaceUsesOfWith(I0, Arg0);
  EXPECT_TRUE(Replaced);
  EXPECT_EQ(Ret->getOperand(0), Arg0);

  // Check RAUW on constant.
  auto *Glob0 = cast<sandboxir::Constant>(I1->getOperand(0));
  auto *Glob1 = cast<sandboxir::Constant>(I2->getOperand(0));
  auto *Glob0Op = Glob0->getOperand(0);
  Glob0->replaceUsesOfWith(Glob0Op, Glob1);
  EXPECT_EQ(Glob0->getOperand(0), Glob1);
}

TEST_F(SandboxIRTest, RAUW_RUWIf) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ld0 = load float, ptr %ptr
  %ld1 = load float, ptr %ptr
  store float %ld0, ptr %ptr
  store float %ld0, ptr %ptr
  ret void
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();

  Ctx.createFunction(&LLVMF);
  auto *BB = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMBB));
  auto It = BB->begin();
  sandboxir::Instruction *Ld0 = &*It++;
  sandboxir::Instruction *Ld1 = &*It++;
  sandboxir::Instruction *St0 = &*It++;
  sandboxir::Instruction *St1 = &*It++;
  // Check RUWIf when the lambda returns false.
  Ld0->replaceUsesWithIf(Ld1, [](const sandboxir::Use &Use) { return false; });
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);
  // Check RUWIf when the lambda returns true.
  Ld0->replaceUsesWithIf(Ld1, [](const sandboxir::Use &Use) { return true; });
  EXPECT_EQ(St0->getOperand(0), Ld1);
  EXPECT_EQ(St1->getOperand(0), Ld1);
  St0->setOperand(0, Ld0);
  St1->setOperand(0, Ld0);
  // Check RUWIf user == St0.
  Ld0->replaceUsesWithIf(
      Ld1, [St0](const sandboxir::Use &Use) { return Use.getUser() == St0; });
  EXPECT_EQ(St0->getOperand(0), Ld1);
  EXPECT_EQ(St1->getOperand(0), Ld0);
  St0->setOperand(0, Ld0);
  // Check RUWIf user == St1.
  Ld0->replaceUsesWithIf(
      Ld1, [St1](const sandboxir::Use &Use) { return Use.getUser() == St1; });
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld1);
  St1->setOperand(0, Ld0);
  // Check RAUW.
  Ld1->replaceAllUsesWith(Ld0);
  EXPECT_EQ(St0->getOperand(0), Ld0);
  EXPECT_EQ(St1->getOperand(0), Ld0);
}

// Check that the operands/users are counted correctly.
//  I1
// /  \
// \  /
//  I2
TEST_F(SandboxIRTest, DuplicateUses) {
  parseIR(C, R"IR(
define void @foo(i8 %v) {
  %I1 = add i8 %v, %v
  %I2 = add i8 %I1, %I1
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *I1 = &*It++;
  auto *I2 = &*It++;
  EXPECT_EQ(range_size(I1->users()), 2u);
  EXPECT_EQ(range_size(I2->operands()), 2u);
}

TEST_F(SandboxIRTest, Function) {
  parseIR(C, R"IR(
define void @foo(i32 %arg0, i32 %arg1) {
bb0:
  br label %bb1
bb1:
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  llvm::Argument *LLVMArg0 = LLVMF->getArg(0);
  llvm::Argument *LLVMArg1 = LLVMF->getArg(1);

  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);

  // Check F arguments
  EXPECT_EQ(F->arg_size(), 2u);
  EXPECT_FALSE(F->arg_empty());
  EXPECT_EQ(F->getArg(0), Ctx.getValue(LLVMArg0));
  EXPECT_EQ(F->getArg(1), Ctx.getValue(LLVMArg1));

  // Check F.begin(), F.end(), Function::iterator
  llvm::BasicBlock *LLVMBB = &*LLVMF->begin();
  for (sandboxir::BasicBlock &BB : *F) {
    EXPECT_EQ(&BB, Ctx.getValue(LLVMBB));
    LLVMBB = LLVMBB->getNextNode();
  }

#ifndef NDEBUG
  {
    // Check F.dumpNameAndArgs()
    std::string Buff;
    raw_string_ostream BS(Buff);
    F->dumpNameAndArgs(BS);
    EXPECT_EQ(Buff, "void @foo(i32 %arg0, i32 %arg1)");
  }
  {
    // Check F.dump()
    std::string Buff;
    raw_string_ostream BS(Buff);
    BS << "\n";
    F->dump(BS);
    EXPECT_EQ(Buff, R"IR(
void @foo(i32 %arg0, i32 %arg1) {
bb0:
  br label %bb1 ; SB3. (Opaque)

bb1:
  ret void ; SB5. (Opaque)
}
)IR");
  }
#endif // NDEBUG
}

TEST_F(SandboxIRTest, BasicBlock) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
bb0:
  br label %bb1
bb1:
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  llvm::BasicBlock *LLVMBB0 = getBasicBlockByName(*LLVMF, "bb0");
  llvm::BasicBlock *LLVMBB1 = getBasicBlockByName(*LLVMF, "bb1");

  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto &BB0 = cast<sandboxir::BasicBlock>(*Ctx.getValue(LLVMBB0));
  auto &BB1 = cast<sandboxir::BasicBlock>(*Ctx.getValue(LLVMBB1));

  // Check BB::classof()
  EXPECT_TRUE(isa<sandboxir::Value>(BB0));
  EXPECT_FALSE(isa<sandboxir::User>(BB0));
  EXPECT_FALSE(isa<sandboxir::Instruction>(BB0));
  EXPECT_FALSE(isa<sandboxir::Constant>(BB0));
  EXPECT_FALSE(isa<sandboxir::Argument>(BB0));

  // Check BB.getParent()
  EXPECT_EQ(BB0.getParent(), F);
  EXPECT_EQ(BB1.getParent(), F);

  // Check BBIterator, BB.begin(), BB.end().
  llvm::Instruction *LLVMI = &*LLVMBB0->begin();
  for (sandboxir::Instruction &I : BB0) {
    EXPECT_EQ(&I, Ctx.getValue(LLVMI));
    LLVMI = LLVMI->getNextNode();
  }
  LLVMI = &*LLVMBB1->begin();
  for (sandboxir::Instruction &I : BB1) {
    EXPECT_EQ(&I, Ctx.getValue(LLVMI));
    LLVMI = LLVMI->getNextNode();
  }

  // Check BB.getTerminator()
  EXPECT_EQ(BB0.getTerminator(), Ctx.getValue(LLVMBB0->getTerminator()));
  EXPECT_EQ(BB1.getTerminator(), Ctx.getValue(LLVMBB1->getTerminator()));

  // Check BB.rbegin(), BB.rend()
  EXPECT_EQ(&*BB0.rbegin(), BB0.getTerminator());
  EXPECT_EQ(&*std::prev(BB0.rend()), &*BB0.begin());

#ifndef NDEBUG
  {
    // Check BB.dump()
    std::string Buff;
    raw_string_ostream BS(Buff);
    BS << "\n";
    BB0.dump(BS);
    EXPECT_EQ(Buff, R"IR(
bb0:
  br label %bb1 ; SB2. (Opaque)
)IR");
  }
#endif // NDEBUG
}

TEST_F(SandboxIRTest, Instruction) {
  parseIR(C, R"IR(
define void @foo(i8 %v1) {
  %add0 = add i8 %v1, %v1
  %sub1 = sub i8 %add0, %v1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *Arg = F->getArg(0);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *I0 = &*It++;
  auto *I1 = &*It++;
  auto *Ret = &*It++;

  // Check getPrevNode().
  EXPECT_EQ(Ret->getPrevNode(), I1);
  EXPECT_EQ(I1->getPrevNode(), I0);
  EXPECT_EQ(I0->getPrevNode(), nullptr);

  // Check getNextNode().
  EXPECT_EQ(I0->getNextNode(), I1);
  EXPECT_EQ(I1->getNextNode(), Ret);
  EXPECT_EQ(Ret->getNextNode(), nullptr);

  // Check getIterator().
  EXPECT_EQ(I0->getIterator(), std::next(BB->begin(), 0));
  EXPECT_EQ(I1->getIterator(), std::next(BB->begin(), 1));
  EXPECT_EQ(Ret->getIterator(), std::next(BB->begin(), 2));

  // Check getOpcode().
  EXPECT_EQ(I0->getOpcode(), sandboxir::Instruction::Opcode::Opaque);
  EXPECT_EQ(I1->getOpcode(), sandboxir::Instruction::Opcode::Opaque);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Opaque);

  // Check moveBefore(I).
  I1->moveBefore(I0);
  EXPECT_EQ(I0->getPrevNode(), I1);
  EXPECT_EQ(I1->getNextNode(), I0);

  // Check moveAfter(I).
  I1->moveAfter(I0);
  EXPECT_EQ(I0->getNextNode(), I1);
  EXPECT_EQ(I1->getPrevNode(), I0);

  // Check moveBefore(BB, It).
  I1->moveBefore(*BB, BB->begin());
  EXPECT_EQ(I1->getPrevNode(), nullptr);
  EXPECT_EQ(I1->getNextNode(), I0);
  I1->moveBefore(*BB, BB->end());
  EXPECT_EQ(I1->getNextNode(), nullptr);
  EXPECT_EQ(Ret->getNextNode(), I1);
  I1->moveBefore(*BB, std::next(BB->begin()));
  EXPECT_EQ(I0->getNextNode(), I1);
  EXPECT_EQ(I1->getNextNode(), Ret);

  // Check removeFromParent().
  I0->removeFromParent();
#ifndef NDEBUG
  EXPECT_DEATH(I0->getPrevNode(), ".*Detached.*");
  EXPECT_DEATH(I0->getNextNode(), ".*Detached.*");
#endif // NDEBUG
  EXPECT_EQ(I0->getParent(), nullptr);
  EXPECT_EQ(I1->getPrevNode(), nullptr);
  EXPECT_EQ(I0->getOperand(0), Arg);

  // Check insertBefore().
  I0->insertBefore(I1);
  EXPECT_EQ(I1->getPrevNode(), I0);

  // Check insertInto().
  I0->removeFromParent();
  I0->insertInto(BB, BB->end());
  EXPECT_EQ(Ret->getNextNode(), I0);
  I0->moveBefore(I1);
  EXPECT_EQ(I0->getNextNode(), I1);

  // Check eraseFromParent().
#ifndef NDEBUG
  EXPECT_DEATH(I0->eraseFromParent(), "Still connected to users.*");
#endif
  I1->eraseFromParent();
  EXPECT_EQ(I0->getNumUses(), 0u);
  EXPECT_EQ(I0->getNextNode(), Ret);
}
