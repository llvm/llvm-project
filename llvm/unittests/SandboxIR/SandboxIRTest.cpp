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

  EXPECT_TRUE(isa<sandboxir::User>(F));
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
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

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
Def:  i32 %v0 ; SB2. (Argument)
User:   %add0 = add i32 %v0, %v1 ; SB5. (Opaque)
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
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

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
  br label %bb1 ; SB4. (Br)

bb1:
  ret void ; SB6. (Ret)
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
  br label %bb1 ; SB3. (Br)
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
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

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
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);

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

TEST_F(SandboxIRTest, SelectInst) {
  parseIR(C, R"IR(
define void @foo(i1 %c0, i8 %v0, i8 %v1, i1 %c1) {
  %sel = select i1 %c0, i8 %v0, i8 %v1
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *Cond0 = F->getArg(0);
  auto *V0 = F->getArg(1);
  auto *V1 = F->getArg(2);
  auto *Cond1 = F->getArg(3);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Select = cast<sandboxir::SelectInst>(&*It++);
  auto *Ret = &*It++;

  // Check getCondition().
  EXPECT_EQ(Select->getCondition(), Cond0);
  // Check getTrueValue().
  EXPECT_EQ(Select->getTrueValue(), V0);
  // Check getFalseValue().
  EXPECT_EQ(Select->getFalseValue(), V1);
  // Check setCondition().
  Select->setCondition(Cond1);
  EXPECT_EQ(Select->getCondition(), Cond1);
  // Check setTrueValue().
  Select->setTrueValue(V1);
  EXPECT_EQ(Select->getTrueValue(), V1);
  // Check setFalseValue().
  Select->setFalseValue(V0);
  EXPECT_EQ(Select->getFalseValue(), V0);

  {
    // Check SelectInst::create() InsertBefore.
    auto *NewSel = cast<sandboxir::SelectInst>(sandboxir::SelectInst::create(
        Cond0, V0, V1, /*InsertBefore=*/Ret, Ctx));
    EXPECT_EQ(NewSel->getCondition(), Cond0);
    EXPECT_EQ(NewSel->getTrueValue(), V0);
    EXPECT_EQ(NewSel->getFalseValue(), V1);
    EXPECT_EQ(NewSel->getNextNode(), Ret);
  }
  {
    // Check SelectInst::create() InsertAtEnd.
    auto *NewSel = cast<sandboxir::SelectInst>(
        sandboxir::SelectInst::create(Cond0, V0, V1, /*InsertAtEnd=*/BB, Ctx));
    EXPECT_EQ(NewSel->getCondition(), Cond0);
    EXPECT_EQ(NewSel->getTrueValue(), V0);
    EXPECT_EQ(NewSel->getFalseValue(), V1);
    EXPECT_EQ(NewSel->getPrevNode(), Ret);
  }
  {
    // Check SelectInst::create() Folded.
    auto *False =
        sandboxir::Constant::createInt(llvm::Type::getInt1Ty(C), 0, Ctx,
                                       /*IsSigned=*/false);
    auto *FortyTwo =
        sandboxir::Constant::createInt(llvm::Type::getInt1Ty(C), 42, Ctx,
                                       /*IsSigned=*/false);
    auto *NewSel =
        sandboxir::SelectInst::create(False, FortyTwo, FortyTwo, Ret, Ctx);
    EXPECT_TRUE(isa<sandboxir::Constant>(NewSel));
    EXPECT_EQ(NewSel, FortyTwo);
  }
}

TEST_F(SandboxIRTest, BranchInst) {
  parseIR(C, R"IR(
define void @foo(i1 %cond0, i1 %cond2) {
 bb0:
   br i1 %cond0, label %bb1, label %bb2
 bb1:
   ret void
 bb2:
   ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *Cond0 = F->getArg(0);
  auto *Cond1 = F->getArg(1);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(*LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(*LLVMF, "bb1")));
  auto *Ret1 = BB1->getTerminator();
  auto *BB2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(*LLVMF, "bb2")));
  auto *Ret2 = BB2->getTerminator();
  auto It = BB0->begin();
  auto *Br0 = cast<sandboxir::BranchInst>(&*It++);
  // Check isUnconditional().
  EXPECT_FALSE(Br0->isUnconditional());
  // Check isConditional().
  EXPECT_TRUE(Br0->isConditional());
  // Check getCondition().
  EXPECT_EQ(Br0->getCondition(), Cond0);
  // Check setCondition().
  Br0->setCondition(Cond1);
  EXPECT_EQ(Br0->getCondition(), Cond1);
  // Check getNumSuccessors().
  EXPECT_EQ(Br0->getNumSuccessors(), 2u);
  // Check getSuccessor().
  EXPECT_EQ(Br0->getSuccessor(0), BB1);
  EXPECT_EQ(Br0->getSuccessor(1), BB2);
  // Check swapSuccessors().
  Br0->swapSuccessors();
  EXPECT_EQ(Br0->getSuccessor(0), BB2);
  EXPECT_EQ(Br0->getSuccessor(1), BB1);
  // Check successors().
  EXPECT_EQ(range_size(Br0->successors()), 2u);
  unsigned SuccIdx = 0;
  SmallVector<sandboxir::BasicBlock *> ExpectedSuccs({BB1, BB2});
  for (sandboxir::BasicBlock *Succ : Br0->successors())
    EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);

  {
    // Check unconditional BranchInst::create() InsertBefore.
    auto *Br = sandboxir::BranchInst::create(BB1, /*InsertBefore=*/Ret1, Ctx);
    EXPECT_FALSE(Br->isConditional());
    EXPECT_TRUE(Br->isUnconditional());
#ifndef NDEBUG
    EXPECT_DEATH(Br->getCondition(), ".*condition.*");
#endif // NDEBUG
    unsigned SuccIdx = 0;
    SmallVector<sandboxir::BasicBlock *> ExpectedSuccs({BB1});
    for (sandboxir::BasicBlock *Succ : Br->successors())
      EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);
    EXPECT_EQ(Br->getNextNode(), Ret1);
  }
  {
    // Check unconditional BranchInst::create() InsertAtEnd.
    auto *Br = sandboxir::BranchInst::create(BB1, /*InsertAtEnd=*/BB1, Ctx);
    EXPECT_FALSE(Br->isConditional());
    EXPECT_TRUE(Br->isUnconditional());
#ifndef NDEBUG
    EXPECT_DEATH(Br->getCondition(), ".*condition.*");
#endif // NDEBUG
    unsigned SuccIdx = 0;
    SmallVector<sandboxir::BasicBlock *> ExpectedSuccs({BB1});
    for (sandboxir::BasicBlock *Succ : Br->successors())
      EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);
    EXPECT_EQ(Br->getPrevNode(), Ret1);
  }
  {
    // Check conditional BranchInst::create() InsertBefore.
    auto *Br = sandboxir::BranchInst::create(BB1, BB2, Cond0,
                                             /*InsertBefore=*/Ret1, Ctx);
    EXPECT_TRUE(Br->isConditional());
    EXPECT_EQ(Br->getCondition(), Cond0);
    unsigned SuccIdx = 0;
    SmallVector<sandboxir::BasicBlock *> ExpectedSuccs({BB2, BB1});
    for (sandboxir::BasicBlock *Succ : Br->successors())
      EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);
    EXPECT_EQ(Br->getNextNode(), Ret1);
  }
  {
    // Check conditional BranchInst::create() InsertAtEnd.
    auto *Br = sandboxir::BranchInst::create(BB1, BB2, Cond0,
                                             /*InsertAtEnd=*/BB2, Ctx);
    EXPECT_TRUE(Br->isConditional());
    EXPECT_EQ(Br->getCondition(), Cond0);
    unsigned SuccIdx = 0;
    SmallVector<sandboxir::BasicBlock *> ExpectedSuccs({BB2, BB1});
    for (sandboxir::BasicBlock *Succ : Br->successors())
      EXPECT_EQ(Succ, ExpectedSuccs[SuccIdx++]);
    EXPECT_EQ(Br->getPrevNode(), Ret2);
  }
}

TEST_F(SandboxIRTest, LoadInst) {
  parseIR(C, R"IR(
define void @foo(ptr %arg0, ptr %arg1) {
  %ld = load i8, ptr %arg0, align 64
  %vld = load volatile i8, ptr %arg0, align 64
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *Arg0 = F->getArg(0);
  auto *Arg1 = F->getArg(1);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Ld = cast<sandboxir::LoadInst>(&*It++);
  auto *VLd = cast<sandboxir::LoadInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check isVolatile()
  EXPECT_FALSE(Ld->isVolatile());
  // Check isVolatile()
  EXPECT_TRUE(VLd->isVolatile());
  // Check getPointerOperand()
  EXPECT_EQ(Ld->getPointerOperand(), Arg0);
  // Check getAlign()
  EXPECT_EQ(Ld->getAlign(), 64);
  // Check create(InsertBefore)
  sandboxir::LoadInst *NewLd =
      sandboxir::LoadInst::create(Ld->getType(), Arg1, Align(8),
                                  /*InsertBefore=*/Ret, Ctx, "NewLd");
  // Checking if create() was volatile
  EXPECT_FALSE(NewLd->isVolatile());
  EXPECT_EQ(NewLd->getType(), Ld->getType());
  EXPECT_EQ(NewLd->getPointerOperand(), Arg1);
  EXPECT_EQ(NewLd->getAlign(), 8);
  EXPECT_EQ(NewLd->getName(), "NewLd");

  sandboxir::LoadInst *NewVLd =
      sandboxir::LoadInst::create(VLd->getType(), Arg1, Align(8),
                                  /*InsertBefore=*/Ret,
                                  /*IsVolatile=*/true, Ctx, "NewVLd");

  // Checking if create() was volatile
  EXPECT_TRUE(NewVLd->isVolatile());
  EXPECT_EQ(NewVLd->getName(), "NewVLd");

  // Check create(InsertAtEnd)
  sandboxir::LoadInst *NewLdEnd =
      sandboxir::LoadInst::create(Ld->getType(), Arg1, Align(8),
                                  /*InsertAtEnd=*/BB, Ctx, "NewLdEnd");
  EXPECT_FALSE(NewLdEnd->isVolatile());
  EXPECT_EQ(NewLdEnd->getName(), "NewLdEnd");
  // Check create(InsertAtEnd)
  sandboxir::LoadInst *NewVLdEnd =
      sandboxir::LoadInst::create(VLd->getType(), Arg1, Align(8),
                                  /*InsertAtEnd=*/BB,
                                  /*IsVolatile=*/true, Ctx, "NewVLdEnd");
  EXPECT_TRUE(NewVLdEnd->isVolatile());
  EXPECT_EQ(NewVLdEnd->getName(), "NewVLdEnd");
}

TEST_F(SandboxIRTest, StoreInst) {
  parseIR(C, R"IR(
define void @foo(i8 %val, ptr %ptr) {
  store i8 %val, ptr %ptr, align 64
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *Val = F->getArg(0);
  auto *Ptr = F->getArg(1);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *St = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check that the StoreInst has been created correctly.
  // Check getPointerOperand()
  EXPECT_EQ(St->getValueOperand(), Val);
  EXPECT_EQ(St->getPointerOperand(), Ptr);
  // Check getAlign()
  EXPECT_EQ(St->getAlign(), 64);
  // Check create(InsertBefore)
  sandboxir::StoreInst *NewSt =
      sandboxir::StoreInst::create(Val, Ptr, Align(8),
                                   /*InsertBefore=*/Ret, Ctx);
  EXPECT_EQ(NewSt->getType(), St->getType());
  EXPECT_EQ(NewSt->getValueOperand(), Val);
  EXPECT_EQ(NewSt->getPointerOperand(), Ptr);
  EXPECT_EQ(NewSt->getAlign(), 8);
}

TEST_F(SandboxIRTest, ReturnInst) {
  parseIR(C, R"IR(
define i8 @foo(i8 %val) {
  %add = add i8 %val, 42
  ret i8 %val
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *Val = F->getArg(0);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  It++;
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check that the ReturnInst has been created correctly.
  // Check getReturnValue().
  EXPECT_EQ(Ret->getReturnValue(), Val);

  // Check create(InsertBefore) a void ReturnInst.
  auto *NewRet1 = cast<sandboxir::ReturnInst>(
      sandboxir::ReturnInst::create(nullptr, /*InsertBefore=*/Ret, Ctx));
  EXPECT_EQ(NewRet1->getReturnValue(), nullptr);
  // Check create(InsertBefore) a non-void ReturnInst.
  auto *NewRet2 = cast<sandboxir::ReturnInst>(
      sandboxir::ReturnInst::create(Val, /*InsertBefore=*/Ret, Ctx));
  EXPECT_EQ(NewRet2->getReturnValue(), Val);

  // Check create(InsertAtEnd) a void ReturnInst.
  auto *NewRet3 = cast<sandboxir::ReturnInst>(
      sandboxir::ReturnInst::create(nullptr, /*InsertAtEnd=*/BB, Ctx));
  EXPECT_EQ(NewRet3->getReturnValue(), nullptr);
  // Check create(InsertAtEnd) a non-void ReturnInst.
  auto *NewRet4 = cast<sandboxir::ReturnInst>(
      sandboxir::ReturnInst::create(Val, /*InsertAtEnd=*/BB, Ctx));
  EXPECT_EQ(NewRet4->getReturnValue(), Val);
}

TEST_F(SandboxIRTest, CallBase) {
  parseIR(C, R"IR(
declare void @bar1(i8)
declare void @bar2()
declare void @bar3()
declare void @variadic(ptr, ...)

define i8 @foo(i8 %arg0, i32 %arg1, ptr %indirectFoo) {
  %call = call i8 @foo(i8 %arg0, i32 %arg1)
  call void @bar1(i8 %arg0)
  call void @bar2()
  call void %indirectFoo()
  call void @bar2() noreturn
  tail call fastcc void @bar2()
  call void (ptr, ...) @variadic(ptr %indirectFoo, i32 1)
  ret i8 %call
}
)IR");
  llvm::Function &LLVMF = *M->getFunction("foo");
  unsigned ArgIdx = 0;
  llvm::Argument *LLVMArg0 = LLVMF.getArg(ArgIdx++);
  llvm::Argument *LLVMArg1 = LLVMF.getArg(ArgIdx++);
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();
  SmallVector<llvm::CallBase *, 8> LLVMCalls;
  auto LLVMIt = LLVMBB->begin();
  while (isa<llvm::CallBase>(&*LLVMIt))
    LLVMCalls.push_back(cast<llvm::CallBase>(&*LLVMIt++));

  sandboxir::Context Ctx(C);
  sandboxir::Function &F = *Ctx.createFunction(&LLVMF);

  for (llvm::CallBase *LLVMCall : LLVMCalls) {
    // Check classof(Instruction *).
    auto *Call = cast<sandboxir::CallBase>(Ctx.getValue(LLVMCall));
    // Check classof(Value *).
    EXPECT_TRUE(isa<sandboxir::CallBase>((sandboxir::Value *)Call));
    // Check getFunctionType().
    EXPECT_EQ(Call->getFunctionType(), LLVMCall->getFunctionType());
    // Check data_ops().
    EXPECT_EQ(range_size(Call->data_ops()), range_size(LLVMCall->data_ops()));
    auto DataOpIt = Call->data_operands_begin();
    for (llvm::Use &LLVMUse : LLVMCall->data_ops()) {
      Value *LLVMOp = LLVMUse.get();
      sandboxir::Use Use = *DataOpIt++;
      EXPECT_EQ(Ctx.getValue(LLVMOp), Use.get());
      // Check isDataOperand().
      EXPECT_EQ(Call->isDataOperand(Use), LLVMCall->isDataOperand(&LLVMUse));
      // Check getDataOperandNo().
      EXPECT_EQ(Call->getDataOperandNo(Use),
                LLVMCall->getDataOperandNo(&LLVMUse));
      // Check isArgOperand().
      EXPECT_EQ(Call->isArgOperand(Use), LLVMCall->isArgOperand(&LLVMUse));
      // Check isCallee().
      EXPECT_EQ(Call->isCallee(Use), LLVMCall->isCallee(&LLVMUse));
    }
    // Check data_operands_empty().
    EXPECT_EQ(Call->data_operands_empty(), LLVMCall->data_operands_empty());
    // Check data_operands_size().
    EXPECT_EQ(Call->data_operands_size(), LLVMCall->data_operands_size());
    // Check getNumTotalBundleOperands().
    EXPECT_EQ(Call->getNumTotalBundleOperands(),
              LLVMCall->getNumTotalBundleOperands());
    // Check args().
    EXPECT_EQ(range_size(Call->args()), range_size(LLVMCall->args()));
    auto ArgIt = Call->arg_begin();
    for (llvm::Use &LLVMUse : LLVMCall->args()) {
      Value *LLVMArg = LLVMUse.get();
      sandboxir::Use Use = *ArgIt++;
      EXPECT_EQ(Ctx.getValue(LLVMArg), Use.get());
    }
    // Check arg_empty().
    EXPECT_EQ(Call->arg_empty(), LLVMCall->arg_empty());
    // Check arg_size().
    EXPECT_EQ(Call->arg_size(), LLVMCall->arg_size());
    for (unsigned ArgIdx = 0, E = Call->arg_size(); ArgIdx != E; ++ArgIdx) {
      // Check getArgOperand().
      EXPECT_EQ(Call->getArgOperand(ArgIdx),
                Ctx.getValue(LLVMCall->getArgOperand(ArgIdx)));
      // Check getArgOperandUse().
      sandboxir::Use Use = Call->getArgOperandUse(ArgIdx);
      llvm::Use &LLVMUse = LLVMCall->getArgOperandUse(ArgIdx);
      EXPECT_EQ(Use.get(), Ctx.getValue(LLVMUse.get()));
      // Check getArgOperandNo().
      EXPECT_EQ(Call->getArgOperandNo(Use),
                LLVMCall->getArgOperandNo(&LLVMUse));
    }
    // Check hasArgument().
    SmallVector<llvm::Value *> TestArgs(
        {LLVMArg0, LLVMArg1, &LLVMF, LLVMBB, LLVMCall});
    for (llvm::Value *LLVMV : TestArgs) {
      sandboxir::Value *V = Ctx.getValue(LLVMV);
      EXPECT_EQ(Call->hasArgument(V), LLVMCall->hasArgument(LLVMV));
    }
    // Check getCalledOperand().
    EXPECT_EQ(Call->getCalledOperand(),
              Ctx.getValue(LLVMCall->getCalledOperand()));
    // Check getCalledOperandUse().
    EXPECT_EQ(Call->getCalledOperandUse().get(),
              Ctx.getValue(LLVMCall->getCalledOperandUse()));
    // Check getCalledFunction().
    if (LLVMCall->getCalledFunction() == nullptr)
      EXPECT_EQ(Call->getCalledFunction(), nullptr);
    else {
      auto *LLVMCF = cast<llvm::Function>(LLVMCall->getCalledFunction());
      (void)LLVMCF;
      EXPECT_EQ(Call->getCalledFunction(),
                cast<sandboxir::Function>(
                    Ctx.getValue(LLVMCall->getCalledFunction())));
    }
    // Check isIndirectCall().
    EXPECT_EQ(Call->isIndirectCall(), LLVMCall->isIndirectCall());
    // Check getCaller().
    EXPECT_EQ(Call->getCaller(), Ctx.getValue(LLVMCall->getCaller()));
    // Check isMustTailCall().
    EXPECT_EQ(Call->isMustTailCall(), LLVMCall->isMustTailCall());
    // Check isTailCall().
    EXPECT_EQ(Call->isTailCall(), LLVMCall->isTailCall());
    // Check getIntrinsicID().
    EXPECT_EQ(Call->getIntrinsicID(), LLVMCall->getIntrinsicID());
    // Check getCallingConv().
    EXPECT_EQ(Call->getCallingConv(), LLVMCall->getCallingConv());
    // Check isInlineAsm().
    EXPECT_EQ(Call->isInlineAsm(), LLVMCall->isInlineAsm());
  }

  auto *Arg0 = F.getArg(0);
  auto *Arg1 = F.getArg(1);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Call0 = cast<sandboxir::CallBase>(&*It++);
  [[maybe_unused]] auto *Call1 = cast<sandboxir::CallBase>(&*It++);
  auto *Call2 = cast<sandboxir::CallBase>(&*It++);
  // Check setArgOperand
  Call0->setArgOperand(0, Arg1);
  EXPECT_EQ(Call0->getArgOperand(0), Arg1);
  Call0->setArgOperand(0, Arg0);
  EXPECT_EQ(Call0->getArgOperand(0), Arg0);

  auto *Bar3F = Ctx.createFunction(M->getFunction("bar3"));

  // Check setCalledOperand
  auto *SvOp = Call0->getCalledOperand();
  Call0->setCalledOperand(Bar3F);
  EXPECT_EQ(Call0->getCalledOperand(), Bar3F);
  Call0->setCalledOperand(SvOp);
  // Check setCalledFunction
  Call2->setCalledFunction(Bar3F);
  EXPECT_EQ(Call2->getCalledFunction(), Bar3F);
}

TEST_F(SandboxIRTest, CallInst) {
  parseIR(C, R"IR(
define i8 @foo(i8 %arg) {
  %call = call i8 @foo(i8 %arg)
  ret i8 %call
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg0 = F.getArg(ArgIdx++);
  auto *BB = &*F.begin();
  auto It = BB->begin();
  auto *Call = cast<sandboxir::CallInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);
  EXPECT_EQ(Call->getNumOperands(), 2u);
  EXPECT_EQ(Ret->getOpcode(), sandboxir::Instruction::Opcode::Ret);
  FunctionType *FTy = F.getFunctionType();
  SmallVector<sandboxir::Value *, 1> Args;
  Args.push_back(Arg0);
  {
    // Check create() WhereIt.
    auto *Call = cast<sandboxir::CallInst>(sandboxir::CallInst::create(
        FTy, &F, Args, /*WhereIt=*/Ret->getIterator(), BB, Ctx));
    EXPECT_EQ(Call->getNextNode(), Ret);
    EXPECT_EQ(Call->getCalledFunction(), &F);
    EXPECT_EQ(range_size(Call->args()), 1u);
    EXPECT_EQ(Call->getArgOperand(0), Arg0);
  }
  {
    // Check create() InsertBefore.
    auto *Call = cast<sandboxir::CallInst>(
        sandboxir::CallInst::create(FTy, &F, Args, /*InsertBefore=*/Ret, Ctx));
    EXPECT_EQ(Call->getNextNode(), Ret);
    EXPECT_EQ(Call->getCalledFunction(), &F);
    EXPECT_EQ(range_size(Call->args()), 1u);
    EXPECT_EQ(Call->getArgOperand(0), Arg0);
  }
  {
    // Check create() InsertAtEnd.
    auto *Call = cast<sandboxir::CallInst>(
        sandboxir::CallInst::create(FTy, &F, Args, /*InsertAtEnd=*/BB, Ctx));
    EXPECT_EQ(Call->getPrevNode(), Ret);
    EXPECT_EQ(Call->getCalledFunction(), &F);
    EXPECT_EQ(range_size(Call->args()), 1u);
    EXPECT_EQ(Call->getArgOperand(0), Arg0);
  }
}

TEST_F(SandboxIRTest, InvokeInst) {
  parseIR(C, R"IR(
define void @foo(i8 %arg) {
 bb0:
   invoke i8 @foo(i8 %arg) to label %normal_bb
                       unwind label %exception_bb
 normal_bb:
   ret void
 exception_bb:
   %lpad = landingpad { ptr, i32}
           cleanup
   ret void
 other_bb:
   ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *Arg = F.getArg(0);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *NormalBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "normal_bb")));
  auto *ExceptionBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "exception_bb")));
  auto *LandingPad = &*ExceptionBB->begin();
  auto *OtherBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "other_bb")));
  auto It = BB0->begin();
  // Check classof(Instruction *).
  auto *Invoke = cast<sandboxir::InvokeInst>(&*It++);

  // Check getNormalDest().
  EXPECT_EQ(Invoke->getNormalDest(), NormalBB);
  // Check getUnwindDest().
  EXPECT_EQ(Invoke->getUnwindDest(), ExceptionBB);
  // Check getSuccessor().
  EXPECT_EQ(Invoke->getSuccessor(0), NormalBB);
  EXPECT_EQ(Invoke->getSuccessor(1), ExceptionBB);
  // Check setNormalDest().
  Invoke->setNormalDest(OtherBB);
  EXPECT_EQ(Invoke->getNormalDest(), OtherBB);
  EXPECT_EQ(Invoke->getUnwindDest(), ExceptionBB);
  // Check setUnwindDest().
  Invoke->setUnwindDest(OtherBB);
  EXPECT_EQ(Invoke->getNormalDest(), OtherBB);
  EXPECT_EQ(Invoke->getUnwindDest(), OtherBB);
  // Check setSuccessor().
  Invoke->setSuccessor(0, NormalBB);
  EXPECT_EQ(Invoke->getNormalDest(), NormalBB);
  Invoke->setSuccessor(1, ExceptionBB);
  EXPECT_EQ(Invoke->getUnwindDest(), ExceptionBB);
  // Check getLandingPadInst().
  EXPECT_EQ(Invoke->getLandingPadInst(), LandingPad);

  {
    // Check create() WhereIt, WhereBB.
    SmallVector<sandboxir::Value *> Args({Arg});
    auto *InsertBefore = &*BB0->begin();
    auto *NewInvoke = cast<sandboxir::InvokeInst>(sandboxir::InvokeInst::create(
        F.getFunctionType(), &F, NormalBB, ExceptionBB, Args,
        /*WhereIt=*/InsertBefore->getIterator(), /*WhereBB=*/BB0, Ctx));
    EXPECT_EQ(NewInvoke->getNormalDest(), NormalBB);
    EXPECT_EQ(NewInvoke->getUnwindDest(), ExceptionBB);
    EXPECT_EQ(NewInvoke->getNextNode(), InsertBefore);
  }
  {
    // Check create() InsertBefore.
    SmallVector<sandboxir::Value *> Args({Arg});
    auto *InsertBefore = &*BB0->begin();
    auto *NewInvoke = cast<sandboxir::InvokeInst>(
        sandboxir::InvokeInst::create(F.getFunctionType(), &F, NormalBB,
                                      ExceptionBB, Args, InsertBefore, Ctx));
    EXPECT_EQ(NewInvoke->getNormalDest(), NormalBB);
    EXPECT_EQ(NewInvoke->getUnwindDest(), ExceptionBB);
    EXPECT_EQ(NewInvoke->getNextNode(), InsertBefore);
  }
  {
    // Check create() InsertAtEnd.
    SmallVector<sandboxir::Value *> Args({Arg});
    auto *NewInvoke = cast<sandboxir::InvokeInst>(sandboxir::InvokeInst::create(
        F.getFunctionType(), &F, NormalBB, ExceptionBB, Args,
        /*InsertAtEnd=*/BB0, Ctx));
    EXPECT_EQ(NewInvoke->getNormalDest(), NormalBB);
    EXPECT_EQ(NewInvoke->getUnwindDest(), ExceptionBB);
    EXPECT_EQ(NewInvoke->getParent(), BB0);
    EXPECT_EQ(NewInvoke->getNextNode(), nullptr);
  }
}
