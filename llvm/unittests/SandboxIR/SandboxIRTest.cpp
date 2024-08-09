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
#include "llvm/IR/DataLayout.h"
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
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(Ld));
  auto *VLd = cast<sandboxir::LoadInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);
  bool OrigVolatileValue;

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
  EXPECT_FALSE(NewLd->isVolatile());
  OrigVolatileValue = NewLd->isVolatile();
  NewLd->setVolatile(true);
  EXPECT_TRUE(NewLd->isVolatile());
  NewLd->setVolatile(OrigVolatileValue);
  EXPECT_FALSE(NewLd->isVolatile());
  EXPECT_EQ(NewLd->getType(), Ld->getType());
  EXPECT_EQ(NewLd->getPointerOperand(), Arg1);
  EXPECT_EQ(NewLd->getAlign(), 8);
  EXPECT_EQ(NewLd->getName(), "NewLd");
  // Check create(InsertBefore, IsVolatile=true)
  sandboxir::LoadInst *NewVLd =
      sandboxir::LoadInst::create(VLd->getType(), Arg1, Align(8),
                                  /*InsertBefore=*/Ret,
                                  /*IsVolatile=*/true, Ctx, "NewVLd");

  EXPECT_TRUE(NewVLd->isVolatile());
  OrigVolatileValue = NewVLd->isVolatile();
  NewVLd->setVolatile(false);
  EXPECT_FALSE(NewVLd->isVolatile());
  NewVLd->setVolatile(OrigVolatileValue);
  EXPECT_TRUE(NewVLd->isVolatile());
  EXPECT_EQ(NewVLd->getName(), "NewVLd");
  // Check create(InsertAtEnd)
  sandboxir::LoadInst *NewLdEnd =
      sandboxir::LoadInst::create(Ld->getType(), Arg1, Align(8),
                                  /*InsertAtEnd=*/BB, Ctx, "NewLdEnd");
  EXPECT_FALSE(NewLdEnd->isVolatile());
  EXPECT_EQ(NewLdEnd->getName(), "NewLdEnd");
  EXPECT_EQ(NewLdEnd->getType(), Ld->getType());
  EXPECT_EQ(NewLdEnd->getPointerOperand(), Arg1);
  EXPECT_EQ(NewLdEnd->getAlign(), 8);
  EXPECT_EQ(NewLdEnd->getParent(), BB);
  EXPECT_EQ(NewLdEnd->getNextNode(), nullptr);
  // Check create(InsertAtEnd, IsVolatile=true)
  sandboxir::LoadInst *NewVLdEnd =
      sandboxir::LoadInst::create(VLd->getType(), Arg1, Align(8),
                                  /*InsertAtEnd=*/BB,
                                  /*IsVolatile=*/true, Ctx, "NewVLdEnd");
  EXPECT_TRUE(NewVLdEnd->isVolatile());
  EXPECT_EQ(NewVLdEnd->getName(), "NewVLdEnd");
  EXPECT_EQ(NewVLdEnd->getType(), VLd->getType());
  EXPECT_EQ(NewVLdEnd->getPointerOperand(), Arg1);
  EXPECT_EQ(NewVLdEnd->getAlign(), 8);
  EXPECT_EQ(NewVLdEnd->getParent(), BB);
  EXPECT_EQ(NewVLdEnd->getNextNode(), nullptr);
}

TEST_F(SandboxIRTest, StoreInst) {
  parseIR(C, R"IR(
define void @foo(i8 %val, ptr %ptr) {
  store i8 %val, ptr %ptr, align 64
  store volatile i8 %val, ptr %ptr, align 64
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
  auto *VSt = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);
  bool OrigVolatileValue;

  // Check that the StoreInst has been created correctly.
  EXPECT_FALSE(St->isVolatile());
  EXPECT_TRUE(VSt->isVolatile());
  // Check getPointerOperand()
  EXPECT_EQ(St->getValueOperand(), Val);
  EXPECT_EQ(St->getPointerOperand(), Ptr);
  // Check getAlign()
  EXPECT_EQ(St->getAlign(), 64);
  // Check create(InsertBefore)
  sandboxir::StoreInst *NewSt =
      sandboxir::StoreInst::create(Val, Ptr, Align(8),
                                   /*InsertBefore=*/Ret, Ctx);
  EXPECT_FALSE(NewSt->isVolatile());
  OrigVolatileValue = NewSt->isVolatile();
  NewSt->setVolatile(true);
  EXPECT_TRUE(NewSt->isVolatile());
  NewSt->setVolatile(OrigVolatileValue);
  EXPECT_FALSE(NewSt->isVolatile());
  EXPECT_EQ(NewSt->getType(), St->getType());
  EXPECT_EQ(NewSt->getValueOperand(), Val);
  EXPECT_EQ(NewSt->getPointerOperand(), Ptr);
  EXPECT_EQ(NewSt->getAlign(), 8);
  EXPECT_EQ(NewSt->getNextNode(), Ret);
  // Check create(InsertBefore, IsVolatile=true)
  sandboxir::StoreInst *NewVSt =
      sandboxir::StoreInst::create(Val, Ptr, Align(8),
                                   /*InsertBefore=*/Ret,
                                   /*IsVolatile=*/true, Ctx);
  EXPECT_TRUE(NewVSt->isVolatile());
  OrigVolatileValue = NewVSt->isVolatile();
  NewVSt->setVolatile(false);
  EXPECT_FALSE(NewVSt->isVolatile());
  NewVSt->setVolatile(OrigVolatileValue);
  EXPECT_TRUE(NewVSt->isVolatile());
  EXPECT_EQ(NewVSt->getType(), VSt->getType());
  EXPECT_EQ(NewVSt->getValueOperand(), Val);
  EXPECT_EQ(NewVSt->getPointerOperand(), Ptr);
  EXPECT_EQ(NewVSt->getAlign(), 8);
  EXPECT_EQ(NewVSt->getNextNode(), Ret);
  // Check create(InsertAtEnd)
  sandboxir::StoreInst *NewStEnd =
      sandboxir::StoreInst::create(Val, Ptr, Align(8),
                                   /*InsertAtEnd=*/BB, Ctx);
  EXPECT_FALSE(NewStEnd->isVolatile());
  EXPECT_EQ(NewStEnd->getType(), St->getType());
  EXPECT_EQ(NewStEnd->getValueOperand(), Val);
  EXPECT_EQ(NewStEnd->getPointerOperand(), Ptr);
  EXPECT_EQ(NewStEnd->getAlign(), 8);
  EXPECT_EQ(NewStEnd->getParent(), BB);
  EXPECT_EQ(NewStEnd->getNextNode(), nullptr);
  // Check create(InsertAtEnd, IsVolatile=true)
  sandboxir::StoreInst *NewVStEnd =
      sandboxir::StoreInst::create(Val, Ptr, Align(8),
                                   /*InsertAtEnd=*/BB,
                                   /*IsVolatile=*/true, Ctx);
  EXPECT_TRUE(NewVStEnd->isVolatile());
  EXPECT_EQ(NewVStEnd->getType(), VSt->getType());
  EXPECT_EQ(NewVStEnd->getValueOperand(), Val);
  EXPECT_EQ(NewVStEnd->getPointerOperand(), Ptr);
  EXPECT_EQ(NewVStEnd->getAlign(), 8);
  EXPECT_EQ(NewVStEnd->getParent(), BB);
  EXPECT_EQ(NewVStEnd->getNextNode(), nullptr);
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

TEST_F(SandboxIRTest, CallBrInst) {
  parseIR(C, R"IR(
define void @foo(i8 %arg) {
 bb0:
   callbr void asm "", ""()
               to label %bb1 [label %bb2]
 bb1:
   ret void
 bb2:
   ret void
 other_bb:
   ret void
 bb3:
   callbr void @foo(i8 %arg)
               to label %bb1 [label %bb2]
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  auto *LLVMBB0 = getBasicBlockByName(LLVMF, "bb0");
  auto *LLVMCallBr = cast<llvm::CallBrInst>(&*LLVMBB0->begin());
  sandboxir::Context Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto *Arg = F.getArg(0);
  auto *BB0 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb0")));
  auto *BB1 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb1")));
  auto *BB2 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb2")));
  auto *BB3 = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "bb3")));
  auto *OtherBB = cast<sandboxir::BasicBlock>(
      Ctx.getValue(getBasicBlockByName(LLVMF, "other_bb")));
  auto It = BB0->begin();
  // Check classof(Instruction *).
  auto *CallBr0 = cast<sandboxir::CallBrInst>(&*It++);

  It = BB3->begin();
  auto *CallBr1 = cast<sandboxir::CallBrInst>(&*It++);
  for (sandboxir::CallBrInst *CallBr : {CallBr0, CallBr1}) {
    // Check getNumIndirectDests().
    EXPECT_EQ(CallBr->getNumIndirectDests(), 1u);
    // Check getIndirectDestLabel().
    EXPECT_EQ(CallBr->getIndirectDestLabel(0),
              Ctx.getValue(LLVMCallBr->getIndirectDestLabel(0)));
    // Check getIndirectDestLabelUse().
    EXPECT_EQ(CallBr->getIndirectDestLabelUse(0),
              Ctx.getValue(LLVMCallBr->getIndirectDestLabelUse(0)));
    // Check getDefaultDest().
    EXPECT_EQ(CallBr->getDefaultDest(),
              Ctx.getValue(LLVMCallBr->getDefaultDest()));
    // Check getIndirectDest().
    EXPECT_EQ(CallBr->getIndirectDest(0),
              Ctx.getValue(LLVMCallBr->getIndirectDest(0)));
    // Check getIndirectDests().
    auto Dests = CallBr->getIndirectDests();
    EXPECT_EQ(Dests.size(), LLVMCallBr->getIndirectDests().size());
    EXPECT_EQ(Dests[0], Ctx.getValue(LLVMCallBr->getIndirectDests()[0]));
    // Check getNumSuccessors().
    EXPECT_EQ(CallBr->getNumSuccessors(), LLVMCallBr->getNumSuccessors());
    // Check getSuccessor().
    for (unsigned SuccIdx = 0, E = CallBr->getNumSuccessors(); SuccIdx != E;
         ++SuccIdx)
      EXPECT_EQ(CallBr->getSuccessor(SuccIdx),
                Ctx.getValue(LLVMCallBr->getSuccessor(SuccIdx)));
    // Check setDefaultDest().
    auto *SvDefaultDest = CallBr->getDefaultDest();
    CallBr->setDefaultDest(OtherBB);
    EXPECT_EQ(CallBr->getDefaultDest(), OtherBB);
    CallBr->setDefaultDest(SvDefaultDest);
    // Check setIndirectDest().
    auto *SvIndirectDest = CallBr->getIndirectDest(0);
    CallBr->setIndirectDest(0, OtherBB);
    EXPECT_EQ(CallBr->getIndirectDest(0), OtherBB);
    CallBr->setIndirectDest(0, SvIndirectDest);
  }

  {
    // Check create() WhereIt, WhereBB.
    SmallVector<sandboxir::Value *> Args({Arg});
    auto *NewCallBr = cast<sandboxir::CallBrInst>(sandboxir::CallBrInst::create(
        F.getFunctionType(), &F, BB1, {BB2}, Args, /*WhereIt=*/BB0->end(),
        /*WhereBB=*/BB0, Ctx));
    EXPECT_EQ(NewCallBr->getDefaultDest(), BB1);
    EXPECT_EQ(NewCallBr->getIndirectDests().size(), 1u);
    EXPECT_EQ(NewCallBr->getIndirectDests()[0], BB2);
    EXPECT_EQ(NewCallBr->getNextNode(), nullptr);
    EXPECT_EQ(NewCallBr->getParent(), BB0);
  }
  {
    // Check create() InsertBefore
    SmallVector<sandboxir::Value *> Args({Arg});
    auto *InsertBefore = &*BB0->rbegin();
    auto *NewCallBr = cast<sandboxir::CallBrInst>(sandboxir::CallBrInst::create(
        F.getFunctionType(), &F, BB1, {BB2}, Args, InsertBefore, Ctx));
    EXPECT_EQ(NewCallBr->getDefaultDest(), BB1);
    EXPECT_EQ(NewCallBr->getIndirectDests().size(), 1u);
    EXPECT_EQ(NewCallBr->getIndirectDests()[0], BB2);
    EXPECT_EQ(NewCallBr->getNextNode(), InsertBefore);
  }
  {
    // Check create() InsertAtEnd.
    SmallVector<sandboxir::Value *> Args({Arg});
    auto *NewCallBr = cast<sandboxir::CallBrInst>(
        sandboxir::CallBrInst::create(F.getFunctionType(), &F, BB1, {BB2}, Args,
                                      /*InsertAtEnd=*/BB0, Ctx));
    EXPECT_EQ(NewCallBr->getDefaultDest(), BB1);
    EXPECT_EQ(NewCallBr->getIndirectDests().size(), 1u);
    EXPECT_EQ(NewCallBr->getIndirectDests()[0], BB2);
    EXPECT_EQ(NewCallBr->getNextNode(), nullptr);
    EXPECT_EQ(NewCallBr->getParent(), BB0);
  }
}

TEST_F(SandboxIRTest, GetElementPtrInstruction) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, <2 x ptr> %ptrs) {
  %gep0 = getelementptr i8, ptr %ptr, i32 0
  %gep1 = getelementptr nusw i8, ptr %ptr, i32 0
  %gep2 = getelementptr nuw i8, ptr %ptr, i32 0
  %gep3 = getelementptr inbounds {i32, {i32, i8}}, ptr %ptr, i32 1, i32 0
  %gep4 = getelementptr inbounds {i8, i8, {i32, i16}}, <2 x ptr> %ptrs, i32 2, <2 x i32> <i32 0, i32 0>
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  BasicBlock *LLVMBB = &*LLVMF.begin();
  auto LLVMIt = LLVMBB->begin();
  SmallVector<llvm::GetElementPtrInst *, 4> LLVMGEPs;
  while (isa<llvm::GetElementPtrInst>(&*LLVMIt))
    LLVMGEPs.push_back(cast<llvm::GetElementPtrInst>(&*LLVMIt++));
  auto *LLVMRet = cast<llvm::ReturnInst>(&*LLVMIt++);
  sandboxir::Context Ctx(C);
  [[maybe_unused]] auto &F = *Ctx.createFunction(&LLVMF);

  for (llvm::GetElementPtrInst *LLVMGEP : LLVMGEPs) {
    // Check classof().
    auto *GEP = cast<sandboxir::GetElementPtrInst>(Ctx.getValue(LLVMGEP));
    // Check getSourceElementType().
    EXPECT_EQ(GEP->getSourceElementType(), LLVMGEP->getSourceElementType());
    // Check getResultElementType().
    EXPECT_EQ(GEP->getResultElementType(), LLVMGEP->getResultElementType());
    // Check getAddressSpace().
    EXPECT_EQ(GEP->getAddressSpace(), LLVMGEP->getAddressSpace());
    // Check indices().
    EXPECT_EQ(range_size(GEP->indices()), range_size(LLVMGEP->indices()));
    auto IdxIt = GEP->idx_begin();
    for (llvm::Value *LLVMIdxV : LLVMGEP->indices()) {
      sandboxir::Value *IdxV = *IdxIt++;
      EXPECT_EQ(IdxV, Ctx.getValue(LLVMIdxV));
    }
    // Check getPointerOperand().
    EXPECT_EQ(GEP->getPointerOperand(),
              Ctx.getValue(LLVMGEP->getPointerOperand()));
    // Check getPointerOperandIndex().
    EXPECT_EQ(GEP->getPointerOperandIndex(), LLVMGEP->getPointerOperandIndex());
    // Check getPointerOperandType().
    EXPECT_EQ(GEP->getPointerOperandType(), LLVMGEP->getPointerOperandType());
    // Check getPointerAddressSpace().
    EXPECT_EQ(GEP->getPointerAddressSpace(), LLVMGEP->getPointerAddressSpace());
    // Check getNumIndices().
    EXPECT_EQ(GEP->getNumIndices(), LLVMGEP->getNumIndices());
    // Check hasIndices().
    EXPECT_EQ(GEP->hasIndices(), LLVMGEP->hasIndices());
    // Check hasAllConstantIndices().
    EXPECT_EQ(GEP->hasAllConstantIndices(), LLVMGEP->hasAllConstantIndices());
    // Check getNoWrapFlags().
    EXPECT_EQ(GEP->getNoWrapFlags(), LLVMGEP->getNoWrapFlags());
    // Check isInBounds().
    EXPECT_EQ(GEP->isInBounds(), LLVMGEP->isInBounds());
    // Check hasNoUnsignedWrap().
    EXPECT_EQ(GEP->hasNoUnsignedWrap(), LLVMGEP->hasNoUnsignedWrap());
    // Check accumulateConstantOffset().
    DataLayout DL(M.get());
    APInt Offset1 =
        APInt::getZero(DL.getIndexSizeInBits(GEP->getPointerAddressSpace()));
    APInt Offset2 =
        APInt::getZero(DL.getIndexSizeInBits(GEP->getPointerAddressSpace()));
    EXPECT_EQ(GEP->accumulateConstantOffset(DL, Offset1),
              LLVMGEP->accumulateConstantOffset(DL, Offset2));
    EXPECT_EQ(Offset1, Offset2);
  }

  auto *BB = &*F.begin();
  auto *GEP0 = cast<sandboxir::GetElementPtrInst>(&*BB->begin());
  auto *Ret = cast<sandboxir::ReturnInst>(Ctx.getValue(LLVMRet));
  SmallVector<sandboxir::Value *> Indices(GEP0->indices());

  // Check create() WhereIt, WhereBB.
  auto *NewGEP0 =
      cast<sandboxir::GetElementPtrInst>(sandboxir::GetElementPtrInst::create(
          GEP0->getType(), GEP0->getPointerOperand(), Indices,
          /*WhereIt=*/Ret->getIterator(), /*WhereBB=*/Ret->getParent(), Ctx,
          "NewGEP0"));
  EXPECT_EQ(NewGEP0->getName(), "NewGEP0");
  EXPECT_EQ(NewGEP0->getType(), GEP0->getType());
  EXPECT_EQ(NewGEP0->getPointerOperand(), GEP0->getPointerOperand());
  EXPECT_EQ(range_size(NewGEP0->indices()), range_size(GEP0->indices()));
  for (auto NewIt = NewGEP0->idx_begin(), NewItE = NewGEP0->idx_end(),
            OldIt = GEP0->idx_begin();
       NewIt != NewItE; ++NewIt) {
    sandboxir::Value *NewIdxV = *NewIt;
    sandboxir::Value *OldIdxV = *OldIt;
    EXPECT_EQ(NewIdxV, OldIdxV);
  }
  EXPECT_EQ(NewGEP0->getNextNode(), Ret);

  // Check create() InsertBefore.
  auto *NewGEP1 =
      cast<sandboxir::GetElementPtrInst>(sandboxir::GetElementPtrInst::create(
          GEP0->getType(), GEP0->getPointerOperand(), Indices,
          /*InsertBefore=*/Ret, Ctx, "NewGEP1"));
  EXPECT_EQ(NewGEP1->getName(), "NewGEP1");
  EXPECT_EQ(NewGEP1->getType(), GEP0->getType());
  EXPECT_EQ(NewGEP1->getPointerOperand(), GEP0->getPointerOperand());
  EXPECT_EQ(range_size(NewGEP1->indices()), range_size(GEP0->indices()));
  for (auto NewIt = NewGEP0->idx_begin(), NewItE = NewGEP0->idx_end(),
            OldIt = GEP0->idx_begin();
       NewIt != NewItE; ++NewIt) {
    sandboxir::Value *NewIdxV = *NewIt;
    sandboxir::Value *OldIdxV = *OldIt;
    EXPECT_EQ(NewIdxV, OldIdxV);
  }
  EXPECT_EQ(NewGEP1->getNextNode(), Ret);

  // Check create() InsertAtEnd.
  auto *NewGEP2 =
      cast<sandboxir::GetElementPtrInst>(sandboxir::GetElementPtrInst::create(
          GEP0->getType(), GEP0->getPointerOperand(), Indices,
          /*InsertAtEnd=*/BB, Ctx, "NewGEP2"));
  EXPECT_EQ(NewGEP2->getName(), "NewGEP2");
  EXPECT_EQ(NewGEP2->getType(), GEP0->getType());
  EXPECT_EQ(NewGEP2->getPointerOperand(), GEP0->getPointerOperand());
  EXPECT_EQ(range_size(NewGEP2->indices()), range_size(GEP0->indices()));
  for (auto NewIt = NewGEP0->idx_begin(), NewItE = NewGEP0->idx_end(),
            OldIt = GEP0->idx_begin();
       NewIt != NewItE; ++NewIt) {
    sandboxir::Value *NewIdxV = *NewIt;
    sandboxir::Value *OldIdxV = *OldIt;
    EXPECT_EQ(NewIdxV, OldIdxV);
  }
  EXPECT_EQ(NewGEP2->getPrevNode(), Ret);
  EXPECT_EQ(NewGEP2->getNextNode(), nullptr);
}

TEST_F(SandboxIRTest, AllocaInst) {
  parseIR(C, R"IR(
define void @foo() {
  %allocaScalar = alloca i32, align 1024
  %allocaArray = alloca i32, i32 42
  ret void
}
)IR");
  DataLayout DL(M.get());
  llvm::Function &LLVMF = *M->getFunction("foo");
  llvm::BasicBlock *LLVMBB = &*LLVMF.begin();
  auto LLVMIt = LLVMBB->begin();
  auto *LLVMAllocaScalar = cast<llvm::AllocaInst>(&*LLVMIt++);
  auto *LLVMAllocaArray = cast<llvm::AllocaInst>(&*LLVMIt++);

  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(&LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *AllocaScalar = cast<sandboxir::AllocaInst>(&*It++);
  auto *AllocaArray = cast<sandboxir::AllocaInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  // Check isArrayAllocation().
  EXPECT_EQ(AllocaScalar->isArrayAllocation(),
            LLVMAllocaScalar->isArrayAllocation());
  EXPECT_EQ(AllocaArray->isArrayAllocation(),
            LLVMAllocaArray->isArrayAllocation());
  // Check getArraySize().
  EXPECT_EQ(AllocaScalar->getArraySize(),
            Ctx.getValue(LLVMAllocaScalar->getArraySize()));
  EXPECT_EQ(AllocaArray->getArraySize(),
            Ctx.getValue(LLVMAllocaArray->getArraySize()));
  // Check getType().
  EXPECT_EQ(AllocaScalar->getType(), LLVMAllocaScalar->getType());
  EXPECT_EQ(AllocaArray->getType(), LLVMAllocaArray->getType());
  // Check getAddressSpace().
  EXPECT_EQ(AllocaScalar->getAddressSpace(),
            LLVMAllocaScalar->getAddressSpace());
  EXPECT_EQ(AllocaArray->getAddressSpace(), LLVMAllocaArray->getAddressSpace());
  // Check getAllocationSize().
  EXPECT_EQ(AllocaScalar->getAllocationSize(DL),
            LLVMAllocaScalar->getAllocationSize(DL));
  EXPECT_EQ(AllocaArray->getAllocationSize(DL),
            LLVMAllocaArray->getAllocationSize(DL));
  // Check getAllocationSizeInBits().
  EXPECT_EQ(AllocaScalar->getAllocationSizeInBits(DL),
            LLVMAllocaScalar->getAllocationSizeInBits(DL));
  EXPECT_EQ(AllocaArray->getAllocationSizeInBits(DL),
            LLVMAllocaArray->getAllocationSizeInBits(DL));
  // Check getAllocatedType().
  EXPECT_EQ(AllocaScalar->getAllocatedType(),
            LLVMAllocaScalar->getAllocatedType());
  EXPECT_EQ(AllocaArray->getAllocatedType(),
            LLVMAllocaArray->getAllocatedType());
  // Check setAllocatedType().
  auto *OrigType = AllocaScalar->getAllocatedType();
  auto *NewType = PointerType::get(C, 0);
  EXPECT_NE(NewType, OrigType);
  AllocaScalar->setAllocatedType(NewType);
  EXPECT_EQ(AllocaScalar->getAllocatedType(), NewType);
  AllocaScalar->setAllocatedType(OrigType);
  EXPECT_EQ(AllocaScalar->getAllocatedType(), OrigType);
  // Check getAlign().
  EXPECT_EQ(AllocaScalar->getAlign(), LLVMAllocaScalar->getAlign());
  EXPECT_EQ(AllocaArray->getAlign(), LLVMAllocaArray->getAlign());
  // Check setAlignment().
  Align OrigAlign = AllocaScalar->getAlign();
  Align NewAlign(16);
  EXPECT_NE(NewAlign, OrigAlign);
  AllocaScalar->setAlignment(NewAlign);
  EXPECT_EQ(AllocaScalar->getAlign(), NewAlign);
  AllocaScalar->setAlignment(OrigAlign);
  EXPECT_EQ(AllocaScalar->getAlign(), OrigAlign);
  // Check isStaticAlloca().
  EXPECT_EQ(AllocaScalar->isStaticAlloca(), LLVMAllocaScalar->isStaticAlloca());
  EXPECT_EQ(AllocaArray->isStaticAlloca(), LLVMAllocaArray->isStaticAlloca());
  // Check isUsedWithInAlloca(), setUsedWithInAlloca().
  EXPECT_EQ(AllocaScalar->isUsedWithInAlloca(),
            LLVMAllocaScalar->isUsedWithInAlloca());
  bool OrigUsedWithInAlloca = AllocaScalar->isUsedWithInAlloca();
  bool NewUsedWithInAlloca = true;
  EXPECT_NE(NewUsedWithInAlloca, OrigUsedWithInAlloca);
  AllocaScalar->setUsedWithInAlloca(NewUsedWithInAlloca);
  EXPECT_EQ(AllocaScalar->isUsedWithInAlloca(), NewUsedWithInAlloca);
  AllocaScalar->setUsedWithInAlloca(OrigUsedWithInAlloca);
  EXPECT_EQ(AllocaScalar->isUsedWithInAlloca(), OrigUsedWithInAlloca);

  auto *Ty = Type::getInt32Ty(C);
  unsigned AddrSpace = 42;
  auto *PtrTy = PointerType::get(C, AddrSpace);
  auto *ArraySize = sandboxir::Constant::createInt(Ty, 43, Ctx);
  {
    // Check create() WhereIt, WhereBB.
    auto *NewI = cast<sandboxir::AllocaInst>(sandboxir::AllocaInst::create(
        Ty, AddrSpace, /*WhereIt=*/Ret->getIterator(),
        /*WhereBB=*/Ret->getParent(), Ctx, ArraySize, "NewAlloca1"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), sandboxir::Instruction::Opcode::Alloca);
    // Check getType().
    EXPECT_EQ(NewI->getType(), PtrTy);
    // Check getArraySize().
    EXPECT_EQ(NewI->getArraySize(), ArraySize);
    // Check getAddrSpace().
    EXPECT_EQ(NewI->getAddressSpace(), AddrSpace);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), Ret);
  }
  {
    // Check create() InsertBefore.
    auto *NewI = cast<sandboxir::AllocaInst>(sandboxir::AllocaInst::create(
        Ty, AddrSpace, /*InsertBefore=*/Ret, Ctx, ArraySize, "NewAlloca2"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), sandboxir::Instruction::Opcode::Alloca);
    // Check getType().
    EXPECT_EQ(NewI->getType(), PtrTy);
    // Check getArraySize().
    EXPECT_EQ(NewI->getArraySize(), ArraySize);
    // Check getAddrSpace().
    EXPECT_EQ(NewI->getAddressSpace(), AddrSpace);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), Ret);
  }
  {
    // Check create() InsertAtEnd.
    auto *NewI = cast<sandboxir::AllocaInst>(sandboxir::AllocaInst::create(
        Ty, AddrSpace, /*InsertAtEnd=*/BB, Ctx, ArraySize, "NewAlloca3"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), sandboxir::Instruction::Opcode::Alloca);
    // Check getType().
    EXPECT_EQ(NewI->getType(), PtrTy);
    // Check getArraySize().
    EXPECT_EQ(NewI->getArraySize(), ArraySize);
    // Check getAddrSpace().
    EXPECT_EQ(NewI->getAddressSpace(), AddrSpace);
    // Check instr position.
    EXPECT_EQ(NewI->getParent(), BB);
    EXPECT_EQ(NewI->getNextNode(), nullptr);
  }
}

TEST_F(SandboxIRTest, CastInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg, float %farg, double %darg, ptr %ptr) {
  %zext = zext i32 %arg to i64
  %sext = sext i32 %arg to i64
  %fptoui = fptoui float %farg to i32
  %fptosi = fptosi float %farg to i32
  %fpext = fpext float %farg to double
  %ptrtoint = ptrtoint ptr %ptr to i32
  %inttoptr = inttoptr i32 %arg to ptr
  %sitofp = sitofp i32 %arg to float
  %uitofp = uitofp i32 %arg to float
  %trunc = trunc i32 %arg to i16
  %fptrunc = fptrunc double %darg to float
  %bitcast = bitcast i32 %arg to float
  %addrspacecast = addrspacecast ptr %ptr to ptr addrspace(1)
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg = F->getArg(ArgIdx++);
  auto *BB = &*F->begin();
  auto It = BB->begin();

  Type *Ti64 = Type::getInt64Ty(C);
  Type *Ti32 = Type::getInt32Ty(C);
  Type *Ti16 = Type::getInt16Ty(C);
  Type *Tdouble = Type::getDoubleTy(C);
  Type *Tfloat = Type::getFloatTy(C);
  Type *Tptr = Tfloat->getPointerTo();
  Type *Tptr1 = Tfloat->getPointerTo(1);

  // Check classof(), getOpcode(), getSrcTy(), getDstTy()
  auto *ZExt = cast<sandboxir::CastInst>(&*It++);
  auto *ZExtI = cast<sandboxir::ZExtInst>(ZExt);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(ZExtI));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(ZExtI));
  EXPECT_EQ(ZExt->getOpcode(), sandboxir::Instruction::Opcode::ZExt);
  EXPECT_EQ(ZExt->getSrcTy(), Ti32);
  EXPECT_EQ(ZExt->getDestTy(), Ti64);

  auto *SExt = cast<sandboxir::CastInst>(&*It++);
  auto *SExtI = cast<sandboxir::SExtInst>(SExt);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(SExt));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(SExtI));
  EXPECT_EQ(SExt->getOpcode(), sandboxir::Instruction::Opcode::SExt);
  EXPECT_EQ(SExt->getSrcTy(), Ti32);
  EXPECT_EQ(SExt->getDestTy(), Ti64);

  auto *FPToUI = cast<sandboxir::CastInst>(&*It++);
  auto *FPToUII = cast<sandboxir::FPToUIInst>(FPToUI);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPToUI));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPToUII));
  EXPECT_EQ(FPToUI->getOpcode(), sandboxir::Instruction::Opcode::FPToUI);
  EXPECT_EQ(FPToUI->getSrcTy(), Tfloat);
  EXPECT_EQ(FPToUI->getDestTy(), Ti32);

  auto *FPToSI = cast<sandboxir::CastInst>(&*It++);
  auto *FPToSII = cast<sandboxir::FPToSIInst>(FPToSI);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPToSI));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPToSII));
  EXPECT_EQ(FPToSI->getOpcode(), sandboxir::Instruction::Opcode::FPToSI);
  EXPECT_EQ(FPToSI->getSrcTy(), Tfloat);
  EXPECT_EQ(FPToSI->getDestTy(), Ti32);

  auto *FPExt = cast<sandboxir::CastInst>(&*It++);
  auto *FPExtI = cast<sandboxir::FPExtInst>(FPExt);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPExt));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPExtI));
  EXPECT_EQ(FPExt->getOpcode(), sandboxir::Instruction::Opcode::FPExt);
  EXPECT_EQ(FPExt->getSrcTy(), Tfloat);
  EXPECT_EQ(FPExt->getDestTy(), Tdouble);

  auto *PtrToInt = cast<sandboxir::CastInst>(&*It++);
  auto *PtrToIntI = cast<sandboxir::PtrToIntInst>(PtrToInt);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(PtrToInt));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(PtrToIntI));
  EXPECT_EQ(PtrToInt->getOpcode(), sandboxir::Instruction::Opcode::PtrToInt);
  EXPECT_EQ(PtrToInt->getSrcTy(), Tptr);
  EXPECT_EQ(PtrToInt->getDestTy(), Ti32);

  auto *IntToPtr = cast<sandboxir::CastInst>(&*It++);
  auto *IntToPtrI = cast<sandboxir::IntToPtrInst>(IntToPtr);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(IntToPtr));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(IntToPtrI));
  EXPECT_EQ(IntToPtr->getOpcode(), sandboxir::Instruction::Opcode::IntToPtr);
  EXPECT_EQ(IntToPtr->getSrcTy(), Ti32);
  EXPECT_EQ(IntToPtr->getDestTy(), Tptr);

  auto *SIToFP = cast<sandboxir::CastInst>(&*It++);
  auto *SIToFPI = cast<sandboxir::SIToFPInst>(SIToFP);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(SIToFP));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(SIToFPI));
  EXPECT_EQ(SIToFP->getOpcode(), sandboxir::Instruction::Opcode::SIToFP);
  EXPECT_EQ(SIToFP->getSrcTy(), Ti32);
  EXPECT_EQ(SIToFP->getDestTy(), Tfloat);

  auto *UIToFP = cast<sandboxir::CastInst>(&*It++);
  auto *UIToFPI = cast<sandboxir::UIToFPInst>(UIToFP);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(UIToFP));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(UIToFPI));
  EXPECT_EQ(UIToFP->getOpcode(), sandboxir::Instruction::Opcode::UIToFP);
  EXPECT_EQ(UIToFP->getSrcTy(), Ti32);
  EXPECT_EQ(UIToFP->getDestTy(), Tfloat);

  auto *Trunc = cast<sandboxir::CastInst>(&*It++);
  auto *TruncI = cast<sandboxir::TruncInst>(Trunc);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(Trunc));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(TruncI));
  EXPECT_EQ(Trunc->getOpcode(), sandboxir::Instruction::Opcode::Trunc);
  EXPECT_EQ(Trunc->getSrcTy(), Ti32);
  EXPECT_EQ(Trunc->getDestTy(), Ti16);

  auto *FPTrunc = cast<sandboxir::CastInst>(&*It++);
  auto *FPTruncI = cast<sandboxir::FPTruncInst>(FPTrunc);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPTrunc));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(FPTruncI));
  EXPECT_EQ(FPTrunc->getOpcode(), sandboxir::Instruction::Opcode::FPTrunc);
  EXPECT_EQ(FPTrunc->getSrcTy(), Tdouble);
  EXPECT_EQ(FPTrunc->getDestTy(), Tfloat);

  auto *BitCast = cast<sandboxir::CastInst>(&*It++);
  auto *BitCastI = cast<sandboxir::BitCastInst>(BitCast);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(BitCast));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(BitCastI));
  EXPECT_EQ(BitCast->getOpcode(), sandboxir::Instruction::Opcode::BitCast);
  EXPECT_EQ(BitCast->getSrcTy(), Ti32);
  EXPECT_EQ(BitCast->getDestTy(), Tfloat);

  auto *AddrSpaceCast = cast<sandboxir::CastInst>(&*It++);
  auto *AddrSpaceCastI = cast<sandboxir::AddrSpaceCastInst>(AddrSpaceCast);
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(AddrSpaceCast));
  EXPECT_TRUE(isa<sandboxir::UnaryInstruction>(AddrSpaceCastI));
  EXPECT_EQ(AddrSpaceCast->getOpcode(),
            sandboxir::Instruction::Opcode::AddrSpaceCast);
  EXPECT_EQ(AddrSpaceCast->getSrcTy(), Tptr);
  EXPECT_EQ(AddrSpaceCast->getDestTy(), Tptr1);

  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  {
    // Check create() WhereIt, WhereBB
    auto *NewI = cast<sandboxir::CastInst>(sandboxir::CastInst::create(
        Ti64, sandboxir::Instruction::Opcode::SExt, Arg, /*WhereIt=*/BB->end(),
        /*WhereBB=*/BB, Ctx, "SExt"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), sandboxir::Instruction::Opcode::SExt);
    // Check getSrcTy().
    EXPECT_EQ(NewI->getSrcTy(), Arg->getType());
    // Check getDestTy().
    EXPECT_EQ(NewI->getDestTy(), Ti64);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), nullptr);
    EXPECT_EQ(NewI->getPrevNode(), Ret);
  }

  {
    // Check create() InsertBefore.
    auto *NewI = cast<sandboxir::CastInst>(
        sandboxir::CastInst::create(Ti64, sandboxir::Instruction::Opcode::ZExt,
                                    Arg, /*InsertBefore=*/Ret, Ctx, "ZExt"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), sandboxir::Instruction::Opcode::ZExt);
    // Check getSrcTy().
    EXPECT_EQ(NewI->getSrcTy(), Arg->getType());
    // Check getDestTy().
    EXPECT_EQ(NewI->getDestTy(), Ti64);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), Ret);
  }
  {
    // Check create() InsertAtEnd.
    auto *NewI = cast<sandboxir::CastInst>(
        sandboxir::CastInst::create(Ti64, sandboxir::Instruction::Opcode::ZExt,
                                    Arg, /*InsertAtEnd=*/BB, Ctx, "ZExt"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), sandboxir::Instruction::Opcode::ZExt);
    // Check getSrcTy().
    EXPECT_EQ(NewI->getSrcTy(), Arg->getType());
    // Check getDestTy().
    EXPECT_EQ(NewI->getDestTy(), Ti64);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), nullptr);
    EXPECT_EQ(NewI->getParent(), BB);
  }

  {
#ifndef NDEBUG
    // Check that passing a non-cast opcode crashes.
    EXPECT_DEATH(
        sandboxir::CastInst::create(Ti64, sandboxir::Instruction::Opcode::Store,
                                    Arg, /*InsertBefore=*/Ret, Ctx, "Bad"),
        ".*Opcode.*");
#endif // NDEBUG
  }
}

/// CastInst's subclasses are very similar so we can use a common test function
/// for them.
template <typename SubclassT, sandboxir::Instruction::Opcode OpcodeT>
void testCastInst(llvm::Module &M, Type *SrcTy, Type *DstTy) {
  Function &LLVMF = *M.getFunction("foo");
  sandboxir::Context Ctx(M.getContext());
  sandboxir::Function *F = Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg = F->getArg(ArgIdx++);
  auto *BB = &*F->begin();
  auto It = BB->begin();

  auto *CI = cast<SubclassT>(&*It++);
  EXPECT_EQ(CI->getOpcode(), OpcodeT);
  EXPECT_EQ(CI->getSrcTy(), SrcTy);
  EXPECT_EQ(CI->getDestTy(), DstTy);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  {
    // Check create() WhereIt, WhereBB
    auto *NewI =
        cast<SubclassT>(SubclassT::create(Arg, DstTy, /*WhereIt=*/BB->end(),
                                          /*WhereBB=*/BB, Ctx, "NewCI"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), OpcodeT);
    // Check getSrcTy().
    EXPECT_EQ(NewI->getSrcTy(), Arg->getType());
    // Check getDestTy().
    EXPECT_EQ(NewI->getDestTy(), DstTy);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), nullptr);
    EXPECT_EQ(NewI->getPrevNode(), Ret);
    // Check instr name.
    EXPECT_EQ(NewI->getName(), "NewCI");
  }
  {
    // Check create() InsertBefore.
    auto *NewI =
        cast<SubclassT>(SubclassT::create(Arg, DstTy,
                                          /*InsertBefore=*/Ret, Ctx, "NewCI"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), OpcodeT);
    // Check getSrcTy().
    EXPECT_EQ(NewI->getSrcTy(), Arg->getType());
    // Check getDestTy().
    EXPECT_EQ(NewI->getDestTy(), DstTy);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), Ret);
  }
  {
    // Check create() InsertAtEnd.
    auto *NewI =
        cast<SubclassT>(SubclassT::create(Arg, DstTy,
                                          /*InsertAtEnd=*/BB, Ctx, "NewCI"));
    // Check getOpcode().
    EXPECT_EQ(NewI->getOpcode(), OpcodeT);
    // Check getSrcTy().
    EXPECT_EQ(NewI->getSrcTy(), Arg->getType());
    // Check getDestTy().
    EXPECT_EQ(NewI->getDestTy(), DstTy);
    // Check instr position.
    EXPECT_EQ(NewI->getNextNode(), nullptr);
    EXPECT_EQ(NewI->getParent(), BB);
  }
}

TEST_F(SandboxIRTest, TruncInst) {
  parseIR(C, R"IR(
define void @foo(i64 %arg) {
  %trunc = trunc i64 %arg to i32
  ret void
}
)IR");
  testCastInst<sandboxir::TruncInst, sandboxir::Instruction::Opcode::Trunc>(
      *M,
      /*SrcTy=*/Type::getInt64Ty(C), /*DstTy=*/Type::getInt32Ty(C));
}

TEST_F(SandboxIRTest, ZExtInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %zext = zext i32 %arg to i64
  ret void
}
)IR");
  testCastInst<sandboxir::ZExtInst, sandboxir::Instruction::Opcode::ZExt>(
      *M,
      /*SrcTy=*/Type::getInt32Ty(C), /*DstTy=*/Type::getInt64Ty(C));
}

TEST_F(SandboxIRTest, SExtInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %sext = sext i32 %arg to i64
  ret void
}
)IR");
  testCastInst<sandboxir::SExtInst, sandboxir::Instruction::Opcode::SExt>(
      *M,
      /*SrcTy=*/Type::getInt32Ty(C), /*DstTy=*/Type::getInt64Ty(C));
}

TEST_F(SandboxIRTest, FPTruncInst) {
  parseIR(C, R"IR(
define void @foo(double %arg) {
  %fptrunc = fptrunc double %arg to float
  ret void
}
)IR");
  testCastInst<sandboxir::FPTruncInst, sandboxir::Instruction::Opcode::FPTrunc>(
      *M,
      /*SrcTy=*/Type::getDoubleTy(C), /*DstTy=*/Type::getFloatTy(C));
}

TEST_F(SandboxIRTest, FPExtInst) {
  parseIR(C, R"IR(
define void @foo(float %arg) {
  %fpext = fpext float %arg to double
  ret void
}
)IR");
  testCastInst<sandboxir::FPExtInst, sandboxir::Instruction::Opcode::FPExt>(
      *M,
      /*SrcTy=*/Type::getFloatTy(C), /*DstTy=*/Type::getDoubleTy(C));
}

TEST_F(SandboxIRTest, UIToFPInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %uitofp = uitofp i32 %arg to float
  ret void
}
)IR");
  testCastInst<sandboxir::UIToFPInst, sandboxir::Instruction::Opcode::UIToFP>(
      *M,
      /*SrcTy=*/Type::getInt32Ty(C), /*DstTy=*/Type::getFloatTy(C));
}

TEST_F(SandboxIRTest, SIToFPInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %sitofp = sitofp i32 %arg to float
  ret void
}
)IR");
  testCastInst<sandboxir::SIToFPInst, sandboxir::Instruction::Opcode::SIToFP>(
      *M,
      /*SrcTy=*/Type::getInt32Ty(C),
      /*DstTy=*/Type::getFloatTy(C));
}

TEST_F(SandboxIRTest, FPToUIInst) {
  parseIR(C, R"IR(
define void @foo(float %arg) {
  %fptoui = fptoui float %arg to i32
  ret void
}
)IR");
  testCastInst<sandboxir::FPToUIInst, sandboxir::Instruction::Opcode::FPToUI>(

      *M, /*SrcTy=*/Type::getFloatTy(C), /*DstTy=*/Type::getInt32Ty(C));
}

TEST_F(SandboxIRTest, FPToSIInst) {
  parseIR(C, R"IR(
define void @foo(float %arg) {
  %fptosi = fptosi float %arg to i32
  ret void
}
)IR");
  testCastInst<sandboxir::FPToSIInst, sandboxir::Instruction::Opcode::FPToSI>(
      *M, /*SrcTy=*/Type::getFloatTy(C), /*DstTy=*/Type::getInt32Ty(C));
}

TEST_F(SandboxIRTest, IntToPtrInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %inttoptr = inttoptr i32 %arg to ptr
  ret void
}
)IR");
  testCastInst<sandboxir::IntToPtrInst,
               sandboxir::Instruction::Opcode::IntToPtr>(
      *M,
      /*SrcTy=*/Type::getInt32Ty(C), /*DstTy=*/PointerType::get(C, 0));
}

TEST_F(SandboxIRTest, PtrToIntInst) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %ptrtoint = ptrtoint ptr %ptr to i32
  ret void
}
)IR");
  testCastInst<sandboxir::PtrToIntInst,
               sandboxir::Instruction::Opcode::PtrToInt>(
      *M, /*SrcTy=*/PointerType::get(C, 0), /*DstTy=*/Type::getInt32Ty(C));
}

TEST_F(SandboxIRTest, BitCastInst) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
  %bitcast = bitcast i32 %arg to float
  ret void
}
)IR");
  testCastInst<sandboxir::BitCastInst, sandboxir::Instruction::Opcode::BitCast>(
      *M,
      /*SrcTy=*/Type::getInt32Ty(C), /*DstTy=*/Type::getFloatTy(C));
}

TEST_F(SandboxIRTest, AddrSpaceCastInst) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr) {
  %addrspacecast = addrspacecast ptr %ptr to ptr addrspace(1)
  ret void
}
)IR");
  Type *Tptr0 = PointerType::get(C, 0);
  Type *Tptr1 = PointerType::get(C, 1);
  testCastInst<sandboxir::AddrSpaceCastInst,
               sandboxir::Instruction::Opcode::AddrSpaceCast>(*M,
                                                              /*SrcTy=*/Tptr0,
                                                              /*DstTy=*/Tptr1);
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(&LLVMF);
  unsigned ArgIdx = 0;
  auto *Arg = F->getArg(ArgIdx++);
  auto *BB = &*F->begin();
  auto It = BB->begin();

  auto *AddrSpaceCast = cast<sandboxir::AddrSpaceCastInst>(&*It++);
  EXPECT_EQ(AddrSpaceCast->getOpcode(),
            sandboxir::Instruction::Opcode::AddrSpaceCast);
  EXPECT_EQ(AddrSpaceCast->getPointerOperand(), Arg);
  EXPECT_EQ(sandboxir::AddrSpaceCastInst::getPointerOperandIndex(), 0u);
  EXPECT_EQ(AddrSpaceCast->getSrcAddressSpace(),
            cast<PointerType>(Tptr0)->getPointerAddressSpace());
  EXPECT_EQ(AddrSpaceCast->getDestAddressSpace(),
            cast<PointerType>(Tptr1)->getPointerAddressSpace());
}

TEST_F(SandboxIRTest, PHINode) {
  parseIR(C, R"IR(
define void @foo(i32 %arg) {
bb1:
  br label %bb2

bb2:
  %phi = phi i32 [ %arg, %bb1 ], [ 0, %bb2 ], [ 1, %bb3 ], [ 2, %bb4 ], [ 3, %bb5 ]
  br label %bb2

bb3:
  br label %bb2

bb4:
  br label %bb2

bb5:
  br label %bb2
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  auto *LLVMBB1 = getBasicBlockByName(LLVMF, "bb1");
  auto *LLVMBB2 = getBasicBlockByName(LLVMF, "bb2");
  auto *LLVMBB3 = getBasicBlockByName(LLVMF, "bb3");
  auto LLVMIt = LLVMBB2->begin();
  auto *LLVMPHI = cast<llvm::PHINode>(&*LLVMIt++);
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(&LLVMF);
  auto *Arg = F->getArg(0);
  auto *BB1 = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMBB1));
  auto *BB2 = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMBB2));
  auto *BB3 = cast<sandboxir::BasicBlock>(Ctx.getValue(LLVMBB3));
  auto It = BB2->begin();
  // Check classof().
  auto *PHI = cast<sandboxir::PHINode>(&*It++);
  auto *Br = cast<sandboxir::BranchInst>(&*It++);
  // Check blocks().
  EXPECT_EQ(range_size(PHI->blocks()), range_size(LLVMPHI->blocks()));
  auto BlockIt = PHI->block_begin();
  for (llvm::BasicBlock *LLVMBB : LLVMPHI->blocks()) {
    sandboxir::BasicBlock *BB = *BlockIt++;
    EXPECT_EQ(BB, Ctx.getValue(LLVMBB));
  }
  // Check incoming_values().
  EXPECT_EQ(range_size(PHI->incoming_values()),
            range_size(LLVMPHI->incoming_values()));
  auto IncIt = PHI->incoming_values().begin();
  for (llvm::Value *LLVMV : LLVMPHI->incoming_values()) {
    sandboxir::Value *IncV = *IncIt++;
    EXPECT_EQ(IncV, Ctx.getValue(LLVMV));
  }
  // Check getNumIncomingValues().
  EXPECT_EQ(PHI->getNumIncomingValues(), LLVMPHI->getNumIncomingValues());
  // Check getIncomingValue().
  EXPECT_EQ(PHI->getIncomingValue(0),
            Ctx.getValue(LLVMPHI->getIncomingValue(0)));
  EXPECT_EQ(PHI->getIncomingValue(1),
            Ctx.getValue(LLVMPHI->getIncomingValue(1)));
  // Check setIncomingValue().
  auto *OrigV = PHI->getIncomingValue(0);
  PHI->setIncomingValue(0, PHI);
  EXPECT_EQ(PHI->getIncomingValue(0), PHI);
  PHI->setIncomingValue(0, OrigV);
  // Check getOperandNumForIncomingValue().
  EXPECT_EQ(sandboxir::PHINode::getOperandNumForIncomingValue(0),
            llvm::PHINode::getOperandNumForIncomingValue(0));
  // Check getIncomingValueNumForOperand().
  EXPECT_EQ(sandboxir::PHINode::getIncomingValueNumForOperand(0),
            llvm::PHINode::getIncomingValueNumForOperand(0));
  // Check getIncomingBlock(unsigned).
  EXPECT_EQ(PHI->getIncomingBlock(0),
            Ctx.getValue(LLVMPHI->getIncomingBlock(0)));
  // Check getIncomingBlock(Use).
  llvm::Use &LLVMUse = LLVMPHI->getOperandUse(0);
  sandboxir::Use Use = PHI->getOperandUse(0);
  EXPECT_EQ(PHI->getIncomingBlock(Use),
            Ctx.getValue(LLVMPHI->getIncomingBlock(LLVMUse)));
  // Check setIncomingBlock().
  sandboxir::BasicBlock *OrigBB = PHI->getIncomingBlock(0);
  EXPECT_NE(OrigBB, BB2);
  PHI->setIncomingBlock(0, BB2);
  EXPECT_EQ(PHI->getIncomingBlock(0), BB2);
  PHI->setIncomingBlock(0, OrigBB);
  EXPECT_EQ(PHI->getIncomingBlock(0), OrigBB);
  // Check addIncoming().
  unsigned OrigNumIncoming = PHI->getNumIncomingValues();
  PHI->addIncoming(Arg, BB3);
  EXPECT_EQ(PHI->getNumIncomingValues(), LLVMPHI->getNumIncomingValues());
  EXPECT_EQ(PHI->getNumIncomingValues(), OrigNumIncoming + 1);
  EXPECT_EQ(PHI->getIncomingValue(OrigNumIncoming), Arg);
  EXPECT_EQ(PHI->getIncomingBlock(OrigNumIncoming), BB3);
  // Check removeIncomingValue(unsigned).
  PHI->removeIncomingValue(OrigNumIncoming);
  EXPECT_EQ(PHI->getNumIncomingValues(), OrigNumIncoming);
  // Check removeIncomingValue(BasicBlock *).
  PHI->addIncoming(Arg, BB3);
  PHI->removeIncomingValue(BB3);
  EXPECT_EQ(PHI->getNumIncomingValues(), OrigNumIncoming);
  // Check getBasicBlockIndex().
  EXPECT_EQ(PHI->getBasicBlockIndex(BB1), LLVMPHI->getBasicBlockIndex(LLVMBB1));
  // Check getIncomingValueForBlock().
  EXPECT_EQ(PHI->getIncomingValueForBlock(BB1),
            Ctx.getValue(LLVMPHI->getIncomingValueForBlock(LLVMBB1)));
  // Check hasConstantValue().
  llvm::Value *ConstV = LLVMPHI->hasConstantValue();
  EXPECT_EQ(PHI->hasConstantValue(),
            ConstV != nullptr ? Ctx.getValue(ConstV) : nullptr);
  // Check hasConstantOrUndefValue().
  EXPECT_EQ(PHI->hasConstantOrUndefValue(), LLVMPHI->hasConstantOrUndefValue());
  // Check isComplete().
  EXPECT_EQ(PHI->isComplete(), LLVMPHI->isComplete());
  // Check replaceIncomingValueIf
  EXPECT_EQ(PHI->getNumIncomingValues(), 5u);
  auto *RemainBB0 = PHI->getIncomingBlock(0);
  auto *RemoveBB0 = PHI->getIncomingBlock(1);
  auto *RemainBB1 = PHI->getIncomingBlock(2);
  auto *RemoveBB1 = PHI->getIncomingBlock(3);
  auto *RemainBB2 = PHI->getIncomingBlock(4);
  PHI->removeIncomingValueIf([&](unsigned Idx) {
    return PHI->getIncomingBlock(Idx) == RemoveBB0 ||
           PHI->getIncomingBlock(Idx) == RemoveBB1;
  });
  EXPECT_EQ(PHI->getNumIncomingValues(), 3u);
  EXPECT_EQ(PHI->getIncomingBlock(0), RemainBB0);
  EXPECT_EQ(PHI->getIncomingBlock(1), RemainBB1);
  EXPECT_EQ(PHI->getIncomingBlock(2), RemainBB2);
  // Check replaceIncomingBlockWith
  OrigBB = RemainBB0;
  auto *NewBB = RemainBB1;
  EXPECT_NE(NewBB, OrigBB);
  PHI->replaceIncomingBlockWith(OrigBB, NewBB);
  EXPECT_EQ(PHI->getIncomingBlock(0), NewBB);
  EXPECT_EQ(PHI->getIncomingBlock(1), RemainBB1);
  EXPECT_EQ(PHI->getIncomingBlock(2), RemainBB2);
  // Check create().
  auto *NewPHI = cast<sandboxir::PHINode>(
      sandboxir::PHINode::create(PHI->getType(), 0, Br, Ctx, "NewPHI"));
  EXPECT_EQ(NewPHI->getType(), PHI->getType());
  EXPECT_EQ(NewPHI->getNextNode(), Br);
  EXPECT_EQ(NewPHI->getName(), "NewPHI");
  EXPECT_EQ(NewPHI->getNumIncomingValues(), 0u);
  for (auto [Idx, V] : enumerate(PHI->incoming_values())) {
    sandboxir::BasicBlock *IncBB = PHI->getIncomingBlock(Idx);
    NewPHI->addIncoming(V, IncBB);
  }
  EXPECT_EQ(NewPHI->getNumIncomingValues(), PHI->getNumIncomingValues());
}

TEST_F(SandboxIRTest, UnreachableInst) {
  parseIR(C, R"IR(
define void @foo() {
  unreachable
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *UI = cast<sandboxir::UnreachableInst>(&*It++);

  EXPECT_EQ(UI->getNumSuccessors(), 0u);
  EXPECT_EQ(UI->getNumOfIRInstrs(), 1u);
  // Check create(InsertBefore)
  sandboxir::UnreachableInst *NewUI =
      sandboxir::UnreachableInst::create(/*InsertBefore=*/UI, Ctx);
  EXPECT_EQ(NewUI->getNextNode(), UI);
  // Check create(InsertAtEnd)
  sandboxir::UnreachableInst *NewUIEnd =
      sandboxir::UnreachableInst::create(/*InsertAtEnd=*/BB, Ctx);
  EXPECT_EQ(NewUIEnd->getParent(), BB);
  EXPECT_EQ(NewUIEnd->getNextNode(), nullptr);
}
