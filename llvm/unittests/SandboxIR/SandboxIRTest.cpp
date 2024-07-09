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
