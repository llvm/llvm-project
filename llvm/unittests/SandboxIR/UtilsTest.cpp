//===- UtilsTest.cpp ------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Utils.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct UtilsTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("UtilsTest", errs());
  }
  BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
    for (BasicBlock &BB : F)
      if (BB.getName() == Name)
        return &BB;
    llvm_unreachable("Expected to find basic block!");
  }
};

TEST_F(UtilsTest, getMemoryLocation) {
  parseIR(C, R"IR(
define void @foo(ptr %arg0) {
  %ld = load i8, ptr %arg0
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  auto *LLVMBB = &*LLVMF->begin();
  auto *LLVMLd = cast<llvm::LoadInst>(&*LLVMBB->begin());
  sandboxir::Context Ctx(C);
  sandboxir::Function *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto *Ld = cast<sandboxir::LoadInst>(&*BB->begin());
  EXPECT_EQ(sandboxir::Utils::memoryLocationGetOrNone(Ld),
            MemoryLocation::getOrNone(LLVMLd));
}
