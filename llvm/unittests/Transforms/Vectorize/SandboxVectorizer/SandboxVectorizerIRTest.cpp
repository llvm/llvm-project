//===- SandboxVectorizerIRTest.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerIR.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/BasicBlock.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxVectorizerIRTest : public testing::Test {
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

TEST_F(SandboxVectorizerIRTest, Basic) {
  parseIR(C, R"IR(
define void @foo(i8 %v0, i8 %v1) {
  ret void
}
)IR");
  Function &LLVMF = *M->getFunction("foo");
  sandboxir::SBVecContext Ctx(C);
  auto &F = *Ctx.createFunction(&LLVMF);
  auto &BB = *F.begin();
  auto *Arg0 = F.getArg(0);
  auto *Arg1 = F.getArg(1);

  // Check that we can access the new opcode Opcode::Pack and retrieve its name.
  EXPECT_EQ(llvm::StringRef(sandboxir::Instruction::getOpcodeName(
                sandboxir::Instruction::Opcode::Pack)),
            "Pack");
  auto *PackI = sandboxir::PackInst::create({Arg0, Arg1}, BB.begin(), Ctx);
  // Check the class ID.
  EXPECT_EQ(PackI->getSubclassID(), sandboxir::Value::ClassID::Pack);
}
