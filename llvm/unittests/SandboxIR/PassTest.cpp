//===- PassTest.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/Pass.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm::sandboxir;

struct PassTest : public testing::Test {
  llvm::LLVMContext LLVMCtx;
  std::unique_ptr<llvm::Module> LLVMM;
  std::unique_ptr<Context> Ctx;

  Function *parseFunction(const char *IR, const char *FuncName) {
    llvm::SMDiagnostic Err;
    LLVMM = parseAssemblyString(IR, Err, LLVMCtx);
    if (!LLVMM)
      Err.print("PassTest", llvm::errs());
    Ctx = std::make_unique<Context>(LLVMCtx);
    return Ctx->createFunction(LLVMM->getFunction(FuncName));
  }
};

TEST_F(PassTest, FunctionPass) {
  auto *F = parseFunction(R"IR(
define void @foo() {
  ret void
}
)IR",
                          "foo");
  class TestPass final : public FunctionPass {
    unsigned &BBCnt;

  public:
    TestPass(unsigned &BBCnt) : FunctionPass("test-pass"), BBCnt(BBCnt) {}
    bool runOnFunction(Function &F) final {
      for ([[maybe_unused]] auto &BB : F)
        ++BBCnt;
      return false;
    }
  };
  unsigned BBCnt = 0;
  TestPass TPass(BBCnt);
  // Check getName(),
  EXPECT_EQ(TPass.getName(), "test-pass");
  // Check classof().
  EXPECT_TRUE(llvm::isa<FunctionPass>(TPass));
  // Check runOnFunction();
  TPass.runOnFunction(*F);
  EXPECT_EQ(BBCnt, 1u);
#ifndef NDEBUG
  {
    // Check print().
    std::string Buff;
    llvm::raw_string_ostream SS(Buff);
    TPass.print(SS);
    EXPECT_EQ(Buff, "test-pass");
  }
  {
    // Check operator<<().
    std::string Buff;
    llvm::raw_string_ostream SS(Buff);
    SS << TPass;
    EXPECT_EQ(Buff, "test-pass");
  }
  // Check pass name assertions.
  class TestNamePass final : public FunctionPass {
  public:
    TestNamePass(llvm::StringRef Name) : FunctionPass(Name) {}
    bool runOnFunction(Function &F) { return false; }
  };
  EXPECT_DEATH(TestNamePass("white space"), ".*whitespace.*");
  EXPECT_DEATH(TestNamePass("-dash"), ".*start with.*");
#endif
}
