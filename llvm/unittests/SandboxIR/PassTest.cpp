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
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Context.h"
#include "llvm/SandboxIR/Function.h"
#include "llvm/SandboxIR/PassManager.h"
#include "llvm/SandboxIR/Region.h"
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
    bool runOnFunction(Function &F, const Analyses &A) final {
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
  TPass.runOnFunction(*F, Analyses::emptyForTesting());
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
    bool runOnFunction(Function &F, const Analyses &A) { return false; }
  };
  EXPECT_DEATH(TestNamePass("white space"), ".*whitespace.*");
  EXPECT_DEATH(TestNamePass("-dash"), ".*start with.*");
#endif
}

TEST_F(PassTest, RegionPass) {
  auto *F = parseFunction(R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1, !sandboxvec !0
  %t2 = add i8 %t1, %v1, !sandboxvec !0
  ret i8 %t1
}

!0 = distinct !{!"sandboxregion"}
)IR",
                          "foo");

  class TestPass final : public RegionPass {
    unsigned &InstCount;

  public:
    TestPass(unsigned &InstCount)
        : RegionPass("test-pass"), InstCount(InstCount) {}
    bool runOnRegion(Region &R, const Analyses &A) final {
      for ([[maybe_unused]] auto &Inst : R) {
        ++InstCount;
      }
      return false;
    }
  };
  unsigned InstCount = 0;
  TestPass TPass(InstCount);
  // Check getName(),
  EXPECT_EQ(TPass.getName(), "test-pass");
  // Check runOnRegion();
  llvm::SmallVector<std::unique_ptr<Region>> Regions =
      Region::createRegionsFromMD(*F);
  ASSERT_EQ(Regions.size(), 1u);
  TPass.runOnRegion(*Regions[0], Analyses::emptyForTesting());
  EXPECT_EQ(InstCount, 2u);
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
  class TestNamePass final : public RegionPass {
  public:
    TestNamePass(llvm::StringRef Name) : RegionPass(Name) {}
    bool runOnRegion(Region &F, const Analyses &A) { return false; }
  };
  EXPECT_DEATH(TestNamePass("white space"), ".*whitespace.*");
  EXPECT_DEATH(TestNamePass("-dash"), ".*start with.*");
#endif
}

TEST_F(PassTest, FunctionPassManager) {
  auto *F = parseFunction(R"IR(
define void @foo() {
  ret void
}
)IR",
                          "foo");
  class TestPass1 final : public FunctionPass {
    unsigned &BBCnt;

  public:
    TestPass1(unsigned &BBCnt) : FunctionPass("test-pass1"), BBCnt(BBCnt) {}
    bool runOnFunction(Function &F, const Analyses &A) final {
      for ([[maybe_unused]] auto &BB : F)
        ++BBCnt;
      return false;
    }
  };
  class TestPass2 final : public FunctionPass {
    unsigned &BBCnt;

  public:
    TestPass2(unsigned &BBCnt) : FunctionPass("test-pass2"), BBCnt(BBCnt) {}
    bool runOnFunction(Function &F, const Analyses &A) final {
      for ([[maybe_unused]] auto &BB : F)
        ++BBCnt;
      return false;
    }
  };
  unsigned BBCnt1 = 0;
  unsigned BBCnt2 = 0;

  FunctionPassManager FPM("test-fpm");
  FPM.addPass(std::make_unique<TestPass1>(BBCnt1));
  FPM.addPass(std::make_unique<TestPass2>(BBCnt2));
  // Check runOnFunction().
  FPM.runOnFunction(*F, Analyses::emptyForTesting());
  EXPECT_EQ(BBCnt1, 1u);
  EXPECT_EQ(BBCnt2, 1u);
#ifndef NDEBUG
  // Check dump().
  std::string Buff;
  llvm::raw_string_ostream SS(Buff);
  FPM.print(SS);
  EXPECT_EQ(Buff, "test-fpm(test-pass1,test-pass2)");
#endif // NDEBUG
}

TEST_F(PassTest, RegionPassManager) {
  auto *F = parseFunction(R"IR(
define i8 @foo(i8 %v0, i8 %v1) {
  %t0 = add i8 %v0, 1
  %t1 = add i8 %t0, %v1, !sandboxvec !0
  %t2 = add i8 %t1, %v1, !sandboxvec !0
  ret i8 %t1
}

!0 = distinct !{!"sandboxregion"}
)IR",
                          "foo");

  class TestPass1 final : public RegionPass {
    unsigned &InstCount;

  public:
    TestPass1(unsigned &InstCount)
        : RegionPass("test-pass1"), InstCount(InstCount) {}
    bool runOnRegion(Region &R, const Analyses &A) final {
      for ([[maybe_unused]] auto &Inst : R)
        ++InstCount;
      return false;
    }
  };
  class TestPass2 final : public RegionPass {
    unsigned &InstCount;

  public:
    TestPass2(unsigned &InstCount)
        : RegionPass("test-pass2"), InstCount(InstCount) {}
    bool runOnRegion(Region &R, const Analyses &A) final {
      for ([[maybe_unused]] auto &Inst : R)
        ++InstCount;
      return false;
    }
  };
  unsigned InstCount1 = 0;
  unsigned InstCount2 = 0;

  RegionPassManager RPM("test-rpm");
  RPM.addPass(std::make_unique<TestPass1>(InstCount1));
  RPM.addPass(std::make_unique<TestPass2>(InstCount2));
  // Check runOnRegion().
  llvm::SmallVector<std::unique_ptr<Region>> Regions =
      Region::createRegionsFromMD(*F);
  ASSERT_EQ(Regions.size(), 1u);
  RPM.runOnRegion(*Regions[0], Analyses::emptyForTesting());
  EXPECT_EQ(InstCount1, 2u);
  EXPECT_EQ(InstCount2, 2u);
#ifndef NDEBUG
  // Check dump().
  std::string Buff;
  llvm::raw_string_ostream SS(Buff);
  RPM.print(SS);
  EXPECT_EQ(Buff, "test-rpm(test-pass1,test-pass2)");
#endif // NDEBUG
}

TEST_F(PassTest, SetPassPipeline) {
  auto *F = parseFunction(R"IR(
define void @f() {
  ret void
}
)IR",
                          "f");
  class FooPass final : public FunctionPass {
    std::string &Str;
    std::string Args;

  public:
    FooPass(std::string &Str, llvm::StringRef Args)
        : FunctionPass("foo-pass"), Str(Str), Args(Args.str()) {}
    bool runOnFunction(Function &F, const Analyses &A) final {
      Str += "foo<" + Args + ">";
      return false;
    }
  };
  class BarPass final : public FunctionPass {
    std::string &Str;
    std::string Args;

  public:
    BarPass(std::string &Str, llvm::StringRef Args)
        : FunctionPass("bar-pass"), Str(Str), Args(Args.str()) {}
    bool runOnFunction(Function &F, const Analyses &A) final {
      Str += "bar<" + Args + ">";
      return false;
    }
  };

  std::string Str;
  auto CreatePass =
      [&Str](llvm::StringRef Name,
             llvm::StringRef Args) -> std::unique_ptr<FunctionPass> {
    if (Name == "foo")
      return std::make_unique<FooPass>(Str, Args);
    if (Name == "bar")
      return std::make_unique<BarPass>(Str, Args);
    return nullptr;
  };

  FunctionPassManager FPM("test-fpm");
  FPM.setPassPipeline("foo<abc>,bar<nested1<nested2<nested3>>>,foo",
                      CreatePass);
  FPM.runOnFunction(*F, Analyses::emptyForTesting());
  EXPECT_EQ(Str, "foo<abc>bar<nested1<nested2<nested3>>>foo<>");

  // A second call to setPassPipeline will trigger an assertion in debug mode.
#ifndef NDEBUG
  EXPECT_DEATH(FPM.setPassPipeline("bar,bar,foo", CreatePass),
               "setPassPipeline called on a non-empty sandboxir::PassManager");
#endif

  // Fresh PM for the death tests so they die from bad pipeline strings, rather
  // than from multiple setPassPipeline calls.
  FunctionPassManager FPM2("test-fpm");
  // Bad/empty pass names.
  EXPECT_DEATH(FPM2.setPassPipeline("bad-pass-name", CreatePass),
               ".*not registered.*");
  EXPECT_DEATH(FPM2.setPassPipeline(",", CreatePass), ".*empty pass name.*");
  EXPECT_DEATH(FPM2.setPassPipeline("<>", CreatePass), ".*empty pass name.*");
  EXPECT_DEATH(FPM2.setPassPipeline("<>foo", CreatePass),
               ".*empty pass name.*");
  EXPECT_DEATH(FPM2.setPassPipeline("foo,<>", CreatePass),
               ".*empty pass name.*");

  // Mismatched argument brackets.
  EXPECT_DEATH(FPM2.setPassPipeline("foo<", CreatePass), ".*Missing '>'.*");
  EXPECT_DEATH(FPM2.setPassPipeline("foo<bar", CreatePass), ".*Missing '>'.*");
  EXPECT_DEATH(FPM2.setPassPipeline("foo<bar<>", CreatePass),
               ".*Missing '>'.*");
  EXPECT_DEATH(FPM2.setPassPipeline("foo>", CreatePass), ".*Unexpected '>'.*");
  EXPECT_DEATH(FPM2.setPassPipeline(">foo", CreatePass), ".*Unexpected '>'.*");
  // Extra garbage between args and next delimiter/end-of-string.
  EXPECT_DEATH(FPM2.setPassPipeline("foo<bar<>>>", CreatePass),
               ".*Expected delimiter.*");
  EXPECT_DEATH(FPM2.setPassPipeline("bar<>foo", CreatePass),
               ".*Expected delimiter.*");
  EXPECT_DEATH(FPM2.setPassPipeline("bar<>foo,baz", CreatePass),
               ".*Expected delimiter.*");
  EXPECT_DEATH(FPM2.setPassPipeline("foo<args><more-args>", CreatePass),
               ".*Expected delimiter.*");
  EXPECT_DEATH(FPM2.setPassPipeline("foo<args>bar", CreatePass),
               ".*Expected delimiter.*");
}
