//===- SPIRVConvergenceRegionAnalysisTests.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/SPIRVConvergenceRegionAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/Support/SourceMgr.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <queue>

using ::testing::Contains;
using ::testing::Pair;

using namespace llvm;
using namespace llvm::SPIRV;

template <typename T> struct IsA {
  friend bool operator==(const Value *V, const IsA &) { return isa<T>(V); }
};

class SPIRVConvergenceRegionAnalysisTest : public testing::Test {
protected:
  void SetUp() override {
    // Required for tests.
    FAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });

    // Required for ConvergenceRegionAnalysis.
    FAM.registerPass([&] { return DominatorTreeAnalysis(); });
    FAM.registerPass([&] { return LoopAnalysis(); });

    FAM.registerPass([&] { return SPIRVConvergenceRegionAnalysis(); });
  }

  void TearDown() override { M.reset(); }

  SPIRVConvergenceRegionAnalysis::Result &runAnalysis(StringRef Assembly) {
    assert(M == nullptr &&
           "Calling runAnalysis multiple times is unsafe. See getAnalysis().");

    SMDiagnostic Error;
    M = parseAssemblyString(Assembly, Error, Context);
    assert(M && "Bad assembly. Bad test?");
    auto *F = getFunction();

    ModulePassManager MPM;
    MPM.run(*M, MAM);
    return FAM.getResult<SPIRVConvergenceRegionAnalysis>(*F);
  }

  SPIRVConvergenceRegionAnalysis::Result &getAnalysis() {
    assert(M != nullptr && "Has runAnalysis been called before?");
    return FAM.getResult<SPIRVConvergenceRegionAnalysis>(*getFunction());
  }

  Function *getFunction() const {
    assert(M != nullptr && "Has runAnalysis been called before?");
    return M->getFunction("main");
  }

  const BasicBlock *getBlock(StringRef Name) {
    assert(M != nullptr && "Has runAnalysis been called before?");

    auto *F = getFunction();
    for (BasicBlock &BB : *F) {
      if (BB.getName() == Name)
        return &BB;
    }

    ADD_FAILURE() << "Error: Could not locate requested block. Bad test?";
    return nullptr;
  }

  const ConvergenceRegion *getRegionWithEntry(StringRef Name) {
    assert(M != nullptr && "Has runAnalysis been called before?");

    std::queue<const ConvergenceRegion *> ToProcess;
    ToProcess.push(getAnalysis().getTopLevelRegion());

    while (ToProcess.size() != 0) {
      auto *R = ToProcess.front();
      ToProcess.pop();
      for (auto *Child : R->Children)
        ToProcess.push(Child);

      if (R->Entry->getName() == Name)
        return R;
    }

    ADD_FAILURE() << "Error: Could not locate requested region. Bad test?";
    return nullptr;
  }

  void checkRegionBlocks(const ConvergenceRegion *R,
                         std::initializer_list<const char *> InRegion,
                         std::initializer_list<const char *> NotInRegion) {
    for (const char *Name : InRegion) {
      EXPECT_TRUE(R->contains(getBlock(Name)))
          << "error: " << Name << " not in region " << R->Entry->getName();
    }

    for (const char *Name : NotInRegion) {
      EXPECT_FALSE(R->contains(getBlock(Name)))
          << "error: " << Name << " in region " << R->Entry->getName();
    }
  }

protected:
  LLVMContext Context;
  FunctionAnalysisManager FAM;
  ModuleAnalysisManager MAM;
  std::unique_ptr<Module> M;
};

MATCHER_P(ContainsBasicBlock, label, "") {
  for (const auto *bb : arg)
    if (bb->getName() == label)
      return true;
  return false;
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, DefaultRegion) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ret void
    }
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();

  EXPECT_EQ(CR->Parent, nullptr);
  EXPECT_EQ(CR->ConvergenceToken, std::nullopt);
  EXPECT_EQ(CR->Children.size(), 0u);
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, DefaultRegionWithToken) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();

  EXPECT_EQ(CR->Parent, nullptr);
  EXPECT_EQ(CR->Children.size(), 0u);
  EXPECT_TRUE(CR->ConvergenceToken.has_value());
  EXPECT_EQ(CR->ConvergenceToken.value()->getIntrinsicID(),
            Intrinsic::experimental_convergence_entry);
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, SingleLoopOneRegion) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1

    l1:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_end

    l1_body:
      br label %l1_continue

    l1_continue:
      br label %l1

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();

  EXPECT_EQ(CR->Parent, nullptr);
  EXPECT_EQ(CR->ConvergenceToken.value()->getName(), "t1");
  EXPECT_TRUE(CR->ConvergenceToken.has_value());
  EXPECT_EQ(CR->ConvergenceToken.value()->getIntrinsicID(),
            Intrinsic::experimental_convergence_entry);
  EXPECT_EQ(CR->Children.size(), 1u);
}

TEST_F(SPIRVConvergenceRegionAnalysisTest,
       SingleLoopLoopRegionParentsIsTopLevelRegion) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1

    l1:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_end

    l1_body:
      br label %l1_continue

    l1_continue:
      br label %l1

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();

  EXPECT_EQ(CR->Parent, nullptr);
  EXPECT_EQ(CR->ConvergenceToken.value()->getName(), "t1");
  EXPECT_EQ(CR->Children[0]->Parent, CR);
  EXPECT_EQ(CR->Children[0]->ConvergenceToken.value()->getName(), "tl1");
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, SingleLoopExits) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1

    l1:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_end

    l1_body:
      br label %l1_continue

    l1_continue:
      br label %l1

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = CR->Children[0];

  EXPECT_EQ(L->Exits.size(), 1ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1"));
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, SingleLoopWithBreakExits) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %end.loopexit

    l1_body:
      %2 = icmp ne i32 0, 0
      br i1 %2, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      %call = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    end.loopexit:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = CR->Children[0];

  EXPECT_EQ(L->Exits.size(), 2ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_header"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_condition_true"));

  EXPECT_TRUE(CR->contains(getBlock("l1_header")));
  EXPECT_TRUE(CR->contains(getBlock("l1_condition_true")));
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, SingleLoopWithBreakRegionBlocks) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %end.loopexit

    l1_body:
      %2 = icmp ne i32 0, 0
      br i1 %2, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      %call = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    end.loopexit:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  const auto *CR = runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = CR->Children[0];

  EXPECT_TRUE(CR->contains(getBlock("l1_header")));
  EXPECT_TRUE(L->contains(getBlock("l1_header")));

  EXPECT_TRUE(CR->contains(getBlock("l1_body")));
  EXPECT_TRUE(L->contains(getBlock("l1_body")));

  EXPECT_TRUE(CR->contains(getBlock("l1_condition_true")));
  EXPECT_TRUE(L->contains(getBlock("l1_condition_true")));

  EXPECT_TRUE(CR->contains(getBlock("l1_condition_false")));
  EXPECT_TRUE(L->contains(getBlock("l1_condition_false")));

  EXPECT_TRUE(CR->contains(getBlock("l1_continue")));
  EXPECT_TRUE(L->contains(getBlock("l1_continue")));

  EXPECT_TRUE(CR->contains(getBlock("end.loopexit")));
  EXPECT_FALSE(L->contains(getBlock("end.loopexit")));

  EXPECT_TRUE(CR->contains(getBlock("end")));
  EXPECT_FALSE(L->contains(getBlock("end")));
}

// Exact same test as before, except the 'if() break' condition in the loop is
// not marked with any convergence intrinsic. In such case, it is valid to
// consider it outside of the loop.
TEST_F(SPIRVConvergenceRegionAnalysisTest,
       SingleLoopWithBreakNoConvergenceControl) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %end.loopexit

    l1_body:
      %2 = icmp ne i32 0, 0
      br i1 %2, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      br label %end

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    end.loopexit:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
  )";

  runAnalysis(Assembly);
  const auto *L = getRegionWithEntry("l1_header");

  EXPECT_EQ(L->Entry->getName(), "l1_header");
  EXPECT_EQ(L->Exits.size(), 2ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_header"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_body"));

  EXPECT_TRUE(L->contains(getBlock("l1_header")));
  EXPECT_TRUE(L->contains(getBlock("l1_body")));
  EXPECT_FALSE(L->contains(getBlock("l1_condition_true")));
  EXPECT_TRUE(L->contains(getBlock("l1_condition_false")));
  EXPECT_TRUE(L->contains(getBlock("l1_continue")));
  EXPECT_FALSE(L->contains(getBlock("end.loopexit")));
  EXPECT_FALSE(L->contains(getBlock("end")));
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, TwoLoopsWithControl) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_exit

    l1_body:
      br i1 %1, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      br label %mid

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    l1_exit:
      br label %mid

    mid:
      br label %l2_header

    l2_header:
      %tl2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l2_body, label %l2_exit

    l2_body:
      br i1 %1, label %l2_condition_true, label %l2_condition_false

    l2_condition_true:
      br label %end

    l2_condition_false:
      br label %l2_continue

    l2_continue:
      br label %l2_header

    l2_exit:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
  )";

  runAnalysis(Assembly);

  {
    const auto *L = getRegionWithEntry("l1_header");

    EXPECT_EQ(L->Entry->getName(), "l1_header");
    EXPECT_EQ(L->Exits.size(), 2ul);
    EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_header"));
    EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_body"));

    checkRegionBlocks(
        L, {"l1_header", "l1_body", "l1_condition_false", "l1_continue"},
        {"", "l2_header", "l2_body", "l2_condition_true", "l2_condition_false",
         "l2_continue", "l2_exit", "l1_condition_true", "l1_exit", "end"});
  }
  {
    const auto *L = getRegionWithEntry("l2_header");

    EXPECT_EQ(L->Entry->getName(), "l2_header");
    EXPECT_EQ(L->Exits.size(), 2ul);
    EXPECT_THAT(L->Exits, ContainsBasicBlock("l2_header"));
    EXPECT_THAT(L->Exits, ContainsBasicBlock("l2_body"));

    checkRegionBlocks(
        L, {"l2_header", "l2_body", "l2_condition_false", "l2_continue"},
        {"", "l1_header", "l1_body", "l1_condition_true", "l1_condition_false",
         "l1_continue", "l1_exit", "l2_condition_true", "l2_exit", "end"});
  }
}

// Both branches in the loop condition break. This means the loop continue
// targets are unreachable, meaning no reachable back-edge. This should
// transform the loop condition into a simple condition, meaning we have a
// single convergence region.
TEST_F(SPIRVConvergenceRegionAnalysisTest, LoopBothBranchExits) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_exit

    l1_body:
      br i1 %1, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      %call_true = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    l1_condition_false:
      %call_false = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    l1_continue:
      br label %l1_header

    l1_exit:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  ;
  const auto *R = runAnalysis(Assembly).getTopLevelRegion();

  ASSERT_EQ(R->Children.size(), 0ul);
  EXPECT_EQ(R->Exits.size(), 1ul);
  EXPECT_THAT(R->Exits, ContainsBasicBlock("end"));
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, InnerLoopBreaks) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_exit

    l1_body:
      br label %l2_header

    l2_header:
      %tl2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %tl1) ]
      br i1 %1, label %l2_body, label %l2_exit

    l2_body:
      br i1 %1, label %l2_condition_true, label %l2_condition_false

    l2_condition_true:
      %call_true = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    l2_condition_false:
      br label %l2_continue

    l2_continue:
      br label %l2_header

    l2_exit:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    l1_exit:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  const auto *R = runAnalysis(Assembly).getTopLevelRegion();
  const auto *L1 = getRegionWithEntry("l1_header");
  const auto *L2 = getRegionWithEntry("l2_header");

  EXPECT_EQ(R->Children.size(), 1ul);
  EXPECT_EQ(L1->Children.size(), 1ul);
  EXPECT_EQ(L1->Parent, R);
  EXPECT_EQ(L2->Parent, L1);

  EXPECT_EQ(R->Entry->getName(), "");
  EXPECT_EQ(R->Exits.size(), 1ul);
  EXPECT_THAT(R->Exits, ContainsBasicBlock("end"));

  EXPECT_EQ(L1->Entry->getName(), "l1_header");
  EXPECT_EQ(L1->Exits.size(), 2ul);
  EXPECT_THAT(L1->Exits, ContainsBasicBlock("l1_header"));
  EXPECT_THAT(L1->Exits, ContainsBasicBlock("l2_condition_true"));

  checkRegionBlocks(L1,
                    {"l1_header", "l1_body", "l2_header", "l2_body",
                     "l2_condition_false", "l2_condition_true", "l2_continue",
                     "l2_exit", "l1_continue"},
                    {"", "l1_exit", "end"});

  EXPECT_EQ(L2->Entry->getName(), "l2_header");
  EXPECT_EQ(L2->Exits.size(), 2ul);
  EXPECT_THAT(L2->Exits, ContainsBasicBlock("l2_header"));
  EXPECT_THAT(L2->Exits, ContainsBasicBlock("l2_body"));
  checkRegionBlocks(
      L2, {"l2_header", "l2_body", "l2_condition_false", "l2_continue"},
      {"", "l1_header", "l1_body", "l2_exit", "l1_continue",
       "l2_condition_true", "l1_exit", "end"});
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, SingleLoopMultipleExits) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %cond = icmp ne i32 0, 0
      br label %l1

    l1:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %cond, label %l1_body, label %l1_exit

    l1_body:
      switch i32 0, label %sw.default.exit [
        i32 0, label %sw.bb
        i32 1, label %sw.bb1
        i32 2, label %sw.bb2
      ]

    sw.default.exit:
      br label %sw.default

    sw.default:
      br label %l1_end

    sw.bb:
      br label %l1_end

    sw.bb1:
      br label %l1_continue

    sw.bb2:
      br label %sw.default

    l1_continue:
      br label %l1

    l1_exit:
      br label %l1_end

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
  )";

  runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = getRegionWithEntry("l1");
  ASSERT_NE(L, nullptr);

  EXPECT_EQ(L->Entry, getBlock("l1"));
  EXPECT_EQ(L->Exits.size(), 2ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_body"));

  checkRegionBlocks(L, {"l1", "l1_body", "l1_continue", "sw.bb1"},
                    {"", "sw.default.exit", "sw.default", "l1_end", "end",
                     "sw.bb", "sw.bb2", "l1_exit"});
}

TEST_F(SPIRVConvergenceRegionAnalysisTest,
       SingleLoopMultipleExitsWithPartialConvergence) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %cond = icmp ne i32 0, 0
      br label %l1

    l1:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %cond, label %l1_body, label %l1_exit

    l1_body:
      switch i32 0, label %sw.default.exit [
        i32 0, label %sw.bb
        i32 1, label %sw.bb1
        i32 2, label %sw.bb2
      ]

    sw.default.exit:
      br label %sw.default

    sw.default:
      %call = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %l1_end

    sw.bb:
      br label %l1_end

    sw.bb1:
      br label %l1_continue

    sw.bb2:
      br label %sw.default

    l1_continue:
      br label %l1

    l1_exit:
      br label %l1_end

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = getRegionWithEntry("l1");
  ASSERT_NE(L, nullptr);

  EXPECT_EQ(L->Entry, getBlock("l1"));
  EXPECT_EQ(L->Exits.size(), 3ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_body"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("sw.default"));

  checkRegionBlocks(L,
                    {"l1", "l1_body", "l1_continue", "sw.bb1",
                     "sw.default.exit", "sw.bb2", "sw.default"},
                    {"", "l1_end", "end", "sw.bb", "l1_exit"});
}

TEST_F(SPIRVConvergenceRegionAnalysisTest,
       SingleLoopWithDeepConvergenceBranch) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_end

    l1_body:
      %2 = icmp ne i32 0, 0
      br i1 %2, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      br label %a

    a:
      br label %b

    b:
      br label %c

    c:
      %call = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = getRegionWithEntry("l1_header");
  ASSERT_NE(L, nullptr);

  EXPECT_EQ(L->Entry, getBlock("l1_header"));
  EXPECT_EQ(L->Exits.size(), 2ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_header"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("c"));

  checkRegionBlocks(L,
                    {"l1_header", "l1_body", "l1_continue",
                     "l1_condition_false", "l1_condition_true", "a", "b", "c"},
                    {"", "l1_end", "end"});
}

TEST_F(SPIRVConvergenceRegionAnalysisTest,
       SingleLoopWithDeepConvergenceLateBranch) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_end

    l1_body:
      %2 = icmp ne i32 0, 0
      br i1 %2, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      br label %a

    a:
      br label %b

    b:
      br i1 %2, label %c, label %d

    c:
      %call = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %end

    d:
      br label %end

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()

    ; This intrinsic is not convergent. This is only because the backend doesn't
    ; support convergent operations yet.
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = getRegionWithEntry("l1_header");
  ASSERT_NE(L, nullptr);

  EXPECT_EQ(L->Entry, getBlock("l1_header"));
  EXPECT_EQ(L->Exits.size(), 3ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_header"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("b"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("c"));

  checkRegionBlocks(L,
                    {"l1_header", "l1_body", "l1_continue",
                     "l1_condition_false", "l1_condition_true", "a", "b", "c"},
                    {"", "l1_end", "end", "d"});
}

TEST_F(SPIRVConvergenceRegionAnalysisTest,
       SingleLoopWithNoConvergenceIntrinsics) {
  StringRef Assembly = R"(
    define void @main() "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %1 = icmp ne i32 0, 0
      br label %l1_header

    l1_header:
      br i1 %1, label %l1_body, label %l1_end

    l1_body:
      %2 = icmp ne i32 0, 0
      br i1 %2, label %l1_condition_true, label %l1_condition_false

    l1_condition_true:
      br label %a

    a:
      br label %end

    l1_condition_false:
      br label %l1_continue

    l1_continue:
      br label %l1_header

    l1_end:
      br label %end

    end:
      ret void
    }
  )";

  runAnalysis(Assembly).getTopLevelRegion();
  const auto *L = getRegionWithEntry("l1_header");
  ASSERT_NE(L, nullptr);

  EXPECT_EQ(L->Entry, getBlock("l1_header"));
  EXPECT_EQ(L->Exits.size(), 2ul);
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_header"));
  EXPECT_THAT(L->Exits, ContainsBasicBlock("l1_body"));

  checkRegionBlocks(
      L, {"l1_header", "l1_body", "l1_continue", "l1_condition_false"},
      {"", "l1_end", "end", "l1_condition_true", "a"});
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, SimpleFunction) {
  StringRef Assembly = R"(
    define void @main() "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      ret void
    }
  )";

  const auto *R = runAnalysis(Assembly).getTopLevelRegion();
  ASSERT_NE(R, nullptr);

  EXPECT_EQ(R->Entry, getBlock(""));
  EXPECT_EQ(R->Exits.size(), 1ul);
  EXPECT_THAT(R->Exits, ContainsBasicBlock(""));
  EXPECT_TRUE(R->contains(getBlock("")));
}

TEST_F(SPIRVConvergenceRegionAnalysisTest, NestedLoopInBreak) {
  StringRef Assembly = R"(
    define void @main() convergent "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" {
      %t1 = call token @llvm.experimental.convergence.entry()
      %1 = icmp ne i32 0, 0
      br label %l1

    l1:
      %tl1 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %t1) ]
      br i1 %1, label %l1_body, label %l1_to_end

    l1_body:
      br i1 %1, label %cond_inner, label %l1_continue

    cond_inner:
      br label %l2

    l2:
      %tl2 = call token @llvm.experimental.convergence.loop() [ "convergencectrl"(token %tl1) ]
      br i1 %1, label %l2_body, label %l2_end

    l2_body:
      %call = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl2) ]
      br label %l2_continue

    l2_continue:
      br label %l2

    l2_end:
      br label %l2_exit

    l2_exit:
      %call2 = call spir_func i32 @_Z3absi(i32 0) [ "convergencectrl"(token %tl1) ]
      br label %l1_end

    l1_continue:
      br label %l1

    l1_to_end:
      br label %l1_end

    l1_end:
      br label %end

    end:
      ret void
    }

    declare token @llvm.experimental.convergence.entry()
    declare token @llvm.experimental.convergence.control()
    declare token @llvm.experimental.convergence.loop()
    declare spir_func i32 @_Z3absi(i32) convergent
  )";

  const auto *R = runAnalysis(Assembly).getTopLevelRegion();
  ASSERT_NE(R, nullptr);

  EXPECT_EQ(R->Children.size(), 1ul);

  const auto *L1 = R->Children[0];
  EXPECT_EQ(L1->Children.size(), 1ul);
  EXPECT_EQ(L1->Entry->getName(), "l1");
  EXPECT_EQ(L1->Exits.size(), 2ul);
  EXPECT_THAT(L1->Exits, ContainsBasicBlock("l1"));
  EXPECT_THAT(L1->Exits, ContainsBasicBlock("l2_exit"));
  checkRegionBlocks(L1,
                    {"l1", "l1_body", "l1_continue", "cond_inner", "l2",
                     "l2_body", "l2_end", "l2_continue", "l2_exit"},
                    {"", "l1_to_end", "l1_end", "end"});

  const auto *L2 = L1->Children[0];
  EXPECT_EQ(L2->Children.size(), 0ul);
  EXPECT_EQ(L2->Entry->getName(), "l2");
  EXPECT_EQ(L2->Exits.size(), 1ul);
  EXPECT_THAT(L2->Exits, ContainsBasicBlock("l2"));
  checkRegionBlocks(L2, {"l2", "l2_body", "l2_continue"},
                    {"", "l1_to_end", "l1_end", "end", "l1", "l1_body",
                     "l1_continue", "cond_inner", "l2_end", "l2_exit"});
}
