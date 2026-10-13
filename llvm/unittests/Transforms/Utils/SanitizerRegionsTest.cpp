//===- SanitizerRegionsTest.cpp - Unit tests for sanitizer regions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "gtest/gtest.h"
#include <optional>

using namespace llvm;

namespace {

using Issue = SanitizerInstrumentationRegionIssue;

// Every test function loads from %p into values named "in*" for instructions
// expected to be in a region and "out*" for the rest. For malformed regions,
// the expected problem is given by its kind and by the name of the instruction
// it points at ("" for the unnamed marker or return itself).
class SanitizerRegionsTest : public testing::Test {
protected:
  LLVMContext C;
  std::unique_ptr<Module> M;

  Function *parse(StringRef Body) {
    std::string IR = (Twine(R"(
      declare void @llvm.sanitizer.instrumentation.begin()
      declare void @llvm.sanitizer.instrumentation.end()
      declare void @callee()
      define void @f(ptr %p, i1 %c) {
      )") + Body + "}\n")
                         .str();
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M) {
      Err.print("SanitizerRegionsTest", errs());
      return nullptr;
    }
    return M->getFunction("f");
  }

  void expectRegions(StringRef Body, bool ExpectHasRegions = true,
                     std::optional<Issue> ExpectedIssue = std::nullopt,
                     StringRef ExpectedWhere = "") {
    Function *F = parse(Body);
    ASSERT_TRUE(F);
    EXPECT_EQ(hasSanitizerInstrumentationRegions(*F), ExpectHasRegions);
    SmallPtrSet<const Instruction *, 16> InRegion;
    std::optional<SanitizerInstrumentationRegionProblem> Problem =
        computeSanitizerInstrumentationRegions(*F, InRegion);
    ASSERT_EQ(Problem.has_value(), ExpectedIssue.has_value());
    if (Problem) {
      EXPECT_EQ(Problem->Issue, *ExpectedIssue);
      ASSERT_TRUE(Problem->Where);
      EXPECT_EQ(Problem->Where->getName(), ExpectedWhere);
      switch (*ExpectedIssue) {
      case Issue::UnmatchedEnd:
        EXPECT_TRUE(isSanitizerInstrumentationEnd(*Problem->Where));
        break;
      case Issue::NotClosedAtReturn:
        EXPECT_TRUE(isa<ReturnInst>(Problem->Where));
        break;
      case Issue::MergeOutsideRegion:
        EXPECT_FALSE(InRegion.contains(Problem->Where));
        break;
      case Issue::MergeNestedRegions:
        EXPECT_TRUE(InRegion.contains(Problem->Where));
        break;
      case Issue::NumIssues:
        llvm_unreachable("not an issue");
      }
    }
    unsigned Named = 0;
    for (const Instruction &I : instructions(*F)) {
      if (!I.hasName())
        continue;
      ++Named;
      EXPECT_EQ(InRegion.contains(&I), I.getName().starts_with("in"))
          << I.getName().str();
    }
    EXPECT_GT(Named, 0u);
  }
};

TEST_F(SanitizerRegionsTest, NoRegions) {
  expectRegions(R"(
      %out = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/false);
}

TEST_F(SanitizerRegionsTest, Straight) {
  expectRegions(R"(
      %out1 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.begin()
      %in = load i32, ptr %p
      call void @callee()
      call void @llvm.sanitizer.instrumentation.end()
      %out2 = load i32, ptr %p
      ret void
  )");
}

TEST_F(SanitizerRegionsTest, Nested) {
  expectRegions(R"(
      call void @llvm.sanitizer.instrumentation.begin()
      call void @llvm.sanitizer.instrumentation.begin()
      %in1 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      %in2 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      %out = load i32, ptr %p
      ret void
  )");
}

TEST_F(SanitizerRegionsTest, UnmatchedEnd) {
  expectRegions(R"(
      call void @llvm.sanitizer.instrumentation.end()
      %out1 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.begin()
      %in = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      %out2 = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/true, Issue::UnmatchedEnd);
}

TEST_F(SanitizerRegionsTest, NotClosedAtReturn) {
  expectRegions(R"(
      call void @llvm.sanitizer.instrumentation.begin()
      %in = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/true, Issue::NotClosedAtReturn);
}

// Leaving a function without returning with a region open is not reported
// (see findSanitizerInstrumentationRegionProblem() for why).
TEST_F(SanitizerRegionsTest, NotClosedAtUnreachable) {
  expectRegions(R"(
      call void @llvm.sanitizer.instrumentation.begin()
      %in = load i32, ptr %p
      unreachable
  )");
}

// A block that can be reached without passing a begin() is not in a region.
TEST_F(SanitizerRegionsTest, BeginOnOnePath) {
  expectRegions(R"(
    entry:
      br i1 %c, label %then, label %join
    then:
      call void @llvm.sanitizer.instrumentation.begin()
      %in = load i32, ptr %p
      br label %join
    join:
      %out = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/true, Issue::MergeOutsideRegion, "out");
}

// Likewise for a block that can be reached after an end().
TEST_F(SanitizerRegionsTest, EndOnOnePath) {
  expectRegions(R"(
    entry:
      call void @llvm.sanitizer.instrumentation.begin()
      %in1 = load i32, ptr %p
      br i1 %c, label %then, label %join
    then:
      call void @llvm.sanitizer.instrumentation.end()
      %out1 = load i32, ptr %p
      br label %join
    join:
      %out2 = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/true, Issue::MergeOutsideRegion, "out2");
}

// Paths that differ in nesting depth but are all inside a region: the join is
// still in a region, at the smaller depth, and is reported as such.
TEST_F(SanitizerRegionsTest, MergeNestedRegions) {
  expectRegions(R"(
    entry:
      call void @llvm.sanitizer.instrumentation.begin()
      br i1 %c, label %inner, label %join
    inner:
      call void @llvm.sanitizer.instrumentation.begin()
      %in1 = load i32, ptr %p
      br label %join
    join:
      %in2 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      %out = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/true, Issue::MergeNestedRegions, "in2");
}

TEST_F(SanitizerRegionsTest, BalancedOnBothPaths) {
  expectRegions(R"(
    entry:
      call void @llvm.sanitizer.instrumentation.begin()
      br i1 %c, label %a, label %b
    a:
      %in1 = load i32, ptr %p
      br label %join
    b:
      %in2 = load i32, ptr %p
      br label %join
    join:
      %in3 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      %out = load i32, ptr %p
      ret void
  )");
}

// A loop whose body leaves the region on the back edge.
TEST_F(SanitizerRegionsTest, LoopLeavingRegion) {
  expectRegions(R"(
    entry:
      call void @llvm.sanitizer.instrumentation.begin()
      br label %loop
    loop:
      %out1 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      br i1 %c, label %loop, label %exit
    exit:
      %out2 = load i32, ptr %p
      ret void
  )",
                /*ExpectHasRegions=*/true, Issue::MergeOutsideRegion, "out1");
}

// A loop entirely inside a region.
TEST_F(SanitizerRegionsTest, LoopInsideRegion) {
  expectRegions(R"(
    entry:
      call void @llvm.sanitizer.instrumentation.begin()
      br label %loop
    loop:
      %in1 = load i32, ptr %p
      br i1 %c, label %loop, label %exit
    exit:
      %in2 = load i32, ptr %p
      call void @llvm.sanitizer.instrumentation.end()
      ret void
  )");
}

} // namespace
