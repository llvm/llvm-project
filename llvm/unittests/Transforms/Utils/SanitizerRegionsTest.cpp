//===- SanitizerRegionsTest.cpp - Unit tests for sanitizer regions --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Instrumentation.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace {

// Every test function loads from %p into values named "in*" for instructions
// expected to be in a region and "out*" for the rest.
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

  void expectRegions(StringRef Body, bool ExpectHasRegions = true) {
    Function *F = parse(Body);
    ASSERT_TRUE(F);
    EXPECT_EQ(hasSanitizerInstrumentationRegions(*F), ExpectHasRegions);
    SmallPtrSet<const Instruction *, 16> InRegion;
    computeSanitizerInstrumentationRegions(*F, InRegion);
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
  )");
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
  )");
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
  )");
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
