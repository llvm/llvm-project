//===- AnalysisManagerTest.cpp - AnalysisManager unit tests ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
/// Minimal class definitions for two analyses.
struct MyAnalysis {
  MyAnalysis(Operation *) {}
};
struct OtherAnalysis {
  OtherAnalysis(Operation *) {}
};
struct OpSpecificAnalysis {
  OpSpecificAnalysis(ModuleOp) {}
};

TEST(AnalysisManagerTest, FineGrainModuleAnalysisPreservation) {
  MLIRContext context;

  // Test fine grain invalidation of the module analysis manager.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  // Query two different analyses, but only preserve one before invalidating.
  am.getAnalysis<MyAnalysis>();
  am.getAnalysis<OtherAnalysis>();

  detail::PreservedAnalyses pa;
  pa.preserve<MyAnalysis>();
  am.invalidate(pa);

  // Check that only MyAnalysis is preserved.
  EXPECT_TRUE(am.getCachedAnalysis<MyAnalysis>().hasValue());
  EXPECT_FALSE(am.getCachedAnalysis<OtherAnalysis>().hasValue());
}

TEST(AnalysisManagerTest, FineGrainFunctionAnalysisPreservation) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);

  // Create a function and a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  FuncOp func1 =
      FuncOp::create(builder.getUnknownLoc(), "foo",
                     builder.getFunctionType(llvm::None, llvm::None));
  func1.setPrivate();
  module->push_back(func1);

  // Test fine grain invalidation of the function analysis manager.
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;
  AnalysisManager fam = am.nest(func1);

  // Query two different analyses, but only preserve one before invalidating.
  fam.getAnalysis<MyAnalysis>();
  fam.getAnalysis<OtherAnalysis>();

  detail::PreservedAnalyses pa;
  pa.preserve<MyAnalysis>();
  fam.invalidate(pa);

  // Check that only MyAnalysis is preserved.
  EXPECT_TRUE(fam.getCachedAnalysis<MyAnalysis>().hasValue());
  EXPECT_FALSE(fam.getCachedAnalysis<OtherAnalysis>().hasValue());
}

TEST(AnalysisManagerTest, FineGrainChildFunctionAnalysisPreservation) {
  MLIRContext context;
  context.loadDialect<func::FuncDialect>();
  Builder builder(&context);

  // Create a function and a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  FuncOp func1 =
      FuncOp::create(builder.getUnknownLoc(), "foo",
                     builder.getFunctionType(llvm::None, llvm::None));
  func1.setPrivate();
  module->push_back(func1);

  // Test fine grain invalidation of a function analysis from within a module
  // analysis manager.
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  // Check that the analysis cache is initially empty.
  EXPECT_FALSE(am.getCachedChildAnalysis<MyAnalysis>(func1).hasValue());

  // Query two different analyses, but only preserve one before invalidating.
  am.getChildAnalysis<MyAnalysis>(func1);
  am.getChildAnalysis<OtherAnalysis>(func1);

  detail::PreservedAnalyses pa;
  pa.preserve<MyAnalysis>();
  am.invalidate(pa);

  // Check that only MyAnalysis is preserved.
  EXPECT_TRUE(am.getCachedChildAnalysis<MyAnalysis>(func1).hasValue());
  EXPECT_FALSE(am.getCachedChildAnalysis<OtherAnalysis>(func1).hasValue());
}

/// Test analyses with custom invalidation logic.
struct TestAnalysisSet {};

struct CustomInvalidatingAnalysis {
  CustomInvalidatingAnalysis(Operation *) {}

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<TestAnalysisSet>();
  }
};

TEST(AnalysisManagerTest, CustomInvalidation) {
  MLIRContext context;
  Builder builder(&context);

  // Create a function and a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  detail::PreservedAnalyses pa;

  // Check that the analysis is invalidated properly.
  am.getAnalysis<CustomInvalidatingAnalysis>();
  am.invalidate(pa);
  EXPECT_FALSE(am.getCachedAnalysis<CustomInvalidatingAnalysis>().hasValue());

  // Check that the analysis is preserved properly.
  am.getAnalysis<CustomInvalidatingAnalysis>();
  pa.preserve<TestAnalysisSet>();
  am.invalidate(pa);
  EXPECT_TRUE(am.getCachedAnalysis<CustomInvalidatingAnalysis>().hasValue());
}

TEST(AnalysisManagerTest, OpSpecificAnalysis) {
  MLIRContext context;

  // Create a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  // Query the op specific analysis for the module and verify that its cached.
  am.getAnalysis<OpSpecificAnalysis, ModuleOp>();
  EXPECT_TRUE(am.getCachedAnalysis<OpSpecificAnalysis>().hasValue());
}

struct AnalysisWithDependency {
  AnalysisWithDependency(Operation *, AnalysisManager &am) {
    am.getAnalysis<MyAnalysis>();
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<AnalysisWithDependency>() ||
           !pa.isPreserved<MyAnalysis>();
  }
};

TEST(AnalysisManagerTest, DependentAnalysis) {
  MLIRContext context;

  // Create a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  am.getAnalysis<AnalysisWithDependency>();
  EXPECT_TRUE(am.getCachedAnalysis<AnalysisWithDependency>().hasValue());
  EXPECT_TRUE(am.getCachedAnalysis<MyAnalysis>().hasValue());

  detail::PreservedAnalyses pa;
  pa.preserve<AnalysisWithDependency>();
  am.invalidate(pa);

  EXPECT_FALSE(am.getCachedAnalysis<AnalysisWithDependency>().hasValue());
  EXPECT_FALSE(am.getCachedAnalysis<MyAnalysis>().hasValue());
}

struct AnalysisWithNestedDependency {
  AnalysisWithNestedDependency(Operation *, AnalysisManager &am) {
    am.getAnalysis<AnalysisWithDependency>();
  }

  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<AnalysisWithNestedDependency>() ||
           !pa.isPreserved<AnalysisWithDependency>();
  }
};

TEST(AnalysisManagerTest, NestedDependentAnalysis) {
  MLIRContext context;

  // Create a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  am.getAnalysis<AnalysisWithNestedDependency>();
  EXPECT_TRUE(am.getCachedAnalysis<AnalysisWithNestedDependency>().hasValue());
  EXPECT_TRUE(am.getCachedAnalysis<AnalysisWithDependency>().hasValue());
  EXPECT_TRUE(am.getCachedAnalysis<MyAnalysis>().hasValue());

  detail::PreservedAnalyses pa;
  pa.preserve<AnalysisWithDependency>();
  pa.preserve<AnalysisWithNestedDependency>();
  am.invalidate(pa);

  EXPECT_FALSE(am.getCachedAnalysis<AnalysisWithNestedDependency>().hasValue());
  EXPECT_FALSE(am.getCachedAnalysis<AnalysisWithDependency>().hasValue());
  EXPECT_FALSE(am.getCachedAnalysis<MyAnalysis>().hasValue());
}

struct AnalysisWith2Ctors {
  AnalysisWith2Ctors(Operation *) { ctor1called = true; }

  AnalysisWith2Ctors(Operation *, AnalysisManager &) { ctor2called = true; }

  bool ctor1called = false;
  bool ctor2called = false;
};

TEST(AnalysisManagerTest, DependentAnalysis2Ctors) {
  MLIRContext context;

  // Create a module.
  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  ModuleAnalysisManager mam(*module, /*passInstrumentor=*/nullptr);
  AnalysisManager am = mam;

  auto &an = am.getAnalysis<AnalysisWith2Ctors>();
  EXPECT_FALSE(an.ctor1called);
  EXPECT_TRUE(an.ctor2called);
}

} // namespace
