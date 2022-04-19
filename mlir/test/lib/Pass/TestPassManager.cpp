//===- TestPassManager.cpp - Test pass manager functionality --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace {
struct TestModulePass
    : public PassWrapper<TestModulePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestModulePass)

  void runOnOperation() final {}
  StringRef getArgument() const final { return "test-module-pass"; }
  StringRef getDescription() const final {
    return "Test a module pass in the pass manager";
  }
};
struct TestFunctionPass
    : public PassWrapper<TestFunctionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFunctionPass)

  void runOnOperation() final {}
  StringRef getArgument() const final { return "test-function-pass"; }
  StringRef getDescription() const final {
    return "Test a function pass in the pass manager";
  }
};
struct TestInterfacePass
    : public PassWrapper<TestInterfacePass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestInterfacePass)

  void runOnOperation() final {
    getOperation()->emitRemark() << "Executing interface pass on operation";
  }
  StringRef getArgument() const final { return "test-interface-pass"; }
  StringRef getDescription() const final {
    return "Test an interface pass (running on FunctionOpInterface) in the "
           "pass manager";
  }
};
struct TestOptionsPass
    : public PassWrapper<TestOptionsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOptionsPass)

  struct Options : public PassPipelineOptions<Options> {
    ListOption<int> listOption{*this, "list",
                               llvm::cl::desc("Example list option")};
    ListOption<std::string> stringListOption{
        *this, "string-list", llvm::cl::desc("Example string list option")};
    Option<std::string> stringOption{*this, "string",
                                     llvm::cl::desc("Example string option")};
  };
  TestOptionsPass() = default;
  TestOptionsPass(const TestOptionsPass &) {}
  TestOptionsPass(const Options &options) {
    listOption = options.listOption;
    stringOption = options.stringOption;
    stringListOption = options.stringListOption;
  }

  void runOnOperation() final {}
  StringRef getArgument() const final { return "test-options-pass"; }
  StringRef getDescription() const final {
    return "Test options parsing capabilities";
  }

  ListOption<int> listOption{*this, "list",
                             llvm::cl::desc("Example list option")};
  ListOption<std::string> stringListOption{
      *this, "string-list", llvm::cl::desc("Example string list option")};
  Option<std::string> stringOption{*this, "string",
                                   llvm::cl::desc("Example string option")};
};

/// A test pass that always aborts to enable testing the crash recovery
/// mechanism of the pass manager.
struct TestCrashRecoveryPass
    : public PassWrapper<TestCrashRecoveryPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestCrashRecoveryPass)

  void runOnOperation() final { abort(); }
  StringRef getArgument() const final { return "test-pass-crash"; }
  StringRef getDescription() const final {
    return "Test a pass in the pass manager that always crashes";
  }
};

/// A test pass that always fails to enable testing the failure recovery
/// mechanisms of the pass manager.
struct TestFailurePass : public PassWrapper<TestFailurePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestFailurePass)

  void runOnOperation() final { signalPassFailure(); }
  StringRef getArgument() const final { return "test-pass-failure"; }
  StringRef getDescription() const final {
    return "Test a pass in the pass manager that always fails";
  }
};

/// A test pass that creates an invalid operation in a function body.
struct TestInvalidIRPass
    : public PassWrapper<TestInvalidIRPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestInvalidIRPass)

  TestInvalidIRPass() = default;
  TestInvalidIRPass(const TestInvalidIRPass &other) {}

  StringRef getArgument() const final { return "test-pass-create-invalid-ir"; }
  StringRef getDescription() const final {
    return "Test pass that adds an invalid operation in a function body";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<test::TestDialect>();
  }
  void runOnOperation() final {
    if (signalFailure)
      signalPassFailure();
    if (!emitInvalidIR)
      return;
    OpBuilder b(getOperation().getBody());
    OperationState state(b.getUnknownLoc(), "test.any_attr_of_i32_str");
    b.create(state);
  }
  Option<bool> signalFailure{*this, "signal-pass-failure",
                             llvm::cl::desc("Trigger a pass failure")};
  Option<bool> emitInvalidIR{*this, "emit-invalid-ir", llvm::cl::init(true),
                             llvm::cl::desc("Emit invalid IR")};
};

/// A test pass that always fails to enable testing the failure recovery
/// mechanisms of the pass manager.
struct TestInvalidParentPass
    : public PassWrapper<TestInvalidParentPass,
                         InterfacePass<FunctionOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestInvalidParentPass)

  StringRef getArgument() const final { return "test-pass-invalid-parent"; }
  StringRef getDescription() const final {
    return "Test a pass in the pass manager that makes the parent operation "
           "invalid";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<test::TestDialect>();
  }
  void runOnOperation() final {
    FunctionOpInterface op = getOperation();
    OpBuilder b(getOperation().getBody());
    b.create<test::TestCallOp>(op.getLoc(), TypeRange(), "some_unknown_func",
                               ValueRange());
  }
};

/// A test pass that contains a statistic.
struct TestStatisticPass
    : public PassWrapper<TestStatisticPass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestStatisticPass)

  TestStatisticPass() = default;
  TestStatisticPass(const TestStatisticPass &) {}
  StringRef getArgument() const final { return "test-stats-pass"; }
  StringRef getDescription() const final { return "Test pass statistics"; }

  Statistic opCount{this, "num-ops", "Number of operations counted"};

  void runOnOperation() final {
    getOperation()->walk([&](Operation *) { ++opCount; });
  }
};
} // namespace

static void testNestedPipeline(OpPassManager &pm) {
  // Nest a module pipeline that contains:
  /// A module pass.
  auto &modulePM = pm.nest<ModuleOp>();
  modulePM.addPass(std::make_unique<TestModulePass>());
  /// A nested function pass.
  auto &nestedFunctionPM = modulePM.nest<func::FuncOp>();
  nestedFunctionPM.addPass(std::make_unique<TestFunctionPass>());

  // Nest a function pipeline that contains a single pass.
  auto &functionPM = pm.nest<func::FuncOp>();
  functionPM.addPass(std::make_unique<TestFunctionPass>());
}

static void testNestedPipelineTextual(OpPassManager &pm) {
  (void)parsePassPipeline("test-pm-nested-pipeline", pm);
}

namespace mlir {
void registerPassManagerTestPass() {
  PassRegistration<TestOptionsPass>();

  PassRegistration<TestModulePass>();

  PassRegistration<TestFunctionPass>();

  PassRegistration<TestInterfacePass>();

  PassRegistration<TestCrashRecoveryPass>();
  PassRegistration<TestFailurePass>();
  PassRegistration<TestInvalidIRPass>();
  PassRegistration<TestInvalidParentPass>();

  PassRegistration<TestStatisticPass>();

  PassPipelineRegistration<>("test-pm-nested-pipeline",
                             "Test a nested pipeline in the pass manager",
                             testNestedPipeline);
  PassPipelineRegistration<>("test-textual-pm-nested-pipeline",
                             "Test a nested pipeline in the pass manager",
                             testNestedPipelineTextual);
  PassPipelineRegistration<>(
      "test-dump-pipeline",
      "Dumps the pipeline build so far for debugging purposes",
      [](OpPassManager &pm) {
        pm.printAsTextualPipeline(llvm::errs());
        llvm::errs() << "\n";
      });

  PassPipelineRegistration<TestOptionsPass::Options>
      registerOptionsPassPipeline(
          "test-options-pass-pipeline",
          "Parses options using pass pipeline registration",
          [](OpPassManager &pm, const TestOptionsPass::Options &options) {
            pm.addPass(std::make_unique<TestOptionsPass>(options));
          });
}
} // namespace mlir
