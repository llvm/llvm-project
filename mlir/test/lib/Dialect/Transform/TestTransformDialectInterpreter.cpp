//===- TestTransformDialectInterpreter.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a test pass that interprets Transform dialect operations in
// the module.
//
//===----------------------------------------------------------------------===//

#include "TestTransformDialectExtension.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Simple pass that applies transform dialect ops directly contained in a
/// module.

template <typename Derived>
class OpPassWrapper : public PassWrapper<Derived, OperationPass<>> {};

class TestTransformDialectInterpreterPass
    : public transform::TransformInterpreterPassBase<
          TestTransformDialectInterpreterPass, OpPassWrapper> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformDialectInterpreterPass)

  TestTransformDialectInterpreterPass() = default;
  TestTransformDialectInterpreterPass(
      const TestTransformDialectInterpreterPass &pass)
      : TransformInterpreterPassBase(pass) {}

  StringRef getArgument() const override {
    return "test-transform-dialect-interpreter";
  }

  StringRef getDescription() const override {
    return "apply transform dialect operations one by one";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<transform::TransformDialect>();
  }

  void findOperationsByName(Operation *root, StringRef name,
                            SmallVectorImpl<Operation *> &operations) {
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() == name) {
        operations.push_back(op);
      }
    });
  }

  void createParameterMapping(MLIRContext &context, ArrayRef<int> values,
                              RaggedArray<transform::MappedValue> &result) {
    SmallVector<transform::MappedValue> storage =
        llvm::to_vector(llvm::map_range(values, [&](int v) {
          Builder b(&context);
          return transform::MappedValue(b.getI64IntegerAttr(v));
        }));
    result.push_back(std::move(storage));
  }

  void
  createOpResultMapping(Operation *root, StringRef name,
                        RaggedArray<transform::MappedValue> &extraMapping) {
    SmallVector<Operation *> operations;
    findOperationsByName(root, name, operations);
    SmallVector<Value> results;
    for (Operation *op : operations)
      llvm::append_range(results, op->getResults());
    extraMapping.push_back(results);
  }

  unsigned numberOfSetOptions(const Option<std::string> &ops,
                              const ListOption<int> &params,
                              const Option<std::string> &values) {
    unsigned numSetValues = 0;
    numSetValues += !ops.empty();
    numSetValues += !params.empty();
    numSetValues += !values.empty();
    return numSetValues;
  }

  std::optional<LogicalResult> constructTransformModule(OpBuilder &builder,
                                                        Location loc) {
    if (!testModuleGeneration)
      return std::nullopt;

    builder.create<transform::SequenceOp>(
        loc, TypeRange(), transform::FailurePropagationMode::Propagate,
        builder.getType<transform::AnyOpType>(),
        [](OpBuilder &b, Location nested, Value rootH) {
          b.create<mlir::test::TestPrintRemarkAtOperandOp>(
              nested, rootH, "remark from generated");
          b.create<transform::YieldOp>(nested, ValueRange());
        });
    return success();
  }

  void runOnOperation() override {
    unsigned firstSetOptions =
        numberOfSetOptions(bindFirstExtraToOps, bindFirstExtraToParams,
                           bindFirstExtraToResultsOfOps);
    unsigned secondSetOptions =
        numberOfSetOptions(bindSecondExtraToOps, bindSecondExtraToParams,
                           bindSecondExtraToResultsOfOps);
    auto loc = UnknownLoc::get(&getContext());
    if (firstSetOptions > 1) {
      emitError(loc) << "cannot bind the first extra top-level argument to "
                        "multiple entities";
      return signalPassFailure();
    }
    if (secondSetOptions > 1) {
      emitError(loc) << "cannot bind the second extra top-level argument to "
                        "multiple entities";
      return signalPassFailure();
    }
    if (firstSetOptions == 0 && secondSetOptions != 0) {
      emitError(loc) << "cannot bind the second extra top-level argument "
                        "without bindings the first";
    }

    RaggedArray<transform::MappedValue> extraMapping;
    if (!bindFirstExtraToOps.empty()) {
      SmallVector<Operation *> operations;
      findOperationsByName(getOperation(), bindFirstExtraToOps.getValue(),
                           operations);
      extraMapping.push_back(operations);
    } else if (!bindFirstExtraToParams.empty()) {
      createParameterMapping(getContext(), bindFirstExtraToParams,
                             extraMapping);
    } else if (!bindFirstExtraToResultsOfOps.empty()) {
      createOpResultMapping(getOperation(), bindFirstExtraToResultsOfOps,
                            extraMapping);
    }

    if (!bindSecondExtraToOps.empty()) {
      SmallVector<Operation *> operations;
      findOperationsByName(getOperation(), bindSecondExtraToOps, operations);
      extraMapping.push_back(operations);
    } else if (!bindSecondExtraToParams.empty()) {
      createParameterMapping(getContext(), bindSecondExtraToParams,
                             extraMapping);
    } else if (!bindSecondExtraToResultsOfOps.empty()) {
      createOpResultMapping(getOperation(), bindSecondExtraToResultsOfOps,
                            extraMapping);
    }

    options = options.enableExpensiveChecks(enableExpensiveChecks);
    if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
            getOperation(), getArgument(), getSharedTransformModule(),
            getTransformLibraryModule(), extraMapping, options,
            transformFileName, transformLibraryPaths, debugPayloadRootTag,
            debugTransformRootTag, getBinaryName())))
      return signalPassFailure();
  }

  Option<bool> enableExpensiveChecks{
      *this, "enable-expensive-checks", llvm::cl::init(false),
      llvm::cl::desc("perform expensive checks to better report errors in the "
                     "transform IR")};

  Option<std::string> bindFirstExtraToOps{
      *this, "bind-first-extra-to-ops",
      llvm::cl::desc("bind the first extra argument of the top-level op to "
                     "payload operations of the given kind")};
  ListOption<int> bindFirstExtraToParams{
      *this, "bind-first-extra-to-params",
      llvm::cl::desc("bind the first extra argument of the top-level op to "
                     "the given integer parameters")};
  Option<std::string> bindFirstExtraToResultsOfOps{
      *this, "bind-first-extra-to-results-of-ops",
      llvm::cl::desc("bind the first extra argument of the top-level op to "
                     "results of payload operations of the given kind")};

  Option<std::string> bindSecondExtraToOps{
      *this, "bind-second-extra-to-ops",
      llvm::cl::desc("bind the second extra argument of the top-level op to "
                     "payload operations of the given kind")};
  ListOption<int> bindSecondExtraToParams{
      *this, "bind-second-extra-to-params",
      llvm::cl::desc("bind the second extra argument of the top-level op to "
                     "the given integer parameters")};
  Option<std::string> bindSecondExtraToResultsOfOps{
      *this, "bind-second-extra-to-results-of-ops",
      llvm::cl::desc("bind the second extra argument of the top-level op to "
                     "results of payload operations of the given kind")};

  Option<std::string> transformFileName{
      *this, "transform-file-name", llvm::cl::init(""),
      llvm::cl::desc(
          "Optional filename containing a transform dialect specification to "
          "apply. If left empty, the IR is assumed to contain one top-level "
          "transform dialect operation somewhere in the module.")};
  Option<std::string> debugPayloadRootTag{
      *this, "debug-payload-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as payload IR root. If empty select the pass anchor "
          "operation as the payload IR root.")};
  Option<std::string> debugTransformRootTag{
      *this, "debug-transform-root-tag", llvm::cl::init(""),
      llvm::cl::desc(
          "Select the operation with 'transform.target_tag' attribute having "
          "the given value as container IR for top-level transform ops. This "
          "allows user control on what transformation to apply. If empty, "
          "select the container of the top-level transform op.")};
  ListOption<std::string> transformLibraryPaths{
      *this, "transform-library-paths", llvm::cl::ZeroOrMore,
      llvm::cl::desc("Optional paths to files with modules that should be "
                     "merged into the transform module to provide the "
                     "definitions of external named sequences.")};

  Option<bool> testModuleGeneration{
      *this, "test-module-generation", llvm::cl::init(false),
      llvm::cl::desc("test the generation of the transform module during pass "
                     "initialization, overridden by parsing")};
};

struct TestTransformDialectEraseSchedulePass
    : public PassWrapper<TestTransformDialectEraseSchedulePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TestTransformDialectEraseSchedulePass)

  StringRef getArgument() const final {
    return "test-transform-dialect-erase-schedule";
  }

  StringRef getDescription() const final {
    return "erase transform dialect schedule from the IR";
  }

  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (isa<transform::TransformOpInterface>(nestedOp)) {
        nestedOp->erase();
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

namespace mlir {
namespace test {
/// Registers the test pass for erasing transform dialect ops.
void registerTestTransformDialectEraseSchedulePass() {
  PassRegistration<TestTransformDialectEraseSchedulePass> reg;
}
/// Registers the test pass for applying transform dialect ops.
void registerTestTransformDialectInterpreterPass() {
  PassRegistration<TestTransformDialectInterpreterPass> reg;
}
} // namespace test
} // namespace mlir
