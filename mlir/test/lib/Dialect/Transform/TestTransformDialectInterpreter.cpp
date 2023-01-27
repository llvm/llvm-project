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

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterPassBase.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
/// Simple pass that applies transform dialect ops directly contained in a
/// module.

template <typename Derived>
class ModulePassWrapper : public PassWrapper<Derived, OperationPass<ModuleOp>> {
};

class TestTransformDialectInterpreterPass
    : public transform::TransformInterpreterPassBase<
          TestTransformDialectInterpreterPass, ModulePassWrapper> {
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

  ArrayRef<transform::MappedValue>
  findOperationsByName(Operation *root, StringRef name,
                       SmallVectorImpl<transform::MappedValue> &storage) {
    size_t start = storage.size();
    root->walk([&](Operation *op) {
      if (op->getName().getStringRef() == name) {
        storage.push_back(op);
      }
    });
    return ArrayRef(storage).drop_front(start);
  }

  ArrayRef<transform::MappedValue>
  createParameterMapping(MLIRContext &context, ArrayRef<int> values,
                         SmallVectorImpl<transform::MappedValue> &storage) {
    size_t start = storage.size();
    llvm::append_range(storage, llvm::map_range(values, [&](int v) {
                         Builder b(&context);
                         return transform::MappedValue(b.getI64IntegerAttr(v));
                       }));
    return ArrayRef(storage).drop_front(start);
  }

  void runOnOperation() override {
    if (!bindFirstExtraToOps.empty() && !bindFirstExtraToParams.empty()) {
      emitError(UnknownLoc::get(&getContext()))
          << "cannot bind the first extra top-level argument to both "
             "operations and parameters";
      return signalPassFailure();
    }
    if (!bindSecondExtraToOps.empty() && !bindSecondExtraToParams.empty()) {
      emitError(UnknownLoc::get(&getContext()))
          << "cannot bind the second extra top-level argument to both "
             "operations and parameters";
      return signalPassFailure();
    }
    if ((!bindSecondExtraToOps.empty() || !bindSecondExtraToParams.empty()) &&
        bindFirstExtraToOps.empty() && bindFirstExtraToParams.empty()) {
      emitError(UnknownLoc::get(&getContext()))
          << "cannot bind the second extra top-level argument without binding "
             "the first";
      return signalPassFailure();
    }

    SmallVector<transform::MappedValue> extraMappingStorage;
    SmallVector<ArrayRef<transform::MappedValue>> extraMapping;
    if (!bindFirstExtraToOps.empty()) {
      extraMapping.push_back(findOperationsByName(
          getOperation(), bindFirstExtraToOps.getValue(), extraMappingStorage));
    } else if (!bindFirstExtraToParams.empty()) {
      extraMapping.push_back(createParameterMapping(
          getContext(), bindFirstExtraToParams, extraMappingStorage));
    }
    if (!bindSecondExtraToOps.empty()) {
      extraMapping.push_back(findOperationsByName(
          getOperation(), bindSecondExtraToOps, extraMappingStorage));
    } else if (!bindSecondExtraToParams.empty()) {
      extraMapping.push_back(createParameterMapping(
          getContext(), bindSecondExtraToParams, extraMappingStorage));
    }

    options = options.enableExpensiveChecks(enableExpensiveChecks);
    if (failed(transform::detail::interpreterBaseRunOnOperationImpl(
            getOperation(), getArgument(), getSharedTransformModule(),
            extraMapping, options, transformFileName, debugPayloadRootTag,
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

  Option<std::string> bindSecondExtraToOps{
      *this, "bind-second-extra-to-ops",
      llvm::cl::desc("bind the second extra argument of the top-level op to "
                     "payload operations of the given kind")};
  ListOption<int> bindSecondExtraToParams{
      *this, "bind-second-extra-to-params",
      llvm::cl::desc("bind the second extra argument of the top-level op to "
                     "the given integer parameters")};
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
