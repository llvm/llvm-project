//===- InterpreterPass.cpp - Transform dialect interpreter pass -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace mlir;

namespace mlir {
namespace transform {
#define GEN_PASS_DEF_INTERPRETERPASS
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace mlir

/// Returns the payload operation to be used as payload root:
///   - the operation nested under `passRoot` that has the given tag attribute,
///     must be unique;
///   - the `passRoot` itself if the tag is empty.
static Operation *findPayloadRoot(Operation *passRoot, StringRef tag) {
  // Fast return.
  if (tag.empty())
    return passRoot;

  // Walk to do a lookup.
  Operation *target = nullptr;
  auto tagAttrName = StringAttr::get(
      passRoot->getContext(), transform::TransformDialect::kTargetTagAttrName);
  WalkResult walkResult = passRoot->walk([&](Operation *op) {
    auto attr = op->getAttrOfType<StringAttr>(tagAttrName);
    if (!attr || attr.getValue() != tag)
      return WalkResult::advance();

    if (!target) {
      target = op;
      return WalkResult::advance();
    }

    InFlightDiagnostic diag = op->emitError()
                              << "repeated operation with the target tag '"
                              << tag << "'";
    diag.attachNote(target->getLoc()) << "previously seen operation";
    return WalkResult::interrupt();
  });

  return walkResult.wasInterrupted() ? nullptr : target;
}

namespace {
class InterpreterPass
    : public transform::impl::InterpreterPassBase<InterpreterPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp transformModule =
        transform::detail::getPreloadedTransformModule(context);
    Operation *payloadRoot =
        findPayloadRoot(getOperation(), debugPayloadRootTag);
    if (!payloadRoot)
      return signalPassFailure();
    auto debugBindNames = llvm::map_to_vector(
        debugBindTrailingArgs,
        [&](const std::string &name) { return OperationName(name, context); });
    SmallVector<SmallVector<Operation *>, 2> trailingBindings;
    trailingBindings.resize(debugBindNames.size());
    payloadRoot->walk([&](Operation *payload) {
      for (auto &&[position, name] : llvm::enumerate(debugBindNames)) {
        if (payload->getName() == name)
          trailingBindings[position].push_back(payload);
      }
    });

    Operation *transformEntryPoint = transform::detail::findTransformEntryPoint(
        getOperation(), transformModule, entryPoint);
    if (!transformEntryPoint) {
      getOperation()->emitError()
          << "could not find transform entry point: " << entryPoint
          << " in either payload or transform module";
      return signalPassFailure();
    }

    RaggedArray<transform::MappedValue> bindings;
    bindings.push_back(ArrayRef<Operation *>{payloadRoot});
    for (SmallVector<Operation *> &trailing : trailingBindings)
      bindings.push_back(std::move(trailing));

    if (failed(transform::applyTransformNamedSequence(
            bindings,
            cast<transform::TransformOpInterface>(transformEntryPoint),
            transformModule,
            options.enableExpensiveChecks(!disableExpensiveChecks)))) {
      return signalPassFailure();
    }
  }

private:
  /// Transform interpreter options.
  transform::TransformOptions options;
};
} // namespace
