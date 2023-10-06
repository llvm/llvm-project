//===- PreloadLibraryPass.cpp - Pass to preload a transform library -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"

using namespace mlir;

namespace mlir {
namespace transform {
#define GEN_PASS_DEF_PRELOADLIBRARYPASS
#include "mlir/Dialect/Transform/Transforms/Passes.h.inc"
} // namespace transform
} // namespace mlir

namespace {
class PreloadLibraryPass
    : public transform::impl::PreloadLibraryPassBase<PreloadLibraryPass> {
public:
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    OwningOpRef<ModuleOp> mergedParsedLibraries;
    if (failed(transform::detail::assembleTransformLibraryFromPaths(
            context, transformLibraryPaths, mergedParsedLibraries)))
      return failure();
    // TODO: use a resource blob.
    auto *dialect = context->getOrLoadDialect<transform::TransformDialect>();
    dialect->registerLibraryModule(std::move(mergedParsedLibraries));
    return success();
  }

  void runOnOperation() override {}
};
} // namespace
