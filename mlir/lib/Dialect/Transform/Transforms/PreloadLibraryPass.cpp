//===- PreloadLibraryPass.cpp - Pass to preload a transform library -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"

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

  void runOnOperation() override {
    OwningOpRef<ModuleOp> mergedParsedLibraries;
    if (failed(transform::detail::assembleTransformLibraryFromPaths(
            &getContext(), transformLibraryPaths, mergedParsedLibraries)))
      return signalPassFailure();
    // TODO: investigate using a resource blob if some ownership mode allows it.
    auto *dialect =
        getContext().getOrLoadDialect<transform::TransformDialect>();
    if (failed(
            dialect->loadIntoLibraryModule(std::move(mergedParsedLibraries))))
      signalPassFailure();
  }
};
} // namespace
