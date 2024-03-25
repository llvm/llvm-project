//===- LoopExtension.cpp - Loop extension for the Transform dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/LoopExtension/LoopExtension.h"

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {
/// Loop extension of the Transform dialect. This provides "core" transform
/// operations for loop-like ops.
class LoopExtension
    : public transform::TransformDialectExtension<LoopExtension> {
public:
  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::transform::registerLoopExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<LoopExtension>();
}
