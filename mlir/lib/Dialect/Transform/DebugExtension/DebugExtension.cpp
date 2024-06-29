//===- DebugExtension.cpp - Debug extension for the Transform dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Transform/DebugExtension/DebugExtension.h"

#include "mlir/Dialect/Transform/DebugExtension/DebugExtensionOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {
/// Debug extension of the Transform dialect. This provides operations for
/// debugging transform dialect scripts.
class DebugExtension
    : public transform::TransformDialectExtension<DebugExtension> {
public:
  void init() {
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Transform/DebugExtension/DebugExtensionOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::transform::registerDebugExtension(DialectRegistry &dialectRegistry) {
  dialectRegistry.addExtensions<DebugExtension>();
}
