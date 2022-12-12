//=== AffineTransformOps.cpp - Implementation of Affine transformation ops ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class AffineTransformDialectExtension
    : public transform::TransformDialectExtension<
          AffineTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<AffineDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Affine/TransformOps/AffineTransformOps.cpp.inc"

void mlir::affine::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<AffineTransformDialectExtension>();
}
