//===- DialectExtension.cpp - Linalg transform dialect extension ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<linalg::LinalgDialect>();

    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();
    declareGeneratedDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"
        >();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::linalg::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
