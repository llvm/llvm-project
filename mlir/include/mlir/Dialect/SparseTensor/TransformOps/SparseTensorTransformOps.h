//===- SparseTensorTransformOps.h - sparse tensor transform ops -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPARSETENSOR_TRANSFORMOPS_SPARSETENSORTRANSFORMOPS_H
#define MLIR_DIALECT_SPARSETENSOR_TRANSFORMOPS_SPARSETENSORTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/MatchInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

namespace mlir {
namespace transform {
class TransformHandleTypeInterface;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;

namespace sparse_tensor {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace sparse_tensor
} // namespace mlir

//===----------------------------------------------------------------------===//
// SparseTensor Transform Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.h.inc"

#endif // MLIR_DIALECT_SPARSETENSOR_TRANSFORMOPS_SPARSETENSORTRANSFORMOPS_H
