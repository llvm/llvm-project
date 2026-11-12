//===- DLTITransformOps.h - DLTI transform ops ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_TRANSFORMOPS_DLTITRANSFORMOPS_H
#define MLIR_DIALECT_DLTI_TRANSFORMOPS_DLTITRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

namespace mlir {
namespace transform {
class QueryOp;
} // namespace transform
} // namespace mlir

namespace mlir {
class DialectRegistry;

namespace dlti {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace dlti
} // namespace mlir

////===----------------------------------------------------------------------===//
//// DLTI Transform Operations
////===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/DLTI/TransformOps/DLTITransformOps.h.inc"

#endif // MLIR_DIALECT_DLTI_TRANSFORMOPS_DLTITRANSFORMOPS_H
