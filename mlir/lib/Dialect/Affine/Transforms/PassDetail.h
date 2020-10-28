//===- PassDetail.h - Affine Pass class details -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_AFFINE_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_AFFINE_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

namespace linalg {
class LinalgDialect;
} // end namespace linalg
namespace vector {
class VectorDialect;
} // end namespace vector

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Affine/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_AFFINE_TRANSFORMS_PASSDETAIL_H_
