//===- PassDetail.h - Loop Pass class details -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_LOOPOPS_TRANSFORMS_PASSDETAIL_H_
#define DIALECT_LOOPOPS_TRANSFORMS_PASSDETAIL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
// Forward declaration from Dialect.h
template <typename ConcreteDialect>
void registerDialect(DialectRegistry &registry);

class AffineDialect;

#define GEN_PASS_CLASSES
#include "mlir/Dialect/SCF/Passes.h.inc"

} // end namespace mlir

#endif // DIALECT_LOOPOPS_TRANSFORMS_PASSDETAIL_H_
