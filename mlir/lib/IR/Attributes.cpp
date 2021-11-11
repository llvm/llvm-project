//===- Attributes.cpp - MLIR Affine Expr Classes --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// Attribute
//===----------------------------------------------------------------------===//

/// Return the context this attribute belongs to.
MLIRContext *Attribute::getContext() const { return getDialect().getContext(); }

//===----------------------------------------------------------------------===//
// NamedAttribute
//===----------------------------------------------------------------------===//

bool mlir::operator<(const NamedAttribute &lhs, const NamedAttribute &rhs) {
  return lhs.first.compare(rhs.first) < 0;
}
bool mlir::operator<(const NamedAttribute &lhs, StringRef rhs) {
  return lhs.first.getValue().compare(rhs) < 0;
}
