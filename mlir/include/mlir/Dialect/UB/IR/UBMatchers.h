//===- UBMatchers.h - UB Dialect matchers -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides matchers for the UB dialect, in particular for matching
// poison values and attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_UB_IR_UBMATCHERS_H
#define MLIR_DIALECT_UB_IR_UBMATCHERS_H

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Matchers.h"

namespace mlir::ub {
namespace detail {

/// Matches a poison attribute (any attribute implementing PoisonAttrInterface).
/// Supports matching against both Attribute and Operation* (via constant
/// folding).
struct poison_attr_matcher {
  bool match(Attribute attr) { return isa<PoisonAttrInterface>(attr); }

  bool match(Operation *op) {
    Attribute attr;
    if (!::mlir::detail::constant_op_binder<Attribute>(&attr).match(op))
      return false;
    return match(attr);
  }
};

} // namespace detail

/// Matches a poison constant (any attribute implementing PoisonAttrInterface).
/// Works with `matchPattern` on Value, Operation*, and Attribute.
///
/// Examples:
///   matchPattern(value, ub::m_Poison())   // Matches ub.poison op via Value.
///   matchPattern(op, ub::m_Poison())      // Matches ub.poison op directly.
///   matchPattern(attr, ub::m_Poison())    // Matches PoisonAttr(Interface).
inline detail::poison_attr_matcher m_Poison() {
  return detail::poison_attr_matcher();
}

} // namespace mlir::ub

#endif // MLIR_DIALECT_UB_IR_UBMATCHERS_H
