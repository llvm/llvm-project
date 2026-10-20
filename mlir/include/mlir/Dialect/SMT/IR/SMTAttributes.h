//===- SMTAttributes.h - Declare SMT dialect attributes ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTATTRIBUTES_H
#define MLIR_DIALECT_SMT_IR_SMTATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace smt {
namespace detail {

struct BitVectorAttrStorage;

} // namespace detail
} // namespace smt
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/SMT/IR/SMTAttributes.h.inc"

#endif // MLIR_DIALECT_SMT_IR_SMTATTRIBUTES_H
