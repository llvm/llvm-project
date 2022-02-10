//===- Context.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/PDLL/AST/Context.h"
#include "TypeDetail.h"

using namespace mlir;
using namespace mlir::pdll::ast;

Context::Context() {
  typeUniquer.registerSingletonStorageType<detail::AttributeTypeStorage>();
  typeUniquer.registerSingletonStorageType<detail::ConstraintTypeStorage>();
  typeUniquer.registerSingletonStorageType<detail::RewriteTypeStorage>();
  typeUniquer.registerSingletonStorageType<detail::TypeTypeStorage>();
  typeUniquer.registerSingletonStorageType<detail::ValueTypeStorage>();

  typeUniquer.registerParametricStorageType<detail::OperationTypeStorage>();
  typeUniquer.registerParametricStorageType<detail::RangeTypeStorage>();
  typeUniquer.registerParametricStorageType<detail::TupleTypeStorage>();
}
