//===- Argument.cpp - Argument definitions --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Argument.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

bool NamedTypeConstraint::hasPredicate() const {
  return !constraint.getPredicate().isNull();
}

bool NamedTypeConstraint::isOptional() const { return constraint.isOptional(); }

bool NamedTypeConstraint::isVariadic() const { return constraint.isVariadic(); }
