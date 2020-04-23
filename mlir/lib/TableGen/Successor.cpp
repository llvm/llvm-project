//===- Successor.cpp - Successor class ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Successor wrapper to simplify using TableGen Record defining a MLIR
// Successor.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Successor.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

// Returns true if this successor is variadic.
bool Successor::isVariadic() const {
  return def->isSubClassOf("VariadicSuccessor");
}
