//===- Region.cpp - Region class ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Region wrapper to simplify using TableGen Record defining a MLIR Region.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Region.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;
using namespace mlir::tblgen;

// Returns true if this region is variadic.
bool Region::isVariadic() const { return def->isSubClassOf("VariadicRegion"); }
