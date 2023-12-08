//===- DerivedAttributeOpInterface.cpp -- Derived Attribute interfaces ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for derived attribute op interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/DerivedAttributeOpInterface.h"

using namespace mlir;

namespace mlir {
#include "mlir/Interfaces/DerivedAttributeOpInterface.cpp.inc"
} // namespace mlir
