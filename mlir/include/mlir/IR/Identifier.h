//===- Identifier.h - MLIR Identifier Class ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_IDENTIFIER_H
#define MLIR_IR_IDENTIFIER_H

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
/// NOTICE: Identifier is deprecated and usages of it should be replaced with
/// StringAttr.
using Identifier = StringAttr;
} // namespace mlir

#endif
