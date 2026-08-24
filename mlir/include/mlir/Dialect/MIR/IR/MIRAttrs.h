//===- MIRAttrs.h - MIR dialect attributes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MIR_IR_MIRATTRS_H
#define MLIR_DIALECT_MIR_IR_MIRATTRS_H

#include "mlir/IR/Attributes.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MIR/IR/MIROpsAttributes.h.inc"

#endif // MLIR_DIALECT_MIR_IR_MIRATTRS_H
