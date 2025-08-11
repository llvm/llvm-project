//===- QuantizationInterface.h - Quantzation Interfaces --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_QuantizationInterface_H
#define MLIR_IR_QuantizationInterface_H

#include "mlir/IR/Types.h"

// Forward declarations for the types we need in the implementation
namespace mlir {
class IntegerType;
} // namespace mlir

#include "mlir/IR/QuantizationInterface.h.inc"

#endif // MLIR_IR_QuantizationInterface_H
