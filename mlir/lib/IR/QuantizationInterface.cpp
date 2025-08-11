//===- QuantizationInterface.cpp
//------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "mlir/IR/QuantizationInterface.cpp.inc"
