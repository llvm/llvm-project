//===- XeGPU.h - MLIR dialect for XeGPU -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_XEGPU_IR_XEGPU_H
#define MLIR_DIALECT_XEGPU_IR_XEGPU_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ShapedOpInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace xegpu {
// placeholder
} // namespace xegpu
} // namespace mlir

#include <mlir/Dialect/XeGPU/IR/XeGPUDialect.h.inc>
#include <mlir/Dialect/XeGPU/IR/XeGPUEnums.h.inc>
#define GET_ATTRDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUAttrs.h.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPUTypes.h.inc>
#define GET_OP_CLASSES
#include <mlir/Dialect/XeGPU/IR/XeGPU.h.inc>

#endif // MLIR_DIALECT_XEGPU_IR_XEGPU_H
