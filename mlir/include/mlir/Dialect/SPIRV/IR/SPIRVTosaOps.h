//===- SPIRVTosaOps.h - MLIR SPIR-V Tosa operations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_SPIRVTOSAOPS_H_
#define MLIR_DIALECT_SPIRV_IR_SPIRVTOSAOPS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::spirv {

ParseResult parseSPIRV_I32_1DArmTensor(OpAsmParser &parser,
                                       DenseIntElementsAttr &attr);

void printSPIRV_I32_1DArmTensor(OpAsmPrinter &printer, Operation *,
                                DenseIntElementsAttr attr);

} // namespace mlir::spirv

#endif // MLIR_DIALECT_SPIRV_IR_SPIRVTOSAOPS_H_
