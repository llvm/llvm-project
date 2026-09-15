//===- NVVMOps.h - NVVM operation implementation helpers -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_LLVMIR_IR_NVVMOPS_H
#define MLIR_LIB_DIALECT_LLVMIR_IR_NVVMOPS_H

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

#include <optional>

namespace mlir::NVVM {

mlir::ParseResult parseCTAGroup(mlir::OpAsmParser &parser,
                                mlir::NVVM::CTAGroupKindAttr &groupAttr);
void printCTAGroup(mlir::OpAsmPrinter &printer, mlir::Operation *,
                   mlir::NVVM::CTAGroupKindAttr groupAttr);

void nvvmInferResultRanges(std::optional<mlir::LLVM::ConstantRangeAttr> range,
                           mlir::Value result,
                           mlir::ArrayRef<mlir::ConstantIntRanges> argRanges,
                           mlir::SetIntRangeFn setResultRanges);
mlir::LogicalResult
verifyConstantRangeAttr(mlir::Operation *op,
                        std::optional<mlir::LLVM::ConstantRangeAttr> rangeAttr);

} // namespace mlir::NVVM

#endif // MLIR_LIB_DIALECT_LLVMIR_IR_NVVMOPS_H
