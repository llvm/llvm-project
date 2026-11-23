//===- ROCDLOps.h - ROCDL operation implementation helpers -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_LIB_DIALECT_LLVMIR_IR_ROCDLOPS_H
#define MLIR_LIB_DIALECT_LLVMIR_IR_ROCDLOPS_H

#include "mlir/IR/OpImplementation.h"

namespace mlir::ROCDL {

ParseResult parseCachePolicy(OpAsmParser &parser, Attribute &cachePolicy);
void printCachePolicy(OpAsmPrinter &printer, Operation *,
                      Attribute cachePolicy);

} // namespace mlir::ROCDL

#endif // MLIR_LIB_DIALECT_LLVMIR_IR_ROCDLOPS_H
