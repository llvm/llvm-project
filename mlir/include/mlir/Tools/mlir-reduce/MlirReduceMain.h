//===- MlirReduceMain.h - MLIR Reducer driver -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRREDUCE_MLIRREDUCEMAIN_H
#define MLIR_TOOLS_MLIRREDUCE_MLIRREDUCEMAIN_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {

class MLIRContext;

LogicalResult mlirReduceMain(int argc, char **argv, MLIRContext &context);

} // namespace mlir

#endif // MLIR_TOOLS_MLIRREDUCE_MLIRREDUCEMAIN_H
