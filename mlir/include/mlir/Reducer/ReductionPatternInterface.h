//===- ReducePatternInterface.h - Collecting Reduce Patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_REDUCTIONPATTERNINTERFACE_H
#define MLIR_REDUCER_REDUCTIONPATTERNINTERFACE_H

#include "mlir/IR/DialectInterface.h"
#include "mlir/Reducer/Tester.h"

namespace mlir {
class RewritePatternSet;
} // namespace mlir

#include "mlir/Reducer/DialectReductionPatternInterface.h.inc"

#endif // MLIR_REDUCER_REDUCTIONPATTERNINTERFACE_H
