//===- ArithToSMT.h - Arith to SMT dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOSMT_H
#define MLIR_CONVERSION_ARITHTOSMT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class TypeConverter;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARITHTOSMT
#include "mlir/Conversion/Passes.h.inc"

namespace arith {
/// Get the Arith to SMT conversion patterns.
void populateArithToSMTConversionPatterns(TypeConverter &converter,
                                          RewritePatternSet &patterns);
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOSMT_H
