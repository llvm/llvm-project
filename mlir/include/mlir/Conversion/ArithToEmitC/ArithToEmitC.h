//===- ArithToEmitC.h - Arith to EmitC Patterns -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H
#define MLIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

void populateArithToEmitCPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H
