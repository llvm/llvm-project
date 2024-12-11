//===- MemRefToEmitC.h - Convert MemRef to EmitC --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H
#define MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

void populateMemRefToEmitCTypeConversion(TypeConverter &typeConverter);

void populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                             const TypeConverter &converter);
} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H
