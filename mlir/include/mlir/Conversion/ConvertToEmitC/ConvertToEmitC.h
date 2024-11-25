//===- ConvertToEmitC.h - Convert to EmitC Patterns -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOEMITC_CONVERTTOEMITC_H
#define MLIR_CONVERSION_CONVERTTOEMITC_CONVERTTOEMITC_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

void populateConvertToEmitCPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns);

void populateConvertToEmitCTypeConverter(TypeConverter &typeConverter);

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOEMITC_CONVERTTOEMITC_H
