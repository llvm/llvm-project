//===- ConvertToEmitC.cpp - Convert to EmitC Patterns -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitC.h"

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/FuncToEmitC/FuncToEmitC.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

void mlir::populateConvertToEmitCTypeConverter(TypeConverter &typeConverter) {
  typeConverter.addConversion([](Type type) { return type; });
  populateMemRefToEmitCTypeConversion(typeConverter);
}

/// Populate patterns for each dialect.
void mlir::populateConvertToEmitCPatterns(TypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  populateArithToEmitCPatterns(typeConverter, patterns);
  populateFuncToEmitCPatterns(typeConverter, patterns);
  populateMemRefToEmitCConversionPatterns(patterns, typeConverter);
  populateSCFToEmitCConversionPatterns(patterns);
}
