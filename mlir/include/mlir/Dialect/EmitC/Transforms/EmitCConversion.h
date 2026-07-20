//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_EMITCCONVERSION_H
#define MLIR_DIALECT_EMITC_TRANSFORMS_EMITCCONVERSION_H

#include "mlir/Conversion/EmitCCommon/TypeConverter.h"
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.h.inc"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
void populateBuiltinModuleToEmitCPatterns(
    const EmitCTypeConverter &typeConverter, RewritePatternSet &patterns);
}

#endif