//===- TypeConverter.h - Convert builtin to EmitC dialect types -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides a type converter configuration for converting common builtin types
// to the EmitC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_EMITCCOMMON_TYPECONVERTER_H
#define MLIR_CONVERSION_EMITCCOMMON_TYPECONVERTER_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

/// Conversion from common builtin types to the EmitC dialect.
class EmitCTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  explicit EmitCTypeConverter(MLIRContext *ctx);
};

} // namespace mlir

#endif // MLIR_CONVERSION_EMITCCOMMON_TYPECONVERTER_H
