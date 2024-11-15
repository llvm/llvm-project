//===- DecomposeCallGraphTypes.h - CG type decompositions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Conversion patterns for decomposing types along call graph edges. That is,
// decomposing types for calls, returns, and function args.
//
// TODO: Make this handle dialect-defined functions, calls, and returns.
// Currently, the generic interfaces aren't sophisticated enough for the
// types of mutations that we are doing here.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_FUNC_TRANSFORMS_DECOMPOSECALLGRAPHTYPES_H
#define MLIR_DIALECT_FUNC_TRANSFORMS_DECOMPOSECALLGRAPHTYPES_H

#include "mlir/Transforms/DialectConversion.h"
#include <optional>

namespace mlir {

/// Populates the patterns needed to drive the conversion process for
/// decomposing call graph types with the given `TypeConverter`.
void populateDecomposeCallGraphTypesPatterns(MLIRContext *context,
                                             const TypeConverter &typeConverter,
                                             RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_DIALECT_FUNC_TRANSFORMS_DECOMPOSECALLGRAPHTYPES_H
