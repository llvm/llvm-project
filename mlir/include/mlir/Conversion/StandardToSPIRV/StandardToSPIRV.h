//===- StandardToSPIRV.h - Standard to SPIR-V Patterns --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert Standard dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_STANDARDTOSPIRV_STANDARDTOSPIRV_H
#define MLIR_CONVERSION_STANDARDTOSPIRV_STANDARDTOSPIRV_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class SPIRVTypeConverter;

/// Appends to a pattern list additional patterns for translating standard ops
/// to SPIR-V ops. Also adds the patterns to legalize ops not directly
/// translated to SPIR-V dialect.
void populateStandardToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                     RewritePatternSet &patterns);

/// Appends to a pattern list additional patterns for translating tensor ops
/// to SPIR-V ops.
///
/// Note: Normally tensors will be stored in buffers before converting to
/// SPIR-V, given that is how a large amount of data is sent to the GPU.
/// However, SPIR-V supports converting from tensors directly too. This is
/// for the cases where the tensor just contains a small amount of elements
/// and it makes sense to directly inline them as a small data array in the
/// shader. To handle this, internally the conversion might create new local
/// variables. SPIR-V consumers in GPU drivers may or may not optimize that
/// away. So this has implications over register pressure. Therefore, a
/// threshold is used to control when the patterns should kick in.
void populateTensorToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                   int64_t byteCountThreshold,
                                   RewritePatternSet &patterns);

/// Appends to a pattern list patterns to legalize ops that are not directly
/// lowered to SPIR-V.
void populateStdLegalizationPatternsForSPIRVLowering(
    RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_STANDARDTOSPIRV_STANDARDTOSPIRV_H
