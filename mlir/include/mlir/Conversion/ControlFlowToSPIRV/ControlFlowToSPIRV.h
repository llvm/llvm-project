//===- ControlFlowToSPIRV.h - CF to SPIR-V Patterns --------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides patterns to convert ControlFlow dialect to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONTROLFLOWTOSPIRV_CONTROLFLOWTOSPIRV_H
#define MLIR_CONVERSION_CONTROLFLOWTOSPIRV_CONTROLFLOWTOSPIRV_H

namespace mlir {
class RewritePatternSet;
class SPIRVTypeConverter;

namespace cf {
/// Appends to a pattern list additional patterns for translating ControlFLow
/// ops to SPIR-V ops.
void populateControlFlowToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                        RewritePatternSet &patterns);
} // namespace cf
} // namespace mlir

#endif // MLIR_CONVERSION_CONTROLFLOWTOSPIRV_CONTROLFLOWTOSPIRV_H
