//===- ShapeToSCF.h - Conversion utils from Shape to SCF dialect ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SHAPETOSCF_SHAPETOSCF_H_
#define MLIR_CONVERSION_SHAPETOSCF_SHAPETOSCF_H_

#include <memory>

namespace mlir {

class MLIRContext;
class FunctionPass;
class OwningRewritePatternList;

void populateShapeToSCFConversionPatterns(OwningRewritePatternList &patterns,
                                          MLIRContext *ctx);

std::unique_ptr<FunctionPass> createConvertShapeToSCFPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SHAPETOSCF_SHAPETOSCF_H_
