//===- ShapeToStandard.h - Conversion utils from shape to std dialect -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_SHAPETOSTANDARD_SHAPETOSTANDARD_H_
#define MLIR_CONVERSION_SHAPETOSTANDARD_SHAPETOSTANDARD_H_

#include <memory>

namespace mlir {

class FuncOp;
class ModuleOp;
template <typename T>
class OperationPass;
class OwningRewritePatternList;

void populateShapeToStandardConversionPatterns(
    OwningRewritePatternList &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createConvertShapeToStandardPass();

void populateConvertShapeConstraintsConversionPatterns(
    OwningRewritePatternList &patterns);

std::unique_ptr<OperationPass<FuncOp>> createConvertShapeConstraintsPass();

} // namespace mlir

#endif // MLIR_CONVERSION_SHAPETOSTANDARD_SHAPETOSTANDARD_H_
