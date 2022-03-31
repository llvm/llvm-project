//===- VectorToROCDL.h - Convert Vector to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_VECTORTOROCDL_VECTORTOROCDL_H_
#define MLIR_CONVERSION_VECTORTOROCDL_VECTORTOROCDL_H_

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class ModuleOp;
template <typename OpT>
class OperationPass;
class RewritePatternSet;

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
void populateVectorToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);

/// Create a pass to convert vector operations to the ROCDL dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertVectorToROCDLPass();

} // namespace mlir
#endif // MLIR_CONVERSION_VECTORTOROCDL_VECTORTOROCDL_H_
