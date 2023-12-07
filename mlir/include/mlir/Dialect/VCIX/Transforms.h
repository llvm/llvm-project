//===- Transforms.h - VCIX Dialect Transformation Entrypoints ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VCIX_TRANSFORMS_H
#define MLIR_DIALECT_VCIX_TRANSFORMS_H

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;

/// Collect a set of patterns to lower VCIX ops to ops that map to LLVM
/// intrinsics.
void populateVCIXLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
                                               RewritePatternSet &patterns);

/// Configure the target to support lowering VCIX ops to ops that map to LLVM
/// intrinsics.
void configureVCIXLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // MLIR_DIALECT_VCIX_TRANSFORMS_H
