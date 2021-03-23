//===- ArmSVEToLLVM.h - Conversion Patterns from ArmSVE to LLVM -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARMSVETOLLVM_ARMSVETOLLVM_H_
#define MLIR_CONVERSION_ARMSVETOLLVM_ARMSVETOLLVM_H_

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

/// Collect a set of patterns to convert from the ArmSVE dialect to LLVM.
void populateArmSVEToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                            OwningRewritePatternList &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_ARMSVETOLLVM_ARMSVETOLLVM_H_
