//===- ControlFlowToLLVM.h - ControlFlow to LLVM -----------*- C++ ------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Define conversions from the ControlFlow dialect to the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONTROLFLOWTOLLVM_CONTROLFLOWTOLLVM_H
#define MLIR_CONVERSION_CONTROLFLOWTOLLVM_CONTROLFLOWTOLLVM_H

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class Pass;

namespace cf {
/// Collect the patterns to convert from the ControlFlow dialect to LLVM. The
/// conversion patterns capture the LLVMTypeConverter by reference meaning the
/// references have to remain alive during the entire pattern lifetime.
void populateControlFlowToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns);

/// Creates a pass to convert the ControlFlow dialect into the LLVMIR dialect.
std::unique_ptr<Pass> createConvertControlFlowToLLVMPass();
} // namespace cf
} // namespace mlir

#endif // MLIR_CONVERSION_CONTROLFLOWTOLLVM_CONTROLFLOWTOLLVM_H
