//===- ConvertOpenACCToSCF.h - OpenACC conversion pass entrypoint ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OPENACCTOSCF_CONVERTOPENACCTOSCF_H
#define MLIR_CONVERSION_OPENACCTOSCF_CONVERTOPENACCTOSCF_H

#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

/// Collect the patterns to convert from the OpenACC dialect to OpenACC with
/// SCF dialect.
void populateOpenACCToSCFConversionPatterns(RewritePatternSet &patterns);

/// Create a pass to convert the OpenACC dialect into the LLVMIR dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertOpenACCToSCFPass();

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOSCF_CONVERTOPENACCTOSCF_H
