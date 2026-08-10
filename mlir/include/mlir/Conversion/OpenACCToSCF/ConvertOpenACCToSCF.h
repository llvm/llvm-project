//===- ConvertOpenACCToSCF.h - OpenACC conversion pass entrypoint ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_OPENACCTOSCF_CONVERTOPENACCTOSCF_H
#define MLIR_CONVERSION_OPENACCTOSCF_CONVERTOPENACCTOSCF_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTOPENACCTOSCFPASS
#include "mlir/Conversion/Passes.h.inc"

/// Collect the patterns to convert from the OpenACC dialect to OpenACC with
/// SCF dialect.
void populateOpenACCToSCFConversionPatterns(RewritePatternSet &patterns);

} // namespace mlir

#endif // MLIR_CONVERSION_OPENACCTOSCF_CONVERTOPENACCTOSCF_H
