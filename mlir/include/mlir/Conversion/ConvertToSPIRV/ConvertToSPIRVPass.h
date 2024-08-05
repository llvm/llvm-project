//===- ConvertToSPIRVPass.h - Conversion to SPIR-V pass ---*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRVPASS_H
#define MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRVPASS_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
class DialectRegistry;

#define GEN_PASS_DECL_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"

/// Register the extension that will load dependent dialects for SPIR-V
/// conversion. This is useful to implement a pass similar to
/// "convert-to-spirv".
void registerConvertToSPIRVDependentDialectLoading(DialectRegistry &registry);
} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRVPASS_H
