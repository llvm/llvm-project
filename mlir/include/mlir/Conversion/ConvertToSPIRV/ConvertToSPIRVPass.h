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

namespace mlir {
class Pass;

#define GEN_PASS_DECL_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRVPASS_H
