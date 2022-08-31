//===- ArithmeticToSPIRV.h - Convert Arith to SPIRV dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHMETICTOSPIRV_ARITHMETICTOSPIRV_H
#define MLIR_CONVERSION_ARITHMETICTOSPIRV_ARITHMETICTOSPIRV_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHMETICTOSPIRV
#include "mlir/Conversion/Passes.h.inc"

namespace arith {
void populateArithmeticToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

std::unique_ptr<OperationPass<>> createConvertArithmeticToSPIRVPass();
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHMETICTOSPIRV_ARITHMETICTOSPIRV_H
