//===- ArithToSPIRV.h - Convert Arith to SPIRV dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHTOSPIRV_ARITHTOSPIRV_H
#define MLIR_CONVERSION_ARITHTOSPIRV_ARITHTOSPIRV_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {

class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOSPIRV
#include "mlir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

std::unique_ptr<OperationPass<>> createConvertArithToSPIRVPass();
} // namespace arith
} // namespace mlir

#endif // MLIR_CONVERSION_ARITHTOSPIRV_ARITHTOSPIRV_H
