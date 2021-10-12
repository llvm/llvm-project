//===- ArithmeticToSPIRV.h - Convert Arith to SPIRV dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_ARITHMETICTOSPIRV_ARITHMETICTOSPIRV_H
#define MLIR_CONVERSION_ARITHMETICTOSPIRV_ARITHMETICTOSPIRV_H

#include <memory>

namespace mlir {

class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;

namespace arith {
void populateArithmeticToSPIRVPatterns(SPIRVTypeConverter &typeConverter,
                                       RewritePatternSet &patterns);

std::unique_ptr<Pass> createConvertArithmeticToSPIRVPass();
} // end namespace arith
} // end namespace mlir

#endif // MLIR_CONVERSION_ARITHMETICTOSPIRV_ARITHMETICTOSPIRV_H
