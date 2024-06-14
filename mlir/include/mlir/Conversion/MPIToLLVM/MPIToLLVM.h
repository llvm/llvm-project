//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MPITOLLVM_H
#define MLIR_CONVERSION_MPITOLLVM_H

#include "mlir/IR/DialectRegistry.h"

namespace mlir {

class LLVMTypeConverter;
class RewritePatternSet;

#define GEN_PASS_DECL_MPITOLLVMCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace mpi {
void populateMPIToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

void registerConvertMPIToLLVMInterface(DialectRegistry &registry);

} // namespace mpi
} // namespace mlir

#endif // MLIR_CONVERSION_MPITOLLVM_H
