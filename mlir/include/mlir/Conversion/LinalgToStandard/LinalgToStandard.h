//===- LinalgToStandard.h - Utils to convert from the linalg dialect ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_
#define MLIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;
template <typename T>
class OperationPass;

/// Create a pass to convert Linalg operations to the Standard dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertLinalgToStandardPass();

} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOSTANDARD_LINALGTOSTANDARD_H_
