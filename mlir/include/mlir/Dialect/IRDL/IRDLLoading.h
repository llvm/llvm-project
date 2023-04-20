//===- IRDLRegistration.h - IRDL registration -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Manages the registration of MLIR objects from IRDL operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IRDLREGISTRATION_H
#define MLIR_DIALECT_IRDL_IRDLREGISTRATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace irdl {

/// Load all the dialects defined in the module.
LogicalResult loadDialects(ModuleOp op);

} // namespace irdl
} // namespace mlir

#endif // MLIR_DIALECT_IRDL_IRDLREGISTRATION_H
