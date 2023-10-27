//===- IRDLInterfaces.h - IRDL interfaces definition ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the interfaces used by the IRDL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_
#define MLIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_

#include "mlir/Dialect/IRDL/IRDLVerifiers.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ExtensibleDialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>

namespace mlir {
namespace irdl {
class TypeOp;
class AttributeOp;
} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect Interfaces
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDLInterfaces.h.inc"

#endif //  MLIR_DIALECT_IRDL_IR_IRDLINTERFACES_H_
