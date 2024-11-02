//===- DLTI.h - Data Layout and Target Info MLIR Dialect --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the dialect containing the objects pertaining to target information.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_DLTI_H
#define MLIR_DIALECT_DLTI_DLTI_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace mlir {
namespace detail {
class DataLayoutEntryAttrStorage;
} // namespace detail
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/DLTI/DLTIAttrs.h.inc"
#include "mlir/Dialect/DLTI/DLTIDialect.h.inc"

#endif // MLIR_DIALECT_DLTI_DLTI_H
