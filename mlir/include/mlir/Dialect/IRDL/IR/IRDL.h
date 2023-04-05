//===- IRDL.h - IR Definition Language dialect ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dialect for the IR Definition Language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_IRDL_IR_IRDL_H_
#define MLIR_DIALECT_IRDL_IR_IRDL_H_

#include "mlir/Dialect/IRDL/IR/IRDLTraits.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <memory>

// Forward declaration.
namespace mlir {
namespace irdl {
class OpDef;
class OpDefAttr;
} // namespace irdl
} // namespace mlir

//===----------------------------------------------------------------------===//
// IRDL Dialect
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/IRDL/IR/IRDLDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLTypesGen.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/IRDL/IR/IRDLOps.h.inc"

#endif // MLIR_DIALECT_IRDL_IR_IRDL_H_
