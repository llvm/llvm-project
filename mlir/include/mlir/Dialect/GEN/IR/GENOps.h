//===--- GENOps.h - GEN Dialect Operations ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GEN_IR_GENOPS_H
#define MLIR_DIALECT_GEN_IR_GENOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/GEN/IR/GENOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/GEN/IR/GENOpsAttrDefs.h.inc"

#include "mlir/Dialect/GEN/IR/GENTraits.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/GEN/IR/GENOps.h.inc"

#endif // MLIR_DIALECT_GEN_IR_GENOPS_H
