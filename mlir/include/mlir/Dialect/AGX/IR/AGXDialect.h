//===- AGXDialect.h - MLIR dialect for Apple GPUs ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a dialect for the functionality of Apple GPUs (AGX),
// in the style of the AMDGPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AGX_IR_AGXDIALECT_H_
#define MLIR_DIALECT_AGX_IR_AGXDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/AGX/IR/AGXDialect.h.inc"

#include "mlir/Dialect/AGX/IR/AGXEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AGX/IR/AGXAttrs.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/AGX/IR/AGX.h.inc"

#endif // MLIR_DIALECT_AGX_IR_AGXDIALECT_H_
