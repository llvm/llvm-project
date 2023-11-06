//===- VCIXDialect.h - MLIR Dialect for VCIX --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Target dialect for VCIX in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VCIX_VCIXDIALECT_H_
#define MLIR_DIALECT_VCIX_VCIXDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/VCIX/VCIXAttrs.h"
#include "mlir/Dialect/VCIX/VCIXDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/VCIX/VCIX.h.inc"

#endif // MLIR_DIALECT_VCIX_VCIXDIALECT_H_
