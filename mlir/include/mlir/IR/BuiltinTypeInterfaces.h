//===- BuiltinTypeInterfaces.h - Builtin Type Interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_BUILTINTYPEINTERFACES_H
#define MLIR_IR_BUILTINTYPEINTERFACES_H

#include "mlir/IR/OpAsmSupport.h"
#include "mlir/IR/Types.h"

namespace llvm {
struct fltSemantics;
} // namespace llvm

namespace mlir {
class FloatType;
class MLIRContext;
} // namespace mlir

#include "mlir/IR/BuiltinTypeInterfaces.h.inc"
#include "mlir/IR/OpAsmTypeInterface.h.inc"

#endif // MLIR_IR_BUILTINTYPEINTERFACES_H
