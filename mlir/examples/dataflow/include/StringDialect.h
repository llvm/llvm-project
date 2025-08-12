//===- StringDialect.h - Dialect definition for the String IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the String Dialect for the StringConstantPropagation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_STRING_DIALECT_H_
#define MLIR_TUTORIAL_STRING_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "StringOpsTypes.h.inc"

#include "StringOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "StringOps.h.inc"

#endif // MLIR_TUTORIAL_STRING_DIALECT_H_
