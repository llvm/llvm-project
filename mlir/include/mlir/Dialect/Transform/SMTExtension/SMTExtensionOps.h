//===- SMTExtensionOps.h - SMT extension for Transform dialect --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSIONOPS_H
#define MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSIONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_SMTEXTENSION_SMTEXTENSIONOPS_H
