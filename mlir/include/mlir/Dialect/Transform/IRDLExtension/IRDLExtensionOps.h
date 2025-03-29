//===- IRDLExtensionOps.h - IRDL Transform dialect extension ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS_H
#define MLIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_IRDLEXTENSION_IRDLEXTENSIONOPS_H
