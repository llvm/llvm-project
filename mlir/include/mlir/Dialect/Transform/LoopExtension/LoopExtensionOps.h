//===- LoopExtensionOps.h - Loop ext. for Transform dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRANSFORM_LOOPEXTENSION_LOOPEXTENSIONOPS_H
#define MLIR_DIALECT_TRANSFORM_LOOPEXTENSION_LOOPEXTENSIONOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.h.inc"

#endif // MLIR_DIALECT_TRANSFORM_LOOPEXTENSION_LOOPEXTENSIONOPS_H
