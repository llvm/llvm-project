//===-- MyExtension.h - Transform dialect tutorial --------------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Transform dialect extension operations used in the
// Chapter 3 of the Transform dialect tutorial.
//
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

namespace mlir {
class CallOpInterface;
namespace func {
class CallOp;
} // namespace func
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "MyExtensionTypes.h.inc"

#define GET_OP_CLASSES
#include "MyExtension.h.inc"

// Registers our Transform dialect extension.
void registerMyExtension(::mlir::DialectRegistry &registry);
