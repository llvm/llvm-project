//===-- AIRDialect.h - MLIR AIR target definitions --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_AIR_IR_AIRDIALECT_H_
#define MLIR_DIALECT_AIR_IR_AIRDIALECT_H_

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/AIR/IR/AIROpsAttributes.h.inc"

#include "mlir/Dialect/AIR/IR/AIROpsDialect.h.inc"

#endif // MLIR_DIALECT_AIR_IR_AIRDIALECT_H_
