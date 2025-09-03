//===- MemRefAttrs.h - MLIR MemRef IR dialect attributes --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MemRef dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MEMREF_IR_MEMREFATTRS_H_
#define MLIR_DIALECT_MEMREF_IR_MEMREFATTRS_H_

#include "mlir/IR/OpImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/MemRef/IR/MemRefAttrs.h.inc"

#endif // MLIR_DIALECT_MEMREF_IR_MEMREFATTRS_H_
