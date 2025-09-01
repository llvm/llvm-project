//===-- MemorySpaceInterfaces.h - ptr memory space interfaces ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the ptr dialect memory space interfaces.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H
#define MLIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
class Operation;
namespace ptr {
enum class AtomicBinOp : uint32_t;
enum class AtomicOrdering : uint32_t;
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/MemorySpaceAttrInterfaces.h.inc"

#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h.inc"

#endif // MLIR_DIALECT_PTR_IR_MEMORYSPACEINTERFACES_H
