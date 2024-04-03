//===- LLVMInterfaces.cpp - LLVM Interfaces ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines op interfaces for the LLVM dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::LLVM;

/// Verifies that all elements of `array` are instances of `Attr`.
template <class AttrT>
static LogicalResult isArrayOf(Operation *op, ArrayAttr array) {
  for (Attribute iter : array)
    if (!isa<AttrT>(iter))
      return op->emitOpError("expected op to return array of ")
             << AttrT::getMnemonic() << " attributes";
  return success();
}

SmallVector<Value> mlir::LLVM::MemcpyOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemcpyInlineOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemmoveOp::getAccessedOperands() {
  return {getDst(), getSrc()};
}

SmallVector<Value> mlir::LLVM::MemsetOp::getAccessedOperands() {
  return {getDst()};
}

SmallVector<Value> mlir::LLVM::CallOp::getAccessedOperands() {
  return llvm::to_vector(
      llvm::make_filter_range(getArgOperands(), [](Value arg) {
        return isa<LLVMPointerType>(arg.getType());
      }));
}

#include "mlir/Dialect/LLVMIR/LLVMInterfaces.cpp.inc"
