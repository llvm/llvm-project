//===- MemoryAccessOpInterfaces.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

//===----------------------------------------------------------------------===//
// IndexedAccessOpInterface and IndexedMemCpyOpInterface
//===----------------------------------------------------------------------===//

namespace mlir::memref {
#include "mlir/Dialect/MemRef/IR/MemoryAccessOpInterfaces.cpp.inc"

LogicalResult detail::verifyIndexedAccessOpInterface(Operation *op) {
  auto iface = dyn_cast<IndexedAccessOpInterface>(op);
  if (!iface)
    return failure();

  TypedValue<MemRefType> memref = iface.getAccessedMemref();
  if (!memref) {
    // Some operations can carry tensors, this is fine.
    return success();
  }
  if (memref.getType().getRank() !=
      static_cast<int64_t>(iface.getIndices().size()))
    return op->emitOpError(
               "invalid number of indices for accessed memref, expected ")
           << memref.getType().getRank() << " but got "
           << iface.getIndices().size();
  return success();
}

LogicalResult detail::verifyIndexedMemCopyOpInterface(Operation *op) {
  auto iface = dyn_cast<IndexedMemCopyOpInterface>(op);
  if (!iface)
    return failure();

  TypedValue<MemRefType> src = iface.getSrc();
  TypedValue<MemRefType> dst = iface.getDst();
  if (!src || !dst) {
    // Allow operations to not always have memref arguments.
    return success();
  }
  if (src.getType().getRank() !=
      static_cast<int64_t>(iface.getSrcIndices().size()))
    return op->emitOpError(
        "invalid number of indices for source memref, expected " +
        Twine(src.getType().getRank()) + ", got " +
        Twine(iface.getSrcIndices().size()));
  if (dst.getType().getRank() !=
      static_cast<int64_t>(iface.getDstIndices().size()))
    return op->emitOpError(
               "invalid number of indices for destination memref, expected ")
           << dst.getType().getRank() << ", got "
           << iface.getDstIndices().size();
  return success();
}
} // namespace mlir::memref
