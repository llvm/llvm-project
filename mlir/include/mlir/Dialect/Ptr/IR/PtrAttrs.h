//===- PtrAttrs.h - Pointer dialect attributes ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the Ptr dialect attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTR_IR_PTRATTRS_H
#define MLIR_DIALECT_PTR_IR_PTRATTRS_H

#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace ptr {
/// Base class for Ptr attributes participating in the TBAA graph.
class TBAANodeAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);

  /// Required by DenseMapInfo to create empty and tombstone key.
  static TBAANodeAttr getFromOpaquePointer(const void *pointer) {
    return TBAANodeAttr(reinterpret_cast<const ImplType *>(pointer));
  }
};
} // namespace ptr
} // namespace mlir

#include "mlir/Dialect/Ptr/IR/PtrOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Ptr/IR/PtrOpsAttrs.h.inc"

#endif // MLIR_DIALECT_PTR_IR_PTRATTRS_H
