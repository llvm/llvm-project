//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::memref;

//===----------------------------------------------------------------------===//
// MemRefDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct MemRefInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // end anonymous namespace

SmallVector<Value, 4> mlir::getDynOperands(Location loc, Value val,
                                           OpBuilder &b) {
  SmallVector<Value, 4> dynOperands;
  auto shapedType = val.getType().cast<ShapedType>();
  for (auto dim : llvm::enumerate(shapedType.getShape())) {
    if (dim.value() == MemRefType::kDynamicSize)
      dynOperands.push_back(b.create<memref::DimOp>(loc, val, dim.index()));
  }
  return dynOperands;
}

void mlir::memref::MemRefDialect::initialize() {
  addOperations<DmaStartOp, DmaWaitOp,
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
                >();
  addInterfaces<MemRefInlinerInterface>();
}
