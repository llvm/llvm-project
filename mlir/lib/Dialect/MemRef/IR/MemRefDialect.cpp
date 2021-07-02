//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::memref;

#include "mlir/Dialect/MemRef/IR/MemRefOpsDialect.cpp.inc"

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
    if (dim.value() == ShapedType::kDynamicSize)
      dynOperands.push_back(createOrFoldDimOp(b, loc, val, dim.index()));
  }
  return dynOperands;
}

// Helper function that creates a memref::DimOp or tensor::DimOp depending on
// the type of `source`.
// TODO: Move helper function out of MemRef dialect.
Value mlir::createOrFoldDimOp(OpBuilder &b, Location loc, Value source,
                              int64_t dim) {
  if (source.getType().isa<UnrankedMemRefType, MemRefType>())
    return b.createOrFold<memref::DimOp>(loc, source, dim);
  if (source.getType().isa<UnrankedTensorType, RankedTensorType>())
    return b.createOrFold<tensor::DimOp>(loc, source, dim);
  llvm_unreachable("Expected MemRefType or TensorType");
}

void mlir::memref::MemRefDialect::initialize() {
  addOperations<DmaStartOp, DmaWaitOp,
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/IR/MemRefOps.cpp.inc"
                >();
  addInterfaces<MemRefInlinerInterface>();
}
