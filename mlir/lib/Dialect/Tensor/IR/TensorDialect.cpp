//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tensor;

#include "mlir/Dialect/Tensor/IR/TensorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TensorDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct TensorInlinerInterface : public DialectInlinerInterface {
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
} // namespace

//===----------------------------------------------------------------------===//
// TensorDialect Methods
//===----------------------------------------------------------------------===//

void TensorDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Tensor/IR/TensorOps.cpp.inc"
      >();
  addInterfaces<TensorInlinerInterface>();
}
