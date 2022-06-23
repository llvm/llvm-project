//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::bufferization;

#include "mlir/Dialect/Bufferization/IR/BufferizationOpsDialect.cpp.inc"

/// Attribute name used to mark function arguments who's buffers can be written
/// to during One-Shot Module Bufferize.
constexpr const ::llvm::StringLiteral BufferizationDialect::kWritableAttrName;

/// Attribute name used to mark the bufferization layout for region arguments
/// during One-Shot Module Bufferize.
constexpr const ::llvm::StringLiteral
    BufferizationDialect::kBufferLayoutAttrName;

/// Attribute name used to mark escaping behavior of buffer allocations.
constexpr const ::llvm::StringLiteral BufferizationDialect::kEscapeAttrName;

//===----------------------------------------------------------------------===//
// Bufferization Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
struct BufferizationInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Operations in Bufferization dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool,
                       BlockAndValueMapping &) const final {
    return true;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Bufferization Dialect
//===----------------------------------------------------------------------===//

void mlir::bufferization::BufferizationDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Bufferization/IR/BufferizationOps.cpp.inc"
      >();
  addInterfaces<BufferizationInlinerInterface>();
}

LogicalResult
BufferizationDialect::verifyOperationAttribute(Operation *op,
                                               NamedAttribute attr) {
  using bufferization::BufferizableOpInterface;

  if (attr.getName() == kWritableAttrName) {
    if (!attr.getValue().isa<BoolAttr>()) {
      return op->emitError() << "'" << kWritableAttrName
                             << "' is expected to be a boolean attribute";
    }
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << "expected " << attr.getName()
                             << " to be used on function-like operations";
    return success();
  }
  if (attr.getName() == kBufferLayoutAttrName) {
    if (!attr.getValue().isa<AffineMapAttr>()) {
      return op->emitError() << "'" << kBufferLayoutAttrName
                             << "' is expected to be a affine map attribute";
    }
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << "expected " << attr.getName()
                             << " to be used on function-like operations";
    return success();
  }
  if (attr.getName() == kEscapeAttrName) {
    auto arrayAttr = attr.getValue().dyn_cast<ArrayAttr>();
    if (!arrayAttr)
      return op->emitError() << "'" << kEscapeAttrName
                             << "' is expected to be a bool array attribute";
    if (arrayAttr.size() != op->getNumResults())
      return op->emitError()
             << "'" << kEscapeAttrName
             << "' has wrong number of elements, expected "
             << op->getNumResults() << ", got " << arrayAttr.size();
    auto bufferizableOp = dyn_cast<BufferizableOpInterface>(op);
    if (!bufferizableOp)
      return op->emitError()
             << "'" << kEscapeAttrName << "' only valid on bufferizable ops";
    for (const auto &it : llvm::enumerate(arrayAttr)) {
      auto attr = it.value();
      auto boolAttr = attr.dyn_cast<BoolAttr>();
      if (!boolAttr)
        return op->emitError() << "'" << kEscapeAttrName
                               << "' is expected to be a bool array attribute";
      if (!boolAttr.getValue())
        continue;
      if (!op->getResult(it.index()).getType().isa<TensorType>())
        return op->emitError()
               << "'" << kEscapeAttrName << "' only valid for tensor results";
      if (!bufferizableOp.bufferizesToAllocation(op->getOpResult(it.index())))
        return op->emitError() << "'" << kEscapeAttrName
                               << "' only valid for allocation results";
    }
    return success();
  }

  return op->emitError() << "attribute '" << attr.getName()
                         << "' not supported by the bufferization dialect";
}
