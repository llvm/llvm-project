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
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
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

LogicalResult BufferizationDialect::verifyRegionArgAttribute(
    Operation *op, unsigned /*regionIndex*/, unsigned argIndex,
    NamedAttribute attr) {
  if (attr.getName() == kWritableAttrName) {
    if (!llvm::isa<BoolAttr>(attr.getValue())) {
      return op->emitError() << "'" << kWritableAttrName
                             << "' is expected to be a boolean attribute";
    }
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << "expected '" << kWritableAttrName
                             << "' to be used on function-like operations";
    if (cast<FunctionOpInterface>(op).isExternal())
      return op->emitError() << "'" << kWritableAttrName
                             << "' is invalid on external functions";
    return success();
  }
  if (attr.getName() == kBufferAccessAttrName) {
    if (!llvm::isa<StringAttr>(attr.getValue())) {
      return op->emitError() << "'" << kBufferAccessAttrName
                             << "' is expected to be a string attribute";
    }
    StringRef str = llvm::cast<StringAttr>(attr.getValue()).getValue();
    if (str != "none" && str != "read" && str != "write" && str != "read-write")
      return op->emitError()
             << "invalid value for '" << kBufferAccessAttrName << "'";
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << "expected '" << kBufferAccessAttrName
                             << "' to be used on function-like operations";
    return success();
  }
  if (attr.getName() == kBufferLayoutAttrName) {
    if (!llvm::isa<AffineMapAttr>(attr.getValue())) {
      return op->emitError() << "'" << kBufferLayoutAttrName
                             << "' is expected to be a affine map attribute";
    }
    if (!isa<FunctionOpInterface>(op))
      return op->emitError() << "expected '" << kBufferLayoutAttrName
                             << "' to be used on function-like operations";
    return success();
  }
  return op->emitError() << "attribute '" << kBufferLayoutAttrName
                         << "' not supported as a region arg attribute by the "
                            "bufferization dialect";
}

LogicalResult
BufferizationDialect::verifyOperationAttribute(Operation *op,
                                               NamedAttribute attr) {
  using bufferization::BufferizableOpInterface;

  if (attr.getName() == kEscapeAttrName) {
    auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr.getValue());
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
      auto boolAttr = llvm::dyn_cast<BoolAttr>(attr);
      if (!boolAttr)
        return op->emitError() << "'" << kEscapeAttrName
                               << "' is expected to be a bool array attribute";
      if (!boolAttr.getValue())
        continue;
      if (!llvm::isa<TensorType>(op->getResult(it.index()).getType()))
        return op->emitError()
               << "'" << kEscapeAttrName << "' only valid for tensor results";
      if (!bufferizableOp.bufferizesToAllocation(op->getOpResult(it.index())))
        return op->emitError() << "'" << kEscapeAttrName
                               << "' only valid for allocation results";
    }
    return success();
  }

  return op->emitError()
         << "attribute '" << attr.getName()
         << "' not supported as an op attribute by the bufferization dialect";
}
