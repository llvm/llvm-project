//===- Dialect.cpp - Implementation of the linalg dialect and types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg dialect types and dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// LinalgDialect Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct LinalgInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // We don't have any special restrictions on what can be inlined into
  // destination regions (e.g. while/conditional bodies). Always allow it.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    return true;
  }
  // Operations in Linalg dialect are always legal to inline.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {}
};

} // namespace

//===----------------------------------------------------------------------===//
// LinalgDialect
//===----------------------------------------------------------------------===//

/// Attribute name used to memoize indexing maps for named ops.
constexpr const ::llvm::StringLiteral
    LinalgDialect::kMemoizedIndexingMapsAttrName;

/// Trait to check if T provides a `regionBuilder` method.
template <typename T, typename... Args>
using has_region_builder = decltype(T::regionBuilder);
template <typename T>
using detect_has_region_builder = llvm::is_detected<has_region_builder, T>;

/// SFINAE helper for single C++ class without a `regionBuilder` method (e.g.
/// an OpInterface).
template <typename OpType, typename = std::enable_if_t<
                               !detect_has_region_builder<OpType>::value>>
void addNamedOpBuilderImpl(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  // Do nothing.
}

template <typename OpType,
          typename = std::enable_if_t<detect_has_region_builder<OpType>::value>,
          typename = void>
void addNamedOpBuilderImpl(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  map.insert(std::make_pair(
      OpType::getOperationName(),
      static_cast<LinalgDialect::RegionBuilderFunType>(OpType::regionBuilder)));
}

template <typename... OpTypes>
void addNamedOpBuilders(
    llvm::StringMap<LinalgDialect::RegionBuilderFunType> &map) {
  (addNamedOpBuilderImpl<OpTypes>(map), ...);
}

void mlir::linalg::LinalgDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOpsAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >();

  // Fill the Linalg-specific OpName to RegionBuilder map.
  addNamedOpBuilders<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(namedStructuredOpRegionBuilders);

  addInterfaces<LinalgInlinerInterface>();

  declarePromisedInterface<mesh::ShardingInterface, GenericOp>();
  declarePromisedInterfaces<mesh::ShardingInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
  declarePromisedInterface<SubsetOpInterface, CopyOp>();
  declarePromisedInterface<SubsetInsertionOpInterface, CopyOp>();
  declarePromisedInterface<ValueBoundsOpInterface, IndexOp>();
  declarePromisedInterface<TilingInterface, linalg::GenericOp>();
  declarePromisedInterface<PartialReductionOpInterface, linalg::GenericOp>();
  declarePromisedInterfaces<TilingInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
  declarePromisedInterfaces<PartialReductionOpInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
  declarePromisedInterfaces<bufferization::BufferizableOpInterface,
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();
}

LogicalResult LinalgDialect::verifyOperationAttribute(Operation *op,
                                                      NamedAttribute attr) {
  if (attr.getName() == LinalgDialect::kMemoizedIndexingMapsAttrName)
    return success();
  return op->emitError() << "attribute '" << attr.getName()
                         << "' not supported by the linalg dialect";
}

#include "mlir/Dialect/Linalg/IR/LinalgOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Linalg/IR/LinalgOpsAttrDefs.cpp.inc"

#include "mlir/Dialect/Linalg/IR/LinalgOpsDialect.cpp.inc"
