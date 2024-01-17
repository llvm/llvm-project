//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/SubsetOpInterface.h"
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
                       IRMapping &valueMapping) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool wouldBeCloned,
                       IRMapping &) const final {
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
  declarePromisedInterface<CastOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<CollapseShapeOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<DimOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<EmptyOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<ExpandShapeOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<ExtractSliceOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<ExtractOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<FromElementsOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<GenerateOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<InsertOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<InsertSliceOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<PadOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<ParallelInsertSliceOp,
                           bufferization::BufferizableOpInterface>();
  declarePromisedInterface<RankOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<ReshapeOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<SplatOp, bufferization::BufferizableOpInterface>();
  declarePromisedInterface<CollapseShapeOp,
                           transform::FindPayloadReplacementOpInterface>();
  declarePromisedInterface<ExpandShapeOp,
                           transform::FindPayloadReplacementOpInterface>();
  declarePromisedInterface<ExtractSliceOp,
                           transform::FindPayloadReplacementOpInterface>();
  declarePromisedInterface<InsertSliceOp,
                           transform::FindPayloadReplacementOpInterface>();
  declarePromisedInterface<ReshapeOp,
                           transform::FindPayloadReplacementOpInterface>();
  declarePromisedInterface<ExpandShapeOp, ReifyRankedShapedTypeOpInterface>();
  declarePromisedInterface<CollapseShapeOp, ReifyRankedShapedTypeOpInterface>();
  declarePromisedInterface<PadOp, ReifyRankedShapedTypeOpInterface>();
  declarePromisedInterface<ExtractSliceOp, SubsetOpInterface>();
  declarePromisedInterface<ExtractSliceOp, SubsetExtractionOpInterface>();
  declarePromisedInterface<InsertSliceOp, SubsetOpInterface>();
  declarePromisedInterface<InsertSliceOp, SubsetInsertionOpInterface>();
  declarePromisedInterface<ParallelInsertSliceOp, SubsetOpInterface>();
  declarePromisedInterface<ParallelInsertSliceOp, SubsetInsertionOpInterface>();
  declarePromisedInterface<PadOp, TilingInterface>();
  declarePromisedInterface<PackOp, TilingInterface>();
  declarePromisedInterface<UnPackOp, TilingInterface>();
  declarePromisedInterface<CastOp, ValueBoundsOpInterface>();
  declarePromisedInterface<DimOp, ValueBoundsOpInterface>();
  declarePromisedInterface<EmptyOp, ValueBoundsOpInterface>();
  declarePromisedInterface<ExtractSliceOp, ValueBoundsOpInterface>();
  declarePromisedInterface<PadOp, ValueBoundsOpInterface>();
  declarePromisedInterface<RankOp, ValueBoundsOpInterface>();
}
