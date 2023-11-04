//===- MeshOps.cpp - Mesh Dialect Operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "mesh-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

#include "mlir/Dialect/Mesh/IR/MeshOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Mesh dialect
//===----------------------------------------------------------------------===//

void MeshDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Mesh/IR/MeshOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"
      >();
}

Operation *MeshDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

//===----------------------------------------------------------------------===//
// mesh.cluster op
//===----------------------------------------------------------------------===//

LogicalResult ClusterOp::verify() {
  ArrayRef<int64_t> dimSizes = getDimSizes();
  uint64_t rank = getRank();

  if (rank == 0)
    return emitOpError("rank of cluster is expected to be a positive integer");

  if (dimSizes.size() > rank)
    return emitOpError(
        "rank of dim_sizes is not expected to be larger than rank of cluster");

  for (int64_t dimSize : dimSizes) {
    if (dimSize < 0)
      return emitOpError(
          "dimension size of a mesh cluster is expected to be non-negative");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mesh.shard op
//===----------------------------------------------------------------------===//

LogicalResult
MeshShardingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         SymbolRefAttr, ArrayRef<DenseI32ArrayAttr> splitAxes,
                         ArrayRef<int32_t> partialAxes, Partial) {
  // TODO: At present cluster symbol ref is not verified. This is due to the
  // difficulty in fetching the corresponding symbol op based on an attribute.

  llvm::SmallSet<int32_t, 4> visitedAxes;

  auto checkMeshAxis = [&](ArrayRef<int32_t> axesArray) -> LogicalResult {
    for (int32_t axis : axesArray) {
      if (axis < 0)
        return emitError() << "mesh axis is expected to be non-negative";
      if (!visitedAxes.insert(axis).second)
        return emitError() << "mesh axis duplicated";
    }
    return success();
  };

  for (DenseI32ArrayAttr subAxes : splitAxes) {
    ArrayRef<int32_t> subAxesArray = subAxes.asArrayRef();
    if (failed(checkMeshAxis(subAxesArray)))
      return failure();
  }
  if (failed(checkMeshAxis(partialAxes)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"

#include "mlir/Dialect/Mesh/IR/MeshOpsEnums.cpp.inc"
