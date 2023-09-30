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
  size_t rank = getRank();

  if (rank == 0)
    return emitOpError("rank of cluster is expected to be a positive integer");

  if (dimSizes.size() > rank)
    return emitOpError(
        "rank of dim_sizes is not expected to be larger than rank of cluster");

  return success();
}

//===----------------------------------------------------------------------===//
// mesh.shard op
//===----------------------------------------------------------------------===//

LogicalResult ShardOp::verify() {
  bool asResult = getAsResult();
  if (asResult) {
    Value src = getSrc();
    Operation *defOp = src.getDefiningOp();
    if (llvm::isa_and_nonnull<ShardOp>(defOp))
      return emitOpError("two mesh.shard ops with as_result = true are not "
                         "expected to be stacked together");

    unsigned numShard = llvm::count_if(src.getUsers(), [](Operation *user) {
      return llvm::isa<ShardOp>(user);
    });
    if (numShard > 1)
      return emitOpError(
          "when than one mesh.shard ops operate on the same tensor, all of "
          "their as_result attributes are expected to be false");

  } else {
    ShardOp defShardOp = getSrc().getDefiningOp<ShardOp>();
    if (defShardOp && !defShardOp.getAsResult())
      return emitOpError("two mesh.shard ops with as_result = false are not "
                         "expected to be stacked together");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mesh.shard op
//===----------------------------------------------------------------------===//

LogicalResult
MeshShardingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                         SymbolRefAttr, ArrayRef<DenseI64ArrayAttr> axes) {
  // TODO: At present cluster symbol ref is not verified. This is due to the
  // difficulty in fetching the corresponding symbol op based on an attribute.

  DenseSet<int64_t> visitedAxes;
  for (DenseI64ArrayAttr subAxes : axes) {
    ArrayRef<int64_t> subAxesArray = subAxes.asArrayRef();
    for (int64_t axis : subAxesArray) {
      if (!visitedAxes.insert(axis).second)
        return emitError() << "mesh axis duplicated";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"
