//===- MeshOps.cpp - Mesh Dialect Operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#define DEBUG_TYPE "mesh-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

#include "mlir/Dialect/Mesh/IR/MeshOpsDialect.cpp.inc"

namespace {

template <typename It>
It canonicalizeSetAsArray(It begin, It end) {
  std::sort(begin, end);
  return std::unique(begin, end);
}

template <typename R>
auto canonicalizeSetAsArray(R &&range) {
  return canonicalizeSetAsArray(adl_begin(range), adl_end(range));
}

template <typename T>
SmallVector<T> &canonicalizeSetAsVector(SmallVector<T> &vec) {
  auto newEnd = canonicalizeSetAsArray(vec);
  vec.resize(newEnd - vec.begin());
  return vec;
}

template <typename DimSize>
bool isMeshDimensionDynamic(DimSize size) {
  return size <= DimSize(0);
}

using MeshAxis = int16_t;

struct DimensionSize {
  static DimensionSize dynamic() { return DimensionSize(ShapedType::kDynamic); }
  DimensionSize(int64_t val) : val(val) {}
  int64_t value() const { return val; }
  operator int64_t() const { return val; }
  bool isDynamic() const { return ShapedType::isDynamic(val); }

private:
  int64_t val;
};

DimensionSize operator/(DimensionSize lhs, DimensionSize rhs) {
  if (lhs.isDynamic() || rhs.isDynamic()) {
    return DimensionSize::dynamic();
  }
  return lhs.value() / rhs.value();
}

DimensionSize operator*(DimensionSize lhs, DimensionSize rhs) {
  if (lhs.isDynamic() || rhs.isDynamic()) {
    return DimensionSize::dynamic();
  }
  return lhs.value() * rhs.value();
}

} // namespace

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
// Mesh utilities
//===----------------------------------------------------------------------===//

bool mesh::isReductionLoop(IteratorType iType) {
  return iType != IteratorType::Parallel && iType != IteratorType::Invalid;
}

bool mesh::areReductionAndPartialMatch(IteratorType iType, Partial partial) {
  return (partial == Partial::Generic &&
          iType == IteratorType::ReductionGeneric) ||
         (partial == Partial::Sum && iType == IteratorType::ReductionSum) ||
         (partial == Partial::Max && iType == IteratorType::ReductionMax) ||
         (partial == Partial::Min && iType == IteratorType::ReductionMin);
}

Partial mesh::getPartialTypeFromReduction(IteratorType iType) {
  switch (iType) {
  case IteratorType::ReductionGeneric:
    return Partial::Generic;
  case IteratorType::ReductionSum:
    return Partial::Sum;
  case IteratorType::ReductionMax:
    return Partial::Max;
  case IteratorType::ReductionMin:
    return Partial::Min;
  default:
    llvm_unreachable("No corresponding partial type can be found");
  }
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

SmallVector<int64_t> ClusterOp::canonicalDimSizes() {
  SmallVector<int64_t> result;
  canonicalDimSizes(std::back_inserter(result));
  return result;
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
// collective communication ops
//===----------------------------------------------------------------------===//

namespace {

std::optional<DenseI16ArrayAttr>
canonicalizeAxesSetAttribute(DenseI16ArrayAttr attr) {
  if (!attr) {
    return std::nullopt;
  }
  SmallVector<int16_t> axes = llvm::to_vector(attr.asArrayRef());
  canonicalizeSetAsVector(axes);
  if (axes.empty()) {
    return std::nullopt;
  }
  return DenseI16ArrayAttr::get(attr.getContext(), axes);
}

template <typename Op>
struct AxesSetCanonicalizationPattern : OpRewritePattern<Op> {
  AxesSetCanonicalizationPattern(MLIRContext *context, StringRef axisSetAttr)
      : OpRewritePattern<Op>(context), axisSetAttr(axisSetAttr) {}
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto canonicalMeshAxesAttr = canonicalizeAxesSetAttribute(
        op->template getAttrOfType<DenseI16ArrayAttr>(axisSetAttr));
    if (!canonicalMeshAxesAttr) {
      op->removeAttr(axisSetAttr);
    } else {
      op->setAttr(axisSetAttr, canonicalMeshAxesAttr.value());
    }
    return success();
  }

  std::string axisSetAttr;
};

template <typename Op>
void populateMeshAxesSetCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<AxesSetCanonicalizationPattern<Op>>(context, "mesh_axes");
}

template <typename Op>
LogicalResult verifyMeshSymbolUses(Op op, SymbolTableCollection &symbolTable) {
  FlatSymbolRefAttr symbolAttr = op.getMeshAttr();
  if (!symbolAttr) {
    return op.emitError() << "Unspecified \"mesh\" symbol attribute.";
  }
  SymbolTableCollection symbolTableCollection;
  mesh::ClusterOp mesh =
      symbolTableCollection.lookupNearestSymbolFrom<mesh::ClusterOp>(
          op.getOperation(), symbolAttr);
  if (!mesh) {
    return op.emitError() << "Undefined required mesh symbol \""
                          << symbolAttr.getValue() << "\".";
  }
  DenseI16ArrayAttr meshAxes = op.getMeshAxesAttr();
  if (!meshAxes) {
    return success();
  }
  MeshAxis rank = mesh.getRank();
  for (auto axis : meshAxes.asArrayRef()) {
    if (axis >= rank || axis < 0) {
      return op.emitError()
             << "0-based mesh axis index " << axis
             << " is out of bounds. The referenced mesh \""
             << symbolAttr.getValue() << "\" is of rank " << rank << ".";
    }
  }

  return success();
}

template <typename It>
auto product(It begin, It end) {
  using ElementType = std::decay_t<decltype(*begin)>;
  return std::accumulate(begin, end, ElementType(1),
                         std::multiplies<ElementType>());
}

template <typename R>
auto product(R &&range) {
  return product(adl_begin(range), adl_end(range));
}

int64_t collectiveDeviceGroupSize(ArrayRef<MeshAxis> meshAxes,
                                  ArrayRef<int64_t> meshShape) {
  int64_t res = 1;
  for (MeshAxis axis = 0; axis < MeshAxis(meshShape.size()); ++axis) {
    if (llvm::find(meshAxes, axis) == meshAxes.end()) {
      continue;
    }
    if (isMeshDimensionDynamic(meshShape[axis])) {
      return ShapedType::kDynamic;
    }
    res *= meshShape[axis];
  }
  return res;
}

LogicalResult verifyDimensionCompatibility(Location loc,
                                           int64_t expectedDimSize,
                                           int64_t resultDimSize,
                                           int64_t resultAxis) {
  if (!ShapedType::isDynamic(resultDimSize) &&
      expectedDimSize != resultDimSize) {
    return emitError(loc) << "Dimension size mismatch for result axis "
                          << resultAxis << ". Expected "
                          << (ShapedType::isDynamic(expectedDimSize)
                                  ? Twine("dynamic")
                                  : Twine(expectedDimSize))
                          << ", but got " << resultDimSize << ".";
  }

  return success();
}

LogicalResult verifyGatherOperandAndResultShape(Value operand, Value result,
                                                int64_t gatherAxis,
                                                ArrayRef<MeshAxis> meshAxes,
                                                ArrayRef<int64_t> meshShape) {
  ShapedType operandType = operand.getType().cast<ShapedType>();
  ShapedType resultType = result.getType().cast<ShapedType>();
  auto deviceGroupSize =
      DimensionSize(collectiveDeviceGroupSize(meshAxes, meshShape));
  for (int64_t axis = 0; axis < operandType.getRank(); ++axis) {
    auto operandDimSize = DimensionSize(operandType.getDimSize(axis));
    auto resultDimSize = DimensionSize(resultType.getDimSize(axis));
    auto expectedResultDimSize =
        axis == gatherAxis ? deviceGroupSize * operandDimSize : operandDimSize;
    if (failed(verifyDimensionCompatibility(
            result.getLoc(), expectedResultDimSize, resultDimSize, axis))) {
      return failure();
    }
  }
  return success();
}

template <typename Op>
FailureOr<ClusterOp> getMesh(Op op) {
  SymbolTableCollection symbolTableCollection;
  if (failed(verifyMeshSymbolUses(op, symbolTableCollection))) {
    // We need to check the symbol here since this runs before
    // SymbolUserOpInterface.
    return failure();
  }
  return symbolTableCollection.lookupNearestSymbolFrom<mesh::ClusterOp>(
      op.getOperation(), op.getMeshAttr());
}

template <typename Op>
LogicalResult verifyGather(Op op) {
  auto rank = op.getResult().getType().template cast<ShapedType>().getRank();
  auto gatherAxis = op.getGatherAxis().getSExtValue();
  if (gatherAxis < 0 || gatherAxis >= rank) {
    return op.emitError() << "Gather axis " << gatherAxis
                          << " is out of bounds [0, " << rank << ").";
  }

  auto mesh = getMesh(op);
  if (failed(mesh)) {
    return failure();
  }
  return verifyGatherOperandAndResultShape(op.getOperand(), op.getResult(),
                                           gatherAxis, op.getMeshAxes(),
                                           mesh.value().canonicalDimSizes());
}

LogicalResult verifyAllToAllOperandAndResultShape(Value operand, Value result,
                                                  int64_t splitAxis,
                                                  int64_t concatAxis,
                                                  ArrayRef<MeshAxis> meshAxes,
                                                  ArrayRef<int64_t> meshShape) {
  ShapedType operandType = operand.getType().cast<ShapedType>();
  ShapedType resultType = result.getType().cast<ShapedType>();
  for (int64_t axis = 0; axis < operandType.getRank(); ++axis) {
    if ((axis != splitAxis && axis != concatAxis) || splitAxis == concatAxis) {
      if (failed(verifyDimensionCompatibility(
              result.getLoc(), operandType.getDimSize(axis),
              resultType.getDimSize(axis), axis))) {
        return failure();
      }
    }
  }

  if (splitAxis == concatAxis) {
    return success();
  }

  auto deviceGroupSize =
      DimensionSize(collectiveDeviceGroupSize(meshAxes, meshShape));
  auto operandConcatDimSize = DimensionSize(operandType.getDimSize(concatAxis));
  auto operandSplitDimSize = DimensionSize(operandType.getDimSize(splitAxis));
  if (!operandSplitDimSize.isDynamic() && !deviceGroupSize.isDynamic() &&
      int64_t(operandSplitDimSize) % int64_t(deviceGroupSize) != 0) {
    return emitError(result.getLoc())
           << "Operand dimension size " << int64_t(operandSplitDimSize)
           << " is not divisible by collective device group size "
           << int64_t(deviceGroupSize) << " for split axis " << splitAxis
           << ".";
  }
  DimensionSize expectedResultConcatDimSize =
      operandConcatDimSize * deviceGroupSize;
  DimensionSize expectedResultSplitDimSize =
      operandSplitDimSize / deviceGroupSize;
  if (failed(verifyDimensionCompatibility(
          result.getLoc(), expectedResultConcatDimSize.value(),
          resultType.getDimSize(concatAxis), concatAxis))) {
    return failure();
  }
  if (failed(verifyDimensionCompatibility(
          result.getLoc(), expectedResultSplitDimSize.value(),
          resultType.getDimSize(splitAxis), splitAxis))) {
    return failure();
  }

  return success();
}

LogicalResult verifyReduceScatterOperandAndResultShape(
    Value operand, Value result, int64_t scatterAxis,
    ArrayRef<MeshAxis> meshAxes, ArrayRef<int64_t> meshShape) {
  ShapedType operandType = operand.getType().cast<ShapedType>();
  ShapedType resultType = result.getType().cast<ShapedType>();
  for (int64_t axis = 0; axis < operandType.getRank(); ++axis) {
    if (axis != scatterAxis) {
      if (failed(verifyDimensionCompatibility(
              result.getLoc(), operandType.getDimSize(axis),
              resultType.getDimSize(axis), axis))) {
        return failure();
      }
    }
  }

  auto deviceGroupSize =
      DimensionSize(collectiveDeviceGroupSize(meshAxes, meshShape));
  auto operandScatterDimSize =
      DimensionSize(operandType.getDimSize(scatterAxis));
  if (!operandScatterDimSize.isDynamic() && !deviceGroupSize.isDynamic() &&
      int64_t(operandScatterDimSize) % int64_t(deviceGroupSize) != 0) {
    return emitError(result.getLoc())
           << "Operand dimension size " << int64_t(operandScatterDimSize)
           << " is not divisible by collective device group size "
           << int64_t(deviceGroupSize) << " for scatter axis " << scatterAxis
           << ".";
  }
  DimensionSize expectedResultScatterDimSize =
      operandScatterDimSize / deviceGroupSize;
  if (failed(verifyDimensionCompatibility(
          result.getLoc(), expectedResultScatterDimSize.value(),
          resultType.getDimSize(scatterAxis), scatterAxis))) {
    return failure();
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// mesh.all_reduce op
//===----------------------------------------------------------------------===//

LogicalResult
mlir::mesh::AllReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMeshSymbolUses(*this, symbolTable);
}

void mlir::mesh::AllReduceOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateMeshAxesSetCanonicalizationPatterns<AllReduceOp>(patterns, context);
}

//===----------------------------------------------------------------------===//
// mesh.all_gather op
//===----------------------------------------------------------------------===//

LogicalResult
mlir::mesh::AllGatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMeshSymbolUses(*this, symbolTable);
}

void mlir::mesh::AllGatherOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateMeshAxesSetCanonicalizationPatterns<AllGatherOp>(patterns, context);
}

LogicalResult mlir::mesh::AllGatherOp::verify() { return verifyGather(*this); }

//===----------------------------------------------------------------------===//
// mesh.all_to_all op
//===----------------------------------------------------------------------===//

LogicalResult
mlir::mesh::AllToAllOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyMeshSymbolUses(*this, symbolTable);
}

void mlir::mesh::AllToAllOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateMeshAxesSetCanonicalizationPatterns<AllToAllOp>(patterns, context);
}

LogicalResult mlir::mesh::AllToAllOp::verify() {
  auto mesh = ::getMesh(*this);
  if (failed(mesh)) {
    return failure();
  }
  return verifyAllToAllOperandAndResultShape(
      getOperand(), getResult(), getSplitAxis().getSExtValue(),
      getConcatAxis().getSExtValue(), getMeshAxes(),
      mesh.value().canonicalDimSizes());
}

//===----------------------------------------------------------------------===//
// mesh.reduce_scatter op
//===----------------------------------------------------------------------===//

LogicalResult mlir::mesh::ReduceScatterOp::verifySymbolUses(
    SymbolTableCollection &symbolTable) {
  return verifyMeshSymbolUses(*this, symbolTable);
}

void mlir::mesh::ReduceScatterOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  populateMeshAxesSetCanonicalizationPatterns<ReduceScatterOp>(patterns,
                                                               context);
}

LogicalResult mlir::mesh::ReduceScatterOp::verify() {
  auto mesh = ::getMesh(*this);
  if (failed(mesh)) {
    return failure();
  }
  return verifyReduceScatterOperandAndResultShape(
      getOperand(), getResult(), getScatterAxis().getSExtValue(), getMeshAxes(),
      mesh.value().canonicalDimSizes());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"

#include "mlir/Dialect/Mesh/IR/MeshOpsEnums.cpp.inc"
