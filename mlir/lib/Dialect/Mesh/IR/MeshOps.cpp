//===- MeshOps.cpp - Mesh Dialect Operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
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

template <typename It>
static It canonicalizeSetAsArray(It begin, It end) {
  llvm::sort(begin, end);
  return std::unique(begin, end);
}

template <typename R>
static auto canonicalizeSetAsArray(R &&range) {
  return canonicalizeSetAsArray(adl_begin(range), adl_end(range));
}

template <typename T>
static SmallVector<T> &canonicalizeSetAsVector(SmallVector<T> &vec) {
  auto newEnd = canonicalizeSetAsArray(vec);
  vec.resize(newEnd - vec.begin());
  return vec;
}

using MeshAxis = int16_t;

namespace {

struct DimensionSize {
  static DimensionSize dynamic() { return DimensionSize(ShapedType::kDynamic); }
  DimensionSize(int64_t val) : val(val) {}
  int64_t value() const { return val; }
  operator int64_t() const { return val; }
  bool isDynamic() const { return ShapedType::isDynamic(val); }

private:
  int64_t val;
};

} // namespace

static DimensionSize operator/(DimensionSize lhs, DimensionSize rhs) {
  if (lhs.isDynamic() || rhs.isDynamic()) {
    return DimensionSize::dynamic();
  }
  return lhs.value() / rhs.value();
}

static DimensionSize operator*(DimensionSize lhs, DimensionSize rhs) {
  if (lhs.isDynamic() || rhs.isDynamic()) {
    return DimensionSize::dynamic();
  }
  return lhs.value() * rhs.value();
}

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
    if (dimSize < 0 && !ShapedType::isDynamic(dimSize))
      return emitOpError("dimension size of a mesh cluster is expected to be "
                         "non-negative or dynamic");
  }

  return success();
}

SmallVector<int64_t> ClusterOp::canonicalDimSizes() {
  SmallVector<int64_t> result;
  canonicalDimSizes(std::back_inserter(result));
  result.reserve(getRank());
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

template <typename Op>
struct EmptyMeshAxesCanonicalizationPattern : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto meshAxes = op.getMeshAxes();
    if (!meshAxes.empty()) {
      return failure();
    }
    if (op.getInput().getType() != op.getResult().getType()) {
      return failure();
    }

    rewriter.replaceAllUsesWith(op.getResult(), op.getInput());
    rewriter.eraseOp(op.getOperation());
    return success();
  }
};

} // namespace

static LogicalResult verifyInGroupDevice(Location loc, StringRef deviceName,
                                         ArrayRef<int64_t> device,
                                         Operation::operand_range deviceDynamic,
                                         ArrayRef<MeshAxis> meshAxes,
                                         ArrayRef<int64_t> meshShape) {
  if (device.size() != meshAxes.size()) {
    return emitError(loc) << "In-group device \"" << deviceName
                          << "\" has unexpected multi-index size "
                          << device.size() << ". Expected " << meshAxes.size()
                          << ".";
  }

  for (size_t i = 0; i < device.size(); ++i) {
    if (!ShapedType::isDynamic(device[i]) &&
        !ShapedType::isDynamic(meshShape[meshAxes[i]]) &&
        meshShape[meshAxes[i]] <= device[i]) {
      return emitError(loc)
             << "Out of bounds coordinate " << i << " for in-group device \""
             << deviceName << "\"."
             << " Got " << device[i] << ", but expected value in the range [0, "
             << (meshShape[meshAxes[i]] - 1) << "].";
    }
  }
  return success();
}

static FailureOr<ClusterOp> getMesh(Operation *op, FlatSymbolRefAttr meshSymbol,
                                    SymbolTableCollection &symbolTable) {
  mesh::ClusterOp mesh =
      symbolTable.lookupNearestSymbolFrom<mesh::ClusterOp>(op, meshSymbol);
  if (!mesh) {
    return op->emitError() << "Undefined required mesh symbol \""
                           << meshSymbol.getValue() << "\".";
  }

  return mesh;
}

template <typename It>
bool isUnique(It begin, It end) {
  if (begin == end) {
    return true;
  }
  It next = std::next(begin);
  if (next == end) {
    return true;
  }
  for (; next != end; ++next, ++begin) {
    if (*begin == *next) {
      return false;
    }
  }
  return true;
}

static LogicalResult verifyMeshAxes(Location loc, ArrayRef<MeshAxis> axes,
                                    ClusterOp mesh) {
  SmallVector<MeshAxis> sorted = llvm::to_vector(axes);
  llvm::sort(sorted);
  if (!isUnique(sorted.begin(), sorted.end())) {
    return emitError(loc) << "Mesh axes contains duplicate elements.";
  }

  MeshAxis rank = mesh.getRank();
  for (auto axis : axes) {
    if (axis >= rank || axis < 0) {
      return emitError(loc)
             << "0-based mesh axis index " << axis
             << " is out of bounds. The referenced mesh \"" << mesh.getSymName()
             << "\" is of rank " << rank << ".";
    }
  }

  return success();
}

template <typename Op>
static FailureOr<ClusterOp>
getMeshAndVerifyAxes(Op op, SymbolTableCollection &symbolTable) {
  auto mesh = ::getMesh(op.getOperation(), op.getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyMeshAxes(op.getLoc(), op.getMeshAxes(), mesh.value()))) {
    return failure();
  }
  return mesh;
}

template <typename It>
static auto product(It begin, It end) {
  using ElementType = std::decay_t<decltype(*begin)>;
  return std::accumulate(begin, end, static_cast<ElementType>(1),
                         std::multiplies<ElementType>());
}

template <typename R>
static auto product(R &&range) {
  return product(adl_begin(range), adl_end(range));
}

static int64_t collectiveDeviceGroupSize(ArrayRef<MeshAxis> meshAxes,
                                         ArrayRef<int64_t> meshShape) {
  int64_t res = 1;

  for (MeshAxis axis : meshAxes) {
    if (ShapedType::isDynamic(meshShape[axis])) {
      return ShapedType::kDynamic;
    }
    assert(size_t(axis) < meshShape.size());
    res *= meshShape[axis];
  }

  return res;
}

static LogicalResult verifyDimensionCompatibility(Location loc,
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

static LogicalResult verifyGatherOperandAndResultShape(
    Value operand, Value result, int64_t gatherAxis,
    ArrayRef<MeshAxis> meshAxes, ArrayRef<int64_t> meshShape) {
  auto resultRank = result.getType().template cast<ShapedType>().getRank();
  if (gatherAxis < 0 || gatherAxis >= resultRank) {
    return emitError(result.getLoc())
           << "Gather axis " << gatherAxis << " is out of bounds [0, "
           << resultRank << ").";
  }

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

static LogicalResult verifyAllToAllOperandAndResultShape(
    Value operand, Value result, int64_t splitAxis, int64_t concatAxis,
    ArrayRef<MeshAxis> meshAxes, ArrayRef<int64_t> meshShape) {
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
  DimensionSize expectedResultConcatDimSize =
      operandConcatDimSize * deviceGroupSize;
  DimensionSize expectedResultSplitDimSize =
      operandSplitDimSize / deviceGroupSize;
  if (!expectedResultSplitDimSize.isDynamic() &&
      int64_t(operandSplitDimSize) % int64_t(deviceGroupSize) != 0) {
    expectedResultSplitDimSize = DimensionSize::dynamic();
  }
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

static LogicalResult verifyScatterOperandAndResultShape(
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

//===----------------------------------------------------------------------===//
// mesh.all_gather op
//===----------------------------------------------------------------------===//

LogicalResult
AllGatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto gatherAxis = getGatherAxis().getSExtValue();
  return verifyGatherOperandAndResultShape(getOperand(), getResult(),
                                           gatherAxis, getMeshAxes(),
                                           mesh.value().canonicalDimSizes());
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<AllGatherOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.all_reduce op
//===----------------------------------------------------------------------===//

LogicalResult
AllReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return getMeshAndVerifyAxes(*this, symbolTable);
}

void AllReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<AllReduceOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.all_to_all op
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }

  return verifyAllToAllOperandAndResultShape(
      getOperand(), getResult(), getSplitAxis().getSExtValue(),
      getConcatAxis().getSExtValue(), getMeshAxes(),
      mesh.value().canonicalDimSizes());
}

void AllToAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<AllToAllOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.broadcast op
//===----------------------------------------------------------------------===//

LogicalResult
BroadcastOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto meshShape = mesh.value().canonicalDimSizes();
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(), meshShape))) {
    return failure();
  }

  return success();
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<BroadcastOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.gather op
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto meshShape = mesh.value().canonicalDimSizes();
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(), meshShape))) {
    return failure();
  }

  auto gatherAxis = getGatherAxis().getSExtValue();
  return verifyGatherOperandAndResultShape(getInput(), getResult(), gatherAxis,
                                           getMeshAxes(),
                                           mesh.value().canonicalDimSizes());
}

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<GatherOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.recv op
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto meshShape = mesh.value().canonicalDimSizes();
  if (getSource() && failed(verifyInGroupDevice(
                         getLoc(), getSourceAttrName(), getSource().value(),
                         getSourceDynamic(), getMeshAxes(), meshShape))) {
    return failure();
  }
  return success();
}

void RecvOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<RecvOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.reduce op
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto meshShape = mesh.value().canonicalDimSizes();
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(), meshShape))) {
    return failure();
  }

  return success();
}

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<ReduceOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.reduce_scatter op
//===----------------------------------------------------------------------===//

LogicalResult
ReduceScatterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }

  return verifyScatterOperandAndResultShape(
      getOperand(), getResult(), getScatterAxis().getSExtValue(), getMeshAxes(),
      mesh.value().canonicalDimSizes());
}

void ReduceScatterOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<ReduceScatterOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.scatter op
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto meshShape = mesh.value().canonicalDimSizes();
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(), meshShape))) {
    return failure();
  }

  auto scatterAxis = getScatterAxis().getSExtValue();
  return verifyScatterOperandAndResultShape(getInput(), getResult(),
                                            scatterAxis, getMeshAxes(),
                                            mesh.value().canonicalDimSizes());
}

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<ScatterOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.send op
//===----------------------------------------------------------------------===//

LogicalResult SendOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  auto meshShape = mesh.value().canonicalDimSizes();
  if (failed(verifyInGroupDevice(getLoc(), getDestinationAttrName(),
                                 getDestination(), getDestinationDynamic(),
                                 getMeshAxes(), meshShape))) {
    return failure();
  }
  return success();
}

void SendOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<SendOp>>(context);
}

//===----------------------------------------------------------------------===//
// mesh.shift op
//===----------------------------------------------------------------------===//

LogicalResult ShiftOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }

  auto meshAxes = getMeshAxes();
  auto shiftAxis = getShiftAxis().getZExtValue();
  if (llvm::find(meshAxes, shiftAxis) == meshAxes.end()) {
    return emitError() << "Invalid shift axis " << shiftAxis
                       << ". It must be one of the grouping mesh axes.";
  }

  return success();
}

void ShiftOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  // TODO: remove op when offset is 0 or if it is a rotate with and
  // offset % shift_axis_mesh_dim_size == 0.
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOpsAttributes.cpp.inc"

#include "mlir/Dialect/Mesh/IR/MeshOpsEnums.cpp.inc"
