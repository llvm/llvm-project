//===- MeshOps.cpp - Mesh Dialect Operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/IR/MeshOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Mesh/IR/MeshDialect.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

#define DEBUG_TYPE "mesh-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace mlir::mesh;

#include "mlir/Dialect/Mesh/IR/MeshDialect.cpp.inc"

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
// Inliner
//===----------------------------------------------------------------------===//

namespace {
struct MeshInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  // Currently no restrictions are encoded for inlining.
  bool isLegalToInline(Operation *, Operation *, bool) const final {
    return true;
  }
  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
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
#include "mlir/Dialect/Mesh/IR/MeshAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Mesh/IR/MeshTypes.cpp.inc"
      >();
  addInterface<MeshInlinerInterface>();
}

Operation *MeshDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

//===----------------------------------------------------------------------===//
// Mesh utilities
//===----------------------------------------------------------------------===//

static FailureOr<MeshOp> getMeshAndVerify(Operation *op,
                                          FlatSymbolRefAttr meshSymbol,
                                          SymbolTableCollection &symbolTable) {
  mesh::MeshOp mesh = getMeshOrNull(op, meshSymbol, symbolTable);
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
                                    MeshOp mesh) {
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
static FailureOr<MeshOp>
getMeshAndVerifyAxes(Op op, SymbolTableCollection &symbolTable) {
  auto mesh =
      ::getMeshAndVerify(op.getOperation(), op.getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyMeshAxes(op.getLoc(), op.getMeshAxes(), mesh.value()))) {
    return failure();
  }
  return mesh;
}

template <typename InShape, typename MeshShape, typename SplitAxes,
          typename OutShape>
static void shardShape(const InShape &inShape, const MeshShape &meshShape,
                       const SplitAxes &splitAxes, OutShape &outShape,
                       ArrayRef<int64_t> shardedDimsOffsets = {},
                       ArrayRef<int64_t> haloSizes = {}) {
  // 0d tensors cannot be sharded and must get replicated
  if (inShape.empty()) {
    assert(outShape.empty());
    return;
  }

  std::copy(llvm::adl_begin(inShape), llvm::adl_end(inShape),
            llvm::adl_begin(outShape));

  if (!shardedDimsOffsets.empty()) {
    auto isDynShape = ShapedType::isDynamicShape(meshShape);
    uint64_t pos = 1;
    for (auto [tensorAxis, innerSplitAxes] : llvm::enumerate(splitAxes)) {
      if (!innerSplitAxes.empty()) {
        auto sz = shardedDimsOffsets[pos];
        bool same = !isDynShape;
        if (same) {
          // Find sharded dims in shardedDimsOffsets with same static size on
          // all devices. Use kDynamic for dimensions with dynamic or
          // non-uniform offs in shardedDimsOffsets.
          uint64_t numShards = 0;
          for (auto i : innerSplitAxes.asArrayRef()) {
            numShards += meshShape[i];
          }
          for (size_t i = 1; i < numShards; ++i) {
            if (shardedDimsOffsets[pos + i] - shardedDimsOffsets[pos + i - 1] !=
                sz) {
              same = false;
              break;
            }
          }
          pos += numShards + 1;
        }
        outShape[tensorAxis] = same ? sz : ShapedType::kDynamic;
      }
    }
  } else {
    for (auto [tensorAxis, innerSplitAxes] : llvm::enumerate(splitAxes)) {
      outShape[tensorAxis] = shardDimension(
          inShape[tensorAxis],
          collectiveProcessGroupSize(innerSplitAxes.asArrayRef(), meshShape));
    }

    if (!haloSizes.empty()) {
      // add halo sizes if requested
      int haloAxis = 0;
      for (auto [tensorAxis, innerSplitAxes] : llvm::enumerate(splitAxes)) {
        if (!ShapedType::isDynamic(outShape[tensorAxis]) &&
            !innerSplitAxes.empty()) {
          if (haloSizes[haloAxis * 2] >= 0 &&
              haloSizes[haloAxis * 2 + 1] >= 0) {
            outShape[tensorAxis] +=
                haloSizes[haloAxis * 2] + haloSizes[haloAxis * 2 + 1];
            ++haloAxis;
          } else {
            outShape[tensorAxis] = ShapedType::kDynamic;
          }
        }
      }
    }
  }
}

ShapedType mesh::shardShapedType(ShapedType shape, MeshOp mesh,
                                 MeshSharding sharding) {
  using Dim = std::decay_t<decltype(shape.getDimSize(0))>;
  SmallVector<Dim> resShapeArr(shape.getShape().size());
  shardShape(shape.getShape(), mesh.getShape(), sharding.getSplitAxes(),
             resShapeArr, sharding.getStaticShardedDimsOffsets(),
             sharding.getStaticHaloSizes());
  return shape.clone(resShapeArr);
}

Type mesh::shardType(Type type, MeshOp mesh, MeshSharding sharding) {
  RankedTensorType rankedTensorType = dyn_cast<RankedTensorType>(type);
  if (rankedTensorType && !rankedTensorType.getShape().empty()) {
    return shardShapedType(rankedTensorType, mesh, sharding);
  }
  return type;
}

void mlir::mesh::maybeInsertTargetShardingAnnotation(MeshSharding sharding,
                                                     OpOperand &operand,
                                                     OpBuilder &builder,
                                                     ShardOp &newShardOp) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Value operandValue = operand.get();
  Operation *operandOp = operand.getOwner();
  builder.setInsertionPointAfterValue(operandValue);
  ShardOp shardOp = dyn_cast<ShardOp>(operandOp);
  if (shardOp && sharding == shardOp.getSharding() &&
      !shardOp.getAnnotateForUsers()) {
    // No need for anything if the correct sharding is already set.
    if (!newShardOp) {
      newShardOp = shardOp;
    }
    return;
  }

  if (!newShardOp) {
    auto shardingOp =
        builder.create<ShardingOp>(operandValue.getLoc(), sharding);
    newShardOp =
        builder.create<ShardOp>(operandValue.getLoc(), operandValue, shardingOp,
                                /*annotate_for_users*/ false);
  }
  IRRewriter rewriter(builder);
  rewriter.replaceUsesWithIf(
      operandValue, newShardOp, [operandOp, operandValue](OpOperand &use) {
        return use.getOwner() == operandOp && use.get() == operandValue;
      });

  if (!shardOp || shardOp.getAnnotateForUsers()) {
    return;
  }

  auto newShardOp2 = builder.create<ShardOp>(operandValue.getLoc(), newShardOp,
                                             newShardOp.getSharding(),
                                             /*annotate_for_users*/ true);
  rewriter.replaceAllUsesExcept(newShardOp, newShardOp2, newShardOp2);
}

void mlir::mesh::maybeInsertTargetShardingAnnotation(MeshSharding sharding,
                                                     OpResult result,
                                                     OpBuilder &builder) {
  ShardOp newShardOp;
  for (auto &use : llvm::make_early_inc_range(result.getUses())) {
    maybeInsertTargetShardingAnnotation(sharding, use, builder, newShardOp);
  }
}

void mlir::mesh::maybeInsertSourceShardingAnnotation(MeshSharding sharding,
                                                     OpOperand &operand,
                                                     OpBuilder &builder) {
  OpBuilder::InsertionGuard insertionGuard(builder);
  Value operandValue = operand.get();
  Operation *operandSrcOp = operandValue.getDefiningOp();
  bool isBlockArg = !operandSrcOp;
  {
    [[maybe_unused]] auto opType =
        dyn_cast<mlir::RankedTensorType>(operandValue.getType());
    assert(!opType || opType.getRank() > 0 || isFullReplication(sharding));
  }
  if (!isa<RankedTensorType>(operandValue.getType()) && operandSrcOp &&
      operandSrcOp->hasTrait<OpTrait::ConstantLike>()) {
    return;
  }

  Operation *operandOp = operand.getOwner();
  ShardOp shardOp = dyn_cast_or_null<ShardOp>(operandSrcOp);

  if (shardOp && sharding == shardOp.getSharding() &&
      shardOp.getAnnotateForUsers()) {
    // No need for anything the correct sharding is already set.
    return;
  }

  builder.setInsertionPoint(operandOp);
  auto shardingOp =
      builder.create<ShardingOp>(operand.get().getLoc(), sharding);
  auto newShardOp =
      builder.create<ShardOp>(operandValue.getLoc(), operandValue, shardingOp,
                              /*annotate_for_users*/ true);
  IRRewriter rewriter(builder);
  rewriter.replaceUsesWithIf(
      operandValue, newShardOp, [operandOp, operandValue](OpOperand &use) {
        return use.getOwner() == operandOp && use.get() == operandValue;
      });

  if (isBlockArg || !shardOp || !shardOp.getAnnotateForUsers()) {
    // No need for resharding.
    return;
  }

  builder.setInsertionPoint(newShardOp);
  auto newPreceedingShardOp =
      builder.create<ShardOp>(operandValue.getLoc(), operandValue, shardingOp,
                              /*annotate_for_users*/ false);
  rewriter.replaceUsesWithIf(
      newShardOp.getSrc(), newPreceedingShardOp, [&newShardOp](OpOperand &use) {
        return use.getOwner() == newShardOp.getOperation();
      });
}

//===----------------------------------------------------------------------===//
// mesh.mesh op
//===----------------------------------------------------------------------===//

LogicalResult MeshOp::verify() {
  int64_t rank = getRank();

  if (rank <= 0)
    return emitOpError("rank of mesh is expected to be a positive integer");

  for (int64_t dimSize : getShape()) {
    if (dimSize < 0 && !ShapedType::isDynamic(dimSize))
      return emitOpError("dimension size of a mesh is expected to be "
                         "non-negative or dynamic");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// mesh.mesh_shape op
//===----------------------------------------------------------------------===//

LogicalResult
MeshShapeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = ::getMeshAndVerify(getOperation(), getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyMeshAxes(getLoc(), getAxes(), mesh.value()))) {
    return failure();
  }

  size_t expectedResultsCount =
      getAxes().empty() ? mesh->getRank() : getAxes().size();
  if (getResult().size() != expectedResultsCount) {
    return emitError() << "Unexpected number of results " << getResult().size()
                       << ". Expected " << expectedResultsCount << ".";
  }

  return success();
}

void MeshShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        MeshOp mesh) {
  build(odsBuilder, odsState, mesh, SmallVector<MeshAxis>());
}

void MeshShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        MeshOp mesh, ArrayRef<MeshAxis> axes) {
  build(odsBuilder, odsState,
        SmallVector<Type>(axes.empty() ? mesh.getRank() : axes.size(),
                          odsBuilder.getIndexType()),
        mesh.getSymName(), MeshAxesAttr::get(odsBuilder.getContext(), axes));
}

void MeshShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        StringRef mesh, ArrayRef<MeshAxis> axes) {
  assert(!axes.empty());
  build(odsBuilder, odsState,
        SmallVector<Type>(axes.size(), odsBuilder.getIndexType()), mesh,
        MeshAxesAttr::get(odsBuilder.getContext(), axes));
}

void MeshShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResults()[0], "mesh_shape");
}

//===----------------------------------------------------------------------===//
// mesh.sharding
//===----------------------------------------------------------------------===//

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       FlatSymbolRefAttr mesh,
                       ArrayRef<MeshAxesAttr> split_axes,
                       ArrayRef<MeshAxis> partial_axes,
                       mesh::ReductionKind partial_type,
                       ArrayRef<int64_t> static_halos,
                       ArrayRef<int64_t> static_offsets) {
  return build(
      b, odsState, mesh, MeshAxesArrayAttr::get(b.getContext(), split_axes),
      ::mlir::DenseI16ArrayAttr::get(b.getContext(), partial_axes),
      ::mlir::mesh::ReductionKindAttr::get(b.getContext(), partial_type),
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_halos), {},
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_offsets), {});
}

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       FlatSymbolRefAttr mesh,
                       ArrayRef<MeshAxesAttr> split_axes) {
  return build(
      b, odsState, mesh, MeshAxesArrayAttr::get(b.getContext(), split_axes), {},
      ::mlir::mesh::ReductionKindAttr::get(b.getContext(), ReductionKind::Sum),
      {}, {}, {}, {});
}

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       llvm::StringRef mesh, ArrayRef<MeshAxesAttr> split_axes,
                       ArrayRef<int64_t> static_halos,
                       ArrayRef<int64_t> static_offsets) {
  return build(
      b, odsState, FlatSymbolRefAttr::get(b.getContext(), mesh),
      MeshAxesArrayAttr::get(b.getContext(), split_axes), {},
      ::mlir::mesh::ReductionKindAttr::get(b.getContext(), ReductionKind::Sum),
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_halos), {},
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_offsets), {});
}

void ShardingOp::build(
    ::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
    FlatSymbolRefAttr mesh, ArrayRef<MeshAxesAttr> split_axes,
    ::mlir::ArrayRef<::mlir::OpFoldResult> halo_sizes,
    ::mlir::ArrayRef<::mlir::OpFoldResult> sharded_dims_offsets) {
  mlir::SmallVector<int64_t> staticHalos, staticDims;
  mlir::SmallVector<mlir::Value> dynamicHalos, dynamicDims;
  dispatchIndexOpFoldResults(halo_sizes, dynamicHalos, staticHalos);
  dispatchIndexOpFoldResults(sharded_dims_offsets, dynamicDims, staticDims);
  return build(
      b, odsState, mesh, MeshAxesArrayAttr::get(b.getContext(), split_axes), {},
      ::mlir::mesh::ReductionKindAttr::get(b.getContext(), ReductionKind::Sum),
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), staticHalos), dynamicHalos,
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), staticDims), dynamicDims);
}

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       mlir::mesh::MeshSharding from) {

  build(b, odsState, ShardingType::get(b.getContext()), from.getMeshAttr(),
        MeshAxesArrayAttr::get(b.getContext(), from.getSplitAxes()),
        from.getPartialAxes().empty()
            ? DenseI16ArrayAttr()
            : b.getDenseI16ArrayAttr(from.getPartialAxes()),
        ::mlir::mesh::ReductionKindAttr::get(b.getContext(),
                                             from.getPartialType()),
        from.getStaticShardedDimsOffsets().empty()
            ? DenseI64ArrayAttr()
            : b.getDenseI64ArrayAttr(from.getStaticShardedDimsOffsets()),
        from.getDynamicShardedDimsOffsets(),
        from.getStaticHaloSizes().empty()
            ? DenseI64ArrayAttr()
            : b.getDenseI64ArrayAttr(from.getStaticHaloSizes()),
        from.getDynamicHaloSizes());
}

LogicalResult ShardingOp::verify() {
  llvm::SmallSet<MeshAxis, 4> visitedAxes;

  auto checkMeshAxis = [&](ArrayRef<MeshAxis> axesArray) -> LogicalResult {
    for (MeshAxis axis : axesArray) {
      if (axis < 0)
        return emitError() << "mesh axis is expected to be non-negative";
      if (!visitedAxes.insert(axis).second)
        return emitError() << "mesh axis duplicated";
    }
    return success();
  };

  for (auto subAxes : getSplitAxes().getAxes()) {
    ArrayRef<MeshAxis> subAxesArray = subAxes.asArrayRef();
    if (failed(checkMeshAxis(subAxesArray)))
      return failure();
  }
  if (getPartialAxes().has_value() &&
      failed(checkMeshAxis(getPartialAxes().value())))
    return failure();

  if (!getStaticHaloSizes().empty() && !getStaticShardedDimsOffsets().empty()) {
    return emitOpError("halo sizes and shard offsets are mutually exclusive");
  }

  if (!getStaticHaloSizes().empty()) {
    auto numSplitAxes = getSplitAxes().getAxes().size();
    for (auto splitAxis : getSplitAxes().getAxes()) {
      if (splitAxis.empty()) {
        --numSplitAxes;
      }
    }
    if (getStaticHaloSizes().size() != numSplitAxes * 2) {
      return emitError() << "halo sizes must be specified for all split axes.";
    }
  }

  return success();
}

void ShardingOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "sharding");
}

LogicalResult ShardingOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = ::getMeshAndVerify(getOperation(), getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (mlir::ShapedType::isDynamicShape(mesh->getShape()) &&
      getStaticShardedDimsOffsets().size() > 0) {
    return emitError() << "sharded dims offsets are not allowed for "
                          "devices meshes with dynamic shape.";
  }

  auto shardedDimsOffsets = getStaticShardedDimsOffsets();
  if (!shardedDimsOffsets.empty()) {
    auto meshShape = mesh.value().getShape();
    assert(!ShapedType::isDynamicShape(meshShape));
    uint64_t pos = 0;
    for (auto [tensorAxis, innerSplitAxes] : llvm::enumerate(getSplitAxes())) {
      if (!innerSplitAxes.empty()) {
        int64_t numShards = 0, off = 0;
        for (auto i : innerSplitAxes.asArrayRef()) {
          numShards += meshShape[i];
        }
        for (int64_t i = 0; i <= numShards; ++i) {
          if (shardedDimsOffsets.size() <= pos + i) {
            return emitError() << "sharded dims offsets has wrong size.";
          }
          if (!ShapedType::isDynamic(shardedDimsOffsets[pos + i])) {
            if (shardedDimsOffsets[pos + i] < off) {
              return emitError()
                     << "sharded dims offsets must be non-decreasing.";
            }
            off = shardedDimsOffsets[pos + i];
          }
        }
        pos += numShards + 1;
      }
    }
  }
  return success();
}

namespace {
// Sharding annotations "halo sizes" and "sharded dims offsets"
// are a mix of attributes and dynamic values. This canonicalization moves
// constant values to the respective attribute lists, minimizing the number
// of values.
// It also removes sharded_dims_sizes and halos if they are effectively "empty".
class NormalizeSharding final : public OpRewritePattern<ShardingOp> {
public:
  using OpRewritePattern<ShardingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShardingOp op,
                                PatternRewriter &b) const override {
    auto mixedHalos =
        getMixedValues(op.getStaticHaloSizes(), op.getDynamicHaloSizes(), b);
    auto mixedOffs = getMixedValues(op.getStaticShardedDimsOffsets(),
                                    op.getDynamicShardedDimsOffsets(), b);

    // No constant operands were folded, just return;
    bool modified = succeeded(foldDynamicIndexList(mixedHalos, true)) ||
                    succeeded(foldDynamicIndexList(mixedOffs, true));

    auto [staticHalos, dynamicHalos] = decomposeMixedValues(mixedHalos);
    auto [staticOffs, dynamicOffs] = decomposeMixedValues(mixedOffs);

    if (dynamicHalos.empty() && !staticHalos.empty()) {
      if (staticHalos[0] == 0 && llvm::all_equal(staticHalos)) {
        staticHalos.clear();
        modified = true;
      }
    }

    // Remove sharded dims offsets if they are effectively the default values,
    // e.g. if they define equi-distance between all neighboring shards.
    // Requires static-only offsets. Compares the first distance as the
    // difference between the first two offsets. Only if all consecutive
    // distances are the same, the offsets are removed.
    if (dynamicOffs.empty() && !staticOffs.empty()) {
      assert(staticOffs.size() >= 2);
      auto diff = staticOffs[1] - staticOffs[0];
      bool all_same = staticOffs.size() > 2;
      for (auto i = 2u; i < staticOffs.size(); ++i) {
        if (staticOffs[i] - staticOffs[i - 1] != diff) {
          all_same = false;
          break;
        }
      }
      if (all_same) {
        staticOffs.clear();
        modified = true;
      }
    }

    if (!modified) {
      return failure();
    }

    op.setStaticHaloSizes(staticHalos);
    op.getDynamicHaloSizesMutable().assign(dynamicHalos);
    op.setStaticShardedDimsOffsets(staticOffs);
    op.getDynamicShardedDimsOffsetsMutable().assign(dynamicOffs);

    return success();
  }
};
} // namespace

void ShardingOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             mlir::MLIRContext *context) {
  results.add<NormalizeSharding>(context);
}

//===----------------------------------------------------------------------===//
// MeshSharding
//===----------------------------------------------------------------------===//

bool MeshSharding::equalSplitAndPartialAxes(const MeshSharding &rhs) const {
  if (getMesh() != rhs.getMesh()) {
    return false;
  }

  if (getPartialAxes().size() != rhs.getPartialAxes().size() ||
      (!getPartialAxes().empty() && getPartialType() != rhs.getPartialType()) ||
      !llvm::equal(getPartialAxes(), rhs.getPartialAxes())) {
    return false;
  }

  auto minSize = std::min(getSplitAxes().size(), rhs.getSplitAxes().size());
  if (!llvm::equal(llvm::make_range(getSplitAxes().begin(),
                                    getSplitAxes().begin() + minSize),
                   llvm::make_range(rhs.getSplitAxes().begin(),
                                    rhs.getSplitAxes().begin() + minSize))) {
    return false;
  }

  return llvm::all_of(llvm::drop_begin(getSplitAxes(), minSize),
                      std::mem_fn(&MeshAxesAttr::empty)) &&
         llvm::all_of(llvm::drop_begin(rhs.getSplitAxes(), minSize),
                      std::mem_fn(&MeshAxesAttr::empty));
}

bool MeshSharding::equalHaloAndShardSizes(const MeshSharding &rhs) const {
  return equalShardSizes(rhs) && equalHaloSizes(rhs);
}

bool MeshSharding::equalShardSizes(const MeshSharding &rhs) const {
  if (rhs.getStaticShardedDimsOffsets().size() !=
          getStaticShardedDimsOffsets().size() ||
      !llvm::equal(getStaticShardedDimsOffsets(),
                   rhs.getStaticShardedDimsOffsets())) {
    return false;
  }
  if (rhs.getDynamicShardedDimsOffsets().size() !=
          getDynamicShardedDimsOffsets().size() ||
      !llvm::equal(getDynamicShardedDimsOffsets(),
                   rhs.getDynamicShardedDimsOffsets())) {
    return false;
  }
  return true;
}

bool MeshSharding::equalHaloSizes(const MeshSharding &rhs) const {
  if (rhs.getStaticHaloSizes().size() != getStaticHaloSizes().size() ||
      !llvm::equal(getStaticHaloSizes(), rhs.getStaticHaloSizes())) {
    return false;
  }
  if (rhs.getDynamicHaloSizes().size() != getDynamicHaloSizes().size() ||
      !llvm::equal(getDynamicHaloSizes(), rhs.getDynamicHaloSizes())) {
    return false;
  }
  return true;
}

bool MeshSharding::operator==(Value rhs) const {
  return equalSplitAndPartialAxes(rhs) && equalHaloAndShardSizes(rhs);
}

bool MeshSharding::operator!=(Value rhs) const { return !(*this == rhs); }

bool MeshSharding::operator==(const MeshSharding &rhs) const {
  return equalSplitAndPartialAxes(rhs) && equalHaloAndShardSizes(rhs);
}

bool MeshSharding::operator!=(const MeshSharding &rhs) const {
  return !(*this == rhs);
}

MeshSharding::MeshSharding(::mlir::FlatSymbolRefAttr mesh_) : mesh(mesh_) {}

MeshSharding::MeshSharding(Value rhs) {
  auto shardingOp = mlir::dyn_cast<ShardingOp>(rhs.getDefiningOp());
  assert(shardingOp && "expected sharding op");
  auto splitAxes = shardingOp.getSplitAxes().getAxes();
  auto partialAxes = shardingOp.getPartialAxes().value_or(ArrayRef<MeshAxis>());
  // If splitAxes and partialAxes are empty, use "empty" constructor.
  if (splitAxes.empty() && partialAxes.empty()) {
    *this = MeshSharding(shardingOp.getMeshAttr());
    return;
  }
  *this = get(shardingOp.getMeshAttr(), splitAxes, partialAxes,
              shardingOp.getPartialType().value_or(ReductionKind::Sum),
              shardingOp.getStaticHaloSizes(),
              shardingOp.getStaticShardedDimsOffsets(),
              SmallVector<Value>(shardingOp.getDynamicHaloSizes()),
              SmallVector<Value>(shardingOp.getDynamicShardedDimsOffsets()));
}

MeshSharding MeshSharding::get(::mlir::FlatSymbolRefAttr mesh_,
                               ArrayRef<MeshAxesAttr> split_axes_,
                               ArrayRef<MeshAxis> partial_axes_,
                               ReductionKind partial_type_,
                               ArrayRef<int64_t> static_halo_sizes_,
                               ArrayRef<int64_t> static_sharded_dims_offsets_,
                               ArrayRef<Value> dynamic_halo_sizes_,
                               ArrayRef<Value> dynamic_sharded_dims_offsets_) {
  MeshSharding res(mesh_);
  if (split_axes_.empty() && partial_axes_.empty()) {
    return res;
  }

  res.split_axes.resize(split_axes_.size());
  for (auto [i, axis] : llvm::enumerate(split_axes_)) {
    res.split_axes[i] =
        MeshAxesAttr::get(mesh_.getContext(), axis.asArrayRef());
  }

  auto clone = [](const auto src, auto &dst) {
    dst.resize(src.size());
    llvm::copy(src, dst.begin());
  };

  clone(partial_axes_, res.partial_axes);
  res.partial_type = partial_type_;
  clone(static_halo_sizes_, res.static_halo_sizes);
  clone(static_sharded_dims_offsets_, res.static_sharded_dims_offsets);
  clone(dynamic_halo_sizes_, res.dynamic_halo_sizes);
  clone(dynamic_sharded_dims_offsets_, res.dynamic_sharded_dims_offsets);

  return res;
}

//===----------------------------------------------------------------------===//
// mesh.shard_shape
//===----------------------------------------------------------------------===//

void ShardShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult()[0], "shard_shape");
}

void ShardShapeOp::build(::mlir::OpBuilder &odsBuilder,
                         ::mlir::OperationState &odsState,
                         ::llvm::ArrayRef<int64_t> dims,
                         ArrayRef<Value> dims_dyn, ::mlir::Value sharding,
                         ::mlir::ValueRange device) {
  SmallVector<mlir::Type> resType(dims.size(), odsBuilder.getIndexType());
  build(odsBuilder, odsState, resType, dims, dims_dyn, sharding,
        SmallVector<int64_t>(device.size(), ShapedType::kDynamic), device);
}

//===----------------------------------------------------------------------===//
// mesh.shard op
//===----------------------------------------------------------------------===//

void ShardOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "sharding_annotated");
}

namespace {
// Determine if the given ShardOp is a duplicate of another ShardOp
// on the same value. This can happen if constant values are sharded.
class FoldDuplicateShardOp final : public OpRewritePattern<ShardOp> {
public:
  using OpRewritePattern<ShardOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ShardOp op, PatternRewriter &b) const override {
    // Get the use-list of the value being sharded and check if it has more than
    // one use.
    Value value = op.getSrc();
    if (value.hasOneUse() || value.getDefiningOp<ShardOp>()) {
      return failure();
    }

    // Iterate through the uses of the value to find a duplicate ShardOp.
    for (auto &use : value.getUses()) {
      if (use.getOwner() != op.getOperation()) {
        auto otherOp = dyn_cast<ShardOp>(use.getOwner());
        if (!otherOp || !otherOp->isBeforeInBlock(op)) {
          return failure();
        }
        // Create a MeshSharding object for the current and the other ShardOp
        // If the two are equal replace current op with the other op.
        MeshSharding currentSharding(op.getSharding());
        MeshSharding otherSharding(otherOp.getSharding());
        if (currentSharding == otherSharding) {
          b.replaceAllUsesWith(op.getResult(), otherOp.getResult());
          b.eraseOp(op.getOperation());
        } else {
          // use the other sharding as input for op
          op.getSrcMutable().assign(otherOp.getResult());
        }
        return success();
      }
    }

    return failure();
  }
};
} // namespace

void ShardOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                          mlir::MLIRContext *context) {
  results.add<FoldDuplicateShardOp>(context);
}

//===----------------------------------------------------------------------===//
// mesh.process_multi_index op
//===----------------------------------------------------------------------===//

LogicalResult
ProcessMultiIndexOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = ::getMeshAndVerify(getOperation(), getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyMeshAxes(getLoc(), getAxes(), mesh.value()))) {
    return failure();
  }

  size_t expectedResultsCount =
      getAxes().empty() ? mesh->getRank() : getAxes().size();
  if (getResult().size() != expectedResultsCount) {
    return emitError() << "Unexpected number of results " << getResult().size()
                       << ". Expected " << expectedResultsCount << ".";
  }

  return success();
}

void ProcessMultiIndexOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                MeshOp mesh) {
  build(odsBuilder, odsState,
        SmallVector<Type>(mesh.getRank(), odsBuilder.getIndexType()),
        mesh.getSymName(), ArrayRef<MeshAxis>());
}

void ProcessMultiIndexOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                StringRef mesh, ArrayRef<MeshAxis> axes) {
  build(odsBuilder, odsState,
        SmallVector<Type>(axes.size(), odsBuilder.getIndexType()), mesh,
        MeshAxesAttr::get(odsBuilder.getContext(), axes));
}

void ProcessMultiIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResults()[0], "proc_linear_idx");
}

//===----------------------------------------------------------------------===//
// mesh.process_linear_index op
//===----------------------------------------------------------------------===//

LogicalResult
ProcessLinearIndexOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = ::getMeshAndVerify(getOperation(), getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  return success();
}

void ProcessLinearIndexOp::build(OpBuilder &odsBuilder,
                                 OperationState &odsState, MeshOp mesh) {
  build(odsBuilder, odsState, mesh.getSymName());
}

void ProcessLinearIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "proc_linear_idx");
}

//===----------------------------------------------------------------------===//
// mesh.neighbors_linear_indices op
//===----------------------------------------------------------------------===//

LogicalResult
NeighborsLinearIndicesOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = ::getMeshAndVerify(getOperation(), getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  return success();
}

void NeighborsLinearIndicesOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getNeighborDown(), "down_linear_idx");
  setNameFn(getNeighborUp(), "up_linear_idx");
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
  auto resultRank = cast<ShapedType>(result.getType()).getRank();
  if (gatherAxis < 0 || gatherAxis >= resultRank) {
    return emitError(result.getLoc())
           << "Gather axis " << gatherAxis << " is out of bounds [0, "
           << resultRank << ").";
  }

  ShapedType operandType = cast<ShapedType>(operand.getType());
  ShapedType resultType = cast<ShapedType>(result.getType());
  auto deviceGroupSize =
      DimensionSize(collectiveProcessGroupSize(meshAxes, meshShape));
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
  ShapedType operandType = cast<ShapedType>(operand.getType());
  ShapedType resultType = cast<ShapedType>(result.getType());
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
      DimensionSize(collectiveProcessGroupSize(meshAxes, meshShape));
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

static LogicalResult verifyScatterOrSliceOperandAndResultShape(
    Value operand, Value result, int64_t tensorAxis,
    ArrayRef<MeshAxis> meshAxes, ArrayRef<int64_t> meshShape) {
  ShapedType operandType = cast<ShapedType>(operand.getType());
  ShapedType resultType = cast<ShapedType>(result.getType());
  for (int64_t axis = 0; axis < operandType.getRank(); ++axis) {
    if (axis != tensorAxis) {
      if (failed(verifyDimensionCompatibility(
              result.getLoc(), operandType.getDimSize(axis),
              resultType.getDimSize(axis), axis))) {
        return failure();
      }
    }
  }

  auto deviceGroupSize =
      DimensionSize(collectiveProcessGroupSize(meshAxes, meshShape));
  auto operandScatterDimSize =
      DimensionSize(operandType.getDimSize(tensorAxis));
  if (!operandScatterDimSize.isDynamic() && !deviceGroupSize.isDynamic() &&
      int64_t(operandScatterDimSize) % int64_t(deviceGroupSize) != 0) {
    return emitError(result.getLoc())
           << "Operand dimension size " << int64_t(operandScatterDimSize)
           << " is not divisible by collective device group size "
           << int64_t(deviceGroupSize) << " for tensor axis " << tensorAxis
           << ".";
  }
  DimensionSize expectedResultTensorDimSize =
      operandScatterDimSize / deviceGroupSize;
  if (failed(verifyDimensionCompatibility(
          result.getLoc(), expectedResultTensorDimSize.value(),
          resultType.getDimSize(tensorAxis), tensorAxis))) {
    return failure();
  }

  return success();
}

static RankedTensorType sliceResultType(Type operandType, MeshOp mesh,
                                        ArrayRef<MeshAxis> meshAxes,
                                        int64_t sliceAxis) {
  RankedTensorType operandRankedTensorType =
      cast<RankedTensorType>(operandType);
  DimensionSize operandSliceAxisSize =
      operandRankedTensorType.getShape()[sliceAxis];
  SmallVector<int64_t> resultShape =
      llvm::to_vector(operandRankedTensorType.getShape());

  resultShape[sliceAxis] =
      operandSliceAxisSize /
      DimensionSize(collectiveProcessGroupSize(meshAxes, mesh));
  return operandRankedTensorType.clone(resultShape);
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
                                           mesh.value().getShape());
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<AllGatherOp>>(context);
}

void AllGatherOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_gather");
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

void AllReduceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        Value input, StringRef mesh,
                        ArrayRef<MeshAxis> meshAxes, ReductionKind reduction) {
  build(odsBuilder, odsState, input.getType(), mesh, meshAxes, input,
        reduction);
}

void AllReduceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_reduce");
}

//===----------------------------------------------------------------------===//
// mesh.all_slice op
//===----------------------------------------------------------------------===//

LogicalResult AllSliceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  return verifyScatterOrSliceOperandAndResultShape(
      getOperand(), getResult(), getSliceAxis().getSExtValue(), getMeshAxes(),
      mesh.value().getShape());
}

void AllSliceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<AllSliceOp>>(context);
}

void AllSliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Value input, MeshOp mesh, ArrayRef<MeshAxis> meshAxes,
                       int64_t sliceAxis) {
  Type resultType = sliceResultType(input.getType(), mesh, meshAxes, sliceAxis);
  build(odsBuilder, odsState, resultType, input, mesh.getSymName(), meshAxes,
        sliceAxis);
}

void AllSliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Type resultType, Value input, StringRef mesh,
                       ArrayRef<MeshAxis> meshAxes, int64_t sliceAxis) {
  build(odsBuilder, odsState, resultType, mesh, meshAxes, input,
        APInt(sizeof(sliceAxis) * CHAR_BIT, sliceAxis));
}

void AllSliceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_slice");
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
      getConcatAxis().getSExtValue(), getMeshAxes(), mesh.value().getShape());
}

void AllToAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<AllToAllOp>>(context);
}

void AllToAllOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_to_all");
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
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(),
                                 mesh.value().getShape()))) {
    return failure();
  }

  return success();
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<BroadcastOp>>(context);
}

void BroadcastOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "broadcast");
}

//===----------------------------------------------------------------------===//
// mesh.gather op
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(),
                                 mesh.value().getShape()))) {
    return failure();
  }

  auto gatherAxis = getGatherAxis().getSExtValue();
  return verifyGatherOperandAndResultShape(getInput(), getResult(), gatherAxis,
                                           getMeshAxes(),
                                           mesh.value().getShape());
}

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<GatherOp>>(context);
}

void GatherOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "gather");
}

//===----------------------------------------------------------------------===//
// mesh.recv op
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (getSource() &&
      failed(verifyInGroupDevice(getLoc(), getSourceAttrName(),
                                 getSource().value(), getSourceDynamic(),
                                 getMeshAxes(), mesh.value().getShape()))) {
    return failure();
  }
  return success();
}

void RecvOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<RecvOp>>(context);
}

void RecvOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "recv");
}

//===----------------------------------------------------------------------===//
// mesh.reduce op
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(),
                                 mesh.value().getShape()))) {
    return failure();
  }

  return success();
}

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<ReduceOp>>(context);
}

void ReduceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "reduce");
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

  return verifyScatterOrSliceOperandAndResultShape(
      getOperand(), getResult(), getScatterAxis().getSExtValue(), getMeshAxes(),
      mesh.value().getShape());
}

void ReduceScatterOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<ReduceScatterOp>>(context);
}

void ReduceScatterOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "reduce_scatter");
}

//===----------------------------------------------------------------------===//
// mesh.scatter op
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getMeshAxes(),
                                 mesh.value().getShape()))) {
    return failure();
  }

  auto scatterAxis = getScatterAxis().getSExtValue();
  return verifyScatterOrSliceOperandAndResultShape(getInput(), getResult(),
                                                   scatterAxis, getMeshAxes(),
                                                   mesh.value().getShape());
}

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<ScatterOp>>(context);
}

void ScatterOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "scatter");
}

//===----------------------------------------------------------------------===//
// mesh.send op
//===----------------------------------------------------------------------===//

LogicalResult SendOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerifyAxes(*this, symbolTable);
  if (failed(mesh)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getDestinationAttrName(),
                                 getDestination(), getDestinationDynamic(),
                                 getMeshAxes(), mesh.value().getShape()))) {
    return failure();
  }
  return success();
}

void SendOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<EmptyMeshAxesCanonicalizationPattern<SendOp>>(context);
}

void SendOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "send");
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
  if (!llvm::is_contained(meshAxes, shiftAxis)) {
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

void ShiftOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "shift");
}

//===----------------------------------------------------------------------===//
// mesh.update_halo op
//===----------------------------------------------------------------------===//

LogicalResult
UpdateHaloOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto mesh = getMeshAndVerify(getOperation(), getMeshAttr(), symbolTable);
  if (failed(mesh)) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Mesh/IR/MeshTypes.cpp.inc"

#include "mlir/Dialect/Mesh/IR/MeshEnums.cpp.inc"
