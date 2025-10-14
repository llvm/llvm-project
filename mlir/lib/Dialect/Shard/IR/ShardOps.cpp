//===- ShardOps.cpp - Shard Dialect Operations ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Shard/IR/ShardOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
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
#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <optional>
#include <utility>

#define DEBUG_TYPE "shard-ops"

using namespace mlir;
using namespace mlir::shard;

#include "mlir/Dialect/Shard/IR/ShardDialect.cpp.inc"

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

SmallVector<Value>
mlir::shard::getMixedAsValues(OpBuilder b, const Location &loc,
                              llvm::ArrayRef<int64_t> statics,
                              ValueRange dynamics, Type type) {
  SmallVector<Value> values;
  auto dyn = dynamics.begin();
  Type i64 = b.getI64Type();
  if (!type)
    type = i64;
  assert((i64 == type || b.getIndexType() == type) &&
         "expected an i64 or an intex type");
  for (auto s : statics) {
    if (s == ShapedType::kDynamic) {
      values.emplace_back(*(dyn++));
    } else {
      TypedAttr val = type == i64 ? b.getI64IntegerAttr(s) : b.getIndexAttr(s);
      values.emplace_back(arith::ConstantOp::create(b, loc, type, val));
    }
  }
  return values;
}

//===----------------------------------------------------------------------===//
// Inliner
//===----------------------------------------------------------------------===//

namespace {
struct ShardInlinerinterface : public DialectInlinerInterface {
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
// Shard dialect
//===----------------------------------------------------------------------===//

void ShardDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Shard/IR/ShardOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/Shard/IR/ShardAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/Shard/IR/ShardTypes.cpp.inc"
      >();
  addInterface<ShardInlinerinterface>();
}

Operation *ShardDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

//===----------------------------------------------------------------------===//
// Shard utilities
//===----------------------------------------------------------------------===//

static FailureOr<GridOp> getGridAndVerify(Operation *op,
                                          FlatSymbolRefAttr gridSymbol,
                                          SymbolTableCollection &symbolTable) {
  shard::GridOp grid = getGridOrNull(op, gridSymbol, symbolTable);
  if (!grid) {
    return op->emitError() << "Undefined required grid symbol \""
                           << gridSymbol.getValue() << "\".";
  }

  return grid;
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

static LogicalResult verifyGridAxes(Location loc, ArrayRef<GridAxis> axes,
                                    GridOp grid) {
  SmallVector<GridAxis> sorted = llvm::to_vector(axes);
  llvm::sort(sorted);
  if (!isUnique(sorted.begin(), sorted.end())) {
    return emitError(loc) << "Grid axes contains duplicate elements.";
  }

  GridAxis rank = grid.getRank();
  for (auto axis : axes) {
    if (axis >= rank || axis < 0) {
      return emitError(loc)
             << "0-based grid axis index " << axis
             << " is out of bounds. The referenced grid \"" << grid.getSymName()
             << "\" is of rank " << rank << ".";
    }
  }

  return success();
}

template <typename Op>
static FailureOr<GridOp>
getGridAndVerifyAxes(Op op, SymbolTableCollection &symbolTable) {
  auto grid =
      ::getGridAndVerify(op.getOperation(), op.getGridAttr(), symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyGridAxes(op.getLoc(), op.getGridAxes(), grid.value()))) {
    return failure();
  }
  return grid;
}

template <typename InShape, typename GridShape, typename SplitAxes,
          typename OutShape>
static void shardShape(const InShape &inShape, const GridShape &gridShape,
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
    auto isDynShape = ShapedType::isDynamicShape(gridShape);
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
            numShards += gridShape[i];
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
          collectiveProcessGroupSize(innerSplitAxes.asArrayRef(), gridShape));
    }

    if (!haloSizes.empty()) {
      // add halo sizes if requested
      int haloAxis = 0;
      for (auto [tensorAxis, innerSplitAxes] : llvm::enumerate(splitAxes)) {
        if (ShapedType::isStatic(outShape[tensorAxis]) &&
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

ShapedType shard::shardShapedType(ShapedType shape, GridOp grid,
                                  Sharding sharding) {
  using Dim = std::decay_t<decltype(shape.getDimSize(0))>;
  SmallVector<Dim> resShapeArr(shape.getShape().size());
  shardShape(shape.getShape(), grid.getShape(), sharding.getSplitAxes(),
             resShapeArr, sharding.getStaticShardedDimsOffsets(),
             sharding.getStaticHaloSizes());
  return shape.clone(resShapeArr);
}

Type shard::shardType(Type type, GridOp grid, Sharding sharding) {
  RankedTensorType rankedTensorType = dyn_cast<RankedTensorType>(type);
  if (rankedTensorType && !rankedTensorType.getShape().empty()) {
    return shardShapedType(rankedTensorType, grid, sharding);
  }
  return type;
}

static void maybeInsertTargetShardingAnnotationImpl(Sharding sharding,
                                                    Value &operandValue,
                                                    Operation *operandOp,
                                                    OpBuilder &builder,
                                                    ShardOp &newShardOp) {
  OpBuilder::InsertionGuard insertionGuard(builder);
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
        ShardingOp::create(builder, operandValue.getLoc(), sharding);
    newShardOp = ShardOp::create(builder, operandValue.getLoc(), operandValue,
                                 shardingOp,
                                 /*annotate_for_users*/ false);
  }
  operandValue.replaceUsesWithIf(
      newShardOp, [operandOp, operandValue](OpOperand &use) {
        return use.getOwner() == operandOp && use.get() == operandValue;
      });

  if (!shardOp || shardOp.getAnnotateForUsers()) {
    return;
  }

  auto newShardOp2 = ShardOp::create(builder, operandValue.getLoc(), newShardOp,
                                     newShardOp.getSharding(),
                                     /*annotate_for_users*/ true);
  newShardOp.getResult().replaceAllUsesExcept(newShardOp2, newShardOp2);
}

void mlir::shard::maybeInsertTargetShardingAnnotation(Sharding sharding,
                                                      OpResult result,
                                                      OpBuilder &builder) {
  ShardOp newShardOp;
  SmallVector<std::pair<Value, Operation *>> uses;
  for (auto &use : result.getUses()) {
    uses.emplace_back(use.get(), use.getOwner());
  }
  for (auto &[operandValue, operandOp] : uses) {
    maybeInsertTargetShardingAnnotationImpl(sharding, operandValue, operandOp,
                                            builder, newShardOp);
  }
}

void mlir::shard::maybeInsertSourceShardingAnnotation(Sharding sharding,
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
      ShardingOp::create(builder, operand.get().getLoc(), sharding);
  auto newShardOp =
      ShardOp::create(builder, operandValue.getLoc(), operandValue, shardingOp,
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
      ShardOp::create(builder, operandValue.getLoc(), operandValue, shardingOp,
                      /*annotate_for_users*/ false);
  rewriter.replaceUsesWithIf(
      newShardOp.getSrc(), newPreceedingShardOp, [&newShardOp](OpOperand &use) {
        return use.getOwner() == newShardOp.getOperation();
      });
}

//===----------------------------------------------------------------------===//
// shard.grid op
//===----------------------------------------------------------------------===//

LogicalResult GridOp::verify() {
  int64_t rank = getRank();

  if (rank <= 0)
    return emitOpError("rank of grid is expected to be a positive integer");

  for (int64_t dimSize : getShape()) {
    if (dimSize < 0 && ShapedType::isStatic(dimSize))
      return emitOpError("dimension size of a grid is expected to be "
                         "non-negative or dynamic");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// shard.grid_shape op
//===----------------------------------------------------------------------===//

LogicalResult
GridShapeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = ::getGridAndVerify(getOperation(), getGridAttr(), symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyGridAxes(getLoc(), getAxes(), grid.value()))) {
    return failure();
  }

  size_t expectedResultsCount =
      getAxes().empty() ? grid->getRank() : getAxes().size();
  if (getResult().size() != expectedResultsCount) {
    return emitError() << "Unexpected number of results " << getResult().size()
                       << ". Expected " << expectedResultsCount << ".";
  }

  return success();
}

void GridShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        GridOp grid) {
  build(odsBuilder, odsState, grid, SmallVector<GridAxis>());
}

void GridShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        GridOp grid, ArrayRef<GridAxis> axes) {
  build(odsBuilder, odsState,
        SmallVector<Type>(axes.empty() ? grid.getRank() : axes.size(),
                          odsBuilder.getIndexType()),
        grid.getSymName(), GridAxesAttr::get(odsBuilder.getContext(), axes));
}

void GridShapeOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        StringRef grid, ArrayRef<GridAxis> axes) {
  assert(!axes.empty());
  build(odsBuilder, odsState,
        SmallVector<Type>(axes.size(), odsBuilder.getIndexType()), grid,
        GridAxesAttr::get(odsBuilder.getContext(), axes));
}

void GridShapeOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResults()[0], "grid_shape");
}

//===----------------------------------------------------------------------===//
// shard.sharding
//===----------------------------------------------------------------------===//

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       FlatSymbolRefAttr grid,
                       ArrayRef<GridAxesAttr> split_axes,
                       ArrayRef<int64_t> static_halos,
                       ArrayRef<int64_t> static_offsets) {
  return build(
      b, odsState, grid, GridAxesArrayAttr::get(b.getContext(), split_axes),
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_halos), {},
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_offsets), {});
}

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       llvm::StringRef grid, ArrayRef<GridAxesAttr> split_axes,
                       ArrayRef<int64_t> static_halos,
                       ArrayRef<int64_t> static_offsets) {
  return build(b, odsState, FlatSymbolRefAttr::get(b.getContext(), grid),
               GridAxesArrayAttr::get(b.getContext(), split_axes),
               ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_halos), {},
               ::mlir::DenseI64ArrayAttr::get(b.getContext(), static_offsets),
               {});
}

void ShardingOp::build(
    ::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
    FlatSymbolRefAttr grid, ArrayRef<GridAxesAttr> split_axes,
    ::mlir::ArrayRef<::mlir::OpFoldResult> halo_sizes,
    ::mlir::ArrayRef<::mlir::OpFoldResult> sharded_dims_offsets) {
  mlir::SmallVector<int64_t> staticHalos, staticDims;
  mlir::SmallVector<mlir::Value> dynamicHalos, dynamicDims;
  dispatchIndexOpFoldResults(halo_sizes, dynamicHalos, staticHalos);
  dispatchIndexOpFoldResults(sharded_dims_offsets, dynamicDims, staticDims);
  return build(
      b, odsState, grid, GridAxesArrayAttr::get(b.getContext(), split_axes),
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), staticHalos), dynamicHalos,
      ::mlir::DenseI64ArrayAttr::get(b.getContext(), staticDims), dynamicDims);
}

void ShardingOp::build(::mlir::OpBuilder &b, ::mlir::OperationState &odsState,
                       mlir::shard::Sharding from) {

  build(b, odsState, ShardingType::get(b.getContext()), from.getGridAttr(),
        GridAxesArrayAttr::get(b.getContext(), from.getSplitAxes()),
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
  llvm::SmallSet<GridAxis, 4> visitedAxes;

  auto checkGridAxis = [&](ArrayRef<GridAxis> axesArray) -> LogicalResult {
    for (GridAxis axis : axesArray) {
      if (axis < 0)
        return emitError() << "grid axis is expected to be non-negative";
      if (!visitedAxes.insert(axis).second)
        return emitError() << "grid axis duplicated";
    }
    return success();
  };

  for (auto subAxes : getSplitAxes().getAxes()) {
    ArrayRef<GridAxis> subAxesArray = subAxes.asArrayRef();
    if (failed(checkGridAxis(subAxesArray)))
      return failure();
  }

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
  auto grid = ::getGridAndVerify(getOperation(), getGridAttr(), symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (mlir::ShapedType::isDynamicShape(grid->getShape()) &&
      getStaticShardedDimsOffsets().size() > 0) {
    return emitError() << "sharded dims offsets are not allowed for "
                          "device grids with dynamic shape.";
  }

  auto shardedDimsOffsets = getStaticShardedDimsOffsets();
  if (!shardedDimsOffsets.empty()) {
    auto gridShape = grid.value().getShape();
    assert(ShapedType::isStaticShape(gridShape));
    uint64_t pos = 0;
    for (auto [tensorAxis, innerSplitAxes] : llvm::enumerate(getSplitAxes())) {
      if (!innerSplitAxes.empty()) {
        int64_t numShards = 0, off = 0;
        for (auto i : innerSplitAxes.asArrayRef()) {
          numShards += gridShape[i];
        }
        for (int64_t i = 0; i <= numShards; ++i) {
          if (shardedDimsOffsets.size() <= pos + i) {
            return emitError() << "sharded dims offsets has wrong size.";
          }
          if (ShapedType::isStatic(shardedDimsOffsets[pos + i])) {
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
// Sharding
//===----------------------------------------------------------------------===//

bool Sharding::equalSplitAxes(const Sharding &rhs) const {
  if (getGrid() != rhs.getGrid()) {
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
                      std::mem_fn(&GridAxesAttr::empty)) &&
         llvm::all_of(llvm::drop_begin(rhs.getSplitAxes(), minSize),
                      std::mem_fn(&GridAxesAttr::empty));
}

bool Sharding::equalHaloAndShardSizes(const Sharding &rhs) const {
  return equalShardSizes(rhs) && equalHaloSizes(rhs);
}

bool Sharding::equalShardSizes(const Sharding &rhs) const {
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

bool Sharding::equalHaloSizes(const Sharding &rhs) const {
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

bool Sharding::operator==(Value rhs) const {
  return equalSplitAxes(rhs) && equalHaloAndShardSizes(rhs);
}

bool Sharding::operator!=(Value rhs) const { return !(*this == rhs); }

bool Sharding::operator==(const Sharding &rhs) const {
  return equalSplitAxes(rhs) && equalHaloAndShardSizes(rhs);
}

bool Sharding::operator!=(const Sharding &rhs) const { return !(*this == rhs); }

Sharding::Sharding(::mlir::FlatSymbolRefAttr grid_) : grid(grid_) {}

Sharding::Sharding(Value rhs) {
  auto shardingOp = rhs.getDefiningOp<ShardingOp>();
  assert(shardingOp && "expected sharding op");
  auto splitAxes = shardingOp.getSplitAxes().getAxes();
  // If splitAxes are empty, use "empty" constructor.
  if (splitAxes.empty()) {
    *this = Sharding(shardingOp.getGridAttr());
    return;
  }
  *this =
      get(shardingOp.getGridAttr(), splitAxes, shardingOp.getStaticHaloSizes(),
          shardingOp.getStaticShardedDimsOffsets(),
          SmallVector<Value>(shardingOp.getDynamicHaloSizes()),
          SmallVector<Value>(shardingOp.getDynamicShardedDimsOffsets()));
}

Sharding Sharding::get(::mlir::FlatSymbolRefAttr grid_,
                       ArrayRef<GridAxesAttr> split_axes_,
                       ArrayRef<int64_t> static_halo_sizes_,
                       ArrayRef<int64_t> static_sharded_dims_offsets_,
                       ArrayRef<Value> dynamic_halo_sizes_,
                       ArrayRef<Value> dynamic_sharded_dims_offsets_) {
  Sharding res(grid_);
  if (split_axes_.empty()) {
    return res;
  }

  res.split_axes.resize(split_axes_.size());
  for (auto [i, axis] : llvm::enumerate(split_axes_)) {
    res.split_axes[i] =
        GridAxesAttr::get(grid_.getContext(), axis.asArrayRef());
  }

  auto clone = [](const auto src, auto &dst) {
    dst.resize(src.size());
    llvm::copy(src, dst.begin());
  };

  clone(static_halo_sizes_, res.static_halo_sizes);
  clone(static_sharded_dims_offsets_, res.static_sharded_dims_offsets);
  clone(dynamic_halo_sizes_, res.dynamic_halo_sizes);
  clone(dynamic_sharded_dims_offsets_, res.dynamic_sharded_dims_offsets);

  return res;
}

//===----------------------------------------------------------------------===//
// shard.shard_shape
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
// shard.shard op
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
        // Create a Sharding object for the current and the other ShardOp
        // If the two are equal replace current op with the other op.
        Sharding currentSharding(op.getSharding());
        Sharding otherSharding(otherOp.getSharding());
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
// shard.process_multi_index op
//===----------------------------------------------------------------------===//

LogicalResult
ProcessMultiIndexOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = ::getGridAndVerify(getOperation(), getGridAttr(), symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyGridAxes(getLoc(), getAxes(), grid.value()))) {
    return failure();
  }

  size_t expectedResultsCount =
      getAxes().empty() ? grid->getRank() : getAxes().size();
  if (getResult().size() != expectedResultsCount) {
    return emitError() << "Unexpected number of results " << getResult().size()
                       << ". Expected " << expectedResultsCount << ".";
  }

  return success();
}

void ProcessMultiIndexOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                GridOp grid) {
  build(odsBuilder, odsState,
        SmallVector<Type>(grid.getRank(), odsBuilder.getIndexType()),
        grid.getSymName(), ArrayRef<GridAxis>());
}

void ProcessMultiIndexOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                                StringRef grid, ArrayRef<GridAxis> axes) {
  build(odsBuilder, odsState,
        SmallVector<Type>(axes.size(), odsBuilder.getIndexType()), grid,
        GridAxesAttr::get(odsBuilder.getContext(), axes));
}

void ProcessMultiIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResults()[0], "proc_linear_idx");
}

//===----------------------------------------------------------------------===//
// shard.process_linear_index op
//===----------------------------------------------------------------------===//

LogicalResult
ProcessLinearIndexOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = ::getGridAndVerify(getOperation(), getGridAttr(), symbolTable);
  if (failed(grid)) {
    return failure();
  }
  return success();
}

void ProcessLinearIndexOp::build(OpBuilder &odsBuilder,
                                 OperationState &odsState, GridOp grid) {
  build(odsBuilder, odsState, grid.getSymName());
}

void ProcessLinearIndexOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "proc_linear_idx");
}

//===----------------------------------------------------------------------===//
// shard.neighbors_linear_indices op
//===----------------------------------------------------------------------===//

LogicalResult
NeighborsLinearIndicesOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = ::getGridAndVerify(getOperation(), getGridAttr(), symbolTable);
  if (failed(grid)) {
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
struct EmptyGridAxesCanonicalizationPattern : OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;
  LogicalResult matchAndRewrite(Op op,
                                PatternRewriter &rewriter) const override {
    auto gridAxes = op.getGridAxes();
    if (!gridAxes.empty()) {
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
                                         ArrayRef<GridAxis> gridAxes,
                                         ArrayRef<int64_t> gridShape) {
  if (device.size() != gridAxes.size()) {
    return emitError(loc) << "In-group device \"" << deviceName
                          << "\" has unexpected multi-index size "
                          << device.size() << ". Expected " << gridAxes.size()
                          << ".";
  }

  for (size_t i = 0; i < device.size(); ++i) {
    if (ShapedType::isStatic(device[i]) &&
        ShapedType::isStatic(gridShape[gridAxes[i]]) &&
        gridShape[gridAxes[i]] <= device[i]) {
      return emitError(loc)
             << "Out of bounds coordinate " << i << " for in-group device \""
             << deviceName << "\"."
             << " Got " << device[i] << ", but expected value in the range [0, "
             << (gridShape[gridAxes[i]] - 1) << "].";
    }
  }
  return success();
}

static LogicalResult verifyDimensionCompatibility(Location loc,
                                                  int64_t expectedDimSize,
                                                  int64_t resultDimSize,
                                                  int64_t resultAxis) {
  if (ShapedType::isStatic(resultDimSize) && expectedDimSize != resultDimSize) {
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
    ArrayRef<GridAxis> gridAxes, ArrayRef<int64_t> gridShape) {
  auto resultRank = cast<ShapedType>(result.getType()).getRank();
  if (gatherAxis < 0 || gatherAxis >= resultRank) {
    return emitError(result.getLoc())
           << "Gather axis " << gatherAxis << " is out of bounds [0, "
           << resultRank << ").";
  }

  ShapedType operandType = cast<ShapedType>(operand.getType());
  ShapedType resultType = cast<ShapedType>(result.getType());
  auto deviceGroupSize =
      DimensionSize(collectiveProcessGroupSize(gridAxes, gridShape));
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
    ArrayRef<GridAxis> gridAxes, ArrayRef<int64_t> gridShape) {
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
      DimensionSize(collectiveProcessGroupSize(gridAxes, gridShape));
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
    ArrayRef<GridAxis> gridAxes, ArrayRef<int64_t> gridShape) {
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
      DimensionSize(collectiveProcessGroupSize(gridAxes, gridShape));
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

static RankedTensorType sliceResultType(Type operandType, GridOp grid,
                                        ArrayRef<GridAxis> gridAxes,
                                        int64_t sliceAxis) {
  RankedTensorType operandRankedTensorType =
      cast<RankedTensorType>(operandType);
  DimensionSize operandSliceAxisSize =
      operandRankedTensorType.getShape()[sliceAxis];
  SmallVector<int64_t> resultShape =
      llvm::to_vector(operandRankedTensorType.getShape());

  resultShape[sliceAxis] =
      operandSliceAxisSize /
      DimensionSize(collectiveProcessGroupSize(gridAxes, grid));
  return operandRankedTensorType.clone(resultShape);
}

//===----------------------------------------------------------------------===//
// shard.all_gather op
//===----------------------------------------------------------------------===//

LogicalResult
AllGatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  auto gatherAxis = getGatherAxis().getSExtValue();
  return verifyGatherOperandAndResultShape(getOperand(), getResult(),
                                           gatherAxis, getGridAxes(),
                                           grid.value().getShape());
}

void AllGatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<AllGatherOp>>(context);
}

void AllGatherOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_gather");
}

//===----------------------------------------------------------------------===//
// shard.all_reduce op
//===----------------------------------------------------------------------===//

LogicalResult
AllReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return getGridAndVerifyAxes(*this, symbolTable);
}

void AllReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<AllReduceOp>>(context);
}

void AllReduceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                        Value input, StringRef grid,
                        ArrayRef<GridAxis> gridAxes, ReductionKind reduction) {
  build(odsBuilder, odsState, input.getType(), grid, gridAxes, input,
        reduction);
}

void AllReduceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_reduce");
}

//===----------------------------------------------------------------------===//
// shard.all_slice op
//===----------------------------------------------------------------------===//

LogicalResult AllSliceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  return verifyScatterOrSliceOperandAndResultShape(
      getOperand(), getResult(), getSliceAxis().getSExtValue(), getGridAxes(),
      grid.value().getShape());
}

void AllSliceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<AllSliceOp>>(context);
}

void AllSliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Value input, GridOp grid, ArrayRef<GridAxis> gridAxes,
                       int64_t sliceAxis) {
  Type resultType = sliceResultType(input.getType(), grid, gridAxes, sliceAxis);
  build(odsBuilder, odsState, resultType, input, grid.getSymName(), gridAxes,
        sliceAxis);
}

void AllSliceOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       Type resultType, Value input, StringRef grid,
                       ArrayRef<GridAxis> gridAxes, int64_t sliceAxis) {
  build(odsBuilder, odsState, resultType, grid, gridAxes, input,
        APInt(sizeof(sliceAxis) * CHAR_BIT, sliceAxis));
}

void AllSliceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_slice");
}

//===----------------------------------------------------------------------===//
// shard.all_to_all op
//===----------------------------------------------------------------------===//

LogicalResult AllToAllOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }

  return verifyAllToAllOperandAndResultShape(
      getOperand(), getResult(), getSplitAxis().getSExtValue(),
      getConcatAxis().getSExtValue(), getGridAxes(), grid.value().getShape());
}

void AllToAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<AllToAllOp>>(context);
}

void AllToAllOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "all_to_all");
}

//===----------------------------------------------------------------------===//
// shard.broadcast op
//===----------------------------------------------------------------------===//

LogicalResult
BroadcastOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getGridAxes(),
                                 grid.value().getShape()))) {
    return failure();
  }

  return success();
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<BroadcastOp>>(context);
}

void BroadcastOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "broadcast");
}

//===----------------------------------------------------------------------===//
// shard.gather op
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getGridAxes(),
                                 grid.value().getShape()))) {
    return failure();
  }

  auto gatherAxis = getGatherAxis().getSExtValue();
  return verifyGatherOperandAndResultShape(getInput(), getResult(), gatherAxis,
                                           getGridAxes(),
                                           grid.value().getShape());
}

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<GatherOp>>(context);
}

void GatherOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "gather");
}

//===----------------------------------------------------------------------===//
// shard.recv op
//===----------------------------------------------------------------------===//

LogicalResult RecvOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (getSource() &&
      failed(verifyInGroupDevice(getLoc(), getSourceAttrName(),
                                 getSource().value(), getSourceDynamic(),
                                 getGridAxes(), grid.value().getShape()))) {
    return failure();
  }
  return success();
}

void RecvOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<RecvOp>>(context);
}

void RecvOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "recv");
}

//===----------------------------------------------------------------------===//
// shard.reduce op
//===----------------------------------------------------------------------===//

LogicalResult ReduceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getGridAxes(),
                                 grid.value().getShape()))) {
    return failure();
  }

  return success();
}

void ReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<ReduceOp>>(context);
}

void ReduceOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "reduce");
}

//===----------------------------------------------------------------------===//
// shard.reduce_scatter op
//===----------------------------------------------------------------------===//

LogicalResult
ReduceScatterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }

  return verifyScatterOrSliceOperandAndResultShape(
      getOperand(), getResult(), getScatterAxis().getSExtValue(), getGridAxes(),
      grid.value().getShape());
}

void ReduceScatterOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<ReduceScatterOp>>(context);
}

void ReduceScatterOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "reduce_scatter");
}

//===----------------------------------------------------------------------===//
// shard.scatter op
//===----------------------------------------------------------------------===//

LogicalResult ScatterOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getRootAttrName(), getRoot(),
                                 getRootDynamic(), getGridAxes(),
                                 grid.value().getShape()))) {
    return failure();
  }

  auto scatterAxis = getScatterAxis().getSExtValue();
  return verifyScatterOrSliceOperandAndResultShape(getInput(), getResult(),
                                                   scatterAxis, getGridAxes(),
                                                   grid.value().getShape());
}

void ScatterOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<ScatterOp>>(context);
}

void ScatterOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "scatter");
}

//===----------------------------------------------------------------------===//
// shard.send op
//===----------------------------------------------------------------------===//

LogicalResult SendOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }
  if (failed(verifyInGroupDevice(getLoc(), getDestinationAttrName(),
                                 getDestination(), getDestinationDynamic(),
                                 getGridAxes(), grid.value().getShape()))) {
    return failure();
  }
  return success();
}

void SendOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                         MLIRContext *context) {
  patterns.add<EmptyGridAxesCanonicalizationPattern<SendOp>>(context);
}

void SendOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "send");
}

//===----------------------------------------------------------------------===//
// shard.shift op
//===----------------------------------------------------------------------===//

LogicalResult ShiftOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerifyAxes(*this, symbolTable);
  if (failed(grid)) {
    return failure();
  }

  auto gridAxes = getGridAxes();
  auto shiftAxis = getShiftAxis().getZExtValue();
  if (!llvm::is_contained(gridAxes, shiftAxis)) {
    return emitError() << "Invalid shift axis " << shiftAxis
                       << ". It must be one of the grouping grid axes.";
  }

  return success();
}

void ShiftOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                          MLIRContext *context) {
  // TODO: remove op when offset is 0 or if it is a rotate with and
  // offset % shift_axis_grid_dim_size == 0.
}

void ShiftOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), "shift");
}

//===----------------------------------------------------------------------===//
// shard.update_halo op
//===----------------------------------------------------------------------===//

LogicalResult
UpdateHaloOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto grid = getGridAndVerify(getOperation(), getGridAttr(), symbolTable);
  if (failed(grid)) {
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/Shard/IR/ShardOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Shard/IR/ShardAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/Shard/IR/ShardTypes.cpp.inc"

#include "mlir/Dialect/Shard/IR/ShardEnums.cpp.inc"
