//===- ShardToMPI.cpp - Shard to MPI  dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation of Shard communication ops to MPI ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ShardToMPI/ShardToMPI.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/Transforms/Simplifications.h"
#include "mlir/Dialect/Shard/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "shard-to-mpi"

namespace mlir {
#define GEN_PASS_DEF_CONVERTSHARDTOMPIPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace shard;

namespace {
/// Converts a vector of OpFoldResults (ints) into vector of Values of the
/// provided type.
static SmallVector<Value> getMixedAsValues(OpBuilder b, const Location &loc,
                                           llvm::ArrayRef<int64_t> statics,
                                           ValueRange dynamics,
                                           Type type = Type()) {
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

/// Create operations converting a linear index to a multi-dimensional index.
static SmallVector<Value> linearToMultiIndex(Location loc, OpBuilder b,
                                             Value linearIndex,
                                             ValueRange dimensions) {
  int n = dimensions.size();
  SmallVector<Value> multiIndex(n);

  for (int i = n - 1; i >= 0; --i) {
    multiIndex[i] = arith::RemSIOp::create(b, loc, linearIndex, dimensions[i]);
    if (i > 0)
      linearIndex = arith::DivSIOp::create(b, loc, linearIndex, dimensions[i]);
  }

  return multiIndex;
}

/// Create operations converting a multi-dimensional index to a linear index.
Value multiToLinearIndex(Location loc, OpBuilder b, ValueRange multiIndex,
                         ValueRange dimensions) {

  Value linearIndex = arith::ConstantIndexOp::create(b, loc, 0);
  Value stride = arith::ConstantIndexOp::create(b, loc, 1);

  for (int i = multiIndex.size() - 1; i >= 0; --i) {
    Value off = arith::MulIOp::create(b, loc, multiIndex[i], stride);
    linearIndex = arith::AddIOp::create(b, loc, linearIndex, off);
    stride = arith::MulIOp::create(b, loc, stride, dimensions[i]);
  }

  return linearIndex;
}

/// Replace GetShardingOp with related/dependent ShardingOp.
struct ConvertGetShardingOp : public OpConversionPattern<GetShardingOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetShardingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shardOp = adaptor.getSource().getDefiningOp<ShardOp>();
    if (!shardOp)
      return failure();
    auto shardingOp = shardOp.getSharding().getDefiningOp<ShardingOp>();
    if (!shardingOp)
      return failure();

    rewriter.replaceOp(op, shardingOp.getResult());
    return success();
  }
};

/// Convert a sharding op to a tuple of tensors of its components
///   (SplitAxes, HaloSizes, ShardedDimsOffsets)
/// as defined by type converter.
struct ConvertShardingOp : public OpConversionPattern<ShardingOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShardingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto splitAxes = op.getSplitAxes().getAxes();
    int64_t maxNAxes = 0;
    for (auto axes : splitAxes)
      maxNAxes = std::max<int64_t>(maxNAxes, axes.size());

    // To hold the split axes, create empty 2d tensor with shape
    // {splitAxes.size(), max-size-of-split-groups}.
    // Set trailing elements for smaller split-groups to -1.
    Location loc = op.getLoc();
    auto i16 = rewriter.getI16Type();
    auto i64 = rewriter.getI64Type();
    std::array<int64_t, 2> shape = {static_cast<int64_t>(splitAxes.size()),
                                    maxNAxes};
    Value resSplitAxes = tensor::EmptyOp::create(rewriter, loc, shape, i16);
    auto attr = IntegerAttr::get(i16, -1);
    Value fillValue = arith::ConstantOp::create(rewriter, loc, i16, attr);
    resSplitAxes =
        linalg::FillOp::create(rewriter, loc, fillValue, resSplitAxes)
            .getResult(0);

    // explicitly write values into tensor row by row
    std::array<int64_t, 2> strides = {1, 1};
    int64_t nSplits = 0;
    ValueRange empty = {};
    for (auto [i, axes] : llvm::enumerate(splitAxes)) {
      int64_t size = axes.size();
      if (size > 0)
        ++nSplits;
      std::array<int64_t, 2> offs = {(int64_t)i, 0};
      std::array<int64_t, 2> sizes = {1, size};
      auto tensorType = RankedTensorType::get({size}, i16);
      auto attrs = DenseIntElementsAttr::get(tensorType, axes.asArrayRef());
      auto vals = arith::ConstantOp::create(rewriter, loc, tensorType, attrs);
      resSplitAxes = tensor::InsertSliceOp::create(rewriter, loc, vals,
                                                   resSplitAxes, empty, empty,
                                                   empty, offs, sizes, strides);
    }

    // To hold halos sizes, create 2d Tensor with shape {nSplits, 2}.
    // Store the halo sizes in the tensor.
    SmallVector<Value> haloSizes =
        getMixedAsValues(rewriter, loc, adaptor.getStaticHaloSizes(),
                         adaptor.getDynamicHaloSizes());
    auto type = RankedTensorType::get({nSplits, 2}, i64);
    Value resHaloSizes =
        haloSizes.empty()
            ? tensor::EmptyOp::create(rewriter, loc,
                                      std::array<int64_t, 2>{0, 0}, i64)
                  .getResult()
            : tensor::FromElementsOp::create(rewriter, loc, type, haloSizes)
                  .getResult();

    // To hold sharded dims offsets, create Tensor with shape {nSplits,
    // maxSplitSize+1}. Store the offsets in the tensor but set trailing
    // elements for smaller split-groups to -1. Computing the max size of the
    // split groups needs using collectiveProcessGroupSize (which needs the
    // GridOp)
    Value resOffsets;
    if (adaptor.getStaticShardedDimsOffsets().empty()) {
      resOffsets = tensor::EmptyOp::create(rewriter, loc,
                                           std::array<int64_t, 2>{0, 0}, i64);
    } else {
      SymbolTableCollection symbolTableCollection;
      auto gridOp = getGrid(op, symbolTableCollection);
      int64_t maxSplitSize = 0;
      for (auto axes : splitAxes) {
        int64_t splitSize =
            collectiveProcessGroupSize(axes.asArrayRef(), gridOp.getShape());
        assert(splitSize != ShapedType::kDynamic);
        maxSplitSize = std::max<int64_t>(maxSplitSize, splitSize);
      }
      assert(maxSplitSize);
      ++maxSplitSize; // add one for the total size

      resOffsets = tensor::EmptyOp::create(
          rewriter, loc, std::array<int64_t, 2>{nSplits, maxSplitSize}, i64);
      Value zero = arith::ConstantOp::create(
          rewriter, loc, i64, rewriter.getI64IntegerAttr(ShapedType::kDynamic));
      resOffsets =
          linalg::FillOp::create(rewriter, loc, zero, resOffsets).getResult(0);
      SmallVector<Value> offsets =
          getMixedAsValues(rewriter, loc, adaptor.getStaticShardedDimsOffsets(),
                           adaptor.getDynamicShardedDimsOffsets());
      int64_t curr = 0;
      for (auto [i, axes] : llvm::enumerate(splitAxes)) {
        int64_t splitSize =
            collectiveProcessGroupSize(axes.asArrayRef(), gridOp.getShape());
        assert(splitSize != ShapedType::kDynamic && splitSize < maxSplitSize);
        ++splitSize; // add one for the total size
        ArrayRef<Value> values(&offsets[curr], splitSize);
        Value vals = tensor::FromElementsOp::create(rewriter, loc, values);
        std::array<int64_t, 2> offs = {static_cast<int64_t>(i), 0};
        std::array<int64_t, 2> sizes = {1, splitSize};
        resOffsets = tensor::InsertSliceOp::create(rewriter, loc, vals,
                                                   resOffsets, empty, empty,
                                                   empty, offs, sizes, strides);
        curr += splitSize;
      }
    }

    // return a tuple of tensors as defined by type converter
    SmallVector<Type> resTypes;
    if (failed(getTypeConverter()->convertType(op.getResult().getType(),
                                               resTypes)))
      return failure();

    resSplitAxes =
        tensor::CastOp::create(rewriter, loc, resTypes[0], resSplitAxes);
    resHaloSizes =
        tensor::CastOp::create(rewriter, loc, resTypes[1], resHaloSizes);
    resOffsets = tensor::CastOp::create(rewriter, loc, resTypes[2], resOffsets);

    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, TupleType::get(op.getContext(), resTypes),
        ValueRange{resSplitAxes, resHaloSizes, resOffsets});

    return success();
  }
};

struct ConvertProcessMultiIndexOp
    : public OpConversionPattern<ProcessMultiIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProcessMultiIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Currently converts its linear index to a multi-dimensional index.

    SymbolTableCollection symbolTableCollection;
    Location loc = op.getLoc();
    auto gridOp = getGrid(op, symbolTableCollection);
    // For now we only support static grid shapes
    if (ShapedType::isDynamicShape(gridOp.getShape()))
      return failure();

    SmallVector<Value> dims;
    llvm::transform(
        gridOp.getShape(), std::back_inserter(dims), [&](int64_t i) {
          return arith::ConstantIndexOp::create(rewriter, loc, i).getResult();
        });
    Value rank = ProcessLinearIndexOp::create(rewriter, op.getLoc(), gridOp);
    auto mIdx = linearToMultiIndex(loc, rewriter, rank, dims);

    // optionally extract subset of grid axes
    auto axes = adaptor.getAxes();
    if (!axes.empty()) {
      SmallVector<Value> subIndex;
      for (auto axis : axes) {
        subIndex.emplace_back(mIdx[axis]);
      }
      mIdx = std::move(subIndex);
    }

    rewriter.replaceOp(op, mIdx);
    return success();
  }
};

class ConvertProcessLinearIndexOp
    : public OpConversionPattern<ProcessLinearIndexOp> {

public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProcessLinearIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create mpi::CommRankOp
    Location loc = op.getLoc();
    auto *ctx = op.getContext();
    Value commWorld =
        mpi::CommWorldOp::create(rewriter, loc, mpi::CommType::get(ctx));
    auto rank = mpi::CommRankOp::create(
                    rewriter, loc,
                    TypeRange{mpi::RetvalType::get(ctx), rewriter.getI32Type()},
                    commWorld)
                    .getRank();
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, rewriter.getIndexType(),
                                                    rank);
    return success();
  }
};

struct ConvertNeighborsLinearIndicesOp
    : public OpConversionPattern<NeighborsLinearIndicesOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NeighborsLinearIndicesOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Computes the neighbors indices along a split axis by simply
    // adding/subtracting 1 to the current index in that dimension.
    // Assigns -1 if neighbor is out of bounds.

    auto axes = adaptor.getSplitAxes();
    // For now only single axis sharding is supported
    if (axes.size() != 1)
      return failure();

    Location loc = op.getLoc();
    SymbolTableCollection symbolTableCollection;
    auto gridOp = getGrid(op, symbolTableCollection);
    auto mIdx = adaptor.getDevice();
    auto orgIdx = mIdx[axes[0]];
    SmallVector<Value> dims;
    llvm::transform(
        gridOp.getShape(), std::back_inserter(dims), [&](int64_t i) {
          return arith::ConstantIndexOp::create(rewriter, loc, i).getResult();
        });
    Value dimSz = dims[axes[0]];
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value minus1 = arith::ConstantIndexOp::create(rewriter, loc, -1);
    Value atBorder =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle, orgIdx,
                              arith::ConstantIndexOp::create(rewriter, loc, 0));
    auto down = scf::IfOp::create(
        rewriter, loc, atBorder,
        [&](OpBuilder &builder, Location loc) {
          scf::YieldOp::create(builder, loc, minus1);
        },
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Value> tmp = mIdx;
          tmp[axes[0]] =
              arith::SubIOp::create(rewriter, op.getLoc(), orgIdx, one)
                  .getResult();
          scf::YieldOp::create(builder, loc,
                               multiToLinearIndex(loc, rewriter, tmp, dims));
        });
    atBorder = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::sge, orgIdx,
        arith::SubIOp::create(rewriter, loc, dimSz, one).getResult());
    auto up = scf::IfOp::create(
        rewriter, loc, atBorder,
        [&](OpBuilder &builder, Location loc) {
          scf::YieldOp::create(builder, loc, minus1);
        },
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Value> tmp = mIdx;
          tmp[axes[0]] =
              arith::AddIOp::create(rewriter, op.getLoc(), orgIdx, one);
          scf::YieldOp::create(builder, loc,
                               multiToLinearIndex(loc, rewriter, tmp, dims));
        });
    rewriter.replaceOp(op, ValueRange{down.getResult(0), up.getResult(0)});
    return success();
  }
};

struct ConvertShardShapeOp : public OpConversionPattern<ShardShapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ShardShapeOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sharding = op.getSharding().getDefiningOp<ShardingOp>();
    if (!sharding) {
      return op->emitError()
             << "Expected ShardingOp as defining op for sharding"
             << " but found " << adaptor.getSharding()[0].getDefiningOp();
    }

    // Compute the sharded shape by applying the sharding to the input shape.
    // If shardedDimsOffsets is not defined in the sharding, the shard shape is
    // computed by dividing the dimension size by the number of shards in that
    // dimension (which is given by the size of the grid axes provided in
    // split-axes). Odd elements get distributed to trailing shards. If a
    // shardedDimsOffsets is provided, the shard shape is computed by
    // subtracting the offset of the current shard from the offset of the next
    // shard.

    Location loc = op.getLoc();
    Type index = rewriter.getIndexType();

    // This is a 1:N conversion because the sharding op is a 1:3 conversion.
    // The operands in the adaptor are a vector<ValeRange>. For dims and device
    // we have a 1:1 conversion.
    // For simpler access fill a vector with the dynamic dims.
    SmallVector<Value> dynDims, dynDevice;
    for (auto dim : adaptor.getDimsDynamic()) {
      // type conversion should be 1:1 for ints
      dynDims.emplace_back(llvm::getSingleElement(dim));
    }
    // same for device
    for (auto device : adaptor.getDeviceDynamic()) {
      dynDevice.emplace_back(llvm::getSingleElement(device));
    }

    // To keep the code simple, convert dims/device to values when they are
    // attributes. Count on canonicalization to fold static values.
    SmallVector<Value> shape =
        getMixedAsValues(rewriter, loc, op.getDims(), dynDims, index);
    SmallVector<Value> multiIdx =
        getMixedAsValues(rewriter, loc, adaptor.getDevice(), dynDevice, index);

    // Get the GridOp, the grid shape is needed to compute the sharded shape.
    SymbolTableCollection symbolTableCollection;
    auto gridOp = getGrid(sharding, symbolTableCollection);
    // For now we only support static grid shapes
    if (ShapedType::isDynamicShape(gridOp.getShape()))
      return failure();

    auto splitAxes = sharding.getSplitAxes().getAxes();
    // shardedDimsOffsets are optional and might be Values (not attributes).
    // Also, the shardId might be dynamic which means the position in the
    // shardedDimsOffsets is not statically known. Create a tensor of the
    // shardedDimsOffsets and later extract the offsets for computing the
    // local shard-size.
    Value shardedDimsOffs;
    {
      SmallVector<Value> tmp = getMixedAsValues(
          rewriter, loc, sharding.getStaticShardedDimsOffsets(),
          sharding.getDynamicShardedDimsOffsets(), index);
      if (!tmp.empty())
        shardedDimsOffs = tensor::FromElementsOp::create(
            rewriter, loc, RankedTensorType::get({(int64_t)tmp.size()}, index),
            tmp);
    }

    // With static grid shape the sizes of the split axes are known.
    // Hence the start/pos for each split axes in shardDimsOffsets can be
    // computed statically.
    int64_t pos = 0;
    SmallVector<Value> shardShape;
    Value zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(index));
    Value one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getOneAttr(index));

    // Iterate over the dimensions of the tensor shape, get their split Axes,
    // and compute the sharded shape.
    for (auto [i, dim] : llvm::enumerate(shape)) {
      // Trailing dimensions might not be annotated.
      if (i < splitAxes.size() && !splitAxes[i].empty()) {
        auto axes = splitAxes[i];
        // The current dimension might not be sharded.
        // Create a value from the static position in shardDimsOffsets.
        Value posVal = arith::ConstantOp::create(rewriter, loc,
                                                 rewriter.getIndexAttr(pos));
        // Get the index of the local shard in the grid axis.
        Value idx = multiIdx[axes[0]];
        auto numShards =
            collectiveProcessGroupSize(axes.asArrayRef(), gridOp.getShape());
        if (shardedDimsOffs) {
          // If sharded dims offsets are provided, use them to compute the
          // sharded shape.
          if (axes.size() > 1) {
            return op->emitError() << "Only single axis sharding is "
                                   << "supported for each dimension.";
          }
          idx = arith::AddIOp::create(rewriter, loc, posVal, idx);
          // Compute size = shardedDimsOffs[idx+1] - shardedDimsOffs[idx].
          Value off =
              tensor::ExtractOp::create(rewriter, loc, shardedDimsOffs, idx);
          idx = arith::AddIOp::create(rewriter, loc, idx, one);
          Value nextOff =
              tensor::ExtractOp::create(rewriter, loc, shardedDimsOffs, idx);
          Value sz = arith::SubIOp::create(rewriter, loc, nextOff, off);
          shardShape.emplace_back(sz);
        } else {
          Value numShardsVal = arith::ConstantOp::create(
              rewriter, loc, rewriter.getIndexAttr(numShards));
          // Compute shard dim size by distributing odd elements to trailing
          // shards:
          // sz = dim / numShards
          //      + (idx >= (numShards - (dim % numShards)) ? 1 : 0)
          Value sz = arith::DivSIOp::create(rewriter, loc, dim, numShardsVal);
          Value sz1 = arith::RemSIOp::create(rewriter, loc, dim, numShardsVal);
          sz1 = arith::SubIOp::create(rewriter, loc, numShardsVal, sz1);
          auto cond = arith::CmpIOp::create(
              rewriter, loc, arith::CmpIPredicate::sge, idx, sz1);
          Value odd = arith::SelectOp::create(rewriter, loc, cond, one, zero);
          sz = arith::AddIOp::create(rewriter, loc, sz, odd);
          shardShape.emplace_back(sz);
        }
        pos += numShards + 1; // add one for the total size.
      } // else no sharding if split axis is empty or no split axis
      // If no size was added -> no sharding in this dimension.
      if (shardShape.size() <= i)
        shardShape.emplace_back(dim);
    }
    assert(shardShape.size() == shape.size());
    rewriter.replaceOp(op, shardShape);
    return success();
  }
};

static mpi::MPI_ReductionOpEnumAttr getMPIReductionOp(ReductionKindAttr kind) {
  auto *ctx = kind.getContext();
  auto getReductionOp = [ctx](mpi::MPI_ReductionOpEnum redOp) {
    return mpi::MPI_ReductionOpEnumAttr::get(ctx, redOp);
  };

  switch (kind.getValue()) {
  case ReductionKind::Sum:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_SUM);
  case ReductionKind::Product:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_PROD);
  case ReductionKind::Min:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_MIN);
  case ReductionKind::Max:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_MAX);
  case ReductionKind::BitwiseAnd:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_BAND);
  case ReductionKind::BitwiseOr:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_BOR);
  case ReductionKind::BitwiseXor:
    return getReductionOp(mpi::MPI_ReductionOpEnum::MPI_BXOR);
  default:
    llvm_unreachable("Unknown/unsupported reduction kind");
  }
}

struct ConvertAllReduceOp : public OpConversionPattern<AllReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AllReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SymbolTableCollection symbolTableCollection;
    auto grid = adaptor.getGrid();
    mlir::shard::GridOp gridOp = getGrid(op, symbolTableCollection);
    if (!gridOp)
      return op->emitError() << "No grid found for AllReduceOp";
    if (ShapedType::isDynamicShape(gridOp.getShape()))
      return op->emitError()
             << "Dynamic grid shape not supported in AllReduceOp";

    ImplicitLocOpBuilder iBuilder(op.getLoc(), rewriter);
    Value input = adaptor.getInput();
    auto inputShape = cast<ShapedType>(input.getType()).getShape();

    // If the source is a memref, cast it to a tensor.
    if (isa<RankedTensorType>(input.getType())) {
      auto memrefType = MemRefType::get(
          inputShape, cast<ShapedType>(input.getType()).getElementType());
      input = bufferization::ToBufferOp::create(iBuilder, memrefType, input);
    }
    MemRefType inType = cast<MemRefType>(input.getType());

    // Get the actual shape to allocate the buffer.
    SmallVector<OpFoldResult> shape(inType.getRank());
    for (auto i = 0; i < inType.getRank(); ++i) {
      auto s = inputShape[i];
      if (ShapedType::isDynamic(s))
        shape[i] = memref::DimOp::create(iBuilder, input, s).getResult();
      else
        shape[i] = iBuilder.getIndexAttr(s);
    }

    // Allocate buffer and copy input to buffer.
    Value buffer = memref::AllocOp::create(
        iBuilder, shape, cast<ShapedType>(op.getType()).getElementType());
    linalg::CopyOp::create(iBuilder, input, buffer);

    // Get an MPI_Comm_split for the AllReduce operation.
    // The color is the linear index of the process in the grid along the
    // non-reduced axes. The key is the linear index of the process in the grid
    // along the reduced axes.
    SmallVector<Type> indexResultTypes(gridOp.getShape().size(),
                                       iBuilder.getIndexType());
    SmallVector<Value> myMultiIndex =
        ProcessMultiIndexOp::create(iBuilder, indexResultTypes, grid)
            .getResult();
    Value zero = arith::ConstantIndexOp::create(iBuilder, 0);
    SmallVector<Value> multiKey(myMultiIndex.size(), zero);

    auto redAxes = adaptor.getGridAxes();
    for (auto axis : redAxes) {
      multiKey[axis] = myMultiIndex[axis];
      myMultiIndex[axis] = zero;
    }

    Value color =
        createProcessLinearIndex(grid, myMultiIndex, redAxes, iBuilder);
    color = arith::IndexCastOp::create(iBuilder, iBuilder.getI32Type(), color);
    Value key = createProcessLinearIndex(grid, multiKey, redAxes, iBuilder);
    key = arith::IndexCastOp::create(iBuilder, iBuilder.getI32Type(), key);

    // Finally split the communicator
    auto commType = mpi::CommType::get(op->getContext());
    Value commWorld = mpi::CommWorldOp::create(iBuilder, commType);
    auto comm =
        mpi::CommSplitOp::create(iBuilder, commType, commWorld, color, key)
            .getNewcomm();

    Value buffer1d = buffer;
    // Collapse shape to 1d if needed
    if (inType.getRank() > 1) {
      ReassociationIndices reassociation(inType.getRank());
      std::iota(reassociation.begin(), reassociation.end(), 0);
      buffer1d = memref::CollapseShapeOp::create(
          iBuilder, buffer, ArrayRef<ReassociationIndices>(reassociation));
    }

    // Create the MPI AllReduce operation.
    mpi::AllReduceOp::create(iBuilder, TypeRange(), buffer1d, buffer1d,
                             getMPIReductionOp(adaptor.getReductionAttr()),
                             comm);

    // If the destination is a memref, cast it to a tensor
    if (isa<RankedTensorType>(op.getType()))
      buffer = bufferization::ToTensorOp::create(iBuilder, op.getType(), buffer,
                                                 true);

    rewriter.replaceOp(op, buffer);
    return success();
  }
};

struct ConvertUpdateHaloOp : public OpConversionPattern<UpdateHaloOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UpdateHaloOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The input/output memref is assumed to be in C memory order.
    // Halos are exchanged as 2 blocks per dimension (one for each side: down
    // and up). For each haloed dimension `d`, the exchanged blocks are
    // expressed as multi-dimensional subviews. The subviews include potential
    // halos of higher dimensions `dh > d`, no halos for the lower dimensions
    // `dl < d` and for dimension `d` the currently exchanged halo only.
    // By iterating form higher to lower dimensions this also updates the halos
    // in the 'corners'.
    // memref.subview is used to read and write the halo data from and to the
    // local data. Because subviews and halos can have mixed dynamic and static
    // shapes, OpFoldResults are used whenever possible.

    auto haloSizes = getMixedValues(adaptor.getStaticHaloSizes(),
                                    adaptor.getHaloSizes(), rewriter);
    if (haloSizes.empty()) {
      // no halos -> nothing to do
      rewriter.replaceOp(op, adaptor.getDestination());
      return success();
    }

    SymbolTableCollection symbolTableCollection;
    Location loc = op.getLoc();

    // convert a OpFoldResult into a Value
    auto toValue = [&rewriter, &loc](OpFoldResult &v) -> Value {
      if (auto value = dyn_cast<Value>(v))
        return value;
      return arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getIndexAttr(
              cast<IntegerAttr>(cast<Attribute>(v)).getInt()));
    };

    auto dest = adaptor.getDestination();
    auto dstShape = cast<ShapedType>(dest.getType()).getShape();
    Value array = dest;
    if (isa<RankedTensorType>(array.getType())) {
      // If the destination is a memref, we need to cast it to a tensor
      auto mmemrefType = MemRefType::get(
          dstShape, cast<ShapedType>(array.getType()).getElementType());
      array =
          bufferization::ToBufferOp::create(rewriter, loc, mmemrefType, array);
    }
    auto rank = cast<ShapedType>(array.getType()).getRank();
    auto opSplitAxes = adaptor.getSplitAxes().getAxes();
    auto grid = adaptor.getGrid();
    auto gridOp = getGrid(op, symbolTableCollection);
    // subviews need Index values
    for (auto &sz : haloSizes) {
      if (auto value = dyn_cast<Value>(sz))
        sz = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                        value)
                 .getResult();
    }

    // most of the offset/size/stride data is the same for all dims
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> shape(rank), dimSizes(rank);
    auto currHaloDim = -1; // halo sizes are provided for split dimensions only
    // we need the actual shape to compute offsets and sizes
    for (auto i = 0; i < rank; ++i) {
      auto s = dstShape[i];
      if (ShapedType::isDynamic(s))
        shape[i] = memref::DimOp::create(rewriter, loc, array, s).getResult();
      else
        shape[i] = rewriter.getIndexAttr(s);

      if ((size_t)i < opSplitAxes.size() && !opSplitAxes[i].empty()) {
        ++currHaloDim;
        // the offsets for lower dim sstarts after their down halo
        offsets[i] = haloSizes[currHaloDim * 2];

        // prepare shape and offsets of highest dim's halo exchange
        Value _haloSz = arith::AddIOp::create(
            rewriter, loc, toValue(haloSizes[currHaloDim * 2]),
            toValue(haloSizes[currHaloDim * 2 + 1]));
        // the halo shape of lower dims exlude the halos
        dimSizes[i] =
            arith::SubIOp::create(rewriter, loc, toValue(shape[i]), _haloSz)
                .getResult();
      } else {
        dimSizes[i] = shape[i];
      }
    }

    auto tagAttr = rewriter.getI32IntegerAttr(91); // we just pick something
    auto tag = arith::ConstantOp::create(rewriter, loc, tagAttr);
    auto zeroAttr = rewriter.getI32IntegerAttr(0); // for detecting v<0
    auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);

    SmallVector<Type> indexResultTypes(gridOp.getShape().size(),
                                       rewriter.getIndexType());
    auto myMultiIndex =
        ProcessMultiIndexOp::create(rewriter, loc, indexResultTypes, grid)
            .getResult();
    // traverse all split axes from high to low dim
    for (ssize_t dim = opSplitAxes.size() - 1; dim >= 0; --dim) {
      auto splitAxes = opSplitAxes[dim];
      if (splitAxes.empty())
        continue;
      assert(currHaloDim >= 0 && (size_t)currHaloDim < haloSizes.size() / 2);
      // Get the linearized ids of the neighbors (down and up) for the
      // given split
      auto tmp = NeighborsLinearIndicesOp::create(rewriter, loc, grid,
                                                  myMultiIndex, splitAxes)
                     .getResults();
      // MPI operates on i32...
      Value neighbourIDs[2] = {
          arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                     tmp[0]),
          arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(),
                                     tmp[1])};

      auto lowerRecvOffset = rewriter.getIndexAttr(0);
      auto lowerSendOffset = toValue(haloSizes[currHaloDim * 2]);
      auto upperRecvOffset =
          arith::SubIOp::create(rewriter, loc, toValue(shape[dim]),
                                toValue(haloSizes[currHaloDim * 2 + 1]));
      auto upperSendOffset = arith::SubIOp::create(
          rewriter, loc, upperRecvOffset, toValue(haloSizes[currHaloDim * 2]));

      Value commWorld = mpi::CommWorldOp::create(
          rewriter, loc, mpi::CommType::get(op->getContext()));

      // Make sure we send/recv in a way that does not lead to a dead-lock.
      // The current approach is by far not optimal, this should be at least
      // be a red-black pattern or using MPI_sendrecv.
      // Also, buffers should be re-used.
      // Still using temporary contiguous buffers for MPI communication...
      // Still yielding a "serialized" communication pattern...
      auto genSendRecv = [&](bool upperHalo) {
        auto orgOffset = offsets[dim];
        dimSizes[dim] = upperHalo ? haloSizes[currHaloDim * 2 + 1]
                                  : haloSizes[currHaloDim * 2];
        // Check if we need to send and/or receive
        // Processes on the grid borders have only one neighbor
        auto to = upperHalo ? neighbourIDs[0] : neighbourIDs[1];
        auto from = upperHalo ? neighbourIDs[1] : neighbourIDs[0];
        auto hasFrom = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::sge, from, zero);
        auto hasTo = arith::CmpIOp::create(rewriter, loc,
                                           arith::CmpIPredicate::sge, to, zero);
        auto buffer = memref::AllocOp::create(
            rewriter, loc, dimSizes,
            cast<ShapedType>(array.getType()).getElementType());
        // if has neighbor: copy halo data from array to buffer and send
        scf::IfOp::create(
            rewriter, loc, hasTo, [&](OpBuilder &builder, Location loc) {
              offsets[dim] = upperHalo ? OpFoldResult(lowerSendOffset)
                                       : OpFoldResult(upperSendOffset);
              auto subview = memref::SubViewOp::create(
                  builder, loc, array, offsets, dimSizes, strides);
              memref::CopyOp::create(builder, loc, subview, buffer);
              mpi::SendOp::create(builder, loc, TypeRange{}, buffer, tag, to,
                                  commWorld);
              scf::YieldOp::create(builder, loc);
            });
        // if has neighbor: receive halo data into buffer and copy to array
        scf::IfOp::create(
            rewriter, loc, hasFrom, [&](OpBuilder &builder, Location loc) {
              offsets[dim] = upperHalo ? OpFoldResult(upperRecvOffset)
                                       : OpFoldResult(lowerRecvOffset);
              mpi::RecvOp::create(builder, loc, TypeRange{}, buffer, tag, from,
                                  commWorld);
              auto subview = memref::SubViewOp::create(
                  builder, loc, array, offsets, dimSizes, strides);
              memref::CopyOp::create(builder, loc, buffer, subview);
              scf::YieldOp::create(builder, loc);
            });
        memref::DeallocOp::create(rewriter, loc, buffer);
        offsets[dim] = orgOffset;
      };

      auto doSendRecv = [&](int upOrDown) {
        OpFoldResult &v = haloSizes[currHaloDim * 2 + upOrDown];
        Value haloSz = dyn_cast<Value>(v);
        if (!haloSz)
          haloSz = arith::ConstantOp::create(
              rewriter, loc,
              rewriter.getI32IntegerAttr(
                  cast<IntegerAttr>(cast<Attribute>(v)).getInt()));
        auto hasSize = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::sgt, haloSz, zero);
        scf::IfOp::create(rewriter, loc, hasSize,
                          [&](OpBuilder &builder, Location loc) {
                            genSendRecv(upOrDown > 0);
                            scf::YieldOp::create(builder, loc);
                          });
      };

      doSendRecv(0);
      doSendRecv(1);

      // the shape for lower dims include higher dims' halos
      dimSizes[dim] = shape[dim];
      // -> the offset for higher dims is always 0
      offsets[dim] = rewriter.getIndexAttr(0);
      // on to next halo
      --currHaloDim;
    }

    if (isa<MemRefType>(op.getResult().getType())) {
      rewriter.replaceOp(op, array);
    } else {
      assert(isa<RankedTensorType>(op.getResult().getType()));
      rewriter.replaceOp(op, bufferization::ToTensorOp::create(
                                 rewriter, loc, op.getResult().getType(), array,
                                 /*restrict=*/true, /*writable=*/true));
    }
    return success();
  }
};

struct ConvertShardToMPIPass
    : public impl::ConvertShardToMPIPassBase<ConvertShardToMPIPass> {
  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    auto *ctxt = &getContext();
    RewritePatternSet patterns(ctxt);
    ConversionTarget target(getContext());

    // Define a type converter to convert shard::ShardingType,
    // mostly for use in return operations.
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // convert shard::ShardingType to a tuple of RankedTensorTypes
    typeConverter.addConversion(
        [](ShardingType type,
           SmallVectorImpl<Type> &results) -> std::optional<LogicalResult> {
          auto i16 = IntegerType::get(type.getContext(), 16);
          auto i64 = IntegerType::get(type.getContext(), 64);
          std::array<int64_t, 2> shp = {ShapedType::kDynamic,
                                        ShapedType::kDynamic};
          results.emplace_back(RankedTensorType::get(shp, i16));
          results.emplace_back(RankedTensorType::get(shp, i64)); // actually ?x2
          results.emplace_back(RankedTensorType::get(shp, i64));
          return success();
        });

    // To 'extract' components, a UnrealizedConversionCastOp is expected
    // to define the input
    typeConverter.addTargetMaterialization(
        [&](OpBuilder &builder, TypeRange resultTypes, ValueRange inputs,
            Location loc) {
          // Expecting a single input.
          if (inputs.size() != 1 || !isa<TupleType>(inputs[0].getType()))
            return SmallVector<Value>();
          auto castOp = inputs[0].getDefiningOp<UnrealizedConversionCastOp>();
          // Expecting an UnrealizedConversionCastOp.
          if (!castOp)
            return SmallVector<Value>();
          // Fill a vector with elements of the tuple/castOp.
          SmallVector<Value> results;
          for (auto oprnd : castOp.getInputs()) {
            if (!isa<RankedTensorType>(oprnd.getType()))
              return SmallVector<Value>();
            results.emplace_back(oprnd);
          }
          return results;
        });

    // No shard dialect should left after conversion...
    target.addIllegalDialect<shard::ShardDialect>();
    // ...except the global GridOp. GridShapeOp which will get folded later.
    target.addLegalOp<shard::GridOp, shard::GridShapeOp>();
    // Allow all the stuff that our patterns will convert to
    target.addLegalDialect<
        BuiltinDialect, mpi::MPIDialect, scf::SCFDialect, arith::ArithDialect,
        tensor::TensorDialect, bufferization::BufferizationDialect,
        linalg::LinalgDialect, memref::MemRefDialect, affine::AffineDialect>();
    // Make sure the function signature, calls etc. are legal
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    patterns.add<ConvertUpdateHaloOp, ConvertNeighborsLinearIndicesOp,
                 ConvertProcessMultiIndexOp, ConvertGetShardingOp,
                 ConvertShardingOp, ConvertShardShapeOp, ConvertAllReduceOp,
                 ConvertProcessLinearIndexOp>(typeConverter, ctxt);

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    (void)applyPartialConversion(getOperation(), target, std::move(patterns));

    // Folding patterns cannot be mixed with conversion patterns -> extra pass.
    patterns.clear();
    SymbolTableCollection symbolTableCollection;
    mlir::shard::populateFoldingPatterns(patterns, symbolTableCollection);
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace
