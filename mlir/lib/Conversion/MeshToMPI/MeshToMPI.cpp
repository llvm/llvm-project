//===- MeshToMPI.cpp - Mesh to MPI  dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation of Mesh communication ops tp MPI ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MeshToMPI/MeshToMPI.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "mesh-to-mpi"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
#define GEN_PASS_DEF_CONVERTMESHTOMPIPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mesh;

namespace {
// Create operations converting a linear index to a multi-dimensional index
static SmallVector<Value> linearToMultiIndex(Location loc, OpBuilder b,
                                             Value linearIndex,
                                             ValueRange dimensions) {
  int n = dimensions.size();
  SmallVector<Value> multiIndex(n);

  for (int i = n - 1; i >= 0; --i) {
    multiIndex[i] = b.create<arith::RemSIOp>(loc, linearIndex, dimensions[i]);
    if (i > 0) {
      linearIndex = b.create<arith::DivSIOp>(loc, linearIndex, dimensions[i]);
    }
  }

  return multiIndex;
}

// Create operations converting a multi-dimensional index to a linear index
Value multiToLinearIndex(Location loc, OpBuilder b, ValueRange multiIndex,
                         ValueRange dimensions) {

  auto linearIndex = b.create<arith::ConstantIndexOp>(loc, 0).getResult();
  auto stride = b.create<arith::ConstantIndexOp>(loc, 1).getResult();

  for (int i = multiIndex.size() - 1; i >= 0; --i) {
    auto off = b.create<arith::MulIOp>(loc, multiIndex[i], stride);
    linearIndex = b.create<arith::AddIOp>(loc, linearIndex, off);
    stride = b.create<arith::MulIOp>(loc, stride, dimensions[i]);
  }

  return linearIndex;
}

struct ConvertProcessMultiIndexOp
    : public mlir::OpRewritePattern<mlir::mesh::ProcessMultiIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::mesh::ProcessMultiIndexOp op,
                  mlir::PatternRewriter &rewriter) const override {

    // Currently converts its linear index to a multi-dimensional index.

    SymbolTableCollection symbolTableCollection;
    auto loc = op.getLoc();
    auto meshOp = getMesh(op, symbolTableCollection);
    // For now we only support static mesh shapes
    if (ShapedType::isDynamicShape(meshOp.getShape())) {
      return mlir::failure();
    }

    SmallVector<Value> dims;
    llvm::transform(
        meshOp.getShape(), std::back_inserter(dims), [&](int64_t i) {
          return rewriter.create<arith::ConstantIndexOp>(loc, i).getResult();
        });
    auto rank =
        rewriter.create<ProcessLinearIndexOp>(op.getLoc(), meshOp).getResult();
    auto mIdx = linearToMultiIndex(loc, rewriter, rank, dims);

    // optionally extract subset of mesh axes
    auto axes = op.getAxes();
    if (!axes.empty()) {
      SmallVector<Value> subIndex;
      for (auto axis : axes) {
        subIndex.push_back(mIdx[axis]);
      }
      mIdx = subIndex;
    }

    rewriter.replaceOp(op, mIdx);
    return mlir::success();
  }
};

struct ConvertProcessLinearIndexOp
    : public mlir::OpRewritePattern<mlir::mesh::ProcessLinearIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::mesh::ProcessLinearIndexOp op,
                  mlir::PatternRewriter &rewriter) const override {

    // Finds a global named "static_mpi_rank" it will use that splat value.
    // Otherwise it defaults to mpi.comm_rank.

    auto loc = op.getLoc();
    auto rankOpName = StringAttr::get(op->getContext(), "static_mpi_rank");
    if (auto globalOp = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(
            op, rankOpName)) {
      if (auto initTnsr = globalOp.getInitialValueAttr()) {
        auto val = cast<DenseElementsAttr>(initTnsr).getSplatValue<int64_t>();
        rewriter.replaceOp(op,
                           rewriter.create<arith::ConstantIndexOp>(loc, val));
        return mlir::success();
      }
    }
    auto rank =
        rewriter
            .create<mpi::CommRankOp>(
                op.getLoc(), TypeRange{mpi::RetvalType::get(op->getContext()),
                                       rewriter.getI32Type()})
            .getRank();
    rewriter.replaceOpWithNewOp<arith::IndexCastOp>(op, rewriter.getIndexType(),
                                                    rank);
    return mlir::success();
  }
};

struct ConvertNeighborsLinearIndicesOp
    : public mlir::OpRewritePattern<mlir::mesh::NeighborsLinearIndicesOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::mesh::NeighborsLinearIndicesOp op,
                  mlir::PatternRewriter &rewriter) const override {

    // Computes the neighbors indices along a split axis by simply
    // adding/subtracting 1 to the current index in that dimension.
    // Assigns -1 if neighbor is out of bounds.

    auto axes = op.getSplitAxes();
    // For now only single axis sharding is supported
    if (axes.size() != 1) {
      return mlir::failure();
    }

    auto loc = op.getLoc();
    SymbolTableCollection symbolTableCollection;
    auto meshOp = getMesh(op, symbolTableCollection);
    auto mIdx = op.getDevice();
    auto orgIdx = mIdx[axes[0]];
    SmallVector<Value> dims;
    llvm::transform(
        meshOp.getShape(), std::back_inserter(dims), [&](int64_t i) {
          return rewriter.create<arith::ConstantIndexOp>(loc, i).getResult();
        });
    auto dimSz = dims[axes[0]];
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1).getResult();
    auto minus1 = rewriter.create<arith::ConstantIndexOp>(loc, -1).getResult();
    auto atBorder = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, orgIdx,
        rewriter.create<arith::ConstantIndexOp>(loc, 0).getResult());
    auto down = rewriter.create<scf::IfOp>(
        loc, atBorder,
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::YieldOp>(loc, minus1);
        },
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Value> tmp = mIdx;
          tmp[axes[0]] =
              rewriter.create<arith::SubIOp>(op.getLoc(), orgIdx, one)
                  .getResult();
          builder.create<scf::YieldOp>(
              loc, multiToLinearIndex(loc, rewriter, tmp, dims));
        });
    atBorder = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, orgIdx,
        rewriter.create<arith::SubIOp>(loc, dimSz, one).getResult());
    auto up = rewriter.create<scf::IfOp>(
        loc, atBorder,
        [&](OpBuilder &builder, Location loc) {
          builder.create<scf::YieldOp>(loc, minus1);
        },
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Value> tmp = mIdx;
          tmp[axes[0]] =
              rewriter.create<arith::AddIOp>(op.getLoc(), orgIdx, one)
                  .getResult();
          builder.create<scf::YieldOp>(
              loc, multiToLinearIndex(loc, rewriter, tmp, dims));
        });
    rewriter.replaceOp(op, ValueRange{down.getResult(0), up.getResult(0)});
    return mlir::success();
  }
};

struct ConvertUpdateHaloOp
    : public mlir::OpRewritePattern<mlir::mesh::UpdateHaloOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::mesh::UpdateHaloOp op,
                  mlir::PatternRewriter &rewriter) const override {

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

    SymbolTableCollection symbolTableCollection;
    auto loc = op.getLoc();

    // convert a OpFoldResult into a Value
    auto toValue = [&rewriter, &loc](OpFoldResult &v) {
      return v.is<Value>()
                 ? v.get<Value>()
                 : rewriter.create<::mlir::arith::ConstantOp>(
                       loc,
                       rewriter.getIndexAttr(
                           cast<IntegerAttr>(v.get<Attribute>()).getInt()));
    };

    auto dest = op.getDestination();
    auto dstShape = cast<ShapedType>(dest.getType()).getShape();
    Value array = dest;
    if (isa<RankedTensorType>(array.getType())) {
      // If the destination is a memref, we need to cast it to a tensor
      auto tensorType = MemRefType::get(
          dstShape, cast<ShapedType>(array.getType()).getElementType());
      array = rewriter.create<bufferization::ToMemrefOp>(loc, tensorType, array)
                  .getResult();
    }
    auto rank = cast<ShapedType>(array.getType()).getRank();
    auto opSplitAxes = op.getSplitAxes().getAxes();
    auto mesh = op.getMesh();
    auto meshOp = getMesh(op, symbolTableCollection);
    auto haloSizes =
        getMixedValues(op.getStaticHaloSizes(), op.getHaloSizes(), rewriter);
    // subviews need Index values
    for (auto &sz : haloSizes) {
      if (sz.is<Value>()) {
        sz = rewriter
                 .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                             sz.get<Value>())
                 .getResult();
      }
    }

    // most of the offset/size/stride data is the same for all dims
    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> shape(rank), dimSizes(rank);
    auto currHaloDim = -1; // halo sizes are provided for split dimensions only
    // we need the actual shape to compute offsets and sizes
    for (auto i = 0; i < rank; ++i) {
      auto s = dstShape[i];
      if (ShapedType::isDynamic(s)) {
        shape[i] = rewriter.create<memref::DimOp>(loc, array, s).getResult();
      } else {
        shape[i] = rewriter.getIndexAttr(s);
      }

      if ((size_t)i < opSplitAxes.size() && !opSplitAxes[i].empty()) {
        ++currHaloDim;
        // the offsets for lower dim sstarts after their down halo
        offsets[i] = haloSizes[currHaloDim * 2];

        // prepare shape and offsets of highest dim's halo exchange
        auto _haloSz =
            rewriter
                .create<arith::AddIOp>(loc, toValue(haloSizes[currHaloDim * 2]),
                                       toValue(haloSizes[currHaloDim * 2 + 1]))
                .getResult();
        // the halo shape of lower dims exlude the halos
        dimSizes[i] =
            rewriter.create<arith::SubIOp>(loc, toValue(shape[i]), _haloSz)
                .getResult();
      } else {
        dimSizes[i] = shape[i];
      }
    }

    auto tagAttr = rewriter.getI32IntegerAttr(91); // we just pick something
    auto tag = rewriter.create<::mlir::arith::ConstantOp>(loc, tagAttr);
    auto zeroAttr = rewriter.getI32IntegerAttr(0); // for detecting v<0
    auto zero = rewriter.create<::mlir::arith::ConstantOp>(loc, zeroAttr);

    SmallVector<Type> indexResultTypes(meshOp.getShape().size(),
                                       rewriter.getIndexType());
    auto myMultiIndex =
        rewriter.create<ProcessMultiIndexOp>(loc, indexResultTypes, mesh)
            .getResult();
    // traverse all split axes from high to low dim
    for (ssize_t dim = opSplitAxes.size() - 1; dim >= 0; --dim) {
      auto splitAxes = opSplitAxes[dim];
      if (splitAxes.empty()) {
        continue;
      }
      assert(currHaloDim >= 0 && (size_t)currHaloDim < haloSizes.size() / 2);
      // Get the linearized ids of the neighbors (down and up) for the
      // given split
      auto tmp = rewriter
                     .create<NeighborsLinearIndicesOp>(loc, mesh, myMultiIndex,
                                                       splitAxes)
                     .getResults();
      // MPI operates on i32...
      Value neighbourIDs[2] = {rewriter.create<arith::IndexCastOp>(
                                   loc, rewriter.getI32Type(), tmp[0]),
                               rewriter.create<arith::IndexCastOp>(
                                   loc, rewriter.getI32Type(), tmp[1])};

      auto lowerRecvOffset = rewriter.getIndexAttr(0);
      auto lowerSendOffset = toValue(haloSizes[currHaloDim * 2]);
      auto upperRecvOffset = rewriter.create<arith::SubIOp>(
          loc, toValue(shape[dim]), toValue(haloSizes[currHaloDim * 2 + 1]));
      auto upperSendOffset = rewriter.create<arith::SubIOp>(
          loc, upperRecvOffset, toValue(haloSizes[currHaloDim * 2]));

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
        // Processes on the mesh borders have only one neighbor
        auto to = upperHalo ? neighbourIDs[1] : neighbourIDs[0];
        auto from = upperHalo ? neighbourIDs[0] : neighbourIDs[1];
        auto hasFrom = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, from, zero);
        auto hasTo = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sge, to, zero);
        auto buffer = rewriter.create<memref::AllocOp>(
            loc, dimSizes, cast<ShapedType>(array.getType()).getElementType());
        // if has neighbor: copy halo data from array to buffer and send
        rewriter.create<scf::IfOp>(
            loc, hasTo, [&](OpBuilder &builder, Location loc) {
              offsets[dim] = upperHalo ? OpFoldResult(lowerSendOffset)
                                       : OpFoldResult(upperSendOffset);
              auto subview = builder.create<memref::SubViewOp>(
                  loc, array, offsets, dimSizes, strides);
              builder.create<memref::CopyOp>(loc, subview, buffer);
              builder.create<mpi::SendOp>(loc, TypeRange{}, buffer, tag, to);
              builder.create<scf::YieldOp>(loc);
            });
        // if has neighbor: receive halo data into buffer and copy to array
        rewriter.create<scf::IfOp>(
            loc, hasFrom, [&](OpBuilder &builder, Location loc) {
              offsets[dim] = upperHalo ? OpFoldResult(upperRecvOffset)
                                       : OpFoldResult(lowerRecvOffset);
              builder.create<mpi::RecvOp>(loc, TypeRange{}, buffer, tag, from);
              auto subview = builder.create<memref::SubViewOp>(
                  loc, array, offsets, dimSizes, strides);
              builder.create<memref::CopyOp>(loc, buffer, subview);
              builder.create<scf::YieldOp>(loc);
            });
        rewriter.create<memref::DeallocOp>(loc, buffer);
        offsets[dim] = orgOffset;
      };

      genSendRecv(false);
      genSendRecv(true);

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
      rewriter.replaceOp(op, rewriter.create<bufferization::ToTensorOp>(
                                 loc, op.getResult().getType(), array,
                                 /*restrict=*/true, /*writable=*/true));
    }
    return mlir::success();
  }
};

struct ConvertMeshToMPIPass
    : public impl::ConvertMeshToMPIPassBase<ConvertMeshToMPIPass> {
  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    auto *ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    patterns.insert<ConvertUpdateHaloOp, ConvertNeighborsLinearIndicesOp,
                    ConvertProcessLinearIndexOp, ConvertProcessMultiIndexOp>(
        ctx);

    (void)mlir::applyPatternsGreedily(getOperation(), std::move(patterns));
  }
};

} // namespace

// Create a pass that convert Mesh to MPI
std::unique_ptr<::mlir::Pass> mlir::createConvertMeshToMPIPass() {
  return std::make_unique<ConvertMeshToMPIPass>();
}
