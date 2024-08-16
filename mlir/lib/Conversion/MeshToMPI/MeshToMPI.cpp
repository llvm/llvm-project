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
#include "mlir/Dialect/MPI/IR/MPI.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
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

// This pattern converts the mesh.update_halo operation to MPI calls
struct ConvertUpdateHaloOp
    : public mlir::OpRewritePattern<mlir::mesh::UpdateHaloOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::mesh::UpdateHaloOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Halos are exchanged as 2 blocks per dimension (one for each side: down
    // and up). It is assumed that the last dim in a default memref is
    // contiguous, hence iteration starts with the complete halo on the first
    // dim which should be contiguous (unless the source is not). The size of
    // the exchanged data will decrease when iterating over dimensions. That's
    // good because the halos of last dim will be most fragmented.
    // memref.subview is used to read and write the halo data from and to the
    // local data. subviews and halos have dynamic and static values, so
    // OpFoldResults are used whenever possible.

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

    auto array = op.getInput();
    auto rank = array.getType().getRank();
    auto mesh = op.getMesh();
    auto meshOp = getMesh(op, symbolTableCollection);
    auto haloSizes = getMixedValues(op.getStaticHaloSizes(),
                                    op.getDynamicHaloSizes(), rewriter);
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
    SmallVector<OpFoldResult> shape(rank);
    // we need the actual shape to compute offsets and sizes
    for (auto [i, s] : llvm::enumerate(array.getType().getShape())) {
      if (ShapedType::isDynamic(s)) {
        shape[i] = rewriter.create<memref::DimOp>(loc, array, s).getResult();
      } else {
        shape[i] = rewriter.getIndexAttr(s);
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
    // halo sizes are provided for split dimensions only
    auto currHaloDim = 0;

    for (auto [dim, splitAxes] : llvm::enumerate(op.getSplitAxes())) {
      if (splitAxes.empty()) {
        continue;
      }
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
      // store for later
      auto orgDimSize = shape[dim];
      // this dim's offset to the start of the upper halo
      auto upperOffset = rewriter.create<arith::SubIOp>(
          loc, toValue(shape[dim]), toValue(haloSizes[currHaloDim * 2 + 1]));

      // Make sure we send/recv in a way that does not lead to a dead-lock.
      // The current approach is by far not optimal, this should be at least
      // be a red-black pattern or using MPI_sendrecv.
      // Also, buffers should be re-used.
      // Still using temporary contiguous buffers for MPI communication...
      // Still yielding a "serialized" communication pattern...
      auto genSendRecv = [&](auto dim, bool upperHalo) {
        auto orgOffset = offsets[dim];
        shape[dim] = upperHalo ? haloSizes[currHaloDim * 2 + 1]
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
            loc, shape, array.getType().getElementType());
        // if has neighbor: copy halo data from array to buffer and send
        rewriter.create<scf::IfOp>(
            loc, hasTo, [&](OpBuilder &builder, Location loc) {
              offsets[dim] = upperHalo ? OpFoldResult(builder.getIndexAttr(0))
                                       : OpFoldResult(upperOffset);
              auto subview = builder.create<memref::SubViewOp>(
                  loc, array, offsets, shape, strides);
              builder.create<memref::CopyOp>(loc, subview, buffer);
              builder.create<mpi::SendOp>(loc, TypeRange{}, buffer, tag, to);
              builder.create<scf::YieldOp>(loc);
            });
        // if has neighbor: receive halo data into buffer and copy to array
        rewriter.create<scf::IfOp>(
            loc, hasFrom, [&](OpBuilder &builder, Location loc) {
              offsets[dim] = upperHalo ? OpFoldResult(upperOffset)
                                       : OpFoldResult(builder.getIndexAttr(0));
              builder.create<mpi::RecvOp>(loc, TypeRange{}, buffer, tag, from);
              auto subview = builder.create<memref::SubViewOp>(
                  loc, array, offsets, shape, strides);
              builder.create<memref::CopyOp>(loc, buffer, subview);
              builder.create<scf::YieldOp>(loc);
            });
        rewriter.create<memref::DeallocOp>(loc, buffer);
        offsets[dim] = orgOffset;
      };

      genSendRecv(dim, false);
      genSendRecv(dim, true);

      // prepare shape and offsets for next split dim
      auto _haloSz =
          rewriter
              .create<arith::AddIOp>(loc, toValue(haloSizes[currHaloDim * 2]),
                                     toValue(haloSizes[currHaloDim * 2 + 1]))
              .getResult();
      // the shape for next halo excludes the halo on both ends for the
      // current dim
      shape[dim] =
          rewriter.create<arith::SubIOp>(loc, toValue(orgDimSize), _haloSz)
              .getResult();
      // the offsets for next halo starts after the down halo for the
      // current dim
      offsets[dim] = haloSizes[currHaloDim * 2];
      // on to next halo
      ++currHaloDim;
    }
    rewriter.eraseOp(op);
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

    patterns.insert<ConvertUpdateHaloOp>(ctx);

    (void)mlir::applyPatternsAndFoldGreedily(getOperation(),
                                             std::move(patterns));
  }
};

} // namespace

// Create a pass that convert Mesh to MPI
std::unique_ptr<::mlir::OperationPass<void>> createConvertMeshToMPIPass() {
  return std::make_unique<ConvertMeshToMPIPass>();
}
