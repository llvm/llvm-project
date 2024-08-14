//===- MeshToMPI.cpp - Mesh to MPI  dialect conversion -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a translation of Mesh communicatin ops tp MPI ops.
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

#define DEBUG_TYPE "mesh-to-mpi"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

namespace mlir {
#define GEN_PASS_DEF_CONVERTMESHTOMPIPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::mesh;

namespace {
struct ConvertMeshToMPIPass
    : public impl::ConvertMeshToMPIPassBase<ConvertMeshToMPIPass> {
  using Base::Base;

  /// Run the dialect converter on the module.
  void runOnOperation() override {
    getOperation()->walk([&](UpdateHaloOp op) {
      SymbolTableCollection symbolTableCollection;
      OpBuilder builder(op);
      auto loc = op.getLoc();

      auto toValue = [&builder, &loc](OpFoldResult &v) {
        return v.is<Value>()
                   ? v.get<Value>()
                   : builder.create<::mlir::arith::ConstantOp>(
                         loc,
                         builder.getIndexAttr(
                             cast<IntegerAttr>(v.get<Attribute>()).getInt()));
      };

      auto array = op.getInput();
      auto rank = array.getType().getRank();
      auto mesh = op.getMesh();
      auto meshOp = getMesh(op, symbolTableCollection);
      auto haloSizes = getMixedValues(op.getStaticHaloSizes(),
                                      op.getDynamicHaloSizes(), builder);
      for (auto &sz : haloSizes) {
        if (sz.is<Value>()) {
          sz = builder
                   .create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                               sz.get<Value>())
                   .getResult();
        }
      }

      SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
      SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
      SmallVector<OpFoldResult> shape(rank);
      for (auto [i, s] : llvm::enumerate(array.getType().getShape())) {
        if (ShapedType::isDynamic(s)) {
          shape[i] = builder.create<memref::DimOp>(loc, array, s).getResult();
        } else {
          shape[i] = builder.getIndexAttr(s);
        }
      }

      auto tagAttr = builder.getI32IntegerAttr(91); // whatever
      auto tag = builder.create<::mlir::arith::ConstantOp>(loc, tagAttr);
      auto zeroAttr = builder.getI32IntegerAttr(0); // whatever
      auto zero = builder.create<::mlir::arith::ConstantOp>(loc, zeroAttr);
      SmallVector<Type> indexResultTypes(meshOp.getShape().size(),
                                         builder.getIndexType());
      auto myMultiIndex =
          builder.create<ProcessMultiIndexOp>(loc, indexResultTypes, mesh)
              .getResult();
      auto currHaloDim = 0;

      for (auto [dim, splitAxes] : llvm::enumerate(op.getSplitAxes())) {
        if (!splitAxes.empty()) {
          auto tmp = builder
                         .create<NeighborsLinearIndicesOp>(
                             loc, mesh, myMultiIndex, splitAxes)
                         .getResults();
          Value neighbourIDs[2] = {builder.create<arith::IndexCastOp>(
                                       loc, builder.getI32Type(), tmp[0]),
                                   builder.create<arith::IndexCastOp>(
                                       loc, builder.getI32Type(), tmp[1])};
          auto orgDimSize = shape[dim];
          auto upperOffset = builder.create<arith::SubIOp>(
              loc, toValue(shape[dim]), toValue(haloSizes[dim * 2 + 1]));

          // make sure we send/recv in a way that does not lead to a dead-lock
          // This is by far not optimal, this should be at least MPI_sendrecv
          // and - probably even more importantly - buffers should be re-used
          // Currently using temporary, contiguous buffer for MPI communication
          auto genSendRecv = [&](auto dim, bool upperHalo) {
            auto orgOffset = offsets[dim];
            shape[dim] =
                upperHalo ? haloSizes[dim * 2 + 1] : haloSizes[dim * 2];
            auto to = upperHalo ? neighbourIDs[1] : neighbourIDs[0];
            auto from = upperHalo ? neighbourIDs[0] : neighbourIDs[1];
            auto hasFrom = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sge, from, zero);
            auto hasTo = builder.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::sge, to, zero);
            auto buffer = builder.create<memref::AllocOp>(
                loc, shape, array.getType().getElementType());
            builder.create<scf::IfOp>(
                loc, hasTo, [&](OpBuilder &builder, Location loc) {
                  offsets[dim] = upperHalo
                                     ? OpFoldResult(builder.getIndexAttr(0))
                                     : OpFoldResult(upperOffset);
                  auto subview = builder.create<memref::SubViewOp>(
                      loc, array, offsets, shape, strides);
                  builder.create<memref::CopyOp>(loc, subview, buffer);
                  builder.create<mpi::SendOp>(loc, TypeRange{}, buffer, tag,
                                              to);
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<scf::IfOp>(
                loc, hasFrom, [&](OpBuilder &builder, Location loc) {
                  offsets[dim] = upperHalo
                                     ? OpFoldResult(upperOffset)
                                     : OpFoldResult(builder.getIndexAttr(0));
                  builder.create<mpi::RecvOp>(loc, TypeRange{}, buffer, tag,
                                              from);
                  auto subview = builder.create<memref::SubViewOp>(
                      loc, array, offsets, shape, strides);
                  builder.create<memref::CopyOp>(loc, buffer, subview);
                  builder.create<scf::YieldOp>(loc);
                });
            builder.create<memref::DeallocOp>(loc, buffer);
            offsets[dim] = orgOffset;
          };

          genSendRecv(dim, false);
          genSendRecv(dim, true);

          shape[dim] = builder
                           .create<arith::SubIOp>(
                               loc, toValue(orgDimSize),
                               builder
                                   .create<arith::AddIOp>(
                                       loc, toValue(haloSizes[dim * 2]),
                                       toValue(haloSizes[dim * 2 + 1]))
                                   .getResult())
                           .getResult();
          offsets[dim] = haloSizes[dim * 2];
          ++currHaloDim;
        }
      }
    });
  }
};
} // namespace