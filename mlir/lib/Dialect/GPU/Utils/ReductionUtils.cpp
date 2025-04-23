//===- ReductionUtils.cpp - Distribution tools for GPUOps --------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements distribution utility methods.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/GPU/Utils/ReductionUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <numeric>

using namespace mlir;

FailureOr<ClusterInfo> mlir::getAndValidateClusterInfo(gpu::SubgroupReduceOp op,
                                                 unsigned subgroupSize) {
  assert(llvm::isPowerOf2_32(subgroupSize));

  std::optional<uint32_t> clusterSize = op.getClusterSize();
  assert(!clusterSize ||
         llvm::isPowerOf2_32(*clusterSize)); // Verifier should've caught this.
  if (clusterSize && *clusterSize > subgroupSize)
    return op.emitOpError()
           << "cluster size " << *clusterSize
           << " is greater than subgroup size " << subgroupSize;
  unsigned effectiveClusterSize = clusterSize.value_or(subgroupSize);

  auto clusterStride = op.getClusterStride();
  assert(llvm::isPowerOf2_32(clusterStride)); // Verifier should've caught this.
  if (clusterStride >= subgroupSize)
    return op.emitOpError()
           << "cluster stride " << clusterStride
           << " is not less than subgroup size " << subgroupSize;

  return ClusterInfo{clusterStride, effectiveClusterSize, subgroupSize};
}

FailureOr<Value> mlir::createSubgroupDPPReduction(
    PatternRewriter &rewriter, gpu::SubgroupReduceOp op, Value input,
    gpu::AllReduceOperation mode, const ClusterInfo &ci,
    amdgpu::Chipset chipset, function_ref<Value(Value)> packFn,
    function_ref<Value(Value)> unpackFn) {
  Location loc = op.getLoc();
  Value dpp;
  Value res = input;
  constexpr int allRows = 0xf;
  constexpr int allBanks = 0xf;
  const bool boundCtrl = true;
  if (ci.clusterSize >= 2) {
    // Perform reduction between all lanes N <-> N+1.
    res = packFn(res);
    dpp = rewriter.create<amdgpu::DPPOp>(
        loc, res.getType(), res, res, amdgpu::DPPPerm::quad_perm,
        rewriter.getI32ArrayAttr({1, 0, 3, 2}), allRows, allBanks, boundCtrl);
    dpp = unpackFn(dpp);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }

  if (ci.clusterSize >= 4) {
    // Perform reduction between all lanes N <-> N+2.
    res = packFn(res);
    dpp = rewriter.create<amdgpu::DPPOp>(
        loc, res.getType(), res, res, amdgpu::DPPPerm::quad_perm,
        rewriter.getI32ArrayAttr({2, 3, 0, 1}), allRows, allBanks, boundCtrl);
    dpp = unpackFn(dpp);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 8) {
    // Perform reduction between all lanes N <-> 7-N,
    // e.g lane[0] <-> lane[7], lane[1] <-> lane[6]..., lane[3] <-> lane[4].
    res = packFn(res);
    dpp = rewriter.create<amdgpu::DPPOp>(
        loc, res.getType(), res, res, amdgpu::DPPPerm::row_half_mirror,
        rewriter.getUnitAttr(), allRows, allBanks, boundCtrl);
    dpp = unpackFn(dpp);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 16) {
    // Perform reduction between all lanes N <-> 15-N,
    // e.g lane[0] <-> lane[15], lane[1] <-> lane[14]..., lane[7] <-> lane[8].
    res = packFn(res);
    dpp = rewriter.create<amdgpu::DPPOp>(
        loc, res.getType(), res, res, amdgpu::DPPPerm::row_mirror,
        rewriter.getUnitAttr(), allRows, allBanks, boundCtrl);
    dpp = unpackFn(dpp);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 32) {
    if (chipset.majorVersion <= 9) {
      // Broadcast last value from each row to next row.
      // Use row mask to avoid polluting rows 1 and 3.
      res = packFn(res);
      dpp = rewriter.create<amdgpu::DPPOp>(
          loc, res.getType(), res, res, amdgpu::DPPPerm::row_bcast_15,
          rewriter.getUnitAttr(), 0xa, allBanks,
          /*bound_ctrl*/ false);
      dpp = unpackFn(dpp);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), res, dpp);
    } else if (chipset.majorVersion <= 12) {
      // Use a permute lane to cross rows (row 1 <-> row 0, row 3 <-> row 2).
      Value uint32Max = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(-1));
      res = packFn(res);
      dpp = rewriter.create<ROCDL::PermlaneX16Op>(loc, res.getType(), res, res,
                                                  uint32Max, uint32Max,
                                                  /*fi=*/true,
                                                  /*bound_ctrl=*/false);
      dpp = unpackFn(dpp);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), res, dpp);
      if (ci.subgroupSize == 32) {
        Value lane0 = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        res =
            rewriter.create<ROCDL::ReadlaneOp>(loc, res.getType(), res, lane0);
      }
    } else {
      return rewriter.notifyMatchFailure(
          op, "Subgroup reduce lowering to DPP not currently supported for "
              "this device.");
    }
  }
  if (ci.clusterSize >= 64) {
    if (chipset.majorVersion <= 9) {
      // Broadcast 31st lane value to rows 2 and 3.
      // Use row mask to avoid polluting rows 0 and 1.
      res = packFn(res);
      dpp = rewriter.create<amdgpu::DPPOp>(
          loc, res.getType(), res, res, amdgpu::DPPPerm::row_bcast_31,
          rewriter.getUnitAttr(), 0xc, allBanks,
          /*bound_ctrl*/ false);
      dpp = unpackFn(dpp);

    } else if (chipset.majorVersion <= 12) {
      // Assume reduction across 32 lanes has been done.
      // Perform final reduction manually by summing values in lane 0 and
      // lane 32.
      Value lane0 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value lane32 = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(32));
      dpp = rewriter.create<ROCDL::ReadlaneOp>(loc, res.getType(), res, lane32);
      res = rewriter.create<ROCDL::ReadlaneOp>(loc, res.getType(), res, lane0);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Subgroup reduce lowering to DPP not currently supported for "
              "this device.");
    }
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  assert(res.getType() == input.getType());
  return res;
}