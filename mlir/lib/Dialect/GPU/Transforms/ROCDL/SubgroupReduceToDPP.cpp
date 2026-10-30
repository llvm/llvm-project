//===- SubgroupReduceToDPP.cpp - subgroup_reduce to AMDGPU DPP lowering ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements lowering of `gpu.subgroup_reduce` ops to AMDGPU DPP operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/ROCDL/Passes.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/FormatVariadic.h"
#include <cassert>

using namespace mlir;

namespace {
static FailureOr<Value>
createSubgroupDPPReduction(PatternRewriter &rewriter, gpu::SubgroupReduceOp op,
                           Value input, gpu::AllReduceOperation mode,
                           const gpu::SubgroupReduceClusterInfo &ci,
                           const ROCDL::TargetInfo &target) {
  Location loc = op.getLoc();
  Value dpp;
  Value res = input;
  constexpr int allRows = 0xf;
  constexpr int allBanks = 0xf;
  const bool boundCtrl = true;
  if (ci.clusterSize >= 2) {
    // Perform reduction between all lanes N <-> N+1.
    dpp = amdgpu::DPPOp::create(
        rewriter, loc, res.getType(), res, res, amdgpu::DPPPerm::quad_perm,
        rewriter.getI32ArrayAttr({1, 0, 3, 2}), allRows, allBanks, boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }

  if (ci.clusterSize >= 4) {
    // Perform reduction between all lanes N <-> N+2.
    dpp = amdgpu::DPPOp::create(
        rewriter, loc, res.getType(), res, res, amdgpu::DPPPerm::quad_perm,
        rewriter.getI32ArrayAttr({2, 3, 0, 1}), allRows, allBanks, boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 8) {
    // Perform reduction between all lanes N <-> 7-N,
    // e.g lane[0] <-> lane[7], lane[1] <-> lane[6]..., lane[3] <-> lane[4].
    dpp = amdgpu::DPPOp::create(rewriter, loc, res.getType(), res, res,
                                amdgpu::DPPPerm::row_half_mirror,
                                rewriter.getUnitAttr(), allRows, allBanks,
                                boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 16) {
    // Perform reduction between all lanes N <-> 15-N,
    // e.g lane[0] <-> lane[15], lane[1] <-> lane[14]..., lane[7] <-> lane[8].
    dpp = amdgpu::DPPOp::create(
        rewriter, loc, res.getType(), res, res, amdgpu::DPPPerm::row_mirror,
        rewriter.getUnitAttr(), allRows, allBanks, boundCtrl);
    res = vector::makeArithReduction(rewriter, loc,
                                     gpu::convertReductionKind(mode), res, dpp);
  }
  if (ci.clusterSize >= 32) {
    if (!target.has(llvm::AMDGPU::FEAT_GFX10_INSTS)) {
      // Broadcast last value from each row to next row.
      // Use row mask to avoid polluting row 0 (and row 2 if wave-64).
      dpp = amdgpu::DPPOp::create(rewriter, loc, res.getType(), res, res,
                                  amdgpu::DPPPerm::row_bcast_15,
                                  rewriter.getUnitAttr(), 0xa, allBanks,
                                  /*bound_ctrl*/ false);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), res, dpp);

      // For subgroupSize = 64, at this point lanes [16, 32) contain the full
      // reduction over lanes [0, 32), but lanes [0, 16) do not. Similarly,
      // lanes [48, 64) contain the full reduction over lanes [32, 64), but
      // lanes [32, 48) do not.
      //
      // If subgroup size is 64 and cluster size is 64, we don't need lanes [0,
      // 16) and [32, 48) to have the correct cluster-32 reduction values at
      // this point, because only lane 63's value will ultimately be read in
      // this full-cluster case.
      //
      // If subgroup size is 64 and cluster size is 32, we need to ensure that
      // lanes [0, 16) and [32, 48) have the correct final cluster-32 reduction
      // values (subgroup_reduce guarantees that all lanes within each cluster
      // contain the final reduction value). We do this by broadcasting lane
      // 31's value to lanes [0, 16) and lanes 63's value to lanes [32, 48).
      //
      // See https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations
      // for an illustration of how this within-cluster broadcast works with a
      // swizzle.
      if (ci.subgroupSize == 64 && ci.clusterSize == 32) {
        res =
            amdgpu::SwizzleBitModeOp::create(rewriter, loc, res, /*and_mask=*/0,
                                             /*or_mask=*/31,
                                             /*xor_mask=*/0);
      }
    } else if (!target.has(llvm::AMDGPU::FEAT_GFX13_INSTS)) {
      // Use a permute lane to cross rows (row 1 <-> row 0, row 3 <-> row 2).
      Value uint32Max = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(-1));
      dpp = ROCDL::PermlaneX16Op::create(rewriter, loc, res.getType(), res, res,
                                         uint32Max, uint32Max,
                                         /*fi=*/true,
                                         /*boundControl=*/false);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), res, dpp);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Subgroup reduce lowering to DPP not currently supported for "
              "this device.");
    }
    if (ci.subgroupSize == 32) {
      Value lane31 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(31));
      res =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane31);
    }
  }
  if (ci.clusterSize >= 64) {
    if (!target.has(llvm::AMDGPU::FEAT_GFX10_INSTS)) {
      // Broadcast 31st lane value to rows 2 and 3.
      dpp = amdgpu::DPPOp::create(rewriter, loc, res.getType(), res, res,
                                  amdgpu::DPPPerm::row_bcast_31,
                                  rewriter.getUnitAttr(), 0xf, allBanks,
                                  /*bound_ctrl*/ true);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), dpp, res);
      // Obtain reduction from last rows, the previous rows are polluted.
      Value lane63 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(63));
      res =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane63);

    } else if (!target.has(llvm::AMDGPU::FEAT_GFX13_INSTS)) {
      // Assume reduction across 32 lanes has been done.
      // Perform final reduction manually by summing values in lane 0 and
      // lane 32.
      Value lane31 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(31));
      Value lane63 = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(63));
      lane31 =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane31);
      lane63 =
          ROCDL::ReadlaneOp::create(rewriter, loc, res.getType(), res, lane63);
      res = vector::makeArithReduction(
          rewriter, loc, gpu::convertReductionKind(mode), lane31, lane63);
    } else {
      return rewriter.notifyMatchFailure(
          op, "Subgroup reduce lowering to DPP not currently supported for "
              "this device.");
    }
  }
  assert(res.getType() == input.getType());
  return res;
}

/// Collect a set of patterns to lower `gpu.subgroup_reduce` into `amdgpu.dpp`
/// ops over scalar types. Assumes that the subgroup has
/// `subgroupSize` lanes. Applicable only to AMD GPUs.
struct ScalarSubgroupReduceToDPP final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  ScalarSubgroupReduceToDPP(MLIRContext *ctx, unsigned subgroupSize,
                            bool matchClustered,
                            const ROCDL::TargetInfo &target,
                            PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        matchClustered(matchClustered), target(target) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getClusterSize().has_value() != matchClustered) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("op is {0}clustered but pattern is configured to "
                            "only match {1}clustered ops",
                            matchClustered ? "non-" : "",
                            matchClustered ? "" : "non-"));
    }
    auto ci = gpu::getSubgroupReduceClusterInfo(op, subgroupSize);
    if (failed(ci))
      return failure();

    if (ci->clusterStride != 1)
      return rewriter.notifyMatchFailure(
          op, "Subgroup reductions using DPP are currently only available for "
              "clusters of contiguous lanes.");

    Type valueTy = op.getType();
    if (!valueTy.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "Value type is not a compatible scalar.");

    FailureOr<Value> dpp = createSubgroupDPPReduction(
        rewriter, op, op.getValue(), op.getOp(), *ci, target);
    if (failed(dpp))
      return failure();

    rewriter.replaceOp(op, dpp.value());
    return success();
  }

private:
  unsigned subgroupSize = 0;
  bool matchClustered = false;
  ROCDL::TargetInfo target;
};
} // namespace

void mlir::populateGpuLowerSubgroupReduceToDPPPatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    const ROCDL::TargetInfo &target, PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToDPP>(patterns.getContext(), subgroupSize,
                                          /*matchClustered=*/false, target,
                                          benefit);
}

void mlir::populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    const ROCDL::TargetInfo &target, PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToDPP>(patterns.getContext(), subgroupSize,
                                          /*matchClustered=*/true, target,
                                          benefit);
}
