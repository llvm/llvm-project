//===- GPUToAMDGPU.cpp - GPU to AMDGPU dialect conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToAMDGPU/GPUToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/Support/FormatVariadic.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTGPUTOAMDGPUPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ClusterInfo {
  unsigned clusterStride;
  unsigned clusterSize;
  unsigned subgroupSize;
};

static FailureOr<ClusterInfo>
getAndValidateClusterInfo(gpu::SubgroupReduceOp op, unsigned subgroupSize) {
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

Value createSubgroupDPPReduction(OpBuilder &b, Location loc, Value input,
                                 gpu::AllReduceOperation mode,
                                 const ClusterInfo &ci) {
  Value result = input;
  if (ci.clusterSize >= 2) {
    auto permArg = b.getIntegerAttr(b.getIntegerType(32), 1);
    Value dppResult =
        b.create<amdgpu::DPPOp>(loc, result.getType(), result, result,
                                amdgpu::DPPPerm::row_shl, permArg);
    result = vector::makeArithReduction(b, loc, gpu::convertReductionKind(mode),
                                        result, dppResult);
  }

  if (ci.clusterSize >= 4) {
    auto permArg = b.getIntegerAttr(b.getIntegerType(32), 2);
    Value dppResult =
        b.create<amdgpu::DPPOp>(loc, result.getType(), result, result,
                                amdgpu::DPPPerm::row_shl, permArg);
    result = vector::makeArithReduction(b, loc, gpu::convertReductionKind(mode),
                                        result, dppResult);
  }

  if (ci.clusterSize >= 8) {
    Value dppResult = b.create<amdgpu::DPPOp>(
        loc, result.getType(), result, result, amdgpu::DPPPerm::row_half_mirror,
        b.getUnitAttr());
    result = vector::makeArithReduction(b, loc, gpu::convertReductionKind(mode),
                                        result, dppResult);
  }

  if (ci.clusterSize >= 16) {
    Value dppResult =
        b.create<amdgpu::DPPOp>(loc, result.getType(), result, result,
                                amdgpu::DPPPerm::row_mirror, b.getUnitAttr());
    result = vector::makeArithReduction(b, loc, gpu::convertReductionKind(mode),
                                        result, dppResult);
  }

  const int allRows = 0xf;
  const int allBanks = 0xf;
  auto int32Type = IntegerType::get(b.getContext(), 32);
  if (ci.clusterSize >= 32) {
    auto permArg = b.getIntegerAttr(b.getIntegerType(32), 15);
    Value dppResult = b.create<amdgpu::DPPOp>(
        loc, result.getType(), result, result, amdgpu::DPPPerm::row_bcast_15,
        b.getUnitAttr(), 0xa, allBanks, false);
    result = vector::makeArithReduction(b, loc, gpu::convertReductionKind(mode),
    result, dppResult);
    if (ci.subgroupSize == 32) {
      Value lane01 = b.create<LLVM::ConstantOp>(loc, int32Type, 1);
      result = b.create<ROCDL::ReadlaneOp>(loc, input.getType(), result, lane01);    
    }
  }

  if (ci.clusterSize == 64) {
    auto permArg = b.getIntegerAttr(b.getIntegerType(32), 31);
    Value dppResult = b.create<amdgpu::DPPOp>(
        loc, result.getType(), result, result, amdgpu::DPPPerm::row_bcast_31,
        b.getUnitAttr(), allRows, allBanks, false);
    result = vector::makeArithReduction(b, loc, gpu::convertReductionKind(mode),
                                        result, dppResult);
    Value lane63 = b.create<LLVM::ConstantOp>(loc, int32Type, 63);
    result = b.create<ROCDL::ReadlaneOp>(loc, input.getType(), result, lane63);
  }

  assert(result.getType() == input.getType());
  return result;
}

struct ScalarSubgroupReduceToShuffles final
    : OpRewritePattern<gpu::SubgroupReduceOp> {
  ScalarSubgroupReduceToShuffles(MLIRContext *ctx, unsigned subgroupSize,
                                 bool matchClustered, PatternBenefit benefit)
      : OpRewritePattern(ctx, benefit), subgroupSize(subgroupSize),
        matchClustered(matchClustered) {}

  LogicalResult matchAndRewrite(gpu::SubgroupReduceOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getClusterSize().has_value() != matchClustered) {
      return rewriter.notifyMatchFailure(
          op, llvm::formatv("op is {0}clustered but pattern is configured to "
                            "only match {1}clustered ops",
                            matchClustered ? "non-" : "",
                            matchClustered ? "" : "non-"));
    }

    auto ci = getAndValidateClusterInfo(op, subgroupSize);
    if (failed(ci))
      return failure();

    Location loc = op.getLoc();
    rewriter.replaceOp(op, createSubgroupDPPReduction(
                               rewriter, loc, op.getValue(), op.getOp(), *ci));
    return success();
  }

private:
  unsigned subgroupSize = 0;
  bool matchClustered = false;
};

struct ConvertGPUToAMDGPUPass
    : public impl::ConvertGPUToAMDGPUPassBase<ConvertGPUToAMDGPUPass> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    int subgroupSizeInt = static_cast<int>(subgroupSize);
    populateAMDGPUOptimizedSubgroupReducePatterns(patterns, subgroupSizeInt,
                                           PatternBenefit(1));
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};
} // namespace

void mlir::populateAMDGPUOptimizedSubgroupReducePatterns(RewritePatternSet &patterns,
                                                  unsigned subgroupSize,
                                                  PatternBenefit benefit) {
  patterns.add<ScalarSubgroupReduceToShuffles>(
      patterns.getContext(), subgroupSize, /*matchClustered=*/true, benefit);
}
