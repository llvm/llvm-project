//===- Passes.h - GPU ROCDL pass entrypoints --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of the ROCDL-specific GPU passes and
// patterns.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_ROCDL_PASSES_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_ROCDL_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLTargetInfo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DECL
#include "mlir/Dialect/GPU/Transforms/ROCDL/Passes.h.inc"

/// Collect a set of patterns to lower `gpu.subgroup_reduce` into `amdgpu.dpp`
/// ops over scalar types. Assumes that the subgroup has
/// `subgroupSize` lanes. Applicable only to AMD GPUs.
void populateGpuLowerSubgroupReduceToDPPPatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    const ROCDL::TargetInfo &target, PatternBenefit benefit = 1);

/// Disjoint counterpart of `populateGpuLowerSubgroupReduceToDPPPatterns`
/// that only matches `gpu.subgroup_reduce` ops with a `cluster_size`.
void populateGpuLowerClusteredSubgroupReduceToDPPPatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    const ROCDL::TargetInfo &target, PatternBenefit benefit = 1);

/// Tries to promote `gpu.shuffle`s to specialized AMDGPU intrinsics.
void populateGpuPromoteShuffleToAMDGPUPatterns(
    RewritePatternSet &patterns, std::optional<ROCDL::TargetInfo> target);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Transforms/ROCDL/Passes.h.inc"
} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_ROCDL_PASSES_H_
