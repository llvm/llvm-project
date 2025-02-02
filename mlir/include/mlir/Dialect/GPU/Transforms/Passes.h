//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_GPU_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include <optional>

namespace llvm {
class TargetMachine;
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class TypeConverter;
class ConversionTarget;
namespace func {
class FuncOp;
} // namespace func

#define GEN_PASS_DECL
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"

/// Pass that moves ops which are likely an index computation into gpu.launch
/// body.
std::unique_ptr<Pass> createGpuLauchSinkIndexComputationsPass();

/// Replaces `gpu.launch` with `gpu.launch_func` by moving the region into
/// a separate kernel function.
std::unique_ptr<OperationPass<ModuleOp>>
createGpuKernelOutliningPass(StringRef dataLayoutStr = StringRef());

/// Rewrites a function region so that GPU ops execute asynchronously.
std::unique_ptr<OperationPass<func::FuncOp>> createGpuAsyncRegionPass();

/// Maps the parallel loops found in the given function to workgroups. The first
/// loop encountered will be mapped to the global workgroup and the second loop
/// encountered to the local workgroup. Within each mapping, the first three
/// dimensions are mapped to x/y/z hardware ids and all following dimensions are
/// mapped to sequential loops.
std::unique_ptr<OperationPass<func::FuncOp>> createGpuMapParallelLoopsPass();

/// Collect a set of patterns to rewrite GlobalIdOp op within the GPU dialect.
void populateGpuGlobalIdPatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to rewrite shuffle ops within the GPU dialect.
void populateGpuShufflePatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to rewrite all-reduce ops within the GPU dialect.
void populateGpuAllReducePatterns(RewritePatternSet &patterns);

/// Collect a set of patterns to break down subgroup_reduce ops into smaller
/// ones supported by the target of `size <= maxShuffleBitwidth`, where `size`
/// is the subgroup_reduce value bitwidth.
void populateGpuBreakDownSubgroupReducePatterns(
    RewritePatternSet &patterns, unsigned maxShuffleBitwidth = 32,
    PatternBenefit benefit = 1);

/// Collect a set of patterns to lower `gpu.subgroup_reduce` into `gpu.shuffle`
/// ops over `shuffleBitwidth` scalar types. Assumes that the subgroup has
/// `subgroupSize` lanes. Uses the butterfly shuffle algorithm.
///
/// The patterns populated by this function will ignore ops with the
/// `cluster_size` attribute.
/// `populateGpuLowerClusteredSubgroupReduceToShufflePatterns` is the opposite.
void populateGpuLowerSubgroupReduceToShufflePatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    unsigned shuffleBitwidth = 32, PatternBenefit benefit = 1);

/// Disjoint counterpart of `populateGpuLowerSubgroupReduceToShufflePatterns`
/// that only matches `gpu.subgroup_reduce` ops with a `cluster_size`.
void populateGpuLowerClusteredSubgroupReduceToShufflePatterns(
    RewritePatternSet &patterns, unsigned subgroupSize,
    unsigned shuffleBitwidth = 32, PatternBenefit benefit = 1);

/// Collect all patterns to rewrite ops within the GPU dialect.
inline void populateGpuRewritePatterns(RewritePatternSet &patterns) {
  populateGpuAllReducePatterns(patterns);
  populateGpuGlobalIdPatterns(patterns);
  populateGpuShufflePatterns(patterns);
}

namespace gpu {
/// Searches for all GPU modules in `op` and transforms them into GPU binary
/// operations. The resulting `gpu.binary` has `handler` as its offloading
/// handler attribute.
LogicalResult transformGpuModulesToBinaries(
    Operation *op, OffloadingLLVMTranslationAttrInterface handler = nullptr,
    const gpu::TargetOptions &options = {});
} // namespace gpu

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Collect a set of patterns to decompose memrefs ops.
void populateGpuDecomposeMemrefsPatterns(RewritePatternSet &patterns);

/// Pass decomposes memref ops inside `gpu.launch` body.
std::unique_ptr<Pass> createGpuDecomposeMemrefsPass();

/// Erase barriers that do not enforce conflicting memory side effects.
void populateGpuEliminateBarriersPatterns(RewritePatternSet &patterns);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_GPU_TRANSFORMS_PASSES_H_
