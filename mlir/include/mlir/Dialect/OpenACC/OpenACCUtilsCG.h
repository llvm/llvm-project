//===- OpenACCUtilsCG.h - OpenACC Code Generation Utilities -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines utility functions for OpenACC code generation, including
// data layout and type-related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_
#define MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_

#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {
namespace acc {

class OpenACCSupport;

/// Get the data layout for an operation.
///
/// Attempts to get the data layout from the operation or its parent module.
/// If `allowDefault` is true (default), a default data layout may be
/// constructed when no explicit data layout spec is found.
///
/// \param op The operation to get the data layout for.
/// \param allowDefault If true, allow returning a default data layout.
/// \return The data layout if available, std::nullopt otherwise.
std::optional<DataLayout> getDataLayout(Operation *op,
                                        bool allowDefault = true);

/// Build an `acc.compute_region` operation by cloning a source region.
///
/// Creates a new `acc.compute_region` with the given launch arguments and
/// origin string, then clones the operations from `regionToClone` into its
/// body. Launch operands should be `acc.par_width` results (`index`); the
/// region entry block gets matching `index` block arguments first, then
/// arguments for each `ins` operand. Multi-block regions are wrapped with
/// `scf.execute_region`.
///
/// The `mapping` is used and updated during cloning, allowing callers to
/// track value correspondences. Optional `output`, `kernelFuncName`,
/// `kernelModuleName`, and `stream` arguments are forwarded to the op.
///
/// When `inputArgsToMap` is non-empty, it is used as the key set for the
/// clone mapping (instead of `inputArgs`). Use this when cloning a region
/// that references one set of values (e.g. the source function's args) while
/// the op's operands are another set (e.g. the current block's args).
/// `inputArgsToMap` must have the same size as `inputArgs` when provided.
ComputeRegionOp buildComputeRegion(Location loc, ValueRange launchArgs,
                                   ValueRange inputArgs, llvm::StringRef origin,
                                   Region &regionToClone,
                                   RewriterBase &rewriter, IRMapping &mapping,
                                   ValueRange output = {},
                                   FlatSymbolRefAttr kernelFuncName = {},
                                   FlatSymbolRefAttr kernelModuleName = {},
                                   Value stream = {},
                                   ValueRange inputArgsToMap = {});

/// Insert \p parDim into \p parDims while preserving dimension ordering. If the
/// dimension is already present, this is a no-op.
void insertParDim(llvm::SmallVector<GPUParallelDimAttr> &parDims,
                  GPUParallelDimAttr parDim);

/// Remove \p parDim from \p parDims if present.
void removeParDim(llvm::SmallVector<GPUParallelDimAttr> &parDims,
                  GPUParallelDimAttr parDim);

/// Obtain the parallel dimensions carried by \p op, if any.
GPUParallelDimsAttr getParDimsAttr(Operation *op);

/// Return whether \p op carries parallel dimensions.
bool hasParDimsAttr(Operation *op);

/// Return whether \p op carries sequential parallel dimensions.
bool hasSeqParDims(Operation *op);

/// Set parallel dimensions on \p op.
void setParDimsAttr(Operation *op, GPUParallelDimsAttr attr);

/// Update parallel dimensions on \p op.
void updateParDimsAttr(Operation *op, GPUParallelDimsAttr attr);

/// Copy parallel dimensions from \p from to \p to.
void copyParDimsAttr(Operation *from, Operation *to);

/// Obtain the active parallel dimensions carried by \p op, if any.
ActiveParDimsAttr getActiveParDimsAttr(Operation *op);

/// Return whether \p op carries active parallel dimensions.
bool hasActiveParDimsAttr(Operation *op);

/// Set active parallel dimensions on \p op.
void setActiveParDimsAttr(Operation *op, ActiveParDimsAttr attr);

/// Set active parallel dimensions on \p op from a dimension list.
void setActiveParDimsAttr(Operation *op, ArrayRef<GPUParallelDimAttr> dims);

/// Return whether \p op is marked with the `acc.gpu_block_redundant` attribute,
/// i.e. it executes redundantly across all thread blocks. Such an op must not
/// be assigned block/grid-level work-sharing; only thread-level parallelism may
/// apply, and its enclosing block dimensions are treated as active (not
/// predicated).
bool hasGPUBlockRedundantAttr(Operation *op);

/// Mark \p op with the `acc.gpu_block_redundant` attribute.
void setGPUBlockRedundantAttr(Operation *op);

/// Create a gang dim 1 GPUParallelDimsAttr based on the mapping policy.
inline GPUParallelDimsAttr
getGangDim1ParDimsAttr(MLIRContext *ctx, ACCToGPUMappingPolicy &policy) {
  return GPUParallelDimsAttr::get(
      ctx, {policy.gangDim(ctx, acc::ParLevel::gang_dim1)});
}

/// Create a sequential GPUParallelDimsAttr based on the mapping policy.
inline GPUParallelDimsAttr getSeqParDimsAttr(MLIRContext *ctx,
                                             ACCToGPUMappingPolicy &policy) {
  return GPUParallelDimsAttr::get(ctx, {policy.seqDim(ctx)});
}

/// Tracks aligned byte consumption against a configurable shared memory cap.
class SharedMemoryBudget {
public:
  /// Default allocation alignment (bytes).
  static constexpr int64_t kDefaultAlignmentBytes = 16;

  SharedMemoryBudget(int64_t maxTotalBytes, int64_t initialBytesUsed = 0)
      : bytesUsed_(initialBytesUsed), maxTotalBytes_(maxTotalBytes) {}

  /// Reserve \p bytes, rounding the current offset up to \p alignment first.
  /// Returns false without mutating state if the reservation would exceed the
  /// cap. \p alignment must be a power of two.
  bool tryAllocate(int64_t bytes, int64_t alignment = kDefaultAlignmentBytes);
  int64_t bytesUsed() const { return bytesUsed_; }
  int64_t maxTotalBytes() const { return maxTotalBytes_; }
  void setMaxTotalBytes(int64_t maxTotalBytes) {
    maxTotalBytes_ = maxTotalBytes;
  }

  /// Round \p offset up to the next multiple of \p alignment, which must be a
  /// power of two.
  static int64_t alignOffset(int64_t offset,
                             int64_t alignment = kDefaultAlignmentBytes);

private:
  int64_t bytesUsed_ = 0;
  int64_t maxTotalBytes_ = 0;
};

/// Sum aligned static_upper_bound_bytes for all acc.gpu_shared_memory in \p
/// region.
int64_t sumExistingSharedMemoryBytes(Region &region);

/// Resolve the acc.privatize operation associated with a private local.
PrivatizeOp getPrivatizeOp(PrivateLocalOp privateLocal,
                           ComputeRegionOp computeRegion);

/// Returns the ranked MemRef type used to allocate privatized storage.
///
/// \p baseTy is the `baseTy` parameter of `acc.private_type` (the privatized
/// variable's type).
MemRefType getPrivateBaseMemRefType(Type baseTy, ModuleOp module);

/// Collect parallel dimensions that govern privatization of \p privateLocal.
SmallVector<GPUParallelDimAttr>
collectPrivateLocalParDims(PrivateLocalOp privateLocal,
                           ComputeRegionOp computeRegion);

/// True when \p privateLocal may be placed in shared memory.
FailureOr<bool> isPrivateLocalSharedMemoryCandidate(
    PrivateLocalOp privateLocal, ComputeRegionOp computeRegion, ModuleOp module,
    const ACCToGPUMappingPolicy &policy, OpenACCSupport *support = nullptr);

/// Upper-bound byte size for a shared-memory private_local candidate, or
/// std::nullopt when not eligible or not statically computable.
std::optional<int64_t> getPrivateLocalSharedMemoryUpperBoundBytes(
    PrivateLocalOp privateLocal, ComputeRegionOp computeRegion, ModuleOp module,
    const ACCToGPUMappingPolicy &policy, OpenACCSupport *support = nullptr);

/// Returns true when `mapEntryOp` carries an attach point (`varPtrPtr`).
bool hasAttachPoint(Operation *mapEntryOp);

/// Returns descriptor kind from `acc.map_info`, or `none` for other ops.
DataDescKind getDataDescKind(Operation *mapEntryOp);

/// Returns descriptor value from `acc.map_info`. When `desc` is omitted and
/// `descKind` is not `none`, returns `var` (the mapped object is the
/// descriptor). Returns null for other ops.
Value getDesc(Operation *mapEntryOp);

/// Returns element size in bytes from `acc.map_info`, if present.
std::optional<int64_t> getMapElementSize(Operation *mapEntryOp);

/// Returns the optional `size` operand from `acc.map_info`, or null.
Value getMapSize(Operation *mapEntryOp);

/// Returns offload map-type flags from `acc.map_info`, if present.
std::optional<MapFlags> getMapFlags(Operation *mapEntryOp);

/// Compute the private and parallel-level map flags for privatized storage.
/// Storage that names no parallel dimension is private without being
/// replicated per level, so only the private flag is set.
MapFlags computePrivatizeMapFlags(PrivatizeOp privatizeOp,
                                  const ACCToGPUMappingPolicy &policy);

/// Fold enter (+ paired exit) data-clause semantics into offload map flags.
/// `ptrAndObj` is supplied by the caller from type-specific attach discovery.
MapFlags computeDataClauseMapFlags(Operation *entryOp, bool ptrAndObj);

/// True when another data clause of the same construct maps the same variable
/// with a copy-back and no copy-in. One entry operation is emitted per clause,
/// so a construct such as `create(a) copyout(a)` or `copyin(a) copyout(a)` maps
/// `a` twice. The runtime keeps one mapping per variable and acts on it once,
/// which makes the copy-back depend on whichever entry it processes: the entry
/// that has no copy-back of its own must therefore carry `from` as well.
bool hasCopyOutSibling(Operation *entryOp);

/// Returns the data exit operations paired with the data entry result
/// \p entryResult, which take it as their `accVar`.
SmallVector<Operation *> getPairedDataExitOps(Value entryResult);

/// Compute total mapped byte size for `acc.map_info`.
/// Returns 0 when bounds or a non-`none` descriptor kind carry size, the
/// mappable size when statically known, and -1 when the size cannot be
/// determined statically. \p support sizes the members of aggregate types that
/// belong to a dialect, such as a tuple holding dialect-specific references.
int64_t computeMapInfoSizeBytes(Value var, Type varType, DataDescKind descKind,
                                ValueRange bounds, const DataLayout &dataLayout,
                                OpenACCSupport *support = nullptr);

/// Record known extents of the source array on bounds that may describe a
/// section. Dynamic / unknown extents are left unset.
void populateSourceExtents(ValueRange bounds, ArrayRef<int64_t> shape,
                           OpBuilder &builder);

} // namespace acc
} // namespace mlir

#endif // MLIR_DIALECT_OPENACC_OPENACCUTILSCG_H_
