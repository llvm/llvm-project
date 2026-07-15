//===- ACCCGToGPU.cpp - Lower acc.compute_region to gpu.launch ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers `acc.compute_region` to the GPU dialect. For host-side
// kernels it wraps the region in `gpu.launch`; for specialized acc routines
// already inside a `gpu.func`, the body is lowered in place without emitting
// a launch.
//
// Overview:
// ---------
// `acc.compute_region` is the compute-body representation produced after
// OpenACC compute constructs are decomposed and parallelism has been assigned.
// This pass is the final ACC-to-GPU lowering step for that body: it converts
// nested `scf.parallel` / `scf.for` loops marked with `acc.par_dims` into GPU
// block and thread parallelism, materializes privatization and reductions for
// the device, inserts synchronization where shared state is observed across
// threads, and erases the ACC scaffolding (`acc.compute_region`,
// `acc.par_width`).
//
// Transformations:
// ----------------
// 1. Launch creation: outside a `gpu.func`, each `acc.compute_region` becomes a
//    `gpu.launch` whose grid and block sizes come from `acc.par_width` launch
//    operands (defaulting to 1). Kernel/module name attributes are preserved.
//    Inside a `gpu.func` (specialized acc routine), no launch is emitted.
//
// 2. Parallel loops: `scf.parallel` with a single `acc.par_dims` entry is
//    mapped to the corresponding GPU dimension (`block_*` or `thread_*`).
//    Sequential dimensions remain as `scf.parallel`/`scf.for` loops in the
//    generated kernel body.
//
// 3. Privatization: `acc.privatize` / `acc.private_local` storage is
//    materialized as one of: a per-thread `memref.alloca` (thread-private
//    arrays within the stack budget), an `acc.gpu_shared_memory` buffer
//    (gang-/worker-private arrays that fit the shared-memory budget), or a
//    `memref.alloc` whose pointer is broadcast to the block through a small
//    shared-memory slot (the data lives in global memory; shared memory only
//    holds the broadcast pointer).
//
// 4. Predication: `acc.predicate_region` becomes `scf.if` guarded by active
//    thread/block indices derived from `acc.par_dims` and launch dimensions.
//
// 5. Reductions: `acc.reduction_*` ops are lowered to GPU reduction and
//    synchronization primitives according to each reduction's parallel
//    dimensions and accumulator storage class.
//
// Example:
// --------
// Before:
//   %c128 = arith.constant 128 : index
//   %tx = acc.par_width %c128 {par_dim = #acc.par_dim<thread_x>}
//   acc.compute_region launch(%arg0 = %tx) {
//     %c0 = arith.constant 0 : index
//     %c1 = arith.constant 1 : index
//     scf.parallel (%iv) = (%c0) to (%c128) step (%c1) {
//       ...
//       scf.reduce
//     } {acc.par_dims = #acc<par_dims[thread_x]>}
//     acc.yield
//   } {origin = "acc.parallel"}
//
// After:
//   gpu.launch blocks(%bidx, %bidy, %bidz) in (%gdimx = %c1, ...)
//                threads(%tidx, %tidy, %tidz) in (%bdimx = %c128, ...) {
//     ...
//   }
//
// Requirements:
// -------------
// - Must run on a GPU device type (`device-type` option); host and multicore
//   targets are rejected.
// - Input must already be in the `acc.compute_region` form: nested SCF loops
//   carry `acc.par_dims`, privatization is expressed via `acc.privatize` /
//   `acc.private_local`, and reductions use the `acc.reduction_*` ops.
// - Each `scf.parallel` processed by this pass is expected to have exactly
//   one parallel dimension and one induction variable.
// - For acc routines, the `acc.compute_region` must live inside a `gpu.func`
//   in the GPU module.
// - Uses `acc::OpenACCSupport` for NYI reporting and compiler remarks.
// - Pass options: `max-workgroup-shared-memory`, `max-thread-private-stack`,
//   and `subgroup-size` (used for reductions and block-dimension alignment).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenACC/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils/GPUUtils.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/Analysis/OpenACCSupport.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/OpenACC/OpenACCParMapping.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsCG.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsGPU.h"
#include "mlir/Dialect/OpenACC/OpenACCUtilsReduction.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <optional>
#include <utility>

namespace mlir {
namespace acc {
#define GEN_PASS_DEF_ACCCGTOGPU
#include "mlir/Dialect/OpenACC/Transforms/Passes.h.inc"
} // namespace acc
} // namespace mlir

#define DEBUG_TYPE "acc-cg-to-gpu"

namespace {
using namespace mlir;
using namespace mlir::acc;

enum class PrivateMemScope { Thread, Worker, Gang, None };

/// Device label used in compiler remarks (e.g. "NVIDIA GPU").
static std::string getDeviceRemarkQualifier(DeviceType deviceType) {
  switch (deviceType) {
  case DeviceType::None:
  case DeviceType::Star:
  case DeviceType::Default:
    return "GPU";
  default: {
    std::string name;
    llvm::StringRef deviceName = stringifyDeviceType(deviceType);
    name.reserve(deviceName.size());
    for (char c : deviceName)
      name.push_back(llvm::toUpper(c));
    return name + " GPU";
  }
  }
}

/// True when \p op is inside a specialized acc routine function.
static bool isInsideACCSpecializedRoutine(Operation *op) {
  FunctionOpInterface funcOp = op->getParentOfType<FunctionOpInterface>();
  return funcOp && acc::isSpecializedAccRoutine(funcOp);
}

/// Maps an acc.routine's parallelism clauses to a GPU parallel dimension.
static GPUParallelDimAttr
getAccRoutineParDim(RoutineOp routineOp, MLIRContext *ctx,
                    const ACCToGPUMappingPolicy &policy) {
  if (routineOp.getGangDimValue() ||
      routineOp.getGangDimValue(DeviceType::Nvidia)) {
    int64_t gangDimValue = routineOp.getGangDimValue(DeviceType::Nvidia)
                               ? *routineOp.getGangDimValue(DeviceType::Nvidia)
                               : *routineOp.getGangDimValue();
    ParLevel gangLevel = getGangParLevel(gangDimValue);
    return policy.gangDim(ctx, gangLevel);
  }
  if (routineOp.hasGang() || routineOp.hasGang(DeviceType::Nvidia))
    return policy.gangDim(ctx, ParLevel::gang_dim1);
  if (routineOp.hasWorker() || routineOp.hasWorker(DeviceType::Nvidia))
    return policy.workerDim(ctx);
  if (routineOp.hasVector() || routineOp.hasVector(DeviceType::Nvidia))
    return policy.vectorDim(ctx);
  return policy.seqDim(ctx);
}

/// Looks up the acc.routine symbol associated with \p funcOp.
static RoutineOp getRoutineOpForAccRoutineFunction(FunctionOpInterface funcOp,
                                                   const SymbolTable &symTab) {
  if (isSpecializedAccRoutine(funcOp)) {
    SpecializedRoutineAttr attr = funcOp->getAttrOfType<SpecializedRoutineAttr>(
        getSpecializedRoutineAttrName());
    return symTab.lookup<RoutineOp>(attr.getRoutine().getLeafReference());
  }
  RoutineInfoAttr routineInfo =
      funcOp->getAttrOfType<RoutineInfoAttr>(getRoutineInfoAttrName());
  if (!routineInfo || routineInfo.getAccRoutines().empty())
    return nullptr;
  return symTab.lookup<RoutineOp>(
      routineInfo.getAccRoutines().front().getLeafReference());
}

/// Returns the parallelism level of a specialized acc routine function.
static GPUParallelDimAttr
getSpecializedRoutineDim(FunctionOpInterface funcOp,
                         const ACCToGPUMappingPolicy &policy) {
  SpecializedRoutineAttr specAttr =
      funcOp->getAttrOfType<SpecializedRoutineAttr>(
          getSpecializedRoutineAttrName());
  assert(specAttr && "expected specialized routine attribute");
  return policy.map(funcOp->getContext(), specAttr.getLevel().getValue());
}

/// Returns the parallelism dimension of a callee acc routine, if any.
static GPUParallelDimAttr
getAccRoutineCallParDim(CallOpInterface callOp,
                        const ACCToGPUMappingPolicy &policy) {
  std::optional<CallInterfaceCallable> callee = callOp.getCallableForCallee();
  if (!callee)
    return nullptr;
  SymbolRefAttr calleeSymbolRef = dyn_cast<SymbolRefAttr>(*callee);
  if (!calleeSymbolRef)
    return nullptr;
  ModuleOp moduleOp = callOp->getParentOfType<ModuleOp>();
  if (!moduleOp)
    return nullptr;

  SymbolTable symTab(moduleOp);
  FunctionOpInterface funcOp =
      symTab.lookup<FunctionOpInterface>(calleeSymbolRef.getLeafReference());
  if (!funcOp)
    return nullptr;

  if (isSpecializedAccRoutine(funcOp))
    return getSpecializedRoutineDim(funcOp, policy);
  if (RoutineOp routineOp = getRoutineOpForAccRoutineFunction(funcOp, symTab))
    return getAccRoutineParDim(routineOp, funcOp.getContext(), policy);
  return nullptr;
}

/// Collects parallel dimensions from enclosing loops and the compute region.
static SmallVector<GPUParallelDimAttr> getAncestorParDims(Operation *op) {
  SmallVector<GPUParallelDimAttr> parDimsArray;
  ComputeRegionOp computeRegion = op->getParentOfType<ComputeRegionOp>();
  assert(computeRegion && "missing enclosing acc.compute_region");
  scf::ParallelOp parentLoop = op->getParentOfType<scf::ParallelOp>();
  while (parentLoop) {
    if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(parentLoop))
      for (GPUParallelDimAttr parDim : parDimsAttr.getArray())
        insertParDim(parDimsArray, parDim);
    // A block-redundant loop executes on every block, so its enclosing block
    // dimensions are active ancestors that must not be predicated away.
    if (hasGPUBlockRedundantAttr(parentLoop))
      for (GPUParallelDimAttr parDim : computeRegion.getLaunchParDims())
        insertParDim(parDimsArray, parDim);
    parentLoop = parentLoop->getParentOfType<scf::ParallelOp>();
  }

  if (GPUParallelDimsAttr parDimsAttr = getParDimsAttr(computeRegion))
    for (GPUParallelDimAttr parDim : parDimsAttr.getArray())
      insertParDim(parDimsArray, parDim);
  return parDimsArray;
}

/// Strips index casts to reach the underlying defining value.
static Value stripIndexCastsFromValue(Value x) {
  Operation *op = x.getDefiningOp();
  if (!op)
    return x;
  while (arith::IndexCastOp castOp = dyn_cast<arith::IndexCastOp>(op)) {
    op = castOp->getOperand(0).getDefiningOp();
    if (!op)
      return x;
  }
  return op->getResult(0);
}

/// Extracts a compile-time integer constant from \p x, when known.
static FailureOr<int64_t> extractIntConst(Value x,
                                          bool stripIndexCasts = false) {
  if (stripIndexCasts)
    x = stripIndexCastsFromValue(x);
  Operation *op = x.getDefiningOp();
  if (op) {
    if (arith::ConstantIntOp constOp = dyn_cast<arith::ConstantIntOp>(op)) {
      assert(constOp.getType().getIntOrFloatBitWidth() <= 64);
      return constOp.value();
    }
    if (arith::ConstantIndexOp constOp = dyn_cast<arith::ConstantIndexOp>(op))
      return constOp.value();
  }
  return failure();
}

/// True when \p x is a constant equal to \p y (modulo index casts).
static bool sameEffectiveValue(Value x, int64_t y) {
  x = stripIndexCastsFromValue(x);
  FailureOr<int64_t> conX = extractIntConst(x);
  if (failed(conX))
    return false;
  return *conX == y;
}

/// Continues tracking a memref through view-like and partial-access ops.
static bool getPassThroughResults(Operation *userOp, Value trackedOperand,
                                  SmallVectorImpl<Value> &passThroughResults) {
  if (ViewLikeOpInterface viewLikeOp = dyn_cast<ViewLikeOpInterface>(userOp)) {
    if (viewLikeOp.getViewSource() == trackedOperand) {
      passThroughResults.push_back(viewLikeOp.getViewDest());
      return true;
    }
    return false;
  }

  // Partial-entity accesses (e.g. array element or field access) forward the
  // base entity through to their results, so treat them as pass-through when
  // the base entity is the value being tracked.
  if (acc::PartialEntityAccessOpInterface partialAccess =
          dyn_cast<acc::PartialEntityAccessOpInterface>(userOp)) {
    if (partialAccess.getBaseEntity() == trackedOperand) {
      passThroughResults.append(userOp->result_begin(), userOp->result_end());
      return true;
    }
    return false;
  }
  return false;
}

/// Skips memref view/cast chains to reach the underlying buffer.
static Value unwrapMemRefConversion(Value v) {
  while (Operation *op = v.getDefiningOp()) {
    if (ViewLikeOpInterface viewLike = dyn_cast<ViewLikeOpInterface>(op)) {
      if (isa<MemRefType>(viewLike.getViewSource().getType()) ||
          isa<MemRefType>(viewLike.getViewDest().getType())) {
        v = viewLike.getViewSource();
        continue;
      }
    }
    break;
  }
  return v;
}

/// Casts between pointer-like private types when lowering requires it.
static Value castPointerLikeTypeIfNeeded(OpBuilder &builder, Location loc,
                                         Value value, Type resultType) {
  if (value.getType() == resultType)
    return value;
  if (PointerLikeType ptrLike = dyn_cast<PointerLikeType>(value.getType())) {
    if (Value casted = ptrLike.genCast(builder, loc, value, resultType))
      return casted;
  }
  if (PointerLikeType ptrLike = dyn_cast<PointerLikeType>(resultType)) {
    if (Value casted = ptrLike.genCast(builder, loc, value, resultType))
      return casted;
  }
  emitError(loc) << "unsupported pointer-like type cast from "
                 << value.getType() << " to " << resultType;
  return value;
}

/// Returns the sole user of \p v, or null if it has zero or multiple uses.
static Operation *getOnlyUser(Value v) {
  if (!v.hasOneUse())
    return nullptr;
  return *v.user_begin();
}

/// True when \p privatize is privatized at thread_x parallelism.
static bool isThreadXPrivatize(PrivatizeOp privatize) {
  if (GPUParallelDimsAttr parDimsAttr = privatize.getParDimsAttr())
    return llvm::any_of(parDimsAttr.getArray(),
                        [](GPUParallelDimAttr d) { return d.isThreadX(); });
  return false;
}

/// Emits a workgroup-wide GPU barrier.
static void emitGPUBarrierWorkgroup(OpBuilder &builder, Location loc) {
  gpu::BarrierOp::create(builder, loc);
}

/// Emits a subgroup-scoped GPU barrier.
static void emitGPUBarrierSubgroup(OpBuilder &builder, Location loc) {
  gpu::BarrierOp::create(builder, loc, /*address_spaces=*/ArrayAttr{},
                         /*named_barrier=*/Value{},
                         gpu::BarrierScope::Subgroup);
}

/// Lowers a single `acc.compute_region` to GPU dialect IR.
class ACCCGToGPULowering {
public:
  explicit ACCCGToGPULowering(acc::ComputeRegionOp computeRegion,
                              RewriterBase &rewriter,
                              acc::OpenACCSupport &accSupport,
                              const ACCCGToGPUOptions &options)
      : rewriter(rewriter), computeRegion(computeRegion),
        accSupport(accSupport), options(options),
        sharedMemBudget(
            options.maxWorkgroupSharedMemory,
            sumExistingSharedMemoryBytes(computeRegion.getRegion())) {}

  /// Main entry point: emit launch (if needed) and lower the region body.
  LogicalResult rewrite();

  gpu::LaunchOp getLaunch() const { return launch; }

  bool hasFailed = false;
  bool insideAccumulateGridStride = false;
  Value reductionSharedBuf;
  // Reduction-accumulator slot (memref) -> the block-reduced value stored into
  // it; lets a block combine use the register instead of reloading.
  llvm::DenseMap<Value, Value> reductionAccumValue;
  // Combine reloads recorded before accumulates are lowered, patched up after.
  llvm::SmallVector<std::pair<Value, memref::LoadOp>> pendingCombineReloads;

private:
  /// Lower a parallel loop to the GPU dimension given by its `acc.par_dims`.
  void processParallelOp(scf::ParallelOp parallelOp);
  /// Lower a sequential loop, including any required post-loop barriers.
  template <typename LoopOp>
  void processSeqLoop(LoopOp loopOp);
  /// Lower an `acc.predicate_region` to a predicated `scf.if`.
  void processPredicateRegion(acc::PredicateRegionOp interOp);
  /// Materialize storage for an `acc.private_local`.
  void
  processPrivateLocal(acc::PrivateLocalOp privateLocal,
                      std::optional<int64_t> sharedMemCopies = std::nullopt);
  /// Lower an `acc.privatize` to device storage.
  Value processPrivatize(acc::PrivatizeOp privatize);
  /// Clone and lower an `scf.execute_region`.
  void processExecuteRegion(scf::ExecuteRegionOp op);
  /// Lower `acc.reduction_accumulate`.
  void processAccumulateOp(acc::ReductionAccumulateOp op);
  /// Lower `acc.reduction_accumulate_array`.
  void processAccumulateArrayOp(acc::ReductionAccumulateArrayOp op);
  /// Lower `acc.reduction_init`.
  void processReductionOp(acc::ReductionInitOp op);
  /// Lower `acc.reduction_combine`.
  void processReductionCombineOp(acc::ReductionCombineOp op);
  /// Lower `acc.reduction_combine_region`.
  void processCombineRegionOp(acc::ReductionCombineRegionOp op);
  /// Clone a leaf operation into the lowered region.
  void processGenericOp(Operation *op);
  /// Clone and recursively lower an operation with nested regions.
  void processGenericOpWithRegions(Operation *op);
  /// Dispatch lowering for one operation in the compute-region body.
  void processOp(Operation *op);

  /// Emit an atomic reduction update to \p memref.
  void constructAtomicAccumulation(Location loc, Value memref,
                                   ValueRange indices, Value input,
                                   arith::AtomicRMWKind kind);

  /// Map an ACC reduction operator to an atomic RMW kind.
  FailureOr<arith::AtomicRMWKind> getReductionKind(acc::ReductionOperator redOp,
                                                   Type type, Location loc);

  /// Split launch dimensions into those that execute \p op and those that do
  /// not, for predication and barrier placement.
  std::pair<SmallVector<mlir::acc::GPUParallelDimAttr>,
            SmallVector<mlir::acc::GPUParallelDimAttr>>
  computeActiveAndInactiveParDims(Operation *op, Block *block);

  /// Build a predicate that is true only on inactive parallel dimensions.
  Value
  emitPredicate(Location loc,
                SmallVector<mlir::acc::GPUParallelDimAttr> &inactiveParDims);

  /// True when \p privateLocal may be placed in shared memory; returns the
  /// number of copies needed, or nullopt if ineligible.
  std::optional<int64_t>
  isEligibleForSharedMemory(acc::PrivateLocalOp privateLocal,
                            MemRefType baseTy);

  /// Reserve \p bytes from the shared-memory budget.
  bool tryAllocateSharedMemory(int64_t bytes);

  /// Element size in bytes for \p elementType .
  int64_t getElementSizeInBytes(Location loc, Type elementType) const;

  /// True when a static privatization fits in the per-thread stack budget.
  bool canUseStackAlloca(MemRefType baseTy, Location loc,
                         int64_t maxThreadPrivateStack) const;

  /// Emit a barrier scoped to the parallel dimensions in \p parDimsAttr.
  void createBarrier(Location loc, mlir::acc::GPUParallelDimsAttr parDimsAttr);

  /// Emit a per-row (per-worker) barrier.
  /// Runtime branch on blockDim.y == 1 (workgroup-wide); compile-time choice
  /// between gpu.barrier scope<subgroup> (staticBlockDimX <= subgroupSize)
  /// and a named gpu.barrier (staticBlockDimX > subgroupSize) with tid.y+1.
  void createPerRowBarrier(Location loc);

  /// Insert barriers after a sequential loop when shared private state must be
  /// visible to later loops.
  void createBarrierAfterSeqLoop(Operation *loopOp);

  /// Flush any deferred post-loop barriers that precede \p beforeOp.
  void flushDeferredBarriersBefore(Operation *beforeOp);

  /// True when \p loopOp may write shared memory read by a later sibling loop.
  bool mayWriteSharedMemory(Operation *loopOp);

  /// Parallelism scope (thread, worker, or gang) of a privatized variable.
  PrivateMemScope getPrivateMemScope(acc::PrivatizeOp privatizeOp);

  /// Parallelism scope of the private buffer backing \p memref.
  PrivateMemScope getPrivateScopeForMemref(Value memref);

  /// `acc.privatize` that materialized the private buffer for \p memref.
  acc::PrivatizeOp getPrivatizeForMemref(Value memref);

  /// Whether a predicate region needs a barrier before stores that will be read
  /// by a later parallel loop over the same private memory.
  PrivateMemScope needsPreStoreReuseBarrier(acc::PredicateRegionOp interOp);

  /// Emit `gpu.all_reduce` for a reduction partial.
  void createGPUAllReduceOp(Location loc, Value input, Value memref,
                            arith::AtomicRMWKind kind,
                            mlir::acc::GPUParallelDimsAttr parDimsAttr,
                            ValueRange indices = {});

  /// Finish lowering a deferred `acc.reduction_accumulate`.
  void postprocessAccumulateOp(acc::ReductionAccumulateOp op);

  /// Finish lowering reductions attached to a parallel loop.
  void postprocessLoopReduction(scf::ParallelOp parLoop);

  /// Populate block/thread id and grid/block dimension maps for device
  /// routines.
  static void
  createForAllDimensions(RewriterBase &rewriter, Location loc,
                         llvm::DenseMap<gpu::Processor, Value> &ids,
                         llvm::DenseMap<gpu::Processor, Value> &dims) {
    ids[gpu::Processor::BlockX] = gpu::BlockIdOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
    ids[gpu::Processor::BlockY] = gpu::BlockIdOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::y);
    ids[gpu::Processor::BlockZ] = gpu::BlockIdOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::z);
    ids[gpu::Processor::ThreadX] = gpu::ThreadIdOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
    ids[gpu::Processor::ThreadY] = gpu::ThreadIdOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::y);
    ids[gpu::Processor::ThreadZ] = gpu::ThreadIdOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::z);
    dims[gpu::Processor::BlockX] = gpu::GridDimOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
    dims[gpu::Processor::BlockY] = gpu::GridDimOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::y);
    dims[gpu::Processor::BlockZ] = gpu::GridDimOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::z);
    dims[gpu::Processor::ThreadX] = gpu::BlockDimOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
    dims[gpu::Processor::ThreadY] = gpu::BlockDimOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::y);
    dims[gpu::Processor::ThreadZ] = gpu::BlockDimOp::create(
        rewriter, loc, rewriter.getIndexType(), gpu::Dimension::z);
  }

  /// Return the compute-region block argument for \p outside, adding an `ins`
  /// operand when needed.
  BlockArgument getOrAppendInsBlockArg(Value outside) {
    if (std::optional<BlockArgument> blockArg =
            computeRegion.getBlockArg(outside)) {
      return *blockArg;
    }
    return computeRegion.appendInputArg(outside);
  }

  /// Wire dynamic privatization extents into the compute region as `ins` args.
  void preparePrivatizeExtentInsOperands() {
    computeRegion.walk([&](acc::PrivateLocalOp privateLocal) {
      acc::PrivatizeOp privatizeOp =
          getPrivatizeOp(privateLocal, computeRegion);
      if (privatizeOp->getParentOfType<acc::ComputeRegionOp>() == computeRegion)
        return;
      for (Value extent : privatizeOp.getDynamicSizes())
        getOrAppendInsBlockArg(extent);
    });
  }

  /// Resolve dynamic size operands for a privatized array.
  SmallVector<Value>
  resolvePrivateLocalDynamicExtents(acc::PrivateLocalOp privateLocal) {
    acc::PrivatizeOp privatizeOp = getPrivatizeOp(privateLocal, computeRegion);
    SmallVector<Value> extents;
    for (Value extent : privatizeOp.getDynamicSizes()) {
      if (std::optional<BlockArgument> blockArg =
              computeRegion.getBlockArg(extent)) {
        extents.push_back(mapping.lookupOrDefault(*blockArg));
        continue;
      }
      extents.push_back(mapping.lookupOrDefault(extent));
    }
    return extents;
  }

  RewriterBase &rewriter;
  acc::ComputeRegionOp computeRegion;

  acc::OpenACCSupport &accSupport;
  const ACCCGToGPUOptions &options;
  gpu::LaunchOp launch;
  IRMapping mapping;
  llvm::SmallVector<scf::ParallelOp> loopReductions;
  llvm::DenseMap<gpu::Processor, Value> threadIdMap;
  llvm::DenseMap<gpu::Processor, Value> dimensionMap;
  // True if ThreadY reduction exists, which triggers subgroup alignment
  bool hasThreadYReduction = false;
  // True if any ThreadX routine call exists in the kernel
  bool hasThreadLevelRoutineCall = false;
  // True when a per-row ThreadY barrier is emitted
  bool hasThreadYBarrier = false;

  // Reusable privatize broadcast slots per type; disabled for kernels.
  llvm::DenseMap<Type, Value> privatizeBroadcastCache;

  int64_t staticBlockDimX = 1024;
  acc::DefaultACCToGPUMappingPolicy defaultPolicy;
  SharedMemoryBudget sharedMemBudget;
  SmallVector<std::string> sharedMemPrivateVarNames;
  llvm::SmallVector<Operation *, 4> deferredBarrierSeqLoops;

  Value getThreadId(Location loc, gpu::Dimension dim) {
    return gpu::ThreadIdOp::create(rewriter, loc, rewriter.getIndexType(), dim);
  }

  Value getBlockDim(Location loc, gpu::Dimension dim) {
    return gpu::BlockDimOp::create(rewriter, loc, rewriter.getIndexType(), dim);
  }

  /// Thread id for \p proc, from the launch op or the routine context map.
  Value getGPUThreadIdFor(gpu::Processor proc) {
    return getGPUThreadId(proc, getLaunch(), threadIdMap);
  }

  /// Grid/block dimension for \p proc, from the launch op or routine map.
  Value getGPUSizeFor(gpu::Processor proc) {
    return getGPUSize(proc, getLaunch(), dimensionMap);
  }
};

int64_t ACCCGToGPULowering::getElementSizeInBytes(Location loc,
                                                  Type elementType) const {
  ModuleOp module = computeRegion->getParentOfType<ModuleOp>();
  if (std::optional<acc::TypeSizeAndAlignment> sizeAndAlignment =
          accSupport.getTypeSizeAndAlignment(elementType, module)) {
    return sizeAndAlignment->first.getFixedValue();
  }
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "element size computation for unsupported type: " << elementType;
  (void)accSupport.emitNYI(loc, os.str());
  return 0;
}

bool ACCCGToGPULowering::canUseStackAlloca(
    MemRefType baseTy, Location loc, int64_t maxThreadPrivateStack) const {
  for (int64_t dim : baseTy.getShape())
    if (dim == ShapedType::kDynamic)
      return false;
  int64_t elementSize = getElementSizeInBytes(loc, baseTy.getElementType());
  int64_t numElements = 1;
  for (int64_t dim : baseTy.getShape()) {
    if (numElements > maxThreadPrivateStack / std::max<int64_t>(dim, 1))
      return false;
    numElements *= dim;
  }
  return elementSize * numElements < maxThreadPrivateStack;
}

/// True if the accumulate spans a block dim or is nested in a block-mapped
/// loop, i.e. each block owns the elements it reduces across threads. A
/// thread-only accumulate with no block context grid-strides its element loop
/// onto blocks, so per-thread partials would be dropped; such reductions must
/// stay shared.
static bool reductionHasBlockContext(acc::ReductionAccumulateArrayOp accArr) {
  auto hasBlock = [](mlir::acc::GPUParallelDimsAttr parDims) {
    return parDims && llvm::any_of(parDims.getArray(),
                                   [](auto pd) { return pd.isAnyBlock(); });
  };
  if (hasBlock(accArr.getParDimsAttr()))
    return true;
  for (scf::ParallelOp loop = accArr->getParentOfType<scf::ParallelOp>(); loop;
       loop = loop->getParentOfType<scf::ParallelOp>()) {
    if (hasBlock(mlir::acc::getParDimsAttr(loop)))
      return true;
  }
  return false;
}

/// Returns the array reduction accumulate (through cast/view ops) that \p v
/// feeds if it needs per-thread storage: its par_dims include a thread dim
/// and it has block context so the cross-thread all_reduce is well defined.
static acc::ReductionAccumulateArrayOp perThreadArrayReductionAccum(Value v) {
  SmallVector<Value> worklist{v};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    for (Operation *user : cur.getUsers()) {
      if (acc::ReductionAccumulateArrayOp accArr =
              dyn_cast<acc::ReductionAccumulateArrayOp>(user)) {
        bool hasThread = false;
        for (auto pd : accArr.getParDims().getArray())
          hasThread |= pd.isAnyThread();
        if (hasThread && reductionHasBlockContext(accArr))
          return accArr;
        continue;
      }
      SmallVector<Value> through;
      if (getPassThroughResults(user, cur, through))
        worklist.append(through.begin(), through.end());
      else if (isa<ViewLikeOpInterface>(user))
        worklist.append(user->result_begin(), user->result_end());
    }
  }
  return nullptr;
}

/// Store the reduction identity to every element of a freshly allocated
/// per-thread array accumulator so all lanes start from identity (the original
/// init loop may only run on one lane).
static void initPerThreadArrayAccum(OpBuilder &b, Location loc, Value alloca,
                                    MemRefType baseTy,
                                    arith::AtomicRMWKind kind) {
  assert(baseTy.getRank() == 1 && baseTy.hasStaticShape() &&
         "per-thread array reduction accumulator must be static rank-1");
  Value ident = createIdentityValue(b, loc, baseTy.getElementType(), kind,
                                    /*useOnlyFiniteValue=*/true);
  Value lb = arith::ConstantIndexOp::create(b, loc, 0);
  Value ub = arith::ConstantIndexOp::create(b, loc, baseTy.getShape()[0]);
  Value step = arith::ConstantIndexOp::create(b, loc, 1);
  auto forOp = scf::ForOp::create(b, loc, lb, ub, step);
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(forOp.getBody()->getTerminator());
  memref::StoreOp::create(b, loc, ident, alloca, forOp.getInductionVar());
}

std::optional<int64_t>
ACCCGToGPULowering::isEligibleForSharedMemory(acc::PrivateLocalOp privateLocal,
                                              MemRefType baseTy) {
  // Cross-thread array reduction accumulators must stay per-thread.
  if (perThreadArrayReductionAccum(privateLocal.getResult()))
    return std::nullopt;
  ModuleOp module = computeRegion->getParentOfType<ModuleOp>();
  FailureOr<bool> isCandidate = isPrivateLocalSharedMemoryCandidate(
      privateLocal, computeRegion, module, defaultPolicy, &accSupport);
  if (failed(isCandidate)) {
    hasFailed = true;
    return std::nullopt;
  }
  if (!isCandidate.value())
    return std::nullopt;
  std::optional<int64_t> upperBound =
      getPrivateLocalSharedMemoryUpperBoundBytes(privateLocal, computeRegion,
                                                 module, defaultPolicy);
  assert(upperBound && "candidate private_local must have an upper bound");
  int64_t elementSize =
      getElementSizeInBytes(privateLocal.getLoc(), baseTy.getElementType());
  int64_t numElements = 1;
  for (int64_t dim : baseTy.getShape())
    numElements *= dim;
  return *upperBound / (elementSize * numElements);
}

bool ACCCGToGPULowering::tryAllocateSharedMemory(int64_t bytes) {
  return sharedMemBudget.tryAllocate(bytes);
}

FailureOr<arith::AtomicRMWKind>
ACCCGToGPULowering::getReductionKind(acc::ReductionOperator redOp, Type type,
                                     Location loc) {
  if (std::optional<arith::AtomicRMWKind> kind =
          translateACCReductionOperator(redOp, type))
    return *kind;

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "reduction operator (" << redOp << ") for type " << type;
  (void)accSupport.emitNYI(loc, os.str());
  return failure();
}

LogicalResult ACCCGToGPULowering::rewrite() {

  // Pre-compute if thread-level reductions exist. ThreadY reduction generates
  // shuffles which require subgroup alignment (blockDim.x = subgroupSize),
  // meaning ThreadX lanes exist even without explicit ThreadX parallelism.
  computeRegion->walk([&](acc::ReductionAccumulateOp op) -> WalkResult {
    for (auto parDim : op.getParDimsAttr().getArray()) {
      if (parDim.isThreadY()) {
        hasThreadYReduction = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  // Pre-compute if any thread-level (vector or worker) routine call exists.
  // Such routines partition work across ThreadX/ThreadY and emit workgroup-wide
  // barriers internally (e.g. for shared memory alloca synchronization), so all
  // workgroup threads must reach the call site for those barriers to converge.
  computeRegion->walk([&](CallOpInterface callOp) -> WalkResult {
    if (mlir::acc::GPUParallelDimAttr parDim =
            getAccRoutineCallParDim(callOp, defaultPolicy)) {
      if (parDim.isThreadX() || parDim.isThreadY()) {
        hasThreadLevelRoutineCall = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  Location loc = computeRegion->getLoc();
  Value constantOne = arith::ConstantIndexOp::create(rewriter, loc, 1);

  auto launchArgument = [&](gpu::Processor processor) -> Value {
    mlir::acc::GPUParallelDimAttr parDim = mlir::acc::GPUParallelDimAttr::get(
        computeRegion->getContext(), processor);
    std::optional<Value> maybeLaunchArg =
        computeRegion.getKnownLaunchArg(parDim);
    LLVM_DEBUG(llvm::dbgs() << "ACCCGToGPU: launch-arg: "
                            << " parDim: " << parDim << " gpu: " << processor
                            << " widthValue: "
                            << maybeLaunchArg.value_or(constantOne) << "\n");

    return getValueOrCreateCastToIndexLike(
        rewriter, loc, rewriter.getIndexType(),
        maybeLaunchArg.value_or(constantOne));
  };
  LLVM_DEBUG(llvm::dbgs() << "ACCCGToGPU: creating gpu launch op: \n");

  // acc.compute_region keeps launch argument as block argument, for rewriting
  // we now replace these with gpu.launch dimensions.
  auto mapLaunchArguments = [&](gpu::Processor processor, Value launchArg) {
    mlir::acc::GPUParallelDimAttr parDim = mlir::acc::GPUParallelDimAttr::get(
        computeRegion->getContext(), processor);
    std::optional<Value> kernelArg = computeRegion.getLaunchArg(parDim);
    if (kernelArg)
      mapping.map(computeRegion.gpuParWidth(processor), launchArg);
  };

  llvm::StringRef blockDimXName = "blockDim.x";
  llvm::StringRef blockDimYName = "blockDim.y";
  std::string deviceLabel = getDeviceRemarkQualifier(options.deviceType);

  if (!computeRegion->getParentOfType<gpu::GPUFuncOp>()) {
    Value blockDimX = launchArgument(gpu::Processor::ThreadX);
    APInt bdxVal;
    if (matchPattern(blockDimX, m_ConstantInt(&bdxVal)))
      staticBlockDimX = bdxVal.getSExtValue();
    Value blockDimY = launchArgument(gpu::Processor::ThreadY);
    Value blockDimZ = launchArgument(gpu::Processor::ThreadZ);
    Value gridDimX = launchArgument(gpu::Processor::BlockX);
    Value gridDimY = launchArgument(gpu::Processor::BlockY);
    Value gridDimZ = launchArgument(gpu::Processor::BlockZ);

    // The format of the message is:
    // Generating [serial] {deviceLabel} code with gridDim=32x1x1
    // blockDim=256x1x1
    accSupport.emitRemark(computeRegion, [&]() {
      auto getName = [&](Value val) -> std::string {
        std::string name = accSupport.getVariableName(val);
        return name.empty() ? "(*)" : name;
      };
      bool isEffectivelySerial =
          sameEffectiveValue(blockDimX, 1) &&
          sameEffectiveValue(blockDimY, 1) &&
          sameEffectiveValue(blockDimZ, 1) && sameEffectiveValue(gridDimX, 1) &&
          sameEffectiveValue(gridDimY, 1) && sameEffectiveValue(gridDimZ, 1);
      return (llvm::Twine("Generating ") +
              llvm::Twine(isEffectivelySerial ? "serial " : "") + deviceLabel +
              " code with gridDim=" + getName(gridDimX) + "x" +
              getName(gridDimY) + "x" + getName(gridDimZ) +
              " blockDim=" + getName(blockDimX) + "x" + getName(blockDimY) +
              "x" + getName(blockDimZ))
          .str();
    });

    // Check if kernel has a stream operand for async execution
    if (mlir::Value streamValue = computeRegion.getStream()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\nDEBUG: Creating async gpu.launch with stream: "
                 << streamValue << "\n");
      launch = gpu::LaunchOp::create(
          rewriter, loc, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY,
          blockDimZ,
          /*dynamicSharedMemorySize=*/mlir::Value{},
          /*asyncTokenType=*/
          mlir::gpu::AsyncTokenType::get(rewriter.getContext()));
      // Add the stream as an async dependency
      launch.getAsyncDependenciesMutable().append(streamValue);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "\nDEBUG: No stream, creating sync gpu.launch\n");
      launch = gpu::LaunchOp::create(rewriter, loc, gridDimX, gridDimY,
                                     gridDimZ, blockDimX, blockDimY, blockDimZ);
    }

    // Transfer kernel function name and module name from acc.compute_region to
    // gpu.launch if present
    if (auto kernelFuncName = computeRegion.getKernelFuncNameAttr())
      launch.setFunctionAttr(kernelFuncName);
    if (auto kernelModuleName = computeRegion.getKernelModuleNameAttr())
      launch.setModuleAttr(kernelModuleName);

    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.setInsertionPointToStart(&launch.getBody().front());
    mapLaunchArguments(gpu::Processor::BlockX,
                       gpu::GridDimOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              gpu::Dimension::x));
    mapLaunchArguments(gpu::Processor::BlockY,
                       gpu::GridDimOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              gpu::Dimension::y));
    mapLaunchArguments(gpu::Processor::BlockZ,
                       gpu::GridDimOp::create(rewriter, loc,
                                              rewriter.getIndexType(),
                                              gpu::Dimension::z));
    mapLaunchArguments(gpu::Processor::ThreadX,
                       gpu::BlockDimOp::create(rewriter, loc,
                                               rewriter.getIndexType(),
                                               gpu::Dimension::x));
    mapLaunchArguments(gpu::Processor::ThreadY,
                       gpu::BlockDimOp::create(rewriter, loc,
                                               rewriter.getIndexType(),
                                               gpu::Dimension::y));
    mapLaunchArguments(gpu::Processor::ThreadZ,
                       gpu::BlockDimOp::create(rewriter, loc,
                                               rewriter.getIndexType(),
                                               gpu::Dimension::z));
  } else {
    // Do not create gpu.launch for acc routine and map
    // to block/thread index and block/grid size instead
    // of launch arguments, using created maps.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(computeRegion->getBlock());
    createForAllDimensions(rewriter, loc, threadIdMap, dimensionMap);
    mapLaunchArguments(gpu::Processor::BlockX,
                       dimensionMap[gpu::Processor::BlockX]);
    mapLaunchArguments(gpu::Processor::BlockY,
                       dimensionMap[gpu::Processor::BlockY]);
    mapLaunchArguments(gpu::Processor::BlockZ,
                       dimensionMap[gpu::Processor::BlockZ]);
    mapLaunchArguments(gpu::Processor::ThreadX,
                       dimensionMap[gpu::Processor::ThreadX]);
    mapLaunchArguments(gpu::Processor::ThreadY,
                       dimensionMap[gpu::Processor::ThreadY]);
    mapLaunchArguments(gpu::Processor::ThreadZ,
                       dimensionMap[gpu::Processor::ThreadZ]);
  }

  // Map input arguments for compute region; we go from an IsolatedFromAbove
  // operation to gpu.launch which is not IsolatedFromAbove.
  preparePrivatizeExtentInsOperands();
  Block *body = computeRegion.getBody();
  unsigned numLaunchArgs = computeRegion.getLaunchArgs().size();
  ValueRange inputArgs = computeRegion.getInputArgs();
  for (unsigned i = numLaunchArgs; i < body->getNumArguments(); ++i)
    mapping.map(body->getArgument(i), inputArgs[i - numLaunchArgs]);

  assert(computeRegion.getRegion().hasOneBlock() &&
         "compute region only supports one block region for now");
  // process all operations inside kernel region
  for (auto &op : computeRegion.getRegion().getBlocks().front().getOperations())
    processOp(&op);

  for (auto &parLoop : loopReductions)
    postprocessLoopReduction(parLoop);

  // Replace combine reloads of a reduction slot with the block-reduced value.
  // Only when it dominates the reload; otherwise keep the reload.
  if (!pendingCombineReloads.empty() && launch) {
    DominanceInfo domInfo(launch);
    for (auto &[slot, loadOp] : pendingCombineReloads) {
      llvm::DenseMap<Value, Value>::iterator it =
          reductionAccumValue.find(slot);
      if (it == reductionAccumValue.end())
        continue;
      if (!domInfo.dominates(it->second, loadOp.getOperation()))
        continue;
      rewriter.replaceOp(loadOp, ValueRange{it->second});
    }
  }

  if (launch) {
    const int64_t subgroupSize = options.subgroupSize;
    const int64_t subgroupAlignMask = subgroupSize - 1;

    // Adjust blockDim.x to be a multiple of subgroupSize. This is required
    // because:
    // - Subgroup reductions (gpu.all_reduce) require full subgroups
    // - Per-row workgroup barriers require blockDim.x aligned to subgroupSize
    bool isShuffleEnabled = false;

    launch.walk([&](gpu::AllReduceOp allReduce) -> WalkResult {
      ArrayRef<mlir::acc::GPUParallelDimAttr> parDims =
          mlir::acc::getParDimsAttr(allReduce).getArray();
      for (auto parDim : parDims) {
        if (parDim.isThreadX() || parDim.isThreadY()) {
          // Shuffle are enabled. Need to adjust the ThreadX length.
          isShuffleEnabled = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    // Also check if called routines have ThreadY reductions
    if (!isShuffleEnabled) {
      launch.walk([&](func::CallOp callOp) -> WalkResult {
        if (gpu::GPUFuncOp callee =
                callOp->getParentOfType<ModuleOp>()
                    .lookupSymbol<gpu::GPUFuncOp>(callOp.getCallee())) {
          callee.walk([&](gpu::AllReduceOp allReduce) -> WalkResult {
            ArrayRef<mlir::acc::GPUParallelDimAttr> parDims =
                mlir::acc::getParDimsAttr(allReduce).getArray();
            for (auto parDim : parDims) {
              if (parDim.isThreadX() || parDim.isThreadY()) {
                isShuffleEnabled = true;
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
        }
        return isShuffleEnabled ? WalkResult::interrupt()
                                : WalkResult::advance();
      });
    }

    if (isShuffleEnabled || hasThreadYBarrier) {
      rewriter.setInsertionPoint(launch);

      Value curBlockDimX = launch.getBlockSizeX();
      Value curBlockDimY = launch.getBlockSizeY();

      // Emit a report on changing parallelism.
      accSupport.emitRemark(computeRegion, [&]() {
        auto getName = [&](Value val) -> std::string {
          std::string name = accSupport.getVariableName(val);
          return name.empty() ? "(*)" : name;
        };
        std::string blockDimXValStr = getName(curBlockDimX);
        std::string blockDimYValStr = getName(curBlockDimY);
        llvm::StringRef kind =
            isShuffleEnabled ? "Shuffle reduction" : "ThreadY barrier";
        return (llvm::Twine(kind) +
                " is generated while adjusting the number of threads into "
                "groups of " +
                llvm::Twine(subgroupSize) + ".\n\t" + blockDimXName + ": `" +
                blockDimXValStr + "` to `((" + blockDimXValStr + " + " +
                llvm::Twine(subgroupAlignMask) + ") / " +
                llvm::Twine(subgroupSize) + ") * " + llvm::Twine(subgroupSize) +
                "`\n" + "\t" + blockDimYName + ": `" + blockDimYValStr +
                "` to `max(1, (new-" + blockDimXName + " * " + blockDimYValStr +
                ") / new-" + blockDimXName + ")`")
            .str();
      });

      std::optional<int64_t> constBlockDimX = getConstantIntValue(curBlockDimX);
      std::optional<int64_t> constBlockDimY = getConstantIntValue(curBlockDimY);

      // Skip subgroup alignment only when the total thread count is already
      // below a subgroup (constant blockDim.x in 2..subgroupSize-1 and
      // constant blockDim.y == 1). If blockDim.y > 1 or is unknown, padding
      // blockDim.x to a subgroup is still required so subgroups don't cross
      // row boundaries for row-local shuffle/ThreadY-barrier reductions.
      bool skipAlign = false;
      if (constBlockDimX && constBlockDimY && *constBlockDimX > 1 &&
          *constBlockDimX < subgroupSize && *constBlockDimY == 1) {
        skipAlign = true;
      }

      // Update both the ThreadX length and the number of ThreadY.
      // When the original blockDim.x and blockDim.y are compile-time
      // constants, compute the adjusted dimensions as constants directly so
      // that the GpuKernelOutliningPass can set `known_block_size` on the
      // outlined gpu.func.
      Value newBlockDimX, newBlockDimY;
      if (constBlockDimX && constBlockDimY) {
        int64_t bdx = *constBlockDimX;
        int64_t bdy = *constBlockDimY;
        int64_t alignedBdx =
            ((bdx + subgroupAlignMask) / subgroupSize) * subgroupSize;
        int64_t numThreads = bdx * bdy;
        int64_t newBdy = std::max<int64_t>(1, numThreads / alignedBdx);
        newBlockDimX =
            arith::ConstantIndexOp::create(rewriter, loc, alignedBdx);
        newBlockDimY = arith::ConstantIndexOp::create(rewriter, loc, newBdy);
      } else {
        // numThreads = blockDim.x * blockDim.y
        Value numThreads =
            arith::MulIOp::create(rewriter, loc, curBlockDimX, curBlockDimY);
        // blockDim.x = ((blockDim.x + mask) / subgroupSize) * subgroupSize
        Value cstMask =
            arith::ConstantIndexOp::create(rewriter, loc, subgroupAlignMask);
        Value cstSubgroupSize =
            arith::ConstantIndexOp::create(rewriter, loc, subgroupSize);
        Value padded =
            arith::AddIOp::create(rewriter, loc, curBlockDimX, cstMask);
        Value subgroupsRequired =
            arith::DivUIOp::create(rewriter, loc, padded, cstSubgroupSize);
        newBlockDimX = arith::MulIOp::create(rewriter, loc, subgroupsRequired,
                                             cstSubgroupSize);
        // blockDim.y = max(1, numThreads / blockDim.x)
        Value quotient =
            arith::DivUIOp::create(rewriter, loc, numThreads, newBlockDimX);
        Value cst1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        newBlockDimY = arith::MaxUIOp::create(rewriter, loc, cst1, quotient);
      }

      if (!skipAlign) {
        launch.getBlockSizeXMutable().assign(newBlockDimX);
        launch.getBlockSizeYMutable().assign(newBlockDimY);
      }
    }
  }

  if (hasFailed)
    return failure();

  if (!sharedMemPrivateVarNames.empty()) {
    accSupport.emitRemark(computeRegion, [&]() {
      return (llvm::Twine("GPU shared memory used for ") +
              llvm::join(sharedMemPrivateVarNames, ","))
          .str();
    });
  }

  rewriter.eraseOp(computeRegion);
  return success();
}

/// True when this accumulate is redundant in a nested reduction chain: the
/// value is a load of the destination memref and a sibling
/// acc.reduction_combine with block par_dims has already reduced %M across
/// threads in the block.
///
///   %v = memref.load %M[]
///   acc.reduction_accumulate %v to %M ...
///   acc.reduction_combine %M into %parent ... {block par_dims}
///
/// Lowering the accumulate again would double-count. Detection is structural;
/// nested reductions into per-thread privates do not match because their
/// combines are not block-scoped.
static bool isRedundantChainAccumulate(acc::ReductionAccumulateOp op) {
  Value memref = op.getMemref();
  memref::LoadOp loadOp = op.getValue().getDefiningOp<memref::LoadOp>();
  if (!loadOp || loadOp.getMemRef() != memref)
    return false;
  for (Operation *user : memref.getUsers()) {
    acc::ReductionCombineOp combineOp = dyn_cast<acc::ReductionCombineOp>(user);
    if (!combineOp || combineOp.getDestMemref() != memref)
      continue;
    SmallVector<mlir::acc::GPUParallelDimAttr> parDims =
        getReductionCombineParDims(combineOp);
    if (llvm::any_of(parDims, [](mlir::acc::GPUParallelDimAttr d) {
          return d.isAnyBlock();
        })) {
      return true;
    }
  }
  return false;
}

std::pair<SmallVector<mlir::acc::GPUParallelDimAttr>,
          SmallVector<mlir::acc::GPUParallelDimAttr>>
ACCCGToGPULowering::computeActiveAndInactiveParDims(Operation *op,
                                                    Block *block) {
  MLIRContext *ctx = computeRegion->getContext();
  SmallVector<mlir::acc::GPUParallelDimAttr> ancestorParDims =
      getAncestorParDims(op);
  // Preserve whether there were any structural ancestor par-dims before
  // we start augmenting them based on inner uses (e.g. private_local).
  // This is needed for gang redundancy check - stores to worker-indexed
  // private_local should not disable redundant gang execution.
  bool noStructuralAncestorParDims =
      llvm::none_of(ancestorParDims, [](auto pd) { return !pd.isSeq(); });

  mlir::acc::GPUParallelDimAttr routineParDim;
  if (isInsideACCSpecializedRoutine(computeRegion)) {
    FunctionOpInterface funcOp =
        computeRegion->getParentOfType<FunctionOpInterface>();
    routineParDim = getSpecializedRoutineDim(funcOp, defaultPolicy);
    if (routineParDim.isThreadX()) {
      mlir::acc::insertParDim(ancestorParDims,
                              mlir::acc::GPUParallelDimAttr::threadYDim(ctx));
    }
    mlir::acc::insertParDim(ancestorParDims,
                            mlir::acc::GPUParallelDimAttr::blockXDim(ctx));
  }

  // acc.private_local should use the same par_dims as acc.reduction_accumulate.
  if (acc::PrivateLocalOp privateLocalOp = dyn_cast<acc::PrivateLocalOp>(op)) {
    for (Operation *user : privateLocalOp.getResult().getUsers()) {
      if (acc::ReductionAccumulateOp accumulateOp =
              dyn_cast<acc::ReductionAccumulateOp>(user)) {
        if (accumulateOp.getMemref() == privateLocalOp.getResult()) {
          for (mlir::acc::GPUParallelDimAttr parDim :
               accumulateOp.getParDims().getArray()) {
            mlir::acc::insertParDim(ancestorParDims, parDim);
          }
        }
      }
      // For decomposed complex reductions, the private_local is consumed
      // by an acc.reduction_combine{,_region} (no acc.reduction_accumulate
      // user). Mirror the par_dims so this private_local is treated as the
      // accumulator at the same parallelism level as a scalar reduction
      // would be (per-thread, not block-shared).
      if (acc::ReductionCombineOp combineOp =
              dyn_cast<acc::ReductionCombineOp>(user)) {
        if (combineOp.getSrcMemref() == privateLocalOp.getResult()) {
          for (mlir::acc::GPUParallelDimAttr parDim :
               getReductionCombineParDims(combineOp)) {
            mlir::acc::insertParDim(ancestorParDims, parDim);
          }
        }
      }
      if (auto combineRegionOp =
              dyn_cast<acc::ReductionCombineRegionOp>(user)) {
        if (combineRegionOp.getSrcVar() == privateLocalOp.getResult()) {
          for (mlir::acc::GPUParallelDimAttr parDim :
               getReductionCombineParDims(combineRegionOp)) {
            mlir::acc::insertParDim(ancestorParDims, parDim);
          }
        }
      }
    }
  }

  bool hasBlock = false;
  for (mlir::acc::GPUParallelDimAttr parDim : ancestorParDims)
    if (parDim.isAnyBlock())
      hasBlock = true;

  mlir::acc::GPUParallelDimAttr lowestParDim =
      mlir::acc::GPUParallelDimAttr::threadXDim(ctx);
  if (block) {
    block->walk([&](Operation *op) {
      // Check stores to acc.private_local - add the privatize's par_dims
      // as active dims so predication is correct for per-worker/gang memory.
      if (memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(op)) {
        if (auto privateLocalOp =
                storeOp.getMemref().getDefiningOp<acc::PrivateLocalOp>()) {
          acc::PrivatizeOp privatizeOp =
              getPrivatizeOp(privateLocalOp, computeRegion);
          if (mlir::acc::GPUParallelDimsAttr parDimsAttr =
                  privatizeOp.getParDimsAttr()) {
            for (auto parDim : parDimsAttr.getArray())
              mlir::acc::insertParDim(ancestorParDims, parDim);
          }
        }
      }
      // Consider ACC routine calls; routine calls should be predicated up to
      // one level above the parallel dimension of the callee.
      if (CallOpInterface callOp = dyn_cast<CallOpInterface>(op)) {
        if (mlir::acc::GPUParallelDimAttr parDim =
                getAccRoutineCallParDim(callOp, defaultPolicy)) {
          if (parDim.isBlockZ())
            lowestParDim = parDim;
          else
            lowestParDim = parDim.getOneHigher();
        }
      }
      // acc.reduction_combine_region should be predicated with the par_dims of
      // acc.reduction_accumulate. This is required when using combine between
      // kernel and loop in combined constructs.
      if (acc::ReductionCombineOp reductionCombineOp =
              dyn_cast<acc::ReductionCombineOp>(op)) {
        for (mlir::acc::GPUParallelDimAttr parDim :
             getReductionCombineParDims(reductionCombineOp)) {
          mlir::acc::removeParDim(ancestorParDims, parDim);
        }
      }
      if (acc::ReductionCombineRegionOp combineRegionOp =
              dyn_cast<acc::ReductionCombineRegionOp>(op)) {
        for (mlir::acc::GPUParallelDimAttr parDim :
             getReductionCombineParDims(combineRegionOp)) {
          mlir::acc::removeParDim(ancestorParDims, parDim);
        }
      }
      // An array accumulate reduces across its par_dims via gpu.all_reduce, so
      // all those threads must execute it - treat them as active (unlike the
      // scalar accumulate, which is active through its enclosing scf.parallel).
      if (acc::ReductionAccumulateArrayOp accArrayOp =
              dyn_cast<acc::ReductionAccumulateArrayOp>(op)) {
        for (mlir::acc::GPUParallelDimAttr parDim :
             accArrayOp.getParDims().getArray()) {
          mlir::acc::insertParDim(ancestorParDims, parDim);
        }
      }
      return WalkResult::advance();
    });
  }

  // Obtain launch dimensions
  SmallVector<mlir::acc::GPUParallelDimAttr> launchParDims;
  if (routineParDim) {
    for (mlir::acc::GPUParallelDimAttr parDim = routineParDim;
         parDim.getOrder() >= lowestParDim.getOrder();
         parDim = parDim.getOneLower()) {
      mlir::acc::insertParDim(launchParDims, parDim);
    }
  } else {
    launchParDims = computeRegion.getLaunchParDims();
  }

  // Compute dimensions that execute op
  SmallVector<mlir::acc::GPUParallelDimAttr> activeParDims, inactiveParDims;
  for (mlir::acc::GPUParallelDimAttr launchParDim : launchParDims) {
    if (launchParDim.getOrder() < lowestParDim.getOrder())
      break;
    if (llvm::find(ancestorParDims, launchParDim) != ancestorParDims.end() ||
        (launchParDim.isAnyBlock() &&
         (noStructuralAncestorParDims || hasBlock))) {
      activeParDims.push_back(launchParDim);
    } else {
      inactiveParDims.push_back(launchParDim);
    }
  }

  return std::pair{activeParDims, inactiveParDims};
}

Value ACCCGToGPULowering::emitPredicate(
    Location loc, SmallVector<mlir::acc::GPUParallelDimAttr> &inactiveParDims) {
  Value predicate;
  for (mlir::acc::GPUParallelDimAttr inactiveParDim : inactiveParDims) {
    Value threadId = getGPUThreadIdFor(inactiveParDim.getProcessor());
    TypedAttr zeroAttr = rewriter.getZeroAttr(threadId.getType());
    Value zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
    Value cmp = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      threadId, zero);
    if (predicate)
      predicate = arith::AndIOp::create(rewriter, loc, cmp, predicate);
    else
      predicate = cmp;
  }
  return predicate;
}

void ACCCGToGPULowering::createBarrier(
    Location loc, mlir::acc::GPUParallelDimsAttr parDimsAttr) {
  bool hasAnyBlock = false, hasThreadY = false, hasThreadX = false;
  for (auto parDim : parDimsAttr.getArray()) {
    if (parDim.isAnyBlock())
      hasAnyBlock = true;
    if (parDim.isThreadY())
      hasThreadY = true;
    if (parDim.isThreadX())
      hasThreadX = true;
  }

  if (hasAnyBlock || hasThreadY)
    emitGPUBarrierWorkgroup(rewriter, loc);
  else if (hasThreadX)
    createPerRowBarrier(loc);
}

void ACCCGToGPULowering::createPerRowBarrier(Location loc) {
  hasThreadYBarrier = true;

  if (staticBlockDimX <= options.subgroupSize) {
    emitGPUBarrierSubgroup(rewriter, loc);
    return;
  }

  if (options.deviceType != mlir::acc::DeviceType::Nvidia) {
    (void)accSupport.emitNYI(
        loc,
        "per-row barrier to support worker parallelism on non-NVIDIA device");
  }

  // Per-row barrier with fully runtime branching.
  // Three mutually exclusive paths:
  //   blockDim.y == 1    -> gpu.barrier (workgroup-wide, only one worker)
  //   blockDim.x <= subgroupSize -> gpu.barrier scope<subgroup>
  //   blockDim.x > subgroupSize  -> nvvm.barrier (tid.y + 1), blockDim.x
  //   (named)
  //
  // Per-row barriers use tid.y+1 so IDs start at 1, avoiding clash with
  // barrier 0. When blockDim.y >= 16, worker 15's ID (16) wraps to
  // physical barrier 0; this is safe because named barriers are reusable
  // resources - workgroup-wide and per-row barriers on the same physical
  // barrier execute at different program points and never overlap.
  Value blockDimX = gpu::BlockDimOp::create(
      rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
  Value blockDimY = gpu::BlockDimOp::create(
      rewriter, loc, rewriter.getIndexType(), gpu::Dimension::y);
  Value cst1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value isSingleWorker = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, blockDimY, cst1);

  auto outerIf = scf::IfOp::create(rewriter, loc, isSingleWorker,
                                   /*withElseRegion=*/true);

  // Then: blockDim.y == 1 -> workgroup-wide barrier (safe, only one worker)
  rewriter.setInsertionPointToStart(&outerIf.getThenRegion().front());
  emitGPUBarrierWorkgroup(rewriter, loc);

  // Else: blockDim.y > 1 - choose between subgroup sync and named barrier
  rewriter.setInsertionPointToStart(&outerIf.getElseRegion().front());
  Value cstSubgroupSize =
      arith::ConstantIndexOp::create(rewriter, loc, options.subgroupSize);
  Value isSubgroupSized = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ule, blockDimX, cstSubgroupSize);

  auto innerIf = scf::IfOp::create(rewriter, loc, isSubgroupSized,
                                   /*withElseRegion=*/true);

  // Then: blockDim.x <= subgroupSize -> subgroup barrier (one worker per
  // subgroup)
  rewriter.setInsertionPointToStart(&innerIf.getThenRegion().front());
  emitGPUBarrierSubgroup(rewriter, loc);

  // Else: blockDim.x > subgroupSize -> per-row named barrier with tid.y + 1.
  // The 1024-thread-per-block hardware limit with subgroup-aligned blockDim.x
  // (>= 64 here) guarantees blockDim.y <= 16, so IDs span at most 16
  // physical barriers (0-15) with no aliasing across workers.
  rewriter.setInsertionPointToStart(&innerIf.getElseRegion().front());
  Value threadYId = gpu::ThreadIdOp::create(
      rewriter, loc, rewriter.getIndexType(), gpu::Dimension::y);
  Value barrierId = arith::AddIOp::create(rewriter, loc, threadYId, cst1);
  Type i32Ty = rewriter.getI32Type();
  Value barrierId32 =
      arith::IndexCastOp::create(rewriter, loc, i32Ty, barrierId);
  Value numberOfThreads32 =
      arith::IndexCastOp::create(rewriter, loc, i32Ty, blockDimX);

  // GPU dialect named barriers do not have a means to create a custom barrier
  // id. Thus use nvvm directly.
  assert(options.deviceType == mlir::acc::DeviceType::Nvidia);
  NVVM::BarrierOp::create(rewriter, loc, barrierId32, numberOfThreads32);

  rewriter.setInsertionPointAfter(outerIf);
}

/// Whether any later sibling of \p loopOp (or a loop nested inside one) is a
/// loop, i.e. whether some subsequent loop in the same region may read what
/// \p loopOp wrote. Used to skip a barrier after the last loop, where nothing
/// reads the data afterward.
static bool hasSubsequentLoopSibling(Operation *loopOp) {
  for (Operation *next = loopOp->getNextNode(); next;
       next = next->getNextNode()) {
    if (isa<scf::ParallelOp, scf::ForOp>(next))
      return true;
    bool nested = false;
    next->walk([&](Operation *op) {
      if (isa<scf::ParallelOp, scf::ForOp>(op)) {
        nested = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (nested)
      return true;
  }
  return false;
}

/// Nearest enclosing sequential loop ancestor of \p op.
static LoopLikeOpInterface findFirstSequentialLoop(Operation *op) {
  auto isAllSequentialParDims = [](scf::ParallelOp par) -> bool {
    mlir::acc::GPUParallelDimsAttr pd = mlir::acc::getParDimsAttr(par);
    if (!pd || pd.getArray().empty())
      return false;
    return llvm::all_of(pd.getArray(), [](mlir::acc::GPUParallelDimAttr d) {
      return d.isSeq();
    });
  };

  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    // Do not need to check scf.for op's parents
    if (isa<scf::ForOp>(p))
      return cast<LoopLikeOpInterface>(p);
    if (scf::ParallelOp parOp = dyn_cast<scf::ParallelOp>(p)) {
      if (isAllSequentialParDims(parOp))
        return cast<LoopLikeOpInterface>(p);
    }
  }
  return nullptr;
}

// A sequential loop that uses gang-private shared memory needs a
// workgroup-wide barrier afterward so every thread in the block observes the
// same state. When more work still lies between the loop and the next
// thread-reconvergence point in the same block, that barrier must follow that
// work; not sit immediately after the loop.
//
// The helpers below mark loop-body closure and other reconvergence points
// where any postponed barrier must be inserted.

/// Marks the end of a loop body's iteration in the current block.
static bool isLoopBodyClosureOp(Operation *op) {
  return isa<scf::ReduceOp, scf::YieldOp, acc::YieldOp>(op);
}

/// Thread-reconvergence point where any postponed post-loop barrier for earlier
/// loops in this block must be inserted before proceeding.
static bool isDeferredBarrierFlushPoint(Operation *op) {
  if (isLoopBodyClosureOp(op))
    return true;
  // The next sequential loop may consume shared state produced by the prior
  // one.
  if (isa<scf::ForOp>(op))
    return true;
  if (scf::ParallelOp parallelOp = dyn_cast<scf::ParallelOp>(op)) {
    if (mlir::acc::hasParDimsAttr(parallelOp)) {
      if (mlir::acc::GPUParallelDimsAttr parDims =
              mlir::acc::getParDimsAttr(parallelOp)) {
        if (parDims.getArray().size() == 1 &&
            parDims.getArray().front().isSeq()) {
          return true;
        }
      }
    }
  }
  return false;
}

/// True when a loop is followed by other work in the same block before the
/// loop body closes; the post-loop barrier must wait for that reconvergence
/// point instead of being placed right after the loop.
static bool hasTrailingSideEffectSiblings(Operation *loopOp) {
  for (Operation *next = loopOp->getNextNode(); next;
       next = next->getNextNode()) {
    return !isLoopBodyClosureOp(next);
  }
  return false;
}

// Insert a barrier after a sequential loop when the surrounding parallel
// structure requires it, so that block threads observe the loop's writes to
// gang-/block-private shared memory before a later loop reads them. The barrier
// must land at an enclosing collective (block- or worker-level) point where all
// threads converge, so it cannot deadlock.
//
// The function walks the parent hierarchy of the loop to decide where (if at
// all) to place the barrier:
//   - No enclosing parallel loop (top-level worksharing loop): emit a
//     workgroup barrier when the loop writes shared memory and a later sibling
//     loop may read it.
//   - Nearest scf.parallel parent is itself sequential: a barrier *after* it
//     would be unsafe when threads have varying iteration counts. However, when
//     the loop has thread-level sub-loops collaborating on block-private memory
//     (all threads participate every iteration), walk up to the block-level
//     ancestor and insert the barrier there.
//   - Nearest scf.parallel parent is a non-sequential parallel loop: walk up to
//     the block-level (or worker/thread-y) ancestor and insert the barrier
//     there, handling gang-redundant init loops and grid-stride remainders.
void ACCCGToGPULowering::createBarrierAfterSeqLoop(Operation *loopOp) {
  scf::ParallelOp wsLoop = loopOp->getParentOfType<scf::ParallelOp>();
  if (!wsLoop) {
    // loopOp is a worksharing loop at the kernel-body top level (no enclosing
    // parallel loop). When it writes gang-private shared memory
    // that a later sibling loop reads, a workgroup barrier is needed between
    // them. The point just after a top-level loop is uniform (all threads reach
    // it), so a workgroup-wide barrier here cannot deadlock. Loops writing only
    // global/thread-private memory, or with no subsequent reader, get none.
    if (mayWriteSharedMemory(loopOp) && hasSubsequentLoopSibling(loopOp))
      emitGPUBarrierWorkgroup(rewriter, loopOp->getLoc());
    return;
  }

  bool parentIsSeq = false;
  if (mlir::acc::GPUParallelDimsAttr wsParDims =
          mlir::acc::getParDimsAttr(wsLoop)) {
    if (wsParDims.getArray().size() == 1 &&
        wsParDims.getArray().front().isSeq()) {
      parentIsSeq = true;
    }
  }

  if (parentIsSeq) {
    // loopOp is nested inside a sequential parent loop.
    // When the loop contains thread-level sub-loops,
    // multiple threads collaborate on shared (block-private) memory
    // within each iteration.  Insert a block-level barrier.
    bool hasThreadSubLoop = false;
    loopOp->walk([&](scf::ParallelOp innerPar) -> WalkResult {
      if (innerPar.getOperation() == loopOp)
        return WalkResult::advance();
      if (mlir::acc::GPUParallelDimsAttr dims =
              mlir::acc::getParDimsAttr(innerPar)) {
        for (auto d : dims.getArray()) {
          if (d.isThreadX() || d.isThreadY()) {
            hasThreadSubLoop = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (!hasThreadSubLoop)
      return;
    scf::ParallelOp threadLoop = wsLoop->getParentOfType<scf::ParallelOp>();
    if (!threadLoop)
      return;
    scf::ParallelOp blockLoop = threadLoop->getParentOfType<scf::ParallelOp>();
    if (!blockLoop)
      return;
    mlir::acc::GPUParallelDimsAttr parDimsAttr =
        mlir::acc::getParDimsAttr(blockLoop);
    if (parDimsAttr.hasOnlyBlockLevel())
      createBarrier(loopOp->getLoc(), parDimsAttr);
    return;
  }

  // Parent is a non-sequential parallel loop.  Walk up to find the block-level
  // ancestor and insert a barrier there.
  scf::ParallelOp seqLoop = wsLoop->getParentOfType<scf::ParallelOp>();
  if (!seqLoop) {
    // wsLoop is a worksharing loop directly under the compute region with no
    // gang ancestor: a gang-redundant init loop (e.g. a thread-level loop that
    // every gang runs to fill its own copy of gang-private shared memory). If
    // it writes gang-private (block-only) memory that a later sibling loop
    // reads, insert a workgroup barrier between them. The barrier sits just
    // after the top-level loop where all threads converge, so it cannot
    // deadlock.
    if (mayWriteSharedMemory(loopOp) && hasSubsequentLoopSibling(wsLoop))
      emitGPUBarrierWorkgroup(rewriter, loopOp->getLoc());
    return;
  }
  if (scf::ParallelOp outerParLoop =
          seqLoop->getParentOfType<scf::ParallelOp>()) {
    mlir::acc::GPUParallelDimsAttr parDimsAttr =
        mlir::acc::getParDimsAttr(outerParLoop);
    if (parDimsAttr.hasOnlyBlockLevel()) {
      createBarrier(loopOp->getLoc(), parDimsAttr);
    } else if (parDimsAttr.hasOnlyThreadYLevel()) {
      createPerRowBarrier(loopOp->getLoc());
    } else if (parDimsAttr && parDimsAttr.isSeq()) {
      // outerParLoop is a sequential grid-stride remainder of a partitioned
      // gang loop, not the gang. Walk past the remainder(s) to the block-level
      // gang and barrier there.
      for (Operation *gangLoop =
               outerParLoop->getParentOfType<scf::ParallelOp>();
           gangLoop; gangLoop = gangLoop->getParentOfType<scf::ParallelOp>()) {
        mlir::acc::GPUParallelDimsAttr gangDims =
            mlir::acc::getParDimsAttr(gangLoop);
        if (!gangDims)
          break;
        if (gangDims.hasOnlyBlockLevel()) {
          createBarrier(loopOp->getLoc(), gangDims);
          break;
        }
        if (!gangDims.isSeq())
          break;
      }
    }
    return;
  }

  // No block-level ancestor above seqLoop: the gang loop is seqLoop itself, a
  // gang(+vector) loop directly under the compute region. The fixed-depth
  // lookup above misses it (issue: a gang+vector init loop that writes
  // gang-private shared memory must be followed by a barrier before another
  // loop reads it). Only needed when the loop writes such shared memory; loops
  // writing global/thread-private memory (e.g. a plain gang-vector saxpy) do
  // not need one here.
  mlir::acc::GPUParallelDimsAttr parDimsAttr =
      mlir::acc::getParDimsAttr(seqLoop);
  if (parDimsAttr && parDimsAttr.hasOnlyBlockLevel() &&
      mayWriteSharedMemory(loopOp)) {
    createBarrier(loopOp->getLoc(), parDimsAttr);
  }
}

bool ACCCGToGPULowering::mayWriteSharedMemory(Operation *loopOp) {
  bool found = false;
  loopOp->walk([&](memref::StoreOp storeOp) {
    // Trace the store target back to its backing acc.private_local through
    // view/cast/box ops, then check the privatization is gang-level.
    llvm::SmallVector<Value, 8> worklist{storeOp.getMemref()};
    llvm::SmallPtrSet<Value, 8> seen;
    while (!worklist.empty()) {
      Value v = worklist.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      Operation *def = v.getDefiningOp();
      if (!def)
        continue;
      if (acc::PrivateLocalOp privateLocal =
              dyn_cast<acc::PrivateLocalOp>(def)) {
        acc::PrivatizeOp privatizeOp =
            getPrivatizeOp(privateLocal, computeRegion);
        // Only gang-level (block, no thread) private memory is a single copy
        // shared across the workgroup's threads, so a write needs a workgroup
        // barrier before another thread reads it. A [block_x, thread_x]
        // privatization is thread-private (one copy per thread); barriering on
        // it would deadlock when threads take different grid-stride iteration
        // counts.
        if (mlir::acc::GPUParallelDimsAttr parDims =
                privatizeOp.getParDimsAttr()) {
          bool hasBlock = false, hasThread = false;
          for (mlir::acc::GPUParallelDimAttr d : parDims.getArray()) {
            if (d.isAnyBlock())
              hasBlock = true;
            if (d.isThreadX() || d.isThreadY())
              hasThread = true;
          }
          if (hasBlock && !hasThread) {
            found = true;
            return WalkResult::interrupt();
          }
        }
        continue;
      }
      worklist.append(def->getOperands().begin(), def->getOperands().end());
    }
    return WalkResult::advance();
  });
  return found;
}

PrivateMemScope
ACCCGToGPULowering::getPrivateMemScope(acc::PrivatizeOp privatizeOp) {
  bool hasBlock = false;
  bool hasThreadX = false;
  bool hasThreadY = false;
  if (mlir::acc::GPUParallelDimsAttr parDims = privatizeOp.getParDimsAttr()) {
    for (mlir::acc::GPUParallelDimAttr d : parDims.getArray()) {
      if (d.isAnyBlock())
        hasBlock = true;
      if (d.isThreadX())
        hasThreadX = true;
      if (d.isThreadY())
        hasThreadY = true;
    }
  } else {
    for (mlir::acc::GPUParallelDimAttr d : computeRegion.getLaunchParDims())
      if (d.isAnyBlock())
        hasBlock = true;
    if (hasBlock)
      return PrivateMemScope::Gang;
    return PrivateMemScope::Thread;
  }
  if (hasThreadX)
    return PrivateMemScope::Thread;
  if (hasBlock && hasThreadY)
    return PrivateMemScope::Worker;
  if (hasBlock)
    return PrivateMemScope::Gang;
  return PrivateMemScope::Thread;
}

/// Walks back from a memref use to its defining `acc.private_local`, if any.
static acc::PrivateLocalOp getPrivateLocalForMemref(Value memref) {
  llvm::SmallVector<Value, 8> worklist{memref};
  llvm::SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *def = v.getDefiningOp();
    if (!def)
      continue;
    if (acc::PrivateLocalOp privateLocal = dyn_cast<acc::PrivateLocalOp>(def))
      return privateLocal;
    worklist.append(def->getOperands().begin(), def->getOperands().end());
  }
  return nullptr;
}

PrivateMemScope ACCCGToGPULowering::getPrivateScopeForMemref(Value memref) {
  if (auto privateLocal = getPrivateLocalForMemref(memref))
    return getPrivateMemScope(getPrivatizeOp(privateLocal, computeRegion));
  return PrivateMemScope::None;
}

acc::PrivatizeOp ACCCGToGPULowering::getPrivatizeForMemref(Value memref) {
  if (auto privateLocal = getPrivateLocalForMemref(memref))
    return getPrivatizeOp(privateLocal, computeRegion);
  return acc::PrivatizeOp();
}

PrivateMemScope
ACCCGToGPULowering::needsPreStoreReuseBarrier(acc::PredicateRegionOp interOp) {

  // Check if we need a pre-predicate barrier first.
  // Next, check if the barrier should be gang- or worker-level.
  LoopLikeOpInterface seqLoopOp = findFirstSequentialLoop(interOp);
  if (!seqLoopOp)
    return PrivateMemScope::None;

  // Check if any op in the predicate region stores to gang- or worker-private
  // memory. If not, no barrier is needed.
  PrivateMemScope storeScope = PrivateMemScope::None;
  llvm::SmallPtrSet<Operation *, 4> storePrivatizes;
  interOp.getRegion().walk([&](memref::StoreOp storeOp) {
    PrivateMemScope scope = getPrivateScopeForMemref(storeOp.getMemref());
    if (scope != PrivateMemScope::Gang && scope != PrivateMemScope::Worker)
      return WalkResult::advance();
    if (auto privatize = getPrivatizeForMemref(storeOp.getMemref()))
      storePrivatizes.insert(privatize.getOperation());
    if (storeScope == PrivateMemScope::None)
      storeScope = scope;
    return WalkResult::advance();
  });
  if (storeScope == PrivateMemScope::None || storePrivatizes.empty())
    return PrivateMemScope::None;

  // Check that there is a subsequent parallel region that uses private memory
  bool hasParallelPrivateUse = false;
  seqLoopOp.getOperation()->walk([&](Operation *op) {
    // Ignore loads inside the predicate.
    if (interOp->isAncestor(op))
      return WalkResult::advance();

    Value memref;
    if (memref::LoadOp loadOp = dyn_cast<memref::LoadOp>(op))
      memref = loadOp.getMemref();
    else if (memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(op))
      memref = storeOp.getMemref();
    else
      return WalkResult::advance();

    PrivateMemScope scope = getPrivateScopeForMemref(memref);
    if (scope != storeScope)
      return WalkResult::advance();

    acc::PrivatizeOp usePrivatize = getPrivatizeForMemref(memref);
    if (!usePrivatize || !storePrivatizes.contains(usePrivatize.getOperation()))
      return WalkResult::advance();

    bool insideNestedParallel = false;
    for (Operation *p = op->getParentOp(); p && p != seqLoopOp.getOperation();
         p = p->getParentOp()) {
      if (scf::ParallelOp par = dyn_cast<scf::ParallelOp>(p)) {
        if (mlir::acc::GPUParallelDimsAttr pd =
                mlir::acc::getParDimsAttr(par)) {
          if (llvm::any_of(pd.getArray(), [](mlir::acc::GPUParallelDimAttr d) {
                return !d.isSeq();
              })) {
            insideNestedParallel = true;
            break;
          }
        }
      }
    }
    if (!insideNestedParallel)
      return WalkResult::advance();
    // An inner parallel region uses the same private memory.
    hasParallelPrivateUse = true;
    return WalkResult::interrupt();
  });

  if (!hasParallelPrivateUse)
    return PrivateMemScope::None;

  return storeScope;
}

void ACCCGToGPULowering::processPredicateRegion(
    acc::PredicateRegionOp interOp) {
  LLVM_DEBUG(llvm::dbgs() << "processing predicate region: ";
             interOp->print(llvm::dbgs()); llvm::dbgs() << "\n");
  Location loc = interOp->getLoc();

  std::pair<SmallVector<mlir::acc::GPUParallelDimAttr>,
            SmallVector<mlir::acc::GPUParallelDimAttr>>
      parDimsPair = computeActiveAndInactiveParDims(
          interOp, &interOp.getRegion().front());

  // If ThreadY reduction exists, subgroup alignment is applied
  // (blockDim.x = subgroupSize), so ThreadX lanes exist even without explicit
  // ThreadX parallelism. Add ThreadX to inactiveParDims if not already present.
  // Exception: if this region contains a thread-level (vector or worker)
  // routine call, all ThreadX threads must reach the call so the routine's
  // workgroup-wide barriers (e.g. shared memory alloca sync) converge.
  if (hasThreadYReduction) {
    MLIRContext *ctx = computeRegion->getContext();
    mlir::acc::GPUParallelDimAttr threadXParDim =
        mlir::acc::GPUParallelDimAttr::threadXDim(ctx);
    bool hasThreadXInActive =
        llvm::any_of(parDimsPair.first, [](mlir::acc::GPUParallelDimAttr pd) {
          return pd.isThreadX();
        });
    bool hasThreadXInInactive =
        llvm::any_of(parDimsPair.second, [](mlir::acc::GPUParallelDimAttr pd) {
          return pd.isThreadX();
        });

    // Check if THIS predicate region contains a thread-level routine call.
    // We use the pre-computed hasThreadLevelRoutineCall as an early-out
    // optimization.
    bool regionHasThreadLevelRoutineCall = false;
    if (hasThreadLevelRoutineCall) {
      interOp.getRegion().walk([&](CallOpInterface callOp) {
        if (mlir::acc::GPUParallelDimAttr parDim =
                getAccRoutineCallParDim(callOp, defaultPolicy)) {
          if (parDim.isThreadX() || parDim.isThreadY()) {
            regionHasThreadLevelRoutineCall = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    }

    if (!hasThreadXInActive && !hasThreadXInInactive &&
        !regionHasThreadLevelRoutineCall) {
      parDimsPair.second.push_back(threadXParDim);
    }
  }

  if (Value predicate = emitPredicate(loc, parDimsPair.second)) {
    LLVM_DEBUG(llvm::dbgs() << "predicate: " << predicate << "\n");
    bool isInsideThreadXLoop = false;
    bool isInsideThreadYLoop = false;
    for (auto parDim : parDimsPair.first) {
      if (parDim.isThreadX())
        isInsideThreadXLoop = true;
      if (parDim.isThreadY())
        isInsideThreadYLoop = true;
    }
    // Emits the reconvergence barrier matching this region's predication level,
    // at the current insertion point. Called for both before and after the
    // predicated store. Below is the pre-predicate barrier; the post-predicate
    // barrier is emitted after the ifOp.
    auto emitReconvergenceBarrier = [&]() {
      if (isInsideThreadXLoop) {
        // Inside ThreadX loop - skip barrier
      } else if (isInsideThreadYLoop) {
        // Inside ThreadY loop
        if (!isInsideACCSpecializedRoutine(computeRegion)) {
          // Add barrier if ThreadX is predicated (lane 0 writes must be
          // visible to all lanes before they read).
          bool predicatesThreadX = llvm::any_of(
              parDimsPair.second,
              [](mlir::acc::GPUParallelDimAttr pd) { return pd.isThreadX(); });
          if (predicatesThreadX) {
            createBarrier(loc, mlir::acc::GPUParallelDimsAttr::get(
                                   interOp->getContext(), parDimsPair.second));
          }
        }
        // For acc routine ThreadY routines, skip barrier
      } else if (!parDimsPair.first.empty()) {
        // Inside block loop
        createBarrier(loc, mlir::acc::GPUParallelDimsAttr::get(
                               interOp->getContext(), parDimsPair.first));
      } else {
        // Top level
        createBarrier(loc, mlir::acc::GPUParallelDimsAttr::get(
                               interOp->getContext(), parDimsPair.second));
      }
    };

    // A gang-level (block, no thread) shared slot written here and reused on
    // the next iteration of an enclosing sequential loop must not be
    // overwritten before all threads read the current value. Emit the
    // reconvergence barrier before the store too (the post-store barrier below
    // only orders this iteration's store->read). Restricted to the block-level
    // case: such loops are run uniformly by every workgroup thread, so the
    // workgroup barrier cannot deadlock; worker/vector (thread_y/thread_x)
    // loops may have divergent trip counts.
    PrivateMemScope scope = needsPreStoreReuseBarrier(interOp);
    if (scope == PrivateMemScope::Gang)
      emitGPUBarrierWorkgroup(rewriter, loc);
    else if (scope == PrivateMemScope::Worker)
      createPerRowBarrier(loc);

    auto ifOp = scf::IfOp::create(rewriter, loc, predicate,
                                  /*withElseRegion=*/false);
    Region &thenRegion = ifOp.getThenRegion();
    Block &thenBlock = thenRegion.back();
    rewriter.setInsertionPoint(thenBlock.getTerminator());
    // Ops in a predicate region may need to be further processed, recurse
    for (auto &bodyOp : interOp.getRegion().front().getOperations()) {
      // If the store's value loads from a block-level reduction
      // memref, convert to atomic for cross-block correctness.
      // Only at kernel top level (no active thread dims).
      if (memref::StoreOp storeOp = dyn_cast<memref::StoreOp>(&bodyOp)) {
        std::optional<arith::AtomicRMWKind> blockReduceKind;
        bool failedReductionKind = false;
        Value storeVal = storeOp.getValueToStore();
        if (storeVal.getDefiningOp()) {
          // Walk the epilogue def-chain within the enclosing block to find
          // the block-level accumulate load feeding this store.  Epilogue ops
          // (type conversions, arithmetic, etc.) are traversed transparently.
          // Non-acc loads and values defined outside the block are treated as
          // loop-invariant and return nullopt, bounding the search naturally.
          Block *epilogueBlock = interOp->getBlock();
          auto findBlockAccLoad =
              [&](auto &self,
                  Value val) -> std::optional<arith::AtomicRMWKind> {
            Operation *def = val.getDefiningOp();
            if (!def || def->getBlock() != epilogueBlock)
              return std::nullopt;
            if (memref::LoadOp loadOp = dyn_cast<memref::LoadOp>(def)) {
              for (auto *user : loadOp.getMemRef().getUsers()) {
                if (acc::ReductionAccumulateOp accOp =
                        dyn_cast<acc::ReductionAccumulateOp>(user)) {
                  if (llvm::any_of(accOp.getParDims().getArray(),
                                   [](mlir::acc::GPUParallelDimAttr pd) {
                                     return pd.isAnyBlock();
                                   })) {
                    FailureOr<arith::AtomicRMWKind> kind = getReductionKind(
                        accOp.getReductionOperator(),
                        accOp.getValue().getType(), accOp.getLoc());
                    if (failed(kind)) {
                      failedReductionKind = true;
                      return std::nullopt;
                    }
                    return *kind;
                  }
                }
              }
              return std::nullopt;
            }
            for (Value operand : def->getOperands())
              if (auto kind = self(self, operand))
                return kind;
            return std::nullopt;
          };
          blockReduceKind = findBlockAccLoad(findBlockAccLoad, storeVal);
        }
        if (failedReductionKind)
          return;
        if (blockReduceKind) {
          Value input = mapping.lookupOrDefault(storeOp.getValueToStore());
          Value memref = mapping.lookupOrDefault(storeOp.getMemref());
          bool threadIsActive = llvm::any_of(
              parDimsPair.first, [](mlir::acc::GPUParallelDimAttr pd) {
                return !pd.isAnyBlock();
              });
          if (!threadIsActive &&
              !isa_and_nonnull<memref::AllocaOp>(
                  unwrapMemRefConversion(memref).getDefiningOp())) {
            // Initialize the destination to the reduction identity before
            // cross-block atomics so that the final result reflects pure
            // assignment semantics (e.g. r = sum(a)), not accumulation
            // on top of the pre-kernel value.
            if (launch) {
              MemRefType memrefTy = cast<MemRefType>(memref.getType());
              // Map the store indices for ranked memrefs.
              SmallVector<Value> initIndices;
              for (Value idx : storeOp.getIndices())
                initIndices.push_back(mapping.lookupOrDefault(idx));

              OpBuilder::InsertionGuard guard(rewriter);
              Block &launchBody = launch.getBody().front();
              Operation *insertBefore = nullptr;

              launchBody.walk([&](scf::ParallelOp parOp) -> WalkResult {
                for (Operation *parent = parOp->getParentOp(); parent;
                     parent = parent->getParentOp()) {
                  if (parent == launch.getOperation())
                    break;
                  if (isa<scf::ParallelOp>(parent))
                    return WalkResult::advance();
                }
                insertBefore = parOp.getOperation();
                return WalkResult::interrupt();
              });
              if (insertBefore)
                rewriter.setInsertionPoint(insertBefore);
              else
                rewriter.setInsertionPointToStart(&launchBody);
              // Recursively re-materialize operations whose definitions
              // do not dominate the insertion point.  A single-level clone
              // is insufficient when the value is produced by a chain of
              // operations (e.g. reinterpret_cast depending on box_dims,
              // divsi, convert, etc.) that are all defined after the
              // insertion point.
              DominanceInfo domInfo(launch);
              IRMapping initMapping;
              std::function<Value(Value)> materialize =
                  [&](Value val) -> Value {
                Operation *defOp = val.getDefiningOp();
                if (!defOp)
                  return val;
                if (domInfo.dominates(defOp, &*rewriter.getInsertionPoint()))
                  return val;
                if (auto mapped = initMapping.lookupOrNull(val))
                  return mapped;
                // Recurse on operands; the recursive call seeds
                // initMapping for any operand it clones, which the
                // subsequent rewriter.clone(..., initMapping) picks up.
                for (Value operand : defOp->getOperands())
                  materialize(operand);
                Operation *cloned = rewriter.clone(*defOp, initMapping);
                for (auto [orig, clonedRes] :
                     llvm::zip(defOp->getResults(), cloned->getResults())) {
                  initMapping.map(orig, clonedRes);
                }
                return initMapping.lookup(val);
              };
              Value initMemref = materialize(memref);
              for (auto &idx : initIndices)
                idx = materialize(idx);
              Value identityVal = createIdentityValue(
                  rewriter, loc, memrefTy.getElementType(), *blockReduceKind,
                  /*useOnlyFiniteValue=*/true);
              Value blockId = gpu::BlockIdOp::create(
                  rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
              Value threadId = gpu::ThreadIdOp::create(
                  rewriter, loc, rewriter.getIndexType(), gpu::Dimension::x);
              Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
              Value isBlock0 = arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::eq, blockId, zero);
              Value isThread0 = arith::CmpIOp::create(
                  rewriter, loc, arith::CmpIPredicate::eq, threadId, zero);
              Value isFirstThread =
                  arith::AndIOp::create(rewriter, loc, isBlock0, isThread0);
              auto initIf = scf::IfOp::create(rewriter, loc, isFirstThread,
                                              /*withElseRegion=*/false);
              rewriter.setInsertionPoint(
                  initIf.getThenRegion().back().getTerminator());
              memref::StoreOp::create(rewriter, loc, identityVal, initMemref,
                                      initIndices);
              rewriter.setInsertionPointAfter(initIf);
              gpu::BarrierOp::create(rewriter, loc);
            }
            SmallVector<Value> atomicIndices;
            for (Value idx : storeOp.getIndices())
              atomicIndices.push_back(mapping.lookupOrDefault(idx));
            constructAtomicAccumulation(loc, memref, atomicIndices, input,
                                        *blockReduceKind);
            continue;
          }
        }
      }
      processOp(&bodyOp);
    }
    rewriter.setInsertionPointAfter(ifOp);
    emitReconvergenceBarrier();
  } else {
    // Ops in a predicate region may need to be further processed, recurse
    for (auto &bodyOp : interOp.getRegion().front().getOperations())
      processOp(&bodyOp);
  }
}

// clang-format off
//
// Allocates private storage and broadcasts its pointer through a shared
// memref-of-memref slot, using the same predication utility as predicate
// regions.
//
//      %0 = arith.cmpi eq, %thread_id_y, %c0 : index
//      %1 = arith.cmpi eq, %thread_id_x, %c0 : index
//      %2 = arith.andi %1, %0 : i1
//      scf.if %2 {
//        %alloc = memref.alloc() : memref<10xi32>
//        memref.store %alloc, %arg1[] : memref<memref<10xi32>, #gpu.address_space<workgroup>>
//      }
//      gpu.barrier scope<subgroup>
//      %4 = memref.load %arg1[] : memref<memref<10xi32>, #gpu.address_space<workgroup>>
//
// clang-format on

Value ACCCGToGPULowering::processPrivatize(acc::PrivatizeOp privatize) {
  LLVM_DEBUG(llvm::dbgs() << "processing privatize: ";
             privatize->print(llvm::dbgs()); llvm::dbgs() << "\n");
  Value tracked = privatize.getResult();
  if (acc::ComputeRegionOp insUser =
          dyn_cast<acc::ComputeRegionOp>(getOnlyUser(tracked))) {
    assert(privatize->hasOneUse() &&
           "expected acc.privatize op to have one use");
    tracked = insUser.getBody()->getArgument(
        privatize->use_begin()->getOperandNumber());
  }
  Operation *privatizeUser = getOnlyUser(tracked);
  assert(privatizeUser && "expected PrivateLocalOp user for privatize");

  std::pair<SmallVector<mlir::acc::GPUParallelDimAttr>,
            SmallVector<mlir::acc::GPUParallelDimAttr>>
      parDimsPair = computeActiveAndInactiveParDims(privatizeUser, nullptr);
  // Set `par_dims` only when this `acc.privatize` does not already carry it.
  if (!privatize.getParDimsAttr()) {
    privatize.setParDimsAttr(mlir::acc::GPUParallelDimsAttr::get(
        rewriter.getContext(), parDimsPair.first));
  }

  Location loc = privatize->getLoc();
  acc::PrivateType privTy = cast<acc::PrivateType>(privatize.getType());
  ModuleOp module = computeRegion->getParentOfType<ModuleOp>();
  MemRefType baseTy = getPrivateBaseMemRefType(privTy.getBaseTy(), module);

  gpu::GPUFuncOp gpuFuncOp = computeRegion->getParentOfType<gpu::GPUFuncOp>();
  // acc.privatize is outside this compute_region (e.g. passed via ins).
  // Leave the op unchanged here; processPrivateLocal materializes
  // storage when acc.private_local is lowered.
  if (!gpuFuncOp &&
      privatize->getParentOfType<acc::ComputeRegionOp>() != computeRegion) {
    return privatize.getResult();
  }

  for (mlir::acc::GPUParallelDimAttr parDim : parDimsPair.first) {
    if (parDim.isThreadX() &&
        canUseStackAlloca(baseTy, loc, options.maxThreadPrivateStack)) {
      auto alloca = memref::AllocaOp::create(rewriter, loc, baseTy);
      mapping.map(privatize.getResult(), alloca.getResult());
      return alloca.getResult();
    }
  }

  if (!gpuFuncOp)
    return privatize.getResult();

  // When ThreadY is active, shared memory must be indexed by ThreadY ID to
  // avoid races between ThreadY threads.
  // For ThreadX acc routines, assume ThreadY may be active since the routine
  // can be called from a ThreadY loop at runtime.
  // For ThreadY/block acc routines, do not force ThreadY indexing - variables
  // outside the ThreadY loop should be shared across ThreadY threads.
  bool threadYIsActive =
      llvm::any_of(parDimsPair.first, [](mlir::acc::GPUParallelDimAttr parDim) {
        return parDim.isThreadY();
      });
  // For routines with multiple ThreadY threads, need a workgroup barrier
  // instead of a subgroup barrier.
  // - ThreadX routines: called independently by different ThreadY threads, need
  //   per-ThreadY slots (override threadYIsActive)
  // - ThreadY routines: workgroup barrier, but respect parDimsPair for
  // threadYIsActive
  //   (variables before ThreadY loop are shared, inside are per-ThreadY)
  // - Block routines: single call, ThreadY threads cooperate within the
  // routine, so
  //   variables at routine level are shared across ThreadY threads (single
  //   slot)
  bool needsWorkgroupBarrier = false;
  if (isInsideACCSpecializedRoutine(computeRegion)) {
    FunctionOpInterface funcOp =
        computeRegion->getParentOfType<FunctionOpInterface>();
    mlir::acc::GPUParallelDimAttr routineParDim =
        getSpecializedRoutineDim(funcOp, defaultPolicy);
    if (routineParDim.isThreadX()) {
      // ThreadX routine: always per-ThreadY slots since called from ThreadY
      // loops
      threadYIsActive = true;
    } else if (routineParDim.isThreadY()) {
      // ThreadY routine: workgroup barrier, but keep original threadYIsActive
      // (variables before ThreadY loop shared, inside per-ThreadY)
      needsWorkgroupBarrier = true;
    } else if (routineParDim.isAnyBlock()) {
      needsWorkgroupBarrier = true;
    }
  }

  llvm::SmallVector<Value> mappedDynamicSizes;
  for (auto dynamicSize : privatize.getDynamicSizes()) {
    Value mappedDynamicSize = mapping.lookupOrDefault(dynamicSize);
    mappedDynamicSizes.push_back(mappedDynamicSize);
  }
  if (isInsideACCSpecializedRoutine(computeRegion) &&
      computeRegion.isEffectivelySerial()) {
    if (mappedDynamicSizes.empty()) {
      // Static sizes: use alloca (stack allocation)
      auto alloca =
          memref::AllocaOp::create(rewriter, privatize->getLoc(), baseTy);
      mapping.map(privatize.getResult(), alloca.getResult());
      return alloca.getResult();
    }
    // Dynamic sizes: use alloc (heap allocation) with dealloc
    auto alloc = memref::AllocOp::create(rewriter, privatize->getLoc(), baseTy,
                                         mappedDynamicSizes);

    // Insert dealloc (free) before the function return
    OpBuilder::InsertPoint currentInsertPoint = rewriter.saveInsertionPoint();
    Block &parentBlock = *alloc->getBlock();
    if (parentBlock.mightHaveTerminator()) {
      rewriter.setInsertionPoint(parentBlock.getTerminator());
      memref::DeallocOp::create(rewriter, privatize->getLoc(), alloc);
    }
    rewriter.restoreInsertionPoint(currentInsertPoint);

    mapping.map(privatize.getResult(), alloc.getResult());
    return alloc.getResult();
  }

  // Predication - when threadYIsActive, don't predicate on ThreadY dimension
  // since each ThreadY needs to execute the allocation for its own slot
  SmallVector<mlir::acc::GPUParallelDimAttr> predicateDims;
  for (auto parDim : parDimsPair.second) {
    // Skip ThreadY if threadYIsActive - each ThreadY needs to allocate
    if (threadYIsActive && parDim.isThreadY())
      continue;
    predicateDims.push_back(parDim);
  }
  Value predicate = emitPredicate(loc, predicateDims);
  if (!predicate) {
    predicate = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
  }
  auto ifOp = scf::IfOp::create(rewriter, loc, predicate,
                                /*withElseRegion=*/false);
  Region &thenRegion = ifOp.getThenRegion();
  Block &thenBlock = thenRegion.back();
  rewriter.setInsertionPoint(thenBlock.getTerminator());
  auto mem = memref::AllocOp::create(rewriter, privatize->getLoc(), baseTy,
                                     mappedDynamicSizes);
  // Shared memory allocation
  gpu::AddressSpaceAttr sharedMemoryAddressSpace = gpu::AddressSpaceAttr::get(
      computeRegion->getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  // When ThreadY is active, create a shared memory array indexed by ThreadY ID.
  // Each ThreadY stores to its own slot; ThreadX lanes within a ThreadY share
  // it.
  constexpr int64_t kMaxThreadY = 32;
  MemRefType sharedMemTy =
      threadYIsActive
          ? MemRefType::get({kMaxThreadY}, baseTy, MemRefLayoutAttrInterface{},
                            sharedMemoryAddressSpace)
          : MemRefType::get({}, baseTy, MemRefLayoutAttrInterface{},
                            sharedMemoryAddressSpace);
  // The slot only transiently broadcasts the storage pointer, so reuse one per
  // type across privatizes, barriering (before the predicated store) on reuse.
  bool reuseBroadcast = !gpuFuncOp.isKernel();
  Value alloca;
  llvm::DenseMap<Type, Value>::iterator cachedSlot =
      reuseBroadcast ? privatizeBroadcastCache.find(sharedMemTy)
                     : privatizeBroadcastCache.end();
  if (reuseBroadcast && cachedSlot != privatizeBroadcastCache.end()) {
    alloca = cachedSlot->second;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);
    mlir::acc::GPUParallelDimAttr dim =
        needsWorkgroupBarrier
            ? mlir::acc::GPUParallelDimAttr::threadYDim(rewriter.getContext())
            : mlir::acc::GPUParallelDimAttr::threadXDim(rewriter.getContext());
    createBarrier(
        loc, mlir::acc::GPUParallelDimsAttr::get(rewriter.getContext(), {dim}));
  } else {
    alloca = gpuFuncOp.addWorkgroupAttribution(sharedMemTy,
                                               rewriter.getUnknownLoc());
    // Setting the alignment to 16 because of a bug in the gpu toolchain.
    // The default alignment is 8, but optimizations create a packed store of 16
    // bytes which cause a misalignment error at runtime.
    unsigned index = gpuFuncOp.getNumWorkgroupAttributions() - 1;
    gpuFuncOp.setWorkgroupAttributionAttr(index,
                                          LLVM::LLVMDialect::getAlignAttrName(),
                                          rewriter.getI32IntegerAttr(16));
    if (reuseBroadcast)
      privatizeBroadcastCache[sharedMemTy] = alloca;
  }
  // Store to shared memory, indexed by ThreadY ID when ThreadY is active.
  if (threadYIsActive) {
    Value threadYId = getThreadId(loc, gpu::Dimension::y);
    memref::StoreOp::create(rewriter, privatize->getLoc(), mem, alloca,
                            ValueRange{threadYId});
  } else {
    memref::StoreOp::create(rewriter, privatize->getLoc(), mem, alloca);
  }

  // Sync and load - use workgroup barrier for Block/ThreadY routines,
  // ThreadX barrier for ThreadX-only routines
  rewriter.setInsertionPointAfter(ifOp);
  if (needsWorkgroupBarrier) {
    mlir::acc::GPUParallelDimsAttr threadYDimsAttr =
        mlir::acc::GPUParallelDimsAttr::get(
            rewriter.getContext(),
            {mlir::acc::GPUParallelDimAttr::threadYDim(rewriter.getContext())});
    createBarrier(loc, threadYDimsAttr);
  } else {
    // ThreadX-only: use per-ThreadY barrier
    mlir::acc::GPUParallelDimsAttr threadXDimsAttr =
        mlir::acc::GPUParallelDimsAttr::get(
            rewriter.getContext(),
            {mlir::acc::GPUParallelDimAttr::threadXDim(rewriter.getContext())});
    createBarrier(loc, threadXDimsAttr);
  }
  // Load from shared memory, indexed by ThreadY ID when ThreadY is active.
  Value load;
  if (threadYIsActive) {
    Value threadYId = getThreadId(loc, gpu::Dimension::y);
    load = memref::LoadOp::create(rewriter, privatize->getLoc(), baseTy, alloca,
                                  ValueRange{threadYId});
  } else {
    load =
        memref::LoadOp::create(rewriter, privatize->getLoc(), baseTy, alloca);
  }
  rewriter.setInsertionPointAfter(load.getDefiningOp());
  mapping.map(privatize.getResult(), load);

  // Operations inside the kernel are all rewritten from scratch.
  // But if the privatize op is outside the kernel, it needs to be replaced.
  if (!privatize->getParentOfType<acc::ComputeRegionOp>())
    rewriter.replaceOp(privatize, load);
  // Deallocate
  rewriter.setInsertionPoint(ifOp->getBlock()->getTerminator());
  if (needsWorkgroupBarrier) {
    mlir::acc::GPUParallelDimsAttr workerDimsAttr =
        mlir::acc::GPUParallelDimsAttr::get(
            rewriter.getContext(),
            {mlir::acc::GPUParallelDimAttr::threadYDim(rewriter.getContext())});
    createBarrier(loc, workerDimsAttr);
  } else {
    mlir::acc::GPUParallelDimsAttr vectorDimsAttr =
        mlir::acc::GPUParallelDimsAttr::get(
            rewriter.getContext(),
            {mlir::acc::GPUParallelDimAttr::threadXDim(rewriter.getContext())});
    createBarrier(loc, vectorDimsAttr);
  }
  auto ifOp2 = scf::IfOp::create(rewriter, loc, predicate,
                                 /*withElseRegion=*/false);
  Region &thenRegion2 = ifOp2.getThenRegion();
  Block &thenBlock2 = thenRegion2.back();
  rewriter.setInsertionPoint(thenBlock2.getTerminator());
  memref::DeallocOp::create(rewriter, privatize->getLoc(), load);

  // Return the private memory
  rewriter.setInsertionPointAfter(load.getDefiningOp());

  return load;
}

// Materialize acc.private_local storage from acc.privatize: per-thread alloca
// when possible, otherwise a shared broadcast slot or acc.gpu_shared_memory.

void ACCCGToGPULowering::processPrivateLocal(
    acc::PrivateLocalOp privateLocal, std::optional<int64_t> sharedMemCopies) {
  LLVM_DEBUG(llvm::dbgs() << "processing private local: ";
             privateLocal->print(llvm::dbgs()); llvm::dbgs() << "\n");
  Location loc = privateLocal.getLoc();
  acc::PrivateType privTy =
      cast<acc::PrivateType>(privateLocal.getPrivatized().getType());
  ModuleOp module = computeRegion->getParentOfType<ModuleOp>();
  MemRefType baseTy = getPrivateBaseMemRefType(privTy.getBaseTy(), module);
  MemRefType byteMemrefTy =
      MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());

  acc::PrivatizeOp privatizeOp = getPrivatizeOp(privateLocal, computeRegion);
  Value inputMem;
  if (privatizeOp->getParentOfType<acc::ComputeRegionOp>() == computeRegion) {
    inputMem = mapping.lookupOrNull(privatizeOp);
    if (inputMem) {
      Value result = castPointerLikeTypeIfNeeded(rewriter, loc, inputMem,
                                                 privateLocal.getType());
      mapping.map(privateLocal.getResult(), result);
      return;
    }
  } else {
    // Hoisted acc.privatize: allocate per-thread stack storage in the launch
    // body. Cross-thread array reduction accumulators are per-thread too, so
    // the accumulate can reduce each element across threads.
    acc::ReductionAccumulateArrayOp arrayAccum =
        perThreadArrayReductionAccum(privateLocal.getResult());
    if ((isThreadXPrivatize(privatizeOp) || arrayAccum) &&
        canUseStackAlloca(baseTy, loc, options.maxThreadPrivateStack)) {
      Value alloca = memref::AllocaOp::create(rewriter, loc, baseTy);
      if (arrayAccum) {
        FailureOr<arith::AtomicRMWKind> kind = getReductionKind(
            arrayAccum.getReductionOperator(), baseTy.getElementType(), loc);
        if (failed(kind))
          return;
        initPerThreadArrayAccum(rewriter, loc, alloca, baseTy, *kind);
      }
      Value mem = castPointerLikeTypeIfNeeded(rewriter, loc, alloca,
                                              privateLocal.getType());
      mapping.map(privateLocal.getResult(), mem);
      return;
    }

    // If acc.privatize is outside the kernel, it needs to be converted
    // explicitly.
    std::optional<int64_t> copies =
        sharedMemCopies ? sharedMemCopies
                        : isEligibleForSharedMemory(privateLocal, baseTy);
    if (copies) {
      int64_t numCopies = *copies;
      int64_t elementSize = getElementSizeInBytes(loc, baseTy.getElementType());
      int64_t numElements = 1;
      for (int64_t dim : baseTy.getShape())
        numElements *= dim;
      int64_t upperBound = elementSize * numElements * numCopies;

      if (tryAllocateSharedMemory(upperBound)) {
        std::string varName =
            accSupport.getVariableName(privateLocal.getResult());
        sharedMemPrivateVarNames.push_back(varName.empty() ? "(*)" : varName);

        gpu::AddressSpaceAttr workgroupAS = gpu::AddressSpaceAttr::get(
            computeRegion->getContext(),
            gpu::GPUDialect::getWorkgroupAddressSpace());
        MemRefType sharedMemTy =
            MemRefType::get(baseTy.getShape(), baseTy.getElementType(),
                            MemRefLayoutAttrInterface{}, workgroupAS);
        Value sharedMem = acc::GPUSharedMemoryOp::create(
            rewriter, loc, sharedMemTy, rewriter.getI64IntegerAttr(numCopies),
            rewriter.getI64IntegerAttr(upperBound), ValueRange{}, IntegerAttr{},
            IntegerAttr{});

        Value mem =
            castPointerLikeTypeIfNeeded(rewriter, loc, sharedMem, baseTy);
        Value result = castPointerLikeTypeIfNeeded(rewriter, loc, mem,
                                                   privateLocal.getType());

        mapping.map(privateLocal.getResult(), result);
        return;
      }
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(privatizeOp);
    inputMem = processPrivatize(privatizeOp);
  }

  if (isInsideACCSpecializedRoutine(computeRegion)) {
    assert(inputMem && "expected input mem to be mapped");
    Value result = castPointerLikeTypeIfNeeded(rewriter, loc, inputMem,
                                               privateLocal.getType());
    mapping.map(privateLocal.getResult(), result);
    return;
  }

  // The private element shape augmented with a dimension for each level of
  // parallelism
  SmallVector<int64_t> viewShape;
  // The dynamic sizes of the view (num_gangs, num_workers, vector_length)
  SmallVector<Value> viewDynSizes;
  // The offset of the subview from gpu.block_id/gpu.thread_id dimensions.
  SmallVector<OpFoldResult> subviewOffset;
  // The sizes of the subview where the dimensionality is brought back to the
  // private element. That is size 1 for each active block/thread dimension.
  SmallVector<OpFoldResult> subviewSizes;
  // The strides of the subview
  SmallVector<int64_t> subviewStrides;
  // The shape of the subview
  SmallVector<int64_t> subviewShape;

  std::pair<SmallVector<mlir::acc::GPUParallelDimAttr>,
            SmallVector<mlir::acc::GPUParallelDimAttr>>
      parDimsPair = computeActiveAndInactiveParDims(privateLocal, nullptr);
  acc::ReductionAccumulateArrayOp arrayAccum =
      perThreadArrayReductionAccum(privateLocal.getResult());
  for (mlir::acc::GPUParallelDimAttr parDim : parDimsPair.first) {
    if ((parDim.isThreadX() || arrayAccum) &&
        canUseStackAlloca(baseTy, loc, options.maxThreadPrivateStack)) {
      Value alloca = memref::AllocaOp::create(rewriter, loc, baseTy);
      if (arrayAccum) {
        FailureOr<arith::AtomicRMWKind> kind = getReductionKind(
            arrayAccum.getReductionOperator(), baseTy.getElementType(), loc);
        if (failed(kind))
          return;
        initPerThreadArrayAccum(rewriter, loc, alloca, baseTy, *kind);
      }
      Value mem = castPointerLikeTypeIfNeeded(rewriter, loc, alloca,
                                              privateLocal.getType());
      mapping.map(privateLocal.getResult(), mem);
      return;
    }
  }
  if (parDimsPair.first.empty()) {
    // No parallelism is found above. It's single block execution.
    mlir::acc::insertParDim(
        parDimsPair.first,
        mlir::acc::GPUParallelDimAttr::blockXDim(privateLocal.getContext()));
  }
  for (mlir::acc::GPUParallelDimAttr parDim : parDimsPair.first) {
    gpu::Processor gpuProc = parDim.getProcessor();
    Value gpuSize = getGPUSizeFor(gpuProc);
    viewDynSizes.push_back(gpuSize);
    viewShape.push_back(ShapedType::kDynamic);
    subviewOffset.push_back(getGPUThreadIdFor(gpuProc));
    subviewSizes.push_back(rewriter.getIndexAttr(1));
  }

  SmallVector<Value> innerDynSizes =
      resolvePrivateLocalDynamicExtents(privateLocal);

  unsigned dynIdx = 0;
  for (auto innerDim : baseTy.getShape()) {
    subviewOffset.push_back(rewriter.getIndexAttr(0));
    viewShape.push_back(innerDim);
    subviewShape.push_back(innerDim);
    if (innerDim == ShapedType::kDynamic) {
      assert(dynIdx < innerDynSizes.size() &&
             "not enough dynamic sizes for inner dimensions");
      viewDynSizes.push_back(innerDynSizes[dynIdx]);
      subviewSizes.push_back(innerDynSizes[dynIdx]);
      ++dynIdx;
    } else {
      subviewSizes.push_back(rewriter.getIndexAttr(innerDim));
    }
  }

  // Do the strides in reverse order.
  int64_t stride = 1;
  for (auto innerDimIt = baseTy.getShape().rbegin();
       innerDimIt != baseTy.getShape().rend(); ++innerDimIt) {
    int64_t innerDim = *innerDimIt;
    subviewStrides.insert(subviewStrides.begin(), stride);
    if (innerDim == ShapedType::kDynamic)
      stride = ShapedType::kDynamic;
    if (stride != ShapedType::kDynamic)
      stride *= innerDim;
  }

  Value memBuffer =
      castPointerLikeTypeIfNeeded(rewriter, loc, inputMem, byteMemrefTy);
  auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  MemRefType viewType = MemRefType::get(viewShape, baseTy.getElementType());
  auto view = memref::ViewOp::create(rewriter, loc, viewType, memBuffer,
                                     c0.getResult(), viewDynSizes);

  // memref.subview
  StridedLayoutAttr stridedLayout = StridedLayoutAttr::get(
      computeRegion->getContext(), ShapedType::kDynamic, subviewStrides);
  MemRefType subviewType =
      MemRefType::get(subviewShape, baseTy.getElementType(), stridedLayout);
  SmallVector<OpFoldResult> ones(viewType.getRank(), rewriter.getIndexAttr(1));
  Value subview = memref::SubViewOp::create(rewriter, loc, subviewType, view,
                                            subviewOffset, subviewSizes, ones);

  // Cast subview to the target type, preserving the dynamic offset.
  // Do NOT cast to a plain memref (offset: 0) - the LLVM optimizer
  // would fold the gang offset to zero, making all gangs share memory.
  Value result = castPointerLikeTypeIfNeeded(rewriter, loc, subview,
                                             privateLocal.getType());
  mapping.map(privateLocal.getResult(), result);
}

// Could be scf::for or scf::parallel
template <typename LoopOp>
void ACCCGToGPULowering::processSeqLoop(LoopOp loopOp) {
  // Pre-process shared-memory-eligible private_local ops.  Only direct-child
  // ops are considered; nested private_local ops (e.g. inside predicate_region)
  // are handled by recursive body processing.
  LLVM_DEBUG(llvm::dbgs() << "processing seq loop: ";
             loopOp->print(llvm::dbgs()); llvm::dbgs() << "\n");
  llvm::SmallPtrSet<Operation *, 4> preProcessedPrivateLocals;
  ModuleOp module = computeRegion->getParentOfType<ModuleOp>();
  for (auto &bodyOp : loopOp.getBody()->getOperations()) {
    if (acc::PrivateLocalOp privateLocal =
            dyn_cast<acc::PrivateLocalOp>(&bodyOp)) {
      acc::PrivateType privTy =
          cast<acc::PrivateType>(privateLocal.getPrivatized().getType());
      MemRefType baseTy = getPrivateBaseMemRefType(privTy.getBaseTy(), module);
      if (auto copies = isEligibleForSharedMemory(privateLocal, baseTy)) {
        processPrivateLocal(privateLocal, copies);
        preProcessedPrivateLocals.insert(privateLocal.getOperation());
      }
    }
  }

  LoopOp newLoop =
      dyn_cast<LoopOp>(rewriter.cloneWithoutRegions(*loopOp, mapping));
  rewriter.createBlock(
      &newLoop.getRegion(), newLoop.getRegion().begin(),
      loopOp.getBody()->getArgumentTypes(),
      SmallVector<Location>(loopOp.getBody()->getArgumentTypes().size(),
                            loopOp->getLoc()));
  rewriter.setInsertionPointToStart(&newLoop.getRegion().front());

  // Need to clone all block arguments
  Block::BlockArgListType blockArgs = loopOp.getBody()->getArguments();
  assert(blockArgs.size() && "expected block arguments for loop");
  mapping.map(blockArgs, newLoop.getBody()->getArguments());

  for (auto &bodyOp : loopOp.getBody()->getOperations()) {
    if (preProcessedPrivateLocals.contains(&bodyOp))
      continue;
    processOp(&bodyOp);
  }

  mapping.map(loopOp.getResults(), newLoop.getResults());
  rewriter.setInsertionPointAfter(newLoop);

  // Postpone the barrier when trailing work in this block still separates the
  // loop from the next reconvergence point.
  if (hasTrailingSideEffectSiblings(loopOp.getOperation()))
    deferredBarrierSeqLoops.push_back(loopOp.getOperation());
  else
    createBarrierAfterSeqLoop(loopOp.getOperation());
}

void ACCCGToGPULowering::flushDeferredBarriersBefore(Operation *beforeOp) {
  Block *block = beforeOp->getBlock();
  SmallVector<Operation *, 4> toFlush;
  for (Operation *loopOp : deferredBarrierSeqLoops)
    if (loopOp->getBlock() == block && loopOp->isBeforeInBlock(beforeOp))
      toFlush.push_back(loopOp);
  if (toFlush.empty())
    return;
  llvm::sort(toFlush,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *loopOp : toFlush)
    createBarrierAfterSeqLoop(loopOp);
  deferredBarrierSeqLoops.erase(
      std::remove_if(deferredBarrierSeqLoops.begin(),
                     deferredBarrierSeqLoops.end(),
                     [&](Operation *loopOp) {
                       return loopOp->getBlock() == block &&
                              loopOp->isBeforeInBlock(beforeOp);
                     }),
      deferredBarrierSeqLoops.end());
}

// try to process op as a loop mapped to a gpu parallelism
// failure signifies not a loop and needs different processing
void ACCCGToGPULowering::processParallelOp(scf::ParallelOp parallelOp) {
  LLVM_DEBUG(llvm::dbgs() << "processing par loop: ";
             parallelOp->print(llvm::dbgs()); llvm::dbgs() << "\n");
  assert(mlir::acc::hasParDimsAttr(parallelOp) &&
         "requires parallel dimensions attribute");
  mlir::acc::GPUParallelDimsAttr pDimsAttr =
      mlir::acc::getParDimsAttr(parallelOp);
  // both of these should be dealt with before compiler reaches ACCCGToGPU
  // Invalid parallel-loop structure should be rejected before acc-cg-to-gpu.
  assert(pDimsAttr.getArray().size() == 1 &&
         "expected a single par dim in acc-cg-to-gpu");
  assert(parallelOp.getInductionVars().size() == 1 &&
         "expected a single induction variable in acc-cg-to-gpu");

  mlir::acc::GPUParallelDimAttr parDim = pDimsAttr.getArray().front();

  bool savedGridStrideFlag = insideAccumulateGridStride;
  Value savedReductionBuf = reductionSharedBuf;
  if (parDim.isThreadX()) {
    bool found = false;
    parallelOp.getBody()->walk([&](acc::ReductionAccumulateOp accOp) {
      bool hasBlockDim = false;
      bool hasThreadDim = false;
      for (auto d : accOp.getParDims().getArray()) {
        if (d.isAnyBlock())
          hasBlockDim = true;
        if (d.isThreadX() || d.isThreadY())
          hasThreadDim = true;
      }
      if (hasThreadDim && !hasBlockDim) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (found)
      insideAccumulateGridStride = true;
  }

  // common loop body processing for both par0 and 1+
  auto processLoopBody = [&]() {
    // process inner ops recursively
    for (auto &bodyOp : parallelOp.getBody()->getOperations()) {
      if (bodyOp.hasTrait<OpTrait::IsTerminator>()) {
        // Non-cloned parallel loops reconverge at their terminator.
        flushDeferredBarriersBefore(&bodyOp);
        continue;
      }
      // Non-terminator body ops are processed recursively.
      processOp(&bodyOp);
    }
  };

  if (parDim.isSeq()) {
    LLVM_DEBUG(llvm::dbgs() << "loop: parDim: " << parDim << " as gpu seq\n");
    // Sequential loops here are remainder loops from partitioned parallel
    // loops - clone them as-is but process the body as a parallel region.
    // When blockDim.x >= subgroupSize and the loop contains a thread-level
    // accumulate, inactive grid-stride threads cannot participate in subgroup
    // reductions, so we use atomic-to-shared-memory reduction instead.
    bool needsAtomicReduction = false;
    bool hasAccumulateSibling = false;
    if (scf::ParallelOp parentPar =
            parallelOp->getParentOfType<scf::ParallelOp>()) {
      if (mlir::acc::GPUParallelDimsAttr parentDims =
              mlir::acc::getParDimsAttr(parentPar);
          parentDims && llvm::any_of(parentDims.getArray(),
                                     [](auto d) { return d.isThreadX(); })) {
        for (auto &op : parentPar.getBody()->getOperations()) {
          if (acc::ReductionAccumulateOp acc =
                  dyn_cast<acc::ReductionAccumulateOp>(op)) {
            bool hasBlockDim = false;
            bool hasThreadDim = false;
            for (auto d : acc.getParDims().getArray()) {
              if (d.isAnyBlock())
                hasBlockDim = true;
              if (d.isThreadX() || d.isThreadY())
                hasThreadDim = true;
            }
            if (hasThreadDim && !hasBlockDim)
              hasAccumulateSibling = true;
          }
        }
      }
    }
    if (insideAccumulateGridStride || hasAccumulateSibling) {
      for (auto launchArg : computeRegion.getLaunchArgs()) {
        if (acc::ParWidthOp pw = launchArg.getDefiningOp<acc::ParWidthOp>()) {
          if (pw.getParDim().isThreadX()) {
            if (auto cval = getConstantIntValue(pw.getLaunchArg()))
              needsAtomicReduction = (*cval >= options.subgroupSize);
            else
              needsAtomicReduction = true;
            break;
          }
        }
      }
    }
    if (needsAtomicReduction && !reductionSharedBuf) {
      Type elemTy;
      parallelOp.getBody()->walk([&](acc::ReductionAccumulateOp accOp) {
        Type t = accOp.getValue().getType();
        if (isa<FloatType, IntegerType>(t))
          elemTy = t;
        return elemTy ? WalkResult::interrupt() : WalkResult::advance();
      });
      if (!elemTy)
        needsAtomicReduction = false;
    }
    if (needsAtomicReduction && !reductionSharedBuf) {
      Location seqLoc = parallelOp->getLoc();
      gpu::AddressSpaceAttr workgroupAS = gpu::AddressSpaceAttr::get(
          computeRegion->getContext(),
          gpu::GPUDialect::getWorkgroupAddressSpace());
      Type elemTy;
      parallelOp.getBody()->walk([&](acc::ReductionAccumulateOp accOp) {
        Type t = accOp.getValue().getType();
        if (isa<FloatType, IntegerType>(t))
          elemTy = t;
        return elemTy ? WalkResult::interrupt() : WalkResult::advance();
      });
      assert(elemTy && "expected scalar reduction element type");
      unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
      MemRefType bufTy = MemRefType::get({options.subgroupSize}, elemTy,
                                         AffineMap{}, workgroupAS);
      reductionSharedBuf = acc::GPUSharedMemoryOp::create(
          rewriter, seqLoc, bufTy, rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(options.subgroupSize * elemBytes),
          ValueRange{}, IntegerAttr{}, IntegerAttr{});
      Value tidY = getThreadId(seqLoc, gpu::Dimension::y);
      Value identity;
      if (isa<FloatType>(elemTy)) {
        identity = arith::ConstantOp::create(
            rewriter, seqLoc, elemTy, rewriter.getFloatAttr(elemTy, 0.0));
      } else {
        identity = arith::ConstantIntOp::create(rewriter, seqLoc, elemTy, 0);
      }
      memref::StoreOp::create(rewriter, seqLoc, identity, reductionSharedBuf,
                              tidY);
      createPerRowBarrier(seqLoc);
    }
    processSeqLoop(parallelOp);
    loopReductions.push_back(parallelOp);
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "processing loop: parDim: " << parDim << " as gpu par\n");
    // actual parallel loops get their iv mapped to gpu hierarchy
    // and the loop construct is not cloned to gpu kernel only the
    // ops are cloned with mapping of gpu id for original loop iv
    Value gpuThreadId = getGPUThreadIdFor(parDim.getProcessor());
    mapping.map(parallelOp.getInductionVars()[0], gpuThreadId);

    processLoopBody();

    // Since the loop is not copied over, create dummy mappings for_each
    // of the loop results. These will ultimately by replaced with a
    // reduction
    llvm::for_each(parallelOp.getResults(), [&](Value v) {
      Type valTy = v.getType();
      TypedAttr zeroAttr = rewriter.getZeroAttr(valTy);
      auto zero = arith::ConstantOp::create(rewriter, parallelOp->getLoc(),
                                            valTy, zeroAttr);
      mapping.map(v, zero);
    });
    loopReductions.push_back(parallelOp);
  }
  insideAccumulateGridStride = savedGridStrideFlag;
  if (!insideAccumulateGridStride && !savedReductionBuf)
    reductionSharedBuf = Value();
}

/// Map an atomic RMW kind to the corresponding `gpu.all_reduce` operation.
static gpu::AllReduceOperation
getAllReduceOperation(arith::AtomicRMWKind kind) {
  switch (kind) {
  case arith::AtomicRMWKind::addf:
  case arith::AtomicRMWKind::addi:
    return gpu::AllReduceOperation::ADD;
  case arith::AtomicRMWKind::mulf:
  case arith::AtomicRMWKind::muli:
    return gpu::AllReduceOperation::MUL;
  case arith::AtomicRMWKind::minu:
    return gpu::AllReduceOperation::MINUI;
  case arith::AtomicRMWKind::mins:
    return gpu::AllReduceOperation::MINSI;
  case arith::AtomicRMWKind::minnumf:
    return gpu::AllReduceOperation::MINNUMF;
  case arith::AtomicRMWKind::maxu:
    return gpu::AllReduceOperation::MAXUI;
  case arith::AtomicRMWKind::maxs:
    return gpu::AllReduceOperation::MAXSI;
  case arith::AtomicRMWKind::maxnumf:
    return gpu::AllReduceOperation::MAXNUMF;
  case arith::AtomicRMWKind::ori:
    return gpu::AllReduceOperation::OR;
  case arith::AtomicRMWKind::andi:
    return gpu::AllReduceOperation::AND;
  case arith::AtomicRMWKind::xori:
    return gpu::AllReduceOperation::XOR;
  case arith::AtomicRMWKind::minimumf:
    return gpu::AllReduceOperation::MINIMUMF;
  case arith::AtomicRMWKind::maximumf:
    return gpu::AllReduceOperation::MAXIMUMF;
  case arith::AtomicRMWKind::assign:
    break;
  }
  llvm_unreachable("unsupported atomic kind");
}

void ACCCGToGPULowering::constructAtomicAccumulation(
    Location loc, Value memref, ValueRange indices, Value input,
    arith::AtomicRMWKind kind) {
  assert(!memref.getDefiningOp<memref::AllocaOp>() &&
         "cannot lower atomic accumulation on an stack variable");

  // acc.atomic.update derives the element address from the memref descriptor's
  // base pointer and offset field; it has no subscript operand. When the store
  // being lowered targets a specific array element (e.g. result(idx) =
  // max(...)), fold the indices into the descriptor offset with a subview so
  // the atomic updates the intended element. Otherwise the atomic always hits
  // element 0, so any reduction whose destination index is non-zero is
  // miscompiled (the result lands in element 0 while the intended element keeps
  // its identity-init value).
  Value target = memref;
  if (!indices.empty()) {
    MemRefType memrefTy = cast<MemRefType>(memref.getType());
    unsigned rank = memrefTy.getRank();
    assert(indices.size() == rank && "expected one index per memref dimension");
    SmallVector<OpFoldResult> offsets(indices.begin(), indices.end());
    SmallVector<OpFoldResult> sizes(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
    target = memref::SubViewOp::create(rewriter, loc, memref, offsets, sizes,
                                       strides);
  }

  auto atomicUpdateOp =
      acc::AtomicUpdateOp::create(rewriter, loc, target, /*ifCond=*/Value());
  Region &region = atomicUpdateOp->getRegion(0);
  Block *block =
      rewriter.createBlock(&region, region.begin(), {input.getType()}, {loc});
  rewriter.setInsertionPointToStart(block);
  Value reductionExpr =
      generateReductionOp(rewriter, loc, input, block->getArgument(0), kind);
  acc::YieldOp::create(rewriter, loc, reductionExpr);
  rewriter.setInsertionPointAfter(atomicUpdateOp);
}

void ACCCGToGPULowering::createGPUAllReduceOp(
    Location loc, Value input, Value memref, arith::AtomicRMWKind kind,
    mlir::acc::GPUParallelDimsAttr parDimsAttr, ValueRange indices) {
  gpu::AllReduceOperationAttr attr = gpu::AllReduceOperationAttr::get(
      computeRegion->getContext(), getAllReduceOperation(kind));
  auto allReduceOp = gpu::AllReduceOp::create(rewriter, loc, input, attr, true);
  mlir::acc::setParDimsAttr(allReduceOp, parDimsAttr);
  // Predicate the store on the thread-level dimensions being reduced so that
  // only one thread per reduced group writes the result. Only dimensions in
  // parDimsAttr are included; sweeping over all dimensions between the highest
  // par_dim and thread_x would incorrectly add unrelated dimensions (e.g.
  // thread_y for a thread_x-only reduction), preventing other rows from
  // storing their independent results.
  SmallVector<mlir::acc::GPUParallelDimAttr> inactiveParDims;
  MLIRContext *ctx = computeRegion->getContext();
  bool hasThreadX = false;
  for (auto parDim : parDimsAttr.getArray()) {
    if (parDim.isAnyBlock())
      continue;
    if (parDim.isThreadX())
      hasThreadX = true;
    if (computeRegion.getLaunchArg(parDim) ||
        isInsideACCSpecializedRoutine(computeRegion)) {
      inactiveParDims.push_back(parDim);
    }
  }
  // Subgroup alignment may introduce extra ThreadX lanes even when ThreadX is
  // not part of the reduction. Predicate on ThreadX so only one lane stores.
  if (!hasThreadX)
    inactiveParDims.push_back(mlir::acc::GPUParallelDimAttr::threadXDim(ctx));
  Value predicate = emitPredicate(loc, inactiveParDims);
  // Predication is only needed when the store target is visible to
  // multiple threads (shared/global memory). Per-thread targets like
  // memref.alloca are thread-private: gpu.all_reduce returns the same
  // value on all threads, so each can safely store to its own copy.
  // Detect per-thread storage by walking through conversion ops to
  // find the underlying allocation.
  bool isPerThreadPrivate = isa_and_nonnull<memref::AllocaOp>(
      unwrapMemRefConversion(memref).getDefiningOp());
  // combine
  scf::IfOp ifOp;
  if (predicate && !isPerThreadPrivate) {
    ifOp =
        scf::IfOp::create(rewriter, loc, predicate, /*withElseRegion=*/false);
    Region &thenRegion = ifOp.getThenRegion();
    Block &thenBlock = thenRegion.back();
    rewriter.setInsertionPoint(thenBlock.getTerminator());
  }
  memref::StoreOp::create(rewriter, loc, allReduceOp, memref, indices);
  if (predicate && !isPerThreadPrivate)
    rewriter.setInsertionPointAfter(ifOp);
  // A later block combine reuses this instead of reloading the slot. This only
  // applies to scalar accumulators; array elements are indexed individually.
  if (indices.empty())
    reductionAccumValue[memref] = allReduceOp;
}

void ACCCGToGPULowering::postprocessAccumulateOp(
    acc::ReductionAccumulateOp op) {
  Location loc = op->getLoc();

  rewriter.setInsertionPoint(op);

  // Check whether this accumulate has only block-level par dims (no thread
  // dims).  gpu.all_reduce reduces across threads within a block, which is
  // wrong for block-only reductions - the loop result is already per-thread
  // and only needs a predicated store to the memref.
  bool hasThreadDim = false;
  SmallVector<mlir::acc::GPUParallelDimAttr> threadParDims;
  for (auto parDim : op.getParDims().getArray()) {
    if (!parDim.isAnyBlock()) {
      hasThreadDim = true;
      threadParDims.push_back(parDim);
    }
  }

  std::optional<arith::AtomicRMWKind> kind;
  if (hasThreadDim) {
    FailureOr<arith::AtomicRMWKind> kindOr = getReductionKind(
        op.getReductionOperator(), op.getValue().getType(), loc);
    if (failed(kindOr))
      return;
    kind = *kindOr;
  }

  if (hasThreadDim && reductionSharedBuf &&
      op.getValue().getType() ==
          cast<MemRefType>(reductionSharedBuf.getType()).getElementType()) {
    Value val = op.getValue();
    Value mem = op.getMemref();
    Value tidY = getThreadId(loc, gpu::Dimension::y);
    memref::AtomicRMWOp::create(rewriter, loc, *kind, val, reductionSharedBuf,
                                ValueRange{tidY});
    createPerRowBarrier(loc);
    Value result =
        memref::LoadOp::create(rewriter, loc, reductionSharedBuf, tidY);
    memref::StoreOp::create(rewriter, loc, result, mem);
    reductionAccumValue[mem] = result;
  } else if (hasThreadDim) {
    createGPUAllReduceOp(loc, op.getValue(), op.getMemref(), *kind,
                         op.getParDims());
  } else {
    // Block-only: no gpu.all_reduce needed (all threads have the same
    // value after broadcast).  Just store the value to the memref.
    // Predicate on all thread dims when the target is shared memory.
    Value val = mapping.lookupOrDefault(op.getValue());
    Value mem = mapping.lookupOrDefault(op.getMemref());
    bool isPerThreadPrivate = isa_and_nonnull<memref::AllocaOp>(
        unwrapMemRefConversion(mem).getDefiningOp());
    if (!isPerThreadPrivate) {
      SmallVector<mlir::acc::GPUParallelDimAttr> predDims;
      for (auto parDim : computeRegion.getLaunchParDims())
        if (!parDim.isAnyBlock())
          predDims.push_back(parDim);
      if (predDims.empty()) {
        predDims.push_back(mlir::acc::GPUParallelDimAttr::threadXDim(
            computeRegion->getContext()));
      }
      Value predicate = emitPredicate(loc, predDims);
      auto ifOp =
          scf::IfOp::create(rewriter, loc, predicate, /*withElseRegion=*/false);
      rewriter.setInsertionPoint(ifOp.getThenRegion().back().getTerminator());
      memref::StoreOp::create(rewriter, loc, val, mem);
      rewriter.setInsertionPointAfter(ifOp);
    } else {
      memref::StoreOp::create(rewriter, loc, val, mem);
    }
  }

  // erase acc.reduction_accumulate
  rewriter.eraseOp(op);
}

void ACCCGToGPULowering::postprocessLoopReduction(scf::ParallelOp parLoop) {
  if (parLoop.getNumReductions() == 0)
    return;

  for (unsigned i = 0; i < parLoop.getNumResults(); ++i) {
    for (Operation *user :
         mapping.lookupOrDefault(parLoop.getResult(i)).getUsers()) {
      if (acc::ReductionAccumulateOp accumulateOp =
              dyn_cast<acc::ReductionAccumulateOp>(user)) {
        postprocessAccumulateOp(accumulateOp);
      }
    }
  }
}

void ACCCGToGPULowering::processExecuteRegion(scf::ExecuteRegionOp op) {
  LLVM_DEBUG(llvm::dbgs() << "processing execute region op: ";
             op->print(llvm::dbgs()); llvm::dbgs() << "\n");
  Location loc = op->getLoc();
  auto types = op.getResultTypes();
  Region &oldRegion = op.getRegion();
  // create the executeRegion op inside gpu launch
  auto executeRegionOp = scf::ExecuteRegionOp::create(rewriter, loc, types);
  Region &region = executeRegionOp.getRegion();
  rewriter.createBlock(&region);
  rewriter.setInsertionPointToEnd(&region.front());

  llvm::DenseMap<Block *, Block *> blockMap;
  blockMap[&oldRegion.front()] = &region.front();

  // Create blocks in the new operation corresponding to all blocks in the
  // original op
  for (auto &oldBlock : llvm::drop_begin(oldRegion.getBlocks())) {
    TypeRange argTypes = oldBlock.getArgumentTypes();
    size_t numArgs = argTypes.size();
    // Create new block with same argument types
    Block *newBlock = rewriter.createBlock(&region, region.end(), argTypes,
                                           SmallVector<Location>(numArgs, loc));
    blockMap[&oldBlock] = newBlock;
    // Map block arguments
    mapping.map(oldBlock.getArguments(), newBlock->getArguments());
  }

  // Iterate over all blocks of oldRegion and all operations inside them
  // process all the ops except the terminator
  for (auto [oldBlock, newBlock] :
       llvm::zip(oldRegion.getBlocks(), region.getBlocks())) {
    OpBuilder::InsertionGuard blockGuard(rewriter);
    rewriter.setInsertionPointToStart(&newBlock);
    for (auto &bodyOp : oldBlock.getOperations()) {
      // Skip terminators during normal iteration - handle them separately
      if (bodyOp.hasTrait<OpTrait::IsTerminator>())
        continue;
      processOp(&bodyOp);
    }

    // Copy the terminator from old block to new block
    Operation *oldTerminator = oldBlock.getTerminator();
    rewriter.setInsertionPointToEnd(&newBlock);
    Operation *newTerminator = rewriter.clone(*oldTerminator, mapping);

    // Replace successors with mapped blocks
    for (unsigned i = 0; i < oldTerminator->getNumSuccessors(); ++i) {
      Block *oldDest = oldTerminator->getSuccessor(i);
      Block *newDest = blockMap.lookup(oldDest);
      assert(newDest && "Successor block must be in blockMap");
      newTerminator->setSuccessor(newDest, i);
    }
  }
  mapping.map(op->getResults(), executeRegionOp->getResults());
  rewriter.setInsertionPointAfter(executeRegionOp);
}

void ACCCGToGPULowering::processAccumulateOp(acc::ReductionAccumulateOp op) {
  LLVM_DEBUG(llvm::dbgs() << "processing accumulate op: " << *op << "\n");
  Value accumulateValue = op.getValue();
  if (reductionSharedBuf &&
      mapping.lookupOrDefault(accumulateValue).getType() ==
          cast<MemRefType>(reductionSharedBuf.getType()).getElementType()) {
    Location loc = op->getLoc();
    FailureOr<arith::AtomicRMWKind> kind = getReductionKind(
        op.getReductionOperator(), accumulateValue.getType(), loc);
    if (failed(kind))
      return;
    Value mappedValue = mapping.lookupOrDefault(accumulateValue);
    Value memref = mapping.lookupOrDefault(op.getMemref());
    Value tidY = getThreadId(loc, gpu::Dimension::y);
    memref::AtomicRMWOp::create(rewriter, loc, *kind, mappedValue,
                                reductionSharedBuf, ValueRange{tidY});
    createPerRowBarrier(loc);
    Value result =
        memref::LoadOp::create(rewriter, loc, reductionSharedBuf, tidY);
    memref::StoreOp::create(rewriter, loc, result, memref);
    reductionAccumValue[memref] = result;
    return;
  }
  if (accumulateValue.getDefiningOp<scf::ParallelOp>()) {
    Operation *newOp = rewriter.clone(*op, mapping);
    mapping.map(op->getResults(), newOp->getResults());
  } else if (isRedundantChainAccumulate(op)) {
    // The destination memref already holds the correctly aggregated value
    // (atomically reduced by a preceding acc.reduction_combine with a block
    // par_dim). Skip the redundant gpu.all_reduce + atomic.update; emit a
    // Workgroup-wide barrier so downstream readers see all prior atomic
    // updates.
    LLVM_DEBUG(llvm::dbgs() << "  skipped: redundant chain accumulate\n");
    gpu::BarrierOp::create(rewriter, op->getLoc());
  } else {
    Value mappedValue = mapping.lookupOrDefault(accumulateValue);
    Value memref = mapping.lookupOrDefault(op.getMemref());
    FailureOr<arith::AtomicRMWKind> kind = getReductionKind(
        op.getReductionOperator(), accumulateValue.getType(), op.getLoc());
    if (failed(kind))
      return;
    createGPUAllReduceOp(op->getLoc(), mappedValue, memref, *kind,
                         op.getParDims());
  }
}

void ACCCGToGPULowering::processAccumulateArrayOp(
    acc::ReductionAccumulateArrayOp op) {
  LLVM_DEBUG(llvm::dbgs() << "processing accumulate array op: " << *op << "\n");
  Location loc = op.getLoc();

  Value memref = mapping.lookupOrDefault(op.getMemref());
  MemRefType memrefTy = dyn_cast<MemRefType>(memref.getType());
  assert(memrefTy && memrefTy.getRank() == 1 &&
         "array reduction accumulate expects a rank-1 memref");

  FailureOr<arith::AtomicRMWKind> kindOr = getReductionKind(
      op.getReductionOperator(), memrefTy.getElementType(), loc);
  if (failed(kindOr))
    return;
  arith::AtomicRMWKind kind = *kindOr;

  // The (already mapped/cloned) acc.bounds op describes the element range; it
  // is dead after lowering since we read its operands directly.
  acc::DataBoundsOp boundsOp = mapping.lookupOrDefault(op.getBounds())
                                   .getDefiningOp<acc::DataBoundsOp>();
  assert(boundsOp && "expected acc.bounds defining op for array accumulate");
  auto eraseDeadBounds = [&] {
    if (boundsOp->use_empty())
      rewriter.eraseOp(boundsOp);
  };

  bool hasThreadDim = false;
  bool hasBlockDim = false;
  for (auto pd : op.getParDims().getArray()) {
    hasThreadDim |= pd.isAnyThread();
    hasBlockDim |= pd.isAnyBlock();
  }

  // Block-only (gang) reduction: each element is produced by one gang, so the
  // per-gang copy already holds the result and the combine does the rest.
  if (hasBlockDim && !hasThreadDim) {
    eraseDeadBounds();
    return;
  }

  // A thread-level reduction with no block owner for its elements cannot merge
  // the cross-thread partials, so report NYI.
  if (!reductionHasBlockContext(op)) {
    (void)accSupport.emitNYI(
        loc, "reduction: thread-only array reduction accumulate");
    return;
  }

  // Per-element gpu.all_reduce is only correct for a per-thread alloca; mirror
  // processPrivateLocal's stack-fit decision rather than inspect the memref.
  bool isPerThreadPrivate =
      canUseStackAlloca(memrefTy, loc, options.maxThreadPrivateStack);
  if (!isPerThreadPrivate) {
    // Block-shared accumulator: no-op only when the accumulate spans a block
    // dim (threads distribute distinct elements, so the block partial is in
    // place and the atomic combine finishes it). A thread-only shared
    // reduction, where several threads reduce into the same element, is not yet
    // supported.
    if (hasBlockDim) {
      eraseDeadBounds();
    } else {
      (void)accSupport.emitNYI(
          loc, "reduction: shared-memory array reduction accumulate");
    }
    return;
  }

  // Bounds are normalized to be zero-based.
  auto toIndex = [&](Value v) -> Value {
    if (v.getType().isIndex())
      return v;
    return arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                      v);
  };

  Value lb = boundsOp.getLowerbound()
                 ? toIndex(boundsOp.getLowerbound())
                 : arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value step = boundsOp.getStride()
                   ? toIndex(boundsOp.getStride())
                   : arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  // Exclusive upper bound: prefer extent (count of elements), fall back to the
  // inclusive upperbound.
  Value ub;
  if (boundsOp.getExtent()) {
    ub =
        arith::AddIOp::create(rewriter, loc, lb, toIndex(boundsOp.getExtent()));
  } else {
    assert(boundsOp.getUpperbound() &&
           "acc.bounds must specify an extent or upperbound");
    ub = arith::AddIOp::create(rewriter, loc, toIndex(boundsOp.getUpperbound()),
                               one);
  }

  // Reduce each array element across the requested parallel dimensions.
  auto forOp = scf::ForOp::create(rewriter, loc, lb, ub, step);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(forOp.getBody()->getTerminator());
    Value iv = forOp.getInductionVar();
    Value elem = memref::LoadOp::create(rewriter, loc, memref, ValueRange{iv});
    createGPUAllReduceOp(loc, elem, memref, kind, op.getParDims(),
                         ValueRange{iv});
  }

  eraseDeadBounds();
}

void ACCCGToGPULowering::processReductionOp(acc::ReductionInitOp op) {
  // Clone the inner ops of the reduction op only
  op.getRegion().walk<WalkOrder::PreOrder>([&](Operation *innerOp) {
    if (acc::YieldOp yieldOp = dyn_cast<acc::YieldOp>(innerOp)) {
      op.getResult().replaceAllUsesWith(mapping.lookup(yieldOp.getOperand(0)));
      return WalkResult::interrupt();
    }
    if (innerOp->getNumRegions() > 0) {
      processOp(innerOp);
      return WalkResult::skip();
    }
    rewriter.clone(*innerOp, mapping);
    return WalkResult::advance();
  });
}

void ACCCGToGPULowering::processReductionCombineOp(acc::ReductionCombineOp op) {
  LLVM_DEBUG(llvm::dbgs() << "processing reduction combine op: ";
             op->print(llvm::dbgs()); llvm::dbgs() << "\n");
  Location loc = op.getLoc();
  MemRefType memrefType = dyn_cast<MemRefType>(op.getSrcMemref().getType());
  assert(memrefType && "expected memref type for reduction combine op");
  assert(memrefType.getRank() == 0 &&
         "expected scalar memref type for reduction combine op");
  Type elTy = memrefType.getElementType();
  FailureOr<arith::AtomicRMWKind> kindOr =
      getReductionKind(op.getReductionOperator(), elTy, loc);
  if (failed(kindOr))
    return;
  arith::AtomicRMWKind kind = *kindOr;

  Value srcMemref = mapping.lookupOrDefault(op.getSrcMemref());
  Value destMemref = mapping.lookupOrDefault(op.getDestMemref());

  // A block par_dim normally means the accumulator is shared across blocks and
  // must be updated atomically. But when the destination resolves to a
  // thread-private stack alloca (e.g. an inner-loop reduction combining one
  // private accumulator into another private accumulator for the same thread),
  // the update is not visible to other threads and must not be atomic. Treat
  // such destinations as a plain load/combine/store below.
  bool destIsPerThreadPrivate = isa_and_nonnull<memref::AllocaOp>(
      unwrapMemRefConversion(destMemref).getDefiningOp());

  SmallVector<mlir::acc::GPUParallelDimAttr> parDims =
      getReductionCombineParDims(op);
  for (auto parDim : parDims) {
    if (parDim.isAnyBlock() && !destIsPerThreadPrivate) {
      // Block reduction directly stores to the accumulator using atomic.
      // The predication (tid.x == 0 when subgroup-aligned) is already handled
      // by the parent predicate_region processing.
      // Reloading a grid-shared slot races with other blocks; record
      // it and replace with the block-reduced register value in the fixup.
      auto srcLoad = memref::LoadOp::create(rewriter, loc, srcMemref);
      pendingCombineReloads.push_back({srcMemref, srcLoad});
      constructAtomicAccumulation(loc, destMemref, /*indices=*/{}, srcLoad,
                                  kind);
      return;
    }
  }

  // Atomic construction is not needed; lower this operation to typical
  // reduction update operations. E.g. dest = dest <kind> src
  auto srcLoad = memref::LoadOp::create(rewriter, loc, srcMemref, ValueRange{});
  auto destLoad =
      memref::LoadOp::create(rewriter, loc, destMemref, ValueRange{});
  Value combine = generateReductionOp(rewriter, loc, srcLoad, destLoad, kind);
  memref::StoreOp::create(rewriter, loc, combine, destMemref, ValueRange{});
}

void ACCCGToGPULowering::processCombineRegionOp(
    acc::ReductionCombineRegionOp op) {
  LLVM_DEBUG(llvm::dbgs() << "processing combine region op: ";
             op->print(llvm::dbgs()); llvm::dbgs() << "\n");
  // A block par_dim on a combine into a thread-private stack alloca is not a
  // real cross-block accumulation (the alloca is not shared across blocks), so
  // it must use a plain load/combine/store rather than an atomic update.
  bool destIsPerThreadPrivate = isa_and_nonnull<memref::AllocaOp>(
      unwrapMemRefConversion(mapping.lookupOrDefault(op.getDestVar()))
          .getDefiningOp());
  SmallVector<mlir::acc::GPUParallelDimAttr> parDims =
      getReductionCombineParDims(op);
  for (auto parDim : parDims) {
    if (parDim.isAnyBlock() && !destIsPerThreadPrivate) {
      // Block reduction directly stores to the accumulator using atomic.
      // The predication (tid.x == 0 when subgroup-aligned) is already handled
      // by the parent predicate_region processing.
      for (Operation *user : op.getSrcVar().getUsers()) {
        if (acc::ReductionAccumulateOp accumulateOp =
                dyn_cast<acc::ReductionAccumulateOp>(user)) {
          Location loc = accumulateOp.getLoc();
          FailureOr<arith::AtomicRMWKind> kind =
              getReductionKind(accumulateOp.getReductionOperator(),
                               accumulateOp.getValue().getType(), loc);
          if (failed(kind))
            return;
          Value srcMemref = mapping.lookupOrDefault(accumulateOp.getMemref());
          // Recorded and patched in the fixup to avoid the reload race.
          auto reductionLoad = memref::LoadOp::create(rewriter, loc, srcMemref);
          pendingCombineReloads.push_back({srcMemref, reductionLoad});
          constructAtomicAccumulation(loc,
                                      mapping.lookupOrDefault(op.getDestVar()),
                                      /*indices=*/{}, reductionLoad, *kind);
          return;
        }
      }
      // For decomposed complex reductions, the AccumulateOp was replaced
      // with real/imag AccumulateOps. Load from the private memref which
      // holds the reconstructed complex value.
      Value privateMemref = mapping.lookupOrDefault(op.getSrcVar());
      MemRefType memrefTy = cast<MemRefType>(privateMemref.getType());
      if (isa<ComplexType>(memrefTy.getElementType())) {
        Location loc = op.getLoc();
        Value reductionResult =
            memref::LoadOp::create(rewriter, loc, privateMemref);
        arith::AtomicRMWKind kind = arith::AtomicRMWKind::addf;
        op.getRegion().walk([&](Operation *innerOp) {
          if (isa<complex::MulOp>(innerOp))
            kind = arith::AtomicRMWKind::mulf;
        });
        constructAtomicAccumulation(loc,
                                    mapping.lookupOrDefault(op.getDestVar()),
                                    /*indices=*/{}, reductionResult, kind);
        return;
      }
    }
  }
  op.getRegion().walk<WalkOrder::PreOrder>([&](Operation *innerOp) {
    if (acc::YieldOp yieldOp = dyn_cast<acc::YieldOp>(innerOp))
      return WalkResult::interrupt();
    if (innerOp->getNumRegions() > 0) {
      processOp(innerOp);
      return WalkResult::skip();
    }
    rewriter.clone(*innerOp, mapping);
    return WalkResult::advance();
  });
}

void ACCCGToGPULowering::processGenericOp(Operation *op) {
  // Operations with no regions or operations for which we know
  // no recursive processing is needed can be fully cloned.
  LLVM_DEBUG(llvm::dbgs() << "processing generic op, cloning: ";
             op->print(llvm::dbgs()); llvm::dbgs() << "\n");
  Operation *newOp = rewriter.clone(*op, mapping);
  // update mapping as cloning creates different result values
  mapping.map(op->getResults(), newOp->getResults());
}

void ACCCGToGPULowering::processGenericOpWithRegions(Operation *op) {
  // Generic handling for operations with regions
  LLVM_DEBUG(llvm::dbgs() << "processing generic op with regions: ";
             op->print(llvm::dbgs()); llvm::dbgs() << "\n");

  // Clone the operation structure without its regions
  Operation *newOp = rewriter.cloneWithoutRegions(*op, mapping);

  // Process each region recursively
  for (auto [oldRegion, newRegion] :
       llvm::zip(op->getRegions(), newOp->getRegions())) {
    // Create blocks in the new region corresponding to old region blocks
    for (auto &oldBlock : oldRegion.getBlocks()) {
      TypeRange argTypes = oldBlock.getArgumentTypes();
      size_t numArgs = argTypes.size();
      // Create new block with same argument types
      Block *newBlock =
          rewriter.createBlock(&newRegion, newRegion.end(), argTypes,
                               SmallVector<Location>(numArgs, op->getLoc()));

      // Map block arguments
      mapping.map(oldBlock.getArguments(), newBlock->getArguments());

      // Process each operation in the block
      for (auto &innerOp : oldBlock.getOperations()) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToEnd(newBlock);
        processOp(&innerOp);
      }
    }
  }
  rewriter.setInsertionPointAfter(newOp);

  // Update mapping for results
  mapping.map(op->getResults(), newOp->getResults());
}

// thread through par dim to verify redundant/0 execution modes
void ACCCGToGPULowering::processOp(Operation *op) {
  if (isDeferredBarrierFlushPoint(op))
    flushDeferredBarriersBefore(op);
  if (mlir::acc::hasParDimsAttr(op) && isa<scf::ParallelOp>(op)) {
    // parallel loops require special processing based on parallel dimension
    // this is mutually recursive with processOp
    scf::ParallelOp parallelOp = cast<scf::ParallelOp>(op);
    processParallelOp(parallelOp);
  } else if (scf::ForOp seqLoop = dyn_cast<scf::ForOp>(op)) {
    processSeqLoop(seqLoop);
  } else if (acc::PrivatizeOp privatize = dyn_cast<acc::PrivatizeOp>(op)) {
    processPrivatize(privatize);
  } else if (acc::PrivateLocalOp privateLocal =
                 dyn_cast<acc::PrivateLocalOp>(op)) {
    processPrivateLocal(privateLocal);
  } else if (acc::PredicateRegionOp predicateRegionOp =
                 dyn_cast<acc::PredicateRegionOp>(op)) {
    processPredicateRegion(predicateRegionOp);
  } else if (acc::ReductionAccumulateOp accumulateOp =
                 dyn_cast<acc::ReductionAccumulateOp>(op)) {
    processAccumulateOp(accumulateOp);
  } else if (auto accumulateArrayOp =
                 dyn_cast<acc::ReductionAccumulateArrayOp>(op)) {
    processAccumulateArrayOp(accumulateArrayOp);
  } else if (acc::ReductionInitOp reductionInitOp =
                 dyn_cast<acc::ReductionInitOp>(op)) {
    processReductionOp(reductionInitOp);
  } else if (acc::ReductionCombineOp reductionCombineOp =
                 dyn_cast<acc::ReductionCombineOp>(op)) {
    processReductionCombineOp(reductionCombineOp);
  } else if (auto combineRegionOp =
                 dyn_cast<acc::ReductionCombineRegionOp>(op)) {
    processCombineRegionOp(combineRegionOp);
  } else if (acc::ReductionOp accReductionOp = dyn_cast<acc::ReductionOp>(op)) {
    mapping.map(accReductionOp->getResult(0), accReductionOp.getVarPtr());
  } else if (mapping.contains(op)) {
    // do nothing, operation in mapping signals it is already taken care of
    LLVM_DEBUG(llvm::dbgs() << "skipping mapped op: " << *op << "\n");
  } else if (isa<acc::YieldOp>(op)) {
    for (auto [operand, result] :
         llvm::zip(op->getOperands(), op->getParentOp()->getResults())) {
      result.replaceAllUsesWith(mapping.lookup(operand));
    }
  } else if (isa<scf::ExecuteRegionOp>(op)) {
    processExecuteRegion(cast<scf::ExecuteRegionOp>(op));
  } else if (op->getNumRegions() == 0 ||
             isa<acc::OpenACCDialect>(op->getDialect())) {
    processGenericOp(op);
  } else {
    processGenericOpWithRegions(op);
  }
}

/// Fold `acc.par_width` to its launch operand or constant one.
class RemoveParWidth : public OpRewritePattern<acc::ParWidthOp> {
  using OpRewritePattern<acc::ParWidthOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(acc::ParWidthOp op,
                                PatternRewriter &rewriter) const override {
    if (Value launchArg = op.getLaunchArg()) {
      rewriter.replaceOp(op, launchArg);
    } else {
      Value one = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
      rewriter.replaceOp(op, one);
    }
    return success();
  }
};

/// Rewrite pattern that lowers `acc.compute_region` via ACCCGToGPULowering.
class ACCComputeRegionToGPUPattern
    : public OpRewritePattern<acc::ComputeRegionOp> {
public:
  ACCComputeRegionToGPUPattern(MLIRContext *context,
                               acc::OpenACCSupport &accSupport,
                               const ACCCGToGPUOptions &options)
      : OpRewritePattern<acc::ComputeRegionOp>(context), accSupport(accSupport),
        options(options) {}

  LogicalResult matchAndRewrite(acc::ComputeRegionOp op,
                                PatternRewriter &rewriter) const override {
    ACCCGToGPULowering kernelOpRewriter(op, rewriter, accSupport, options);
    return kernelOpRewriter.rewrite();
  }

private:
  acc::OpenACCSupport &accSupport;
  const ACCCGToGPUOptions &options;
};

class ACCCGToGPU : public acc::impl::ACCCGToGPUBase<ACCCGToGPU> {
public:
  using acc::impl::ACCCGToGPUBase<ACCCGToGPU>::ACCCGToGPUBase;

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();

    assert(deviceType != mlir::acc::DeviceType::Host &&
           deviceType != mlir::acc::DeviceType::Multicore &&
           "ACCCGToGPU only supports GPU device types");
    ACCCGToGPUOptions options;
    options.deviceType = deviceType;
    options.maxWorkgroupSharedMemory = maxWorkgroupSharedMemory;
    options.maxThreadPrivateStack = maxThreadPrivateStack;
    options.subgroupSize = subgroupSize;

    // Try to get cached parent analysis first, fall back to local analysis.
    std::optional<std::reference_wrapper<acc::OpenACCSupport>> cachedAnalysis =
        getCachedParentAnalysis<acc::OpenACCSupport>(funcOp->getParentOp());
    acc::OpenACCSupport &accSupport = cachedAnalysis
                                          ? cachedAnalysis->get()
                                          : getAnalysis<acc::OpenACCSupport>();

    RewritePatternSet patterns(context);
    patterns.insert<ACCComputeRegionToGPUPattern>(context, accSupport, options);
    patterns.insert<RemoveParWidth>(context);
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<acc::ComputeRegionOp, acc::ParWidthOp>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
