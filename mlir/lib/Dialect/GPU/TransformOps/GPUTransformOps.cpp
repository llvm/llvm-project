//===- GPUTransformOps.cpp - Implementation of GPU transform ops ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.h"

#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/TransformOps/Utils.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/DeviceMappingInterface.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::transform;
using namespace mlir::transform::gpu;

#define DEBUG_TYPE "gpu-transforms"
#define DEBUG_TYPE_ALIAS "gpu-transforms-alias"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")
#define DBGS_ALIAS() (llvm::dbgs() << '[' << DEBUG_TYPE_ALIAS << "] ")

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyGPUToNVVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  // NVVM uses alloca in the default address space to represent private
  // memory allocations, so drop private annotations. NVVM uses address
  // space 3 for shared memory. NVVM uses the default address space to
  // represent global memory.
  // Used in populateGpuToNVVMConversionPatternsso attaching here for now.
  // TODO: We should have a single to_nvvm_type_converter.
  populateGpuMemorySpaceAttributeConversions(
      llvmTypeConverter, [](AddressSpace space) -> unsigned {
        switch (space) {
        case AddressSpace::Global:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kGlobalMemorySpace);
        case AddressSpace::Workgroup:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kSharedMemorySpace);
        case AddressSpace::Private:
          return 0;
        }
        llvm_unreachable("unknown address space enum value");
        return 0;
      });
  // Used in GPUToNVVM/WmmaOpsToNvvm.cpp so attaching here for now.
  // TODO: We should have a single to_nvvm_type_converter.
  llvmTypeConverter.addConversion(
      [&](MMAMatrixType type) -> Type { return convertMMAToLLVMType(type); });
  populateGpuToNVVMConversionPatterns(llvmTypeConverter, patterns);
}

LogicalResult
transform::ApplyGPUToNVVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

void transform::ApplyGPUWwmaToNVVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  populateGpuWMMAToNVVMConversionPatterns(llvmTypeConverter, patterns);
}

LogicalResult
transform::ApplyGPUWwmaToNVVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

void transform::ApplyGPUSubgroupReduceToNVVMConversionPatternsOp::
    populatePatterns(TypeConverter &typeConverter,
                     RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  populateGpuSubgroupReduceOpLoweringPattern(llvmTypeConverter, patterns);
}

LogicalResult transform::ApplyGPUSubgroupReduceToNVVMConversionPatternsOp::
    verifyTypeConverter(transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//s

void ApplyGPURewritePatternsOp::populatePatterns(RewritePatternSet &patterns) {
  populateGpuRewritePatterns(patterns);
}

//===----------------------------------------------------------------------===//
// ApplyUnrollVectorsSubgroupMmaOp
//===----------------------------------------------------------------------===//

/// Pick an unrolling order that will allow tensorcore operation to reuse LHS
/// register.
static std::optional<SmallVector<int64_t>>
gpuMmaUnrollOrder(vector::ContractionOp contract) {
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) {
      order.push_back(index);
    }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(expr.cast<AffineDimExpr>().getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) {
      order.push_back(index);
    }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) {
      order.push_back(index);
    }
  }
  return order;
}

/// Returns the target vector size for the target operation based on the native
/// vector size specified with `m`, `n`, and `k`.
static std::optional<SmallVector<int64_t>>
getSubgroupMmaNativeVectorSize(Operation *op, int64_t m, int64_t n, int64_t k) {
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    int64_t contractRank = contract.getIteratorTypes().size();
    if (contractRank < 3)
      return std::nullopt;
    SmallVector<int64_t> nativeSize(contractRank - 3, 1);
    nativeSize.append({m, n, k});
    return nativeSize;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    int64_t writeRank = writeOp.getVectorType().getRank();
    if (writeRank < 2)
      return std::nullopt;
    SmallVector<int64_t> nativeSize(writeRank - 2, 1);
    nativeSize.append({m, n});
    return nativeSize;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Transfer read ops may need different shapes based on how they are being
    // used. For simplicity just match the shape used by the extract strided op.
    VectorType sliceType;
    for (Operation *users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract)
        return std::nullopt;
      auto vecType = extract.getResult().getType().cast<VectorType>();
      if (sliceType && sliceType != vecType)
        return std::nullopt;
      sliceType = vecType;
    }
    return llvm::to_vector(sliceType.getShape());
  }
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // TODO: The condition for unrolling elementwise should be restricted
      // only to operations that need unrolling (connected to the contract).
      if (vecType.getRank() < 2)
        return std::nullopt;

      // First check whether there is a slice to infer the shape from. This is
      // required for cases where the accumulator type differs from the input
      // types, in which case we will see an `arith.ext_` between the contract
      // and transfer_read which needs to be unrolled.
      VectorType sliceType;
      for (Operation *users : op->getUsers()) {
        auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
        if (!extract)
          return std::nullopt;
        auto vecType = extract.getResult().getType().cast<VectorType>();
        if (sliceType && sliceType != vecType)
          return std::nullopt;
        sliceType = vecType;
      }
      if (sliceType)
        return llvm::to_vector(sliceType.getShape());

      // Else unroll for trailing elementwise.
      SmallVector<int64_t> nativeSize(vecType.getRank() - 2, 1);
      // Map elementwise ops to the output shape.
      nativeSize.append({m, n});
      return nativeSize;
    }
  }
  return std::nullopt;
}

void transform::ApplyUnrollVectorsSubgroupMmaOp::populatePatterns(
    RewritePatternSet &patterns) {
  auto unrollOrder = [](Operation *op) -> std::optional<SmallVector<int64_t>> {
    auto contract = dyn_cast<vector::ContractionOp>(op);
    if (!contract)
      return std::nullopt;
    return gpuMmaUnrollOrder(contract);
  };

  int64_t m = getM();
  int64_t n = getN();
  int64_t k = getK();
  auto nativeShapeFn =
      [m, n, k](Operation *op) -> std::optional<SmallVector<int64_t>> {
    return getSubgroupMmaNativeVectorSize(op, m, n, k);
  };
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShapeFn(nativeShapeFn)
                    .setUnrollTraversalOrderFn(unrollOrder));
}

//===----------------------------------------------------------------------===//
// EliminateBarriersOp
//===----------------------------------------------------------------------===//

// The functions below provide interface-like verification, but are too specific
// to barrier elimination to become interfaces.

/// Implement the MemoryEffectsOpInterface in the suitable way.
static bool isKnownNoEffectsOpWithoutInterface(Operation *op) {
  // memref::AssumeAlignment is conceptually pure, but marking it as such would
  // make DCE immediately remove it.
  return isa<memref::AssumeAlignmentOp>(op);
}

/// Returns `true` if the op is defines the parallel region that is subject to
/// barrier synchronization.
static bool isParallelRegionBoundary(Operation *op) {
  if (op->hasAttr("__parallel_region_boundary_for_test"))
    return true;

  return isa<GPUFuncOp, LaunchOp>(op);
}

/// Returns `true` if the op behaves like a sequential loop, e.g., the control
/// flow "wraps around" from the end of the body region back to its start.
static bool isSequentialLoopLike(Operation *op) { return isa<scf::ForOp>(op); }

/// Returns `true` if the regions of the op are guaranteed to be executed at
/// most once. Thus, if an operation in one of the nested regions of `op` is
/// executed than so are all the other operations in this region.
static bool hasSingleExecutionBody(Operation *op) {
  return isa<scf::IfOp, memref::AllocaScopeOp>(op);
}

/// Returns `true` if the operation is known to produce a pointer-like object
/// distinct from any other object produced by a similar operation. For example,
/// an allocation produces such an object.
static bool producesDistinctBase(Operation *op) {
  return isa_and_nonnull<memref::AllocOp, memref::AllocaOp>(op);
}

/// Populates `effects` with all memory effects without associating them to a
/// specific value.
static void addAllValuelessEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Read>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Write>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Allocate>());
  effects.emplace_back(MemoryEffects::Effect::get<MemoryEffects::Free>());
}

/// Collect the memory effects of the given op in 'effects'. Returns 'true' if
/// it could extract the effect information from the op, otherwise returns
/// 'false' and conservatively populates the list with all possible effects
/// associated with no particular value or symbol.
static bool
collectEffects(Operation *op,
               SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
               bool ignoreBarriers = true) {
  // Skip over barriers to avoid infinite recursion (those barriers would ask
  // this barrier again).
  if (ignoreBarriers && isa<BarrierOp>(op))
    return true;

  // Skip over ops that we know have no effects.
  if (isKnownNoEffectsOpWithoutInterface(op))
    return true;

  // Collect effect instances the operation. Note that the implementation of
  // getEffects erases all effect instances that have the type other than the
  // template parameter so we collect them first in a local buffer and then
  // copy.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance> localEffects;
    iface.getEffects(localEffects);
    llvm::append_range(effects, localEffects);
    return true;
  }
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    for (auto &region : op->getRegions()) {
      for (auto &block : region) {
        for (auto &innerOp : block)
          if (!collectEffects(&innerOp, effects, ignoreBarriers))
            return false;
      }
    }
    return true;
  }

  // We need to be conservative here in case the op doesn't have the interface
  // and assume it can have any possible effect.
  addAllValuelessEffects(effects);
  return false;
}

/// Collects memory effects from operations that may be executed before `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
static bool
getEffectsBefore(Operation *op,
                 SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                 bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects before the op.
  if (op != &op->getBlock()->front()) {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (isa<BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        else
          continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }
  }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting above the parent operation.
  if (!getEffectsBefore(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the trailing operations until
  // we hit a barrier because they can executed before the current operation by
  // the previous iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op2` at iteration `i` is known to be executed before the
  // operation `op1` at iteration `i+1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    // Assuming loop terminators have no side effects.
    return getEffectsBefore(op->getBlock()->getTerminator(), effects,
                            /*stopAtBarrier=*/true);
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Collects memory effects from operations that may be executed after `op` in
/// a trivial structured control flow, e.g., without branches. Stops at the
/// parallel region boundary or at the barrier operation if `stopAtBarrier` is
/// set. Returns `true` if the memory effects added to `effects` are exact,
/// `false` if they are a conservative over-approximation. The latter means that
/// `effects` contain instances not associated with a specific value.
static bool
getEffectsAfter(Operation *op,
                SmallVectorImpl<MemoryEffects::EffectInstance> &effects,
                bool stopAtBarrier) {
  if (!op->getBlock())
    return true;

  // If there is a non-structured control flow, bail.
  Region *region = op->getBlock()->getParent();
  if (region && !llvm::hasSingleElement(region->getBlocks())) {
    addAllValuelessEffects(effects);
    return false;
  }

  // Collect all effects after the op.
  if (op != &op->getBlock()->back())
    for (Operation *it = op->getNextNode(); it != nullptr;
         it = it->getNextNode()) {
      if (isa<BarrierOp>(it)) {
        if (stopAtBarrier)
          return true;
        continue;
      }
      if (!collectEffects(it, effects))
        return false;
    }

  // Stop if reached the parallel region boundary.
  if (isParallelRegionBoundary(op->getParentOp()))
    return true;

  // Otherwise, keep collecting below the parent operation.
  if (!getEffectsAfter(op->getParentOp(), effects, stopAtBarrier))
    return false;

  // If the op is loop-like, collect effects from the leading operations until
  // we hit a barrier because they can executed after the current operation by
  // the next iteration of this loop. For example, in the following loop
  //
  //   for i = ... {
  //     op1
  //     ...
  //     barrier
  //     op2
  //   }
  //
  // the operation `op1` at iteration `i` is known to be executed after the
  // operation `op2` at iteration `i-1` and the side effects must be ordered
  // appropriately.
  if (isSequentialLoopLike(op->getParentOp())) {
    if (isa<BarrierOp>(op->getBlock()->front()))
      return true;

    bool exact = collectEffects(&op->getBlock()->front(), effects);
    return getEffectsAfter(&op->getBlock()->front(), effects,
                           /*stopAtBarrier=*/true) &&
           exact;
  }

  // If the parent operation is not guaranteed to execute its (single-block)
  // region once, walk the block.
  bool conservative = false;
  if (!hasSingleExecutionBody(op->getParentOp()))
    op->getParentOp()->walk([&](Operation *in) {
      if (conservative)
        return WalkResult::interrupt();
      if (!collectEffects(in, effects)) {
        conservative = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

  return !conservative;
}

/// Looks through known "view-like" ops to find the base memref.
static Value getBase(Value v) {
  while (true) {
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      break;

    bool shouldContinue =
        TypeSwitch<Operation *, bool>(v.getDefiningOp())
            .Case<memref::CastOp, memref::SubViewOp, memref::ViewOp>(
                [&](auto op) {
                  v = op.getSource();
                  return true;
                })
            .Case<memref::TransposeOp>([&](auto op) {
              v = op.getIn();
              return true;
            })
            .Case<memref::CollapseShapeOp, memref::ExpandShapeOp>([&](auto op) {
              v = op.getSrc();
              return true;
            })
            .Default([](Operation *) { return false; });
    if (!shouldContinue)
      break;
  }
  return v;
}

/// Returns `true` if the value is defined as a function argument.
static bool isFunctionArgument(Value v) {
  auto arg = dyn_cast<BlockArgument>(v);
  return arg && isa<FunctionOpInterface>(arg.getOwner()->getParentOp());
}

/// Returns the operand that the operation "propagates" through it for capture
/// purposes. That is, if the value produced by this operation is captured, then
/// so is the returned value.
static Value propagatesCapture(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case(
          [](ViewLikeOpInterface viewLike) { return viewLike.getViewSource(); })
      .Case([](CastOpInterface castLike) { return castLike->getOperand(0); })
      .Case([](memref::TransposeOp transpose) { return transpose.getIn(); })
      .Case<memref::ExpandShapeOp, memref::CollapseShapeOp>(
          [](auto op) { return op.getSrc(); })
      .Default([](Operation *) { return Value(); });
}

/// Returns `true` if the given operation is known to capture the given value,
/// `false` if it is known not to capture the given value, `nullopt` if neither
/// is known.
static std::optional<bool> getKnownCapturingStatus(Operation *op, Value v) {
  return llvm::TypeSwitch<Operation *, std::optional<bool>>(op)
      // Store-like operations don't capture the destination, but do capture
      // the value.
      .Case<memref::StoreOp, vector::TransferWriteOp>(
          [&](auto op) { return op.getValue() == v; })
      .Case<vector::StoreOp, vector::MaskedStoreOp>(
          [&](auto op) { return op.getValueToStore() == v; })
      // These operations are known not to capture.
      .Case([](memref::DeallocOp) { return false; })
      // By default, we don't know anything.
      .Default([](Operation *) { return std::nullopt; });
}

/// Returns `true` if the value may be captured by any of its users, i.e., if
/// the user may be storing this value into memory. This makes aliasing analysis
/// more conservative as it cannot assume the pointer-like value is only passed
/// around through SSA use-def.
static bool maybeCaptured(Value v) {
  SmallVector<Value> todo = {v};
  while (!todo.empty()) {
    Value v = todo.pop_back_val();
    for (Operation *user : v.getUsers()) {
      // A user that is known to only read cannot capture.
      auto iface = dyn_cast<MemoryEffectOpInterface>(user);
      if (iface) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        iface.getEffects(effects);
        if (llvm::all_of(effects,
                         [](const MemoryEffects::EffectInstance &effect) {
                           return isa<MemoryEffects::Read>(effect.getEffect());
                         })) {
          continue;
        }
      }

      // When an operation is known to create an alias, consider if the
      // source is captured as well.
      if (Value v = propagatesCapture(user)) {
        todo.push_back(v);
        continue;
      }

      std::optional<bool> knownCaptureStatus = getKnownCapturingStatus(user, v);
      if (!knownCaptureStatus || *knownCaptureStatus)
        return true;
    }
  }

  return false;
}

/// Returns true if two values may be referencing aliasing memory. This is a
/// rather naive and conservative analysis. Values defined by different
/// allocation-like operations as well as values derived from those by casts and
/// views cannot alias each other. Similarly, values defined by allocations
/// inside a function cannot alias function arguments. Global values cannot
/// alias each other or local allocations. Values that are captured, i.e.
/// themselves potentially stored in memory, are considered as aliasing with
/// everything. This seems sufficient to achieve barrier removal in structured
/// control flow, more complex cases would require a proper dataflow analysis.
static bool mayAlias(Value first, Value second) {
  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, {
    DBGS_ALIAS() << "checking aliasing between ";
    DBGS_ALIAS() << first << "\n";
    DBGS_ALIAS() << "                      and ";
    DBGS_ALIAS() << second << "\n";
  });

  first = getBase(first);
  second = getBase(second);

  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, {
    DBGS_ALIAS() << "base ";
    DBGS_ALIAS() << first << "\n";
    DBGS_ALIAS() << " and ";
    DBGS_ALIAS() << second << "\n";
  });

  // Values derived from the same base memref do alias (unless we do a more
  // advanced analysis to prove non-overlapping accesses).
  if (first == second) {
    DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, DBGS_ALIAS() << "-> do alias!\n");
    return true;
  }

  // Different globals cannot alias.
  if (auto globFirst = first.getDefiningOp<memref::GetGlobalOp>()) {
    if (auto globSecond = second.getDefiningOp<memref::GetGlobalOp>()) {
      return globFirst.getNameAttr() == globSecond.getNameAttr();
    }
  }

  // Two function arguments marked as noalias do not alias.
  auto isNoaliasFuncArgument = [](Value value) {
    auto bbArg = dyn_cast<BlockArgument>(value);
    if (!bbArg)
      return false;
    auto iface = dyn_cast<FunctionOpInterface>(bbArg.getOwner()->getParentOp());
    if (!iface)
      return false;
    // TODO: we need a way to not depend on the LLVM dialect here.
    return iface.getArgAttr(bbArg.getArgNumber(), "llvm.noalias") != nullptr;
  };
  if (isNoaliasFuncArgument(first) && isNoaliasFuncArgument(second))
    return false;

  bool isDistinct[] = {producesDistinctBase(first.getDefiningOp()),
                       producesDistinctBase(second.getDefiningOp())};
  bool isGlobal[] = {first.getDefiningOp<memref::GetGlobalOp>() != nullptr,
                     second.getDefiningOp<memref::GetGlobalOp>() != nullptr};

  // Non-equivalent distinct bases and globals cannot alias. At this point, we
  // have already filtered out based on values being equal and global name being
  // equal.
  if ((isDistinct[0] || isGlobal[0]) && (isDistinct[1] || isGlobal[1]))
    return false;

  bool isArg[] = {isFunctionArgument(first), isFunctionArgument(second)};

  // Distinct bases (allocations) cannot have been passed as an argument.
  if ((isDistinct[0] && isArg[1]) || (isDistinct[1] && isArg[0]))
    return false;

  // Non-captured base distinct values cannot conflict with another base value.
  if (isDistinct[0] && !maybeCaptured(first))
    return false;
  if (isDistinct[1] && !maybeCaptured(second))
    return false;

  // Otherwise, conservatively assume aliasing.
  DEBUG_WITH_TYPE(DEBUG_TYPE_ALIAS, DBGS_ALIAS() << "-> may alias!\n");
  return true;
}

/// Returns `true` if the effect may be affecting memory aliasing the value. If
/// the effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything.
static bool mayAlias(MemoryEffects::EffectInstance a, Value v2) {
  if (Value v = a.getValue()) {
    return mayAlias(v, v2);
  }
  return true;
}

/// Returns `true` if the two effects may be affecting aliasing memory. If
/// an effect is not associated with any value, it is assumed to affect all
/// memory and therefore aliases with everything. Effects on different resources
/// cannot alias.
static bool mayAlias(MemoryEffects::EffectInstance a,
                     MemoryEffects::EffectInstance b) {
  if (a.getResource()->getResourceID() != b.getResource()->getResourceID())
    return false;
  if (Value v2 = b.getValue()) {
    return mayAlias(a, v2);
  } else if (Value v = a.getValue()) {
    return mayAlias(b, v);
  }
  return true;
}

/// Returns `true` if any of the "before" effect instances has a conflict with
/// any "after" instance for the purpose of barrier elimination. The effects are
/// supposed to be limited to a barrier synchronization scope. A conflict exists
/// if effects instances affect aliasing memory locations and at least on of
/// then as a write. As an exception, if the non-write effect is an allocation
/// effect, there is no conflict since we are only expected to see the
/// allocation happening in the same thread and it cannot be accessed from
/// another thread without capture (which we do handle in alias analysis).
static bool
haveConflictingEffects(ArrayRef<MemoryEffects::EffectInstance> beforeEffects,
                       ArrayRef<MemoryEffects::EffectInstance> afterEffects) {
  for (const MemoryEffects::EffectInstance &before : beforeEffects) {
    for (const MemoryEffects::EffectInstance &after : afterEffects) {
      // If cannot alias, definitely no conflict.
      if (!mayAlias(before, after))
        continue;

      // Read/read is not a conflict.
      if (isa<MemoryEffects::Read>(before.getEffect()) &&
          isa<MemoryEffects::Read>(after.getEffect())) {
        continue;
      }

      // Allocate/* is not a conflict since the allocation happens within the
      // thread context.
      // TODO: This is not the case for */Free unless the allocation happened in
      // the thread context, which we could also check for.
      if (isa<MemoryEffects::Allocate>(before.getEffect()) ||
          isa<MemoryEffects::Allocate>(after.getEffect())) {
        continue;
      }

      // In the particular case that the before effect is a free, we only have 2
      // possibilities:
      //   1. either the program is well-formed and there must be an interleaved
      //      alloc that must limit the scope of effect lookback and we can
      //      safely ignore the free -> read / free -> write and free -> free
      //      conflicts.
      //   2. either the program is ill-formed and we are in undefined behavior
      //      territory.
      if (isa<MemoryEffects::Free>(before.getEffect()))
        continue;

      // Other kinds of effects create a conflict, e.g. read-after-write.
      LLVM_DEBUG(
          DBGS() << "found a conflict between (before): " << before.getValue()
                 << " read:" << isa<MemoryEffects::Read>(before.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(before.getEffect())
                 << " alloc:"
                 << isa<MemoryEffects::Allocate>(before.getEffect()) << " free:"
                 << isa<MemoryEffects::Free>(before.getEffect()) << "\n");
      LLVM_DEBUG(
          DBGS() << "and (after):                " << after.getValue()
                 << " read:" << isa<MemoryEffects::Read>(after.getEffect())
                 << " write:" << isa<MemoryEffects::Write>(after.getEffect())
                 << " alloc:" << isa<MemoryEffects::Allocate>(after.getEffect())
                 << " free:" << isa<MemoryEffects::Free>(after.getEffect())
                 << "\n");
      return true;
    }
  }

  return false;
}

namespace {
/// Barrier elimination pattern. If a barrier does not enforce any conflicting
/// pair of memory effects, including a pair that is enforced by another
/// barrier, it is unnecessary and can be removed. Adapted from
/// "High-Performance GPU-to-CPU Transpilation and Optimization via High-Level
/// Parallel Constructs" by Moses, Ivanov, Domke, Endo, Doerfert, and Zinenko in
/// PPoPP 2023 and implementation in Polygeist.
class BarrierElimination final : public OpRewritePattern<BarrierOp> {
public:
  using OpRewritePattern<BarrierOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(DBGS() << "checking the necessity of: " << barrier << " "
                      << barrier.getLoc() << "\n");

    SmallVector<MemoryEffects::EffectInstance> beforeEffects;
    getEffectsBefore(barrier, beforeEffects, /*stopAtBarrier=*/true);

    SmallVector<MemoryEffects::EffectInstance> afterEffects;
    getEffectsAfter(barrier, afterEffects, /*stopAtBarrier=*/true);

    if (!haveConflictingEffects(beforeEffects, afterEffects)) {
      LLVM_DEBUG(DBGS() << "the surrounding barriers are sufficient, removing "
                        << barrier << "\n");
      rewriter.eraseOp(barrier);
      return success();
    }

    LLVM_DEBUG(DBGS() << "barrier is necessary: " << barrier << " "
                      << barrier.getLoc() << "\n");
    return failure();
  }
};
} // namespace

void EliminateBarriersOp::populatePatterns(RewritePatternSet &patterns) {
  patterns.insert<BarrierElimination>(getContext());
}

//===----------------------------------------------------------------------===//
// Block and thread mapping utilities.
//===----------------------------------------------------------------------===//

namespace {
/// Local types used for mapping verification.
struct MappingKind {};
struct BlockMappingKind : MappingKind {};
struct ThreadMappingKind : MappingKind {};
} // namespace

static DiagnosedSilenceableFailure
definiteFailureHelper(std::optional<TransformOpInterface> transformOp,
                      Operation *target, const Twine &message) {
  if (transformOp.has_value())
    return transformOp->emitDefiniteFailure() << message;
  return emitDefiniteFailure(target, message);
}

/// Check if given mapping attributes are one of the desired attributes
template <typename MappingKindType>
static DiagnosedSilenceableFailure
checkMappingAttributeTypes(std::optional<TransformOpInterface> transformOp,
                           scf::ForallOp forallOp) {
  if (!forallOp.getMapping().has_value()) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall op requires a mapping attribute");
  }

  bool hasBlockMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUBlockMappingAttr>(attr);
      });
  bool hasWarpgroupMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUWarpgroupMappingAttr>(attr);
      });
  bool hasWarpMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUWarpMappingAttr>(attr);
      });
  bool hasThreadMapping =
      llvm::any_of(forallOp.getMapping().value(), [](Attribute attr) {
        return isa<GPUThreadMappingAttr>(attr);
      });
  int64_t countMappingTypes = 0;
  countMappingTypes += hasBlockMapping ? 1 : 0;
  countMappingTypes += hasWarpgroupMapping ? 1 : 0;
  countMappingTypes += hasWarpMapping ? 1 : 0;
  countMappingTypes += hasThreadMapping ? 1 : 0;
  if (countMappingTypes > 1) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix different mapping types, use nesting");
  }
  if (std::is_same<MappingKindType, BlockMappingKind>::value &&
      !hasBlockMapping) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "scf.forall op requires a mapping attribute of kind 'block'");
  }
  if (std::is_same<MappingKindType, ThreadMappingKind>::value &&
      !hasThreadMapping && !hasWarpMapping && !hasWarpgroupMapping) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall op requires a mapping attribute "
                                 "of kind 'thread' or 'warp'");
  }

  DenseSet<Attribute> seen;
  for (Attribute map : forallOp.getMapping()->getValue()) {
    if (seen.contains(map)) {
      return definiteFailureHelper(
          transformOp, forallOp,
          "duplicate attribute, cannot map different loops "
          "to the same mapping id");
    }
    seen.insert(map);
  }

  auto isLinear = [](Attribute a) {
    return cast<DeviceMappingAttrInterface>(a).isLinearMapping();
  };
  if (llvm::any_of(forallOp.getMapping()->getValue(), isLinear) &&
      !llvm::all_of(forallOp.getMapping()->getValue(), isLinear)) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "cannot mix linear and non-linear mapping modes");
  }

  return DiagnosedSilenceableFailure::success();
}

template <typename MappingKindType>
static DiagnosedSilenceableFailure
verifyGpuMapping(std::optional<TransformOpInterface> transformOp,
                 scf::ForallOp forallOp) {
  // Check the types of the mapping attributes match.
  DiagnosedSilenceableFailure typeRes =
      checkMappingAttributeTypes<MappingKindType>(transformOp, forallOp);
  if (!typeRes.succeeded())
    return typeRes;

  // Perform other non-types verifications.
  if (!forallOp.isNormalized())
    return definiteFailureHelper(transformOp, forallOp,
                                 "unsupported non-normalized loops");
  if (forallOp.getNumResults() > 0)
    return definiteFailureHelper(transformOp, forallOp,
                                 "only bufferized scf.forall can be mapped");
  bool useLinearMapping = cast<DeviceMappingAttrInterface>(
                              forallOp.getMapping()->getValue().front())
                              .isLinearMapping();
  // TODO: This would be more natural with support for Optional<EnumParameter>
  // in GPUDeviceMappingAttr.
  int64_t maxNumMappingsSupported =
      useLinearMapping ? (getMaxEnumValForMappingId() -
                          static_cast<uint64_t>(MappingId::DimZ))
                       : 3;
  if (forallOp.getRank() > maxNumMappingsSupported) {
    return definiteFailureHelper(transformOp, forallOp,
                                 "scf.forall with rank > ")
           << maxNumMappingsSupported
           << " does not lower for the specified mapping attribute type";
  }
  auto numParallelIterations =
      getConstantIntValues(forallOp.getMixedUpperBound());
  if (!forallOp.isNormalized() || !numParallelIterations.has_value()) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "requires statically sized, normalized forall op");
  }
  return DiagnosedSilenceableFailure::success();
}

/// Struct to return the result of the rewrite of a forall operation.
struct ForallRewriteResult {
  SmallVector<int64_t> mappingSizes;
  SmallVector<Value> mappingIds;
};

/// Helper to replace ids of dimensions known to be 1 by 0 to simplify the IR.
template <typename OpTy, typename OperationOrBlock>
static void
replaceUnitMappingIdsHelper(RewriterBase &rewriter, Location loc,
                            OperationOrBlock *parent, Value replacement,
                            ArrayRef<int64_t> availableMappingSizes) {
  parent->walk([&](OpTy idOp) {
    if (availableMappingSizes[static_cast<int64_t>(idOp.getDimension())] == 1)
      rewriter.replaceAllUsesWith(idOp.getResult(), replacement);
  });
}

static DiagnosedSilenceableFailure rewriteOneForallCommonImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> availableMappingSizes,
    ForallRewriteResult &result, const GpuIdBuilder &gpuIdBuilder) {
  LDBG("--start rewriteOneForallCommonImpl");

  // Step 1. Complete the mapping to a full mapping (with 1s) if necessary.
  auto numParallelIterations =
      getConstantIntValues(forallOp.getMixedUpperBound());
  assert(forallOp.isNormalized() && numParallelIterations.has_value() &&
         "requires statically sized, normalized forall op");
  SmallVector<int64_t> tmpMappingSizes = numParallelIterations.value();
  SetVector<Attribute> forallMappingAttrs;
  forallMappingAttrs.insert(forallOp.getMapping()->getValue().begin(),
                            forallOp.getMapping()->getValue().end());
  auto comparator = [](Attribute a, Attribute b) -> bool {
    return cast<DeviceMappingAttrInterface>(a).getMappingId() <
           cast<DeviceMappingAttrInterface>(b).getMappingId();
  };

  // Step 1.b. In the linear case, compute the max mapping to avoid needlessly
  // mapping all dimensions. In the 3-D mapping case we need to map all
  // dimensions.
  DeviceMappingAttrInterface maxMapping =
      cast<DeviceMappingAttrInterface>(*std::max_element(
          forallMappingAttrs.begin(), forallMappingAttrs.end(), comparator));
  DeviceMappingAttrInterface maxLinearMapping;
  if (maxMapping.isLinearMapping())
    maxLinearMapping = maxMapping;
  for (auto attr : gpuIdBuilder.mappingAttributes) {
    // If attr overflows, just skip.
    if (maxLinearMapping && comparator(maxLinearMapping, attr))
      continue;
    // Try to insert. If element was already present, just continue.
    if (!forallMappingAttrs.insert(attr))
      continue;
    // Otherwise, we have a new insertion without a size -> use size 1.
    tmpMappingSizes.push_back(1);
  }
  LLVM_DEBUG(
      llvm::interleaveComma(
          tmpMappingSizes,
          DBGS() << "----tmpMappingSizes extracted from scf.forall op: ");
      llvm::dbgs() << "\n");

  // Step 2. sort the values by the corresponding DeviceMappingAttrInterface.
  SmallVector<int64_t> forallMappingSizes = getValuesSortedByKey(
      forallMappingAttrs.getArrayRef(), tmpMappingSizes, comparator);
  LLVM_DEBUG(llvm::interleaveComma(forallMappingSizes,
                                   DBGS() << "----forallMappingSizes: ");
             llvm::dbgs() << "\n"; llvm::interleaveComma(
                 forallMappingAttrs, DBGS() << "----forallMappingAttrs: ");
             llvm::dbgs() << "\n");

  // Step 3. Generate the mappingIdOps using the provided generator.
  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forallOp);
  SmallVector<int64_t> originalBasis(availableMappingSizes);
  bool originalBasisWasProvided = !originalBasis.empty();
  if (!originalBasisWasProvided) {
    originalBasis = forallMappingSizes;
    while (originalBasis.size() < 3)
      originalBasis.push_back(1);
  }

  IdBuilderResult builderResult =
      gpuIdBuilder.idBuilder(rewriter, loc, forallMappingSizes, originalBasis);

  // Step 4. Map the induction variables to the mappingIdOps, this may involve
  // a permutation.
  SmallVector<Value> mappingIdOps = builderResult.mappingIdOps;
  IRMapping bvm;
  for (auto [iv, dim] : llvm::zip_equal(
           forallOp.getInductionVars(),
           forallMappingAttrs.getArrayRef().take_front(forallOp.getRank()))) {
    auto mappingAttr = cast<DeviceMappingAttrInterface>(dim);
    Value peIdOp = mappingIdOps[mappingAttr.getRelativeIndex()];
    bvm.map(iv, peIdOp);
  }

  // Step 5. If the originalBasis is already known, create conditionals to
  // predicate the region. Otherwise, the current forall determines the
  // originalBasis and no predication occurs.
  Value predicate;
  if (originalBasisWasProvided) {
    SmallVector<int64_t> activeMappingSizes = builderResult.activeMappingSizes;
    SmallVector<int64_t> availableMappingSizes =
        builderResult.availableMappingSizes;
    SmallVector<Value> activeIdOps = builderResult.activeIdOps;
    // clang-format off
    LLVM_DEBUG(
        llvm::interleaveComma(
          activeMappingSizes, DBGS() << "----activeMappingSizes: ");
        llvm::dbgs() << "\n"; 
        llvm::interleaveComma(
          availableMappingSizes, DBGS() << "----availableMappingSizes: ");
        llvm::dbgs() << "\n";
        llvm::interleaveComma(activeIdOps, DBGS() << "----activeIdOps: ");
        llvm::dbgs() << "\n");
    // clang-format on
    for (auto [activeId, activeMappingSize, availableMappingSize] :
         llvm::zip_equal(activeIdOps, activeMappingSizes,
                         availableMappingSizes)) {
      if (activeMappingSize > availableMappingSize) {
        return definiteFailureHelper(
            transformOp, forallOp,
            "Trying to map to fewer GPU threads than loop iterations but "
            "overprovisioning is not yet supported. "
            "Try additional tiling of the before mapping or map to more "
            "threads.");
      }
      if (activeMappingSize == availableMappingSize)
        continue;
      Value idx =
          rewriter.create<arith::ConstantIndexOp>(loc, activeMappingSize);
      Value tmpPredicate = rewriter.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::ult, activeId, idx);
      LDBG("----predicate: " << tmpPredicate);
      predicate = predicate ? rewriter.create<arith::AndIOp>(loc, predicate,
                                                             tmpPredicate)
                            : tmpPredicate;
    }
  }

  // Step 6. Move the body of forallOp.
  // Erase the terminator first, it will not be used.
  rewriter.eraseOp(forallOp.getTerminator());
  Block *targetBlock;
  Block::iterator insertionPoint;
  if (predicate) {
    // Step 6.a. If predicated, move at the beginning.
    auto ifOp = rewriter.create<scf::IfOp>(loc, predicate,
                                           /*withElseRegion=*/false);
    targetBlock = ifOp.thenBlock();
    insertionPoint = ifOp.thenBlock()->begin();
  } else {
    // Step 6.b. Otherwise, move inline just at the rewriter insertion
    // point.
    targetBlock = forallOp->getBlock();
    insertionPoint = rewriter.getInsertionPoint();
  }
  Block &sourceBlock = forallOp.getRegion().front();
  targetBlock->getOperations().splice(insertionPoint,
                                      sourceBlock.getOperations());

  // Step 7. RAUW indices.
  for (Value loopIndex : forallOp.getInductionVars()) {
    Value threadIdx = bvm.lookup(loopIndex);
    rewriter.replaceAllUsesWith(loopIndex, threadIdx);
  }

  // Step 8. Erase old op.
  rewriter.eraseOp(forallOp);

  LLVM_DEBUG(llvm::interleaveComma(forallMappingSizes,
                                   DBGS() << "----result forallMappingSizes: ");
             llvm::dbgs() << "\n"; llvm::interleaveComma(
                 mappingIdOps, DBGS() << "----result mappingIdOps: ");
             llvm::dbgs() << "\n");

  result = ForallRewriteResult{forallMappingSizes, mappingIdOps};
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MapForallToBlocks
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure mlir::transform::gpu::mapForallToBlocksImpl(
    RewriterBase &rewriter, TransformOpInterface transformOp,
    scf::ForallOp forallOp, SmallVectorImpl<int64_t> &gridDims,
    const GpuIdBuilder &gpuIdBuilder) {
  LDBG("Start mapForallToBlocksImpl");

  {
    // GPU-specific verifications. There is no better place to anchor
    // those right now: the ForallOp is target-independent and the transform
    // op does not apply to individual ForallOp.
    DiagnosedSilenceableFailure diag =
        verifyGpuMapping<BlockMappingKind>(transformOp, forallOp);
    if (!diag.succeeded())
      return diag;
  }

  Location loc = forallOp.getLoc();
  Block *parentBlock = forallOp->getBlock();
  Value zero;
  {
    // Create an early zero index value for replacements and immediately reset
    // the insertion point.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(parentBlock);
    zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  }

  ForallRewriteResult rewriteResult;
  DiagnosedSilenceableFailure diag = rewriteOneForallCommonImpl(
      rewriter, transformOp, forallOp,
      /*availableMappingSizes=*/gridDims, rewriteResult, gpuIdBuilder);

  // Return if anything goes wrong, use silenceable failure as a match
  // failure.
  if (!diag.succeeded())
    return diag;

  // If gridDims was not provided already, set it from the return.
  if (gridDims.empty()) {
    gridDims = rewriteResult.mappingSizes;
    while (gridDims.size() < 3)
      gridDims.push_back(1);
  }
  assert(gridDims.size() == 3 && "Need 3-D gridDims");

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<BlockDimOp>(rewriter, loc, parentBlock, zero,
                                          rewriteResult.mappingSizes);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure
mlir::transform::gpu::findTopLevelForallOp(Operation *target,
                                           scf::ForallOp &topLevelForallOp,
                                           TransformOpInterface transformOp) {
  auto walkResult = target->walk([&](scf::ForallOp forallOp) {
    if (forallOp->getParentOfType<scf::ForallOp>())
      return WalkResult::advance();
    if (topLevelForallOp)
      // TODO: Handle multiple forall if they are independent.
      return WalkResult::interrupt();
    topLevelForallOp = forallOp;
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted() || !topLevelForallOp)
    return transformOp.emitSilenceableError()
           << "could not find a unique topLevel scf.forall";
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapForallToBlocks::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, transform::TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  if (!getGenerateGpuLaunch() && !gpuLaunch) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "Given target is not gpu.launch, set `generate_gpu_launch` "
           "attribute";
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }

  scf::ForallOp topLevelForallOp;
  DiagnosedSilenceableFailure diag = mlir::transform::gpu::findTopLevelForallOp(
      target, topLevelForallOp, transformOp);
  if (!diag.succeeded()) {
    diag.attachNote(target->getLoc()) << "when applied to this payload op";
    return diag;
  }
  assert(topLevelForallOp && "expect an scf.forall");

  SmallVector<int64_t> gridDims{getGridDims()};
  if (!getGenerateGpuLaunch() && gridDims.size() != 3)
    return transformOp.emitDefiniteFailure("transform require size-3 mapping");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(topLevelForallOp);

  // Generate gpu launch here and move the forall inside
  if (getGenerateGpuLaunch()) {
    DiagnosedSilenceableFailure diag =
        createGpuLaunch(rewriter, target->getLoc(), transformOp, gpuLaunch);
    if (!diag.succeeded())
      return diag;

    rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
    Operation *newForallOp = rewriter.clone(*topLevelForallOp);
    rewriter.eraseOp(topLevelForallOp);
    topLevelForallOp = cast<scf::ForallOp>(newForallOp);
  }

  // The BlockIdBuilder adapts to whatever is thrown at it.
  bool useLinearMapping = false;
  if (topLevelForallOp.getMapping()) {
    auto mappingAttr = cast<DeviceMappingAttrInterface>(
        topLevelForallOp.getMapping()->getValue().front());
    useLinearMapping = mappingAttr.isLinearMapping();
  }
  GpuBlockIdBuilder gpuBlockIdBuilder(getContext(), useLinearMapping);

  diag = mlir::transform::gpu::mapForallToBlocksImpl(
      rewriter, transformOp, topLevelForallOp, gridDims, gpuBlockIdBuilder);
  if (!diag.succeeded())
    return diag;

  // Set the GPU launch configuration for the grid dims late, this is
  // subject to IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch,
                        cast<TransformOpInterface>(getOperation()), gridDims[0],
                        gridDims[1], gridDims[2]);

  results.push_back(gpuLaunch);
  return diag;
}

LogicalResult transform::MapForallToBlocks::verify() {
  if (!getGridDims().empty() && getGridDims().size() != 3) {
    return emitOpError() << "transform requires empty or size-3 grid_dims";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MapNestedForallToThreads
//===----------------------------------------------------------------------===//

static DiagnosedSilenceableFailure checkMappingSpec(
    std::optional<TransformOpInterface> transformOp, scf::ForallOp forallOp,
    ArrayRef<int64_t> numParallelIterations, ArrayRef<int64_t> blockOrGridSizes,
    int factor, bool useLinearMapping = false) {
  if (!useLinearMapping && blockOrGridSizes.front() % factor != 0) {
    auto diag = definiteFailureHelper(
        transformOp, forallOp,
        Twine("3-D mapping: size of threadIdx.x must be a multiple of ") +
            std::to_string(factor));
    return diag;
  }
  if (computeProduct(numParallelIterations) * factor >
      computeProduct(blockOrGridSizes)) {
    auto diag = definiteFailureHelper(
        transformOp, forallOp,
        Twine("the number of required parallel resources (blocks or "
              "threads) ") +
            std::to_string(computeProduct(numParallelIterations) * factor) +
            std::string(" overflows the number of available resources ") +
            std::to_string(computeProduct(blockOrGridSizes)));
    return diag;
  }
  return DiagnosedSilenceableFailure::success();
}

static DiagnosedSilenceableFailure
getThreadIdBuilder(std::optional<TransformOpInterface> transformOp,
                   scf::ForallOp forallOp, ArrayRef<int64_t> blockSizes,
                   int64_t warpSize, GpuIdBuilder &gpuIdBuilder) {
  auto mappingAttr = cast<DeviceMappingAttrInterface>(
      forallOp.getMapping()->getValue().front());
  bool useLinearMapping = mappingAttr.isLinearMapping();

  // Sanity checks that may result in runtime verification errors.
  auto numParallelIterations =
      getConstantIntValues((forallOp.getMixedUpperBound()));
  if (!forallOp.isNormalized() || !numParallelIterations.has_value()) {
    return definiteFailureHelper(
        transformOp, forallOp,
        "requires statically sized, normalized forall op");
  }
  int64_t factor = 1;
  if (isa<GPUWarpgroupMappingAttr>(mappingAttr)) {
    factor = GpuWarpgroupIdBuilder::kNumWarpsPerGroup * warpSize;
  } else if (isa<GPUWarpMappingAttr>(mappingAttr)) {
    factor = warpSize;
  }
  DiagnosedSilenceableFailure diag =
      checkMappingSpec(transformOp, forallOp, numParallelIterations.value(),
                       blockSizes, factor, useLinearMapping);
  if (!diag.succeeded())
    return diag;

  // Start mapping.
  MLIRContext *ctx = forallOp.getContext();
  gpuIdBuilder =
      TypeSwitch<DeviceMappingAttrInterface, GpuIdBuilder>(mappingAttr)
          .Case([&](GPUWarpgroupMappingAttr) {
            return GpuWarpgroupIdBuilder(ctx, warpSize, useLinearMapping);
          })
          .Case([&](GPUWarpMappingAttr) {
            return GpuWarpIdBuilder(ctx, warpSize, useLinearMapping);
          })
          .Case([&](GPUThreadMappingAttr) {
            return GpuThreadIdBuilder(ctx, useLinearMapping);
          })
          .Default([&](DeviceMappingAttrInterface) -> GpuIdBuilder {
            llvm_unreachable("unknown mapping attribute");
          });
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapOneForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    scf::ForallOp forallOp, ArrayRef<int64_t> blockSizes, int64_t warpSize,
    bool syncAfterDistribute) {

  {
    // GPU-specific verifications. There is no better place to anchor
    // those right now: the ForallOp is target-independent and the transform
    // op does not apply to individual ForallOp.
    DiagnosedSilenceableFailure diag =
        verifyGpuMapping<ThreadMappingKind>(transformOp, forallOp);
    if (!diag.succeeded())
      return diag;
  }

  GpuIdBuilder gpuIdBuilder;
  {
    // Try to construct the id builder, if it fails, return.
    DiagnosedSilenceableFailure diag = getThreadIdBuilder(
        transformOp, forallOp, blockSizes, warpSize, gpuIdBuilder);
    if (!diag.succeeded())
      return diag;
  }

  Location loc = forallOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  // Insert after to allow for syncthreads after `forall` is erased.
  rewriter.setInsertionPointAfter(forallOp);
  ForallRewriteResult rewriteResult;
  DiagnosedSilenceableFailure diag = rewriteOneForallCommonImpl(
      rewriter, transformOp, forallOp, blockSizes, rewriteResult, gpuIdBuilder);
  if (!diag.succeeded())
    return diag;
  // Add a syncthreads if needed. TODO: warpsync
  if (syncAfterDistribute)
    rewriter.create<BarrierOp>(loc);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure mlir::transform::gpu::mapNestedForallToThreadsImpl(
    RewriterBase &rewriter, std::optional<TransformOpInterface> transformOp,
    Operation *target, ArrayRef<int64_t> blockDims, int64_t warpSize,
    bool syncAfterDistribute) {
  LDBG("Start mapNestedForallToThreadsImpl");
  if (blockDims.size() != 3) {
    return definiteFailureHelper(transformOp, target,
                                 "requires size-3 thread mapping");
  }

  // Create an early zero index value for replacements.
  Location loc = target->getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  DiagnosedSilenceableFailure diag = DiagnosedSilenceableFailure::success();
  WalkResult walkResult = target->walk([&](scf::ForallOp forallOp) {
    diag = mlir::transform::gpu::mapOneForallToThreadsImpl(
        rewriter, transformOp, forallOp, blockDims, warpSize,
        syncAfterDistribute);
    if (diag.isDefiniteFailure())
      return WalkResult::interrupt();
    if (diag.succeeded())
      return WalkResult::skip();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return diag;

  // Replace ids of dimensions known to be 1 by 0 to simplify the IR.
  // Here, the result of mapping determines the available mapping sizes.
  replaceUnitMappingIdsHelper<ThreadIdOp>(rewriter, loc, target, zero,
                                          blockDims);

  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::MapNestedForallToThreads::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  LaunchOp gpuLaunch = dyn_cast<LaunchOp>(target);
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Basic high-level verifications.
  if (!gpuLaunch)
    return emitSilenceableError() << "Given target is not a gpu.launch";

  // Mapping to block ids.
  SmallVector<int64_t> blockDims{getBlockDims()};
  DiagnosedSilenceableFailure diag =
      checkGpuLimits(transformOp, std::nullopt, std::nullopt, std::nullopt,
                     blockDims[0], blockDims[1], blockDims[2]);
  if (diag.isSilenceableFailure()) {
    diag.attachNote(getLoc()) << getBlockDimsAttrName() << " is too large";
    return diag;
  }

  // Set the GPU launch configuration for the block dims early, this is not
  // subject to IR inspection.
  diag = alterGpuLaunch(rewriter, gpuLaunch, transformOp, std::nullopt,
                        std::nullopt, std::nullopt, blockDims[0], blockDims[1],
                        blockDims[2]);

  rewriter.setInsertionPointToStart(&gpuLaunch.getBody().front());
  diag =
      mapNestedForallToThreadsImpl(rewriter, transformOp, gpuLaunch, blockDims,
                                   getWarpSize(), getSyncAfterDistribute());

  results.push_back(gpuLaunch.getOperation());
  return diag;
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class GPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          GPUTransformDialectExtension> {
public:
  GPUTransformDialectExtension() {
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<GPUDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/TransformOps/GPUTransformOps.cpp.inc"

void mlir::gpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<GPUTransformDialectExtension>();
}
