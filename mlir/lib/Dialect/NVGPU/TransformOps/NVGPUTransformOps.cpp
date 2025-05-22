//===- NVGPUTransformOps.cpp - Implementation of NVGPU transform ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::nvgpu;
using namespace mlir::NVVM;
using namespace mlir::transform;

#define DEBUG_TYPE "nvgpu-transforms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyNVGPUToNVVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  auto &llvmTypeConverter = static_cast<LLVMTypeConverter &>(typeConverter);
  /// device-side async tokens cannot be materialized in nvvm. We just
  /// convert them to a dummy i32 type in order to easily drop them during
  /// conversion.
  populateGpuMemorySpaceAttributeConversions(
      llvmTypeConverter, [](gpu::AddressSpace space) -> unsigned {
        switch (space) {
        case gpu::AddressSpace::Global:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kGlobalMemorySpace);
        case gpu::AddressSpace::Workgroup:
          return static_cast<unsigned>(
              NVVM::NVVMMemorySpace::kSharedMemorySpace);
        case gpu::AddressSpace::Private:
          return 0;
        }
        llvm_unreachable("unknown address space enum value");
        return 0;
      });
  llvmTypeConverter.addConversion(
      [&](nvgpu::DeviceAsyncTokenType type) -> Type {
        return llvmTypeConverter.convertType(
            IntegerType::get(type.getContext(), 32));
      });
  llvmTypeConverter.addConversion([&](nvgpu::MBarrierTokenType type) -> Type {
    return llvmTypeConverter.convertType(
        IntegerType::get(type.getContext(), 64));
  });
  llvmTypeConverter.addConversion(
      [&](nvgpu::WarpgroupAccumulatorType type) -> Type {
        Type elemType = type.getFragmented().getElementType();
        int64_t sizeM = type.getFragmented().getDimSize(0);
        int64_t sizeN = type.getFragmented().getDimSize(1);

        unsigned numMembers;
        if (elemType.isF32() || elemType.isInteger(32))
          numMembers = sizeN / 2;
        else if (elemType.isF16())
          numMembers = sizeN / 4;
        else
          llvm_unreachable("unsupported type for warpgroup accumulator");

        SmallVector<Type> innerStructBody;
        for (unsigned i = 0; i < numMembers; i++)
          innerStructBody.push_back(elemType);
        auto innerStructType = LLVM::LLVMStructType::getLiteral(
            type.getContext(), innerStructBody);

        SmallVector<Type> structBody;
        for (int i = 0; i < sizeM; i += kWgmmaSizeM)
          structBody.push_back(innerStructType);

        auto convertedType =
            LLVM::LLVMStructType::getLiteral(type.getContext(), structBody);
        return llvmTypeConverter.convertType(convertedType);
      });
  llvmTypeConverter.addConversion([&](nvgpu::MBarrierGroupType type) -> Type {
    return llvmTypeConverter.convertType(
        getMBarrierMemrefType(type.getContext(), type));
  });
  llvmTypeConverter.addConversion(
      [&](nvgpu::WarpgroupMatrixDescriptorType type) -> Type {
        return llvmTypeConverter.convertType(
            IntegerType::get(type.getContext(), 64));
      });
  llvmTypeConverter.addConversion(
      [&](nvgpu::TensorMapDescriptorType type) -> Type {
        return LLVM::LLVMPointerType::get(type.getContext());
      });
  populateNVGPUToNVVMConversionPatterns(llvmTypeConverter, patterns);
}

LogicalResult
transform::ApplyNVGPUToNVVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===---------------------------------------------------------------------===//
// CreateAsyncGroupsOp
//===---------------------------------------------------------------------===//

void transform::CreateAsyncGroupsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform::CreateAsyncGroupsOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  nvgpu::createAsyncGroups(rewriter, target, getBypassL1());
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PipelineSharedMemoryCopiesOp
//===----------------------------------------------------------------------===//

/// Returns true if the given type has the default memory space.
static bool hasDefaultMemorySpace(BaseMemRefType type) {
  return !type.getMemorySpace() || type.getMemorySpaceAsInt() == 0;
}

/// Returns true if the given type has the shared (workgroup) memory space.
static bool hasSharedMemorySpace(BaseMemRefType type) {
  auto space =
      dyn_cast_if_present<gpu::AddressSpaceAttr>(type.getMemorySpace());
  return space &&
         space.getValue() == gpu::GPUDialect::getWorkgroupAddressSpace();
}

/// Returns the value produced by a load from the default memory space. Returns
/// null if the operation is not such a load.
static Value getValueLoadedFromGlobal(Operation *op) {
  // TODO: consider an interface or leveraging the memory effects interface.
  auto load = dyn_cast<vector::TransferReadOp>(op);
  if (!load)
    return nullptr;

  auto loadType = dyn_cast<MemRefType>(load.getBase().getType());
  if (!loadType || !hasDefaultMemorySpace(loadType))
    return nullptr;
  return load;
}

/// Returns true if the operation is storing the given value into shared memory.
static bool isStoreToShared(Operation *op, Value v) {
  // TOD: consider an interface or leveraging the memory effects interface.
  auto store = dyn_cast<vector::TransferWriteOp>(op);
  if (!store || store.getVector() != v)
    return false;

  auto storeType = dyn_cast<MemRefType>(store.getBase().getType());
  return storeType || hasSharedMemorySpace(storeType);
}

/// Returns true if the operation is a load from the default memory space the
/// result of which is only stored into the shared memory space.
static bool isLoadFromGlobalStoredToShared(Operation *op) {
  Value loaded = getValueLoadedFromGlobal(op);
  if (!loaded || !loaded.hasOneUse())
    return false;

  return isStoreToShared(*loaded.getUsers().begin(), loaded);
}

/// Populate `ops` with the set of operations that belong to the stage 0 of the
/// pipelined version of the given loop when pipelining copies to shared memory.
/// Specifically, this collects:
///
///   1. all loads from global memory, both sync and async;
///   2. the barriers for async loads.
///
/// In particular, barriers are omitted if they do not dominate at least one
/// async load for which there is not yet a barrier.
static LogicalResult
collectStage0PipeliningOps(scf::ForOp forOp,
                           llvm::SmallPtrSet<Operation *, 16> &ops) {

  llvm::SmallPtrSet<Operation *, 4> barriers;
  for (Operation &op : *forOp.getBody()) {
    // Bail on nested ops for now.
    if (op.getNumRegions() > 0)
      return failure();

    if (isa<gpu::BarrierOp>(op)) {
      barriers.insert(&op);
      continue;
    }

    if (isa<nvgpu::DeviceAsyncCopyOp, nvgpu::DeviceAsyncCreateGroupOp>(op)) {
      ops.insert(&op);
      ops.insert(std::make_move_iterator(barriers.begin()),
                 std::make_move_iterator(barriers.end()));
      assert(barriers.empty() &&
             "expected to have moved the barriers into another set");
      continue;
    }

    if (isLoadFromGlobalStoredToShared(&op)) {
      ops.insert(&op);
      continue;
    }
  }

  return success();
}

/// Hook for the loop pipeliner that sets the "num groups in flight" attribute
/// of async wait operations corresponding to pipelined shared memory copies.
// TODO: this currently assumes that there are no groups that could be in flight
// in the existing code.
static void
setAsyncWaitGroupsInFlight(OpBuilder &builder, Operation *op,
                           scf::PipeliningOption::PipelinerPart part,
                           unsigned iteration, unsigned depth) {
  // Based on the order of copies within the loop we need to set the number
  // of copies in flight, unless it is already set.
  auto waitOp = dyn_cast<nvgpu::DeviceAsyncWaitOp>(op);
  if (!waitOp || waitOp.getNumGroups())
    return;

  int numGroupInFlight = 0;
  if (part == scf::PipeliningOption::PipelinerPart::Kernel ||
      part == scf::PipeliningOption::PipelinerPart::Prologue) {
    numGroupInFlight = depth - 1;
  } else {
    // By construction there should be no wait op in the prologue as all the
    // wait should be in the last stage.
    assert(part == scf::PipeliningOption::PipelinerPart::Epilogue);
    // Based on the schedule we pick we know how many groups are in flight for
    // each iteration of the epilogue.
    numGroupInFlight = depth - 1 - iteration;
  }
  waitOp.setNumGroups(numGroupInFlight);
}

/// Hook for the loop pipeliner that populates `ops` with the stage information
/// as follows:
///
///   - operations in `stage0Ops` (typically loads from global memory and
///     related barriers) are at stage 0;
///   - operations in the backward slice of any stage0Ops are all at stage 0;
///   - other operations are at stage `depth`;
///   - the internal order of the pipelined loop has ops at stage `depth` first,
///   then those at stage 0, with relative order within each group preserved.
///
static void getPipelineStages(
    scf::ForOp forOp,
    std::vector<std::pair<Operation *, unsigned>> &opsWithPipelineStages,
    unsigned depth, llvm::SmallPtrSetImpl<Operation *> &stage0Ops) {
  SetVector<Operation *> dependencies;
  BackwardSliceOptions options([&](Operation *visited) {
    return visited->getBlock() == forOp.getBody();
  });
  options.inclusive = true;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (stage0Ops.contains(&op)) {
      LogicalResult result = getBackwardSlice(&op, &dependencies, options);
      assert(result.succeeded() && "expected a backward slice");
    }
  }

  for (Operation &op : forOp.getBody()->getOperations()) {
    if (!dependencies.contains(&op) && !isa<scf::YieldOp>(op))
      opsWithPipelineStages.emplace_back(&op, depth);
  }
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (dependencies.contains(&op))
      opsWithPipelineStages.emplace_back(&op, 0);
  }
}

/// Hook for the loop pipeliner. Replaces op with a predicated version and
/// returns the resulting operation. Returns the original op if the predication
/// isn't necessary for the given op. Returns null if predication is needed but
/// not supported.
static Operation *replaceOpWithPredicatedOp(RewriterBase &rewriter,
                                            Operation *op, Value predicate) {
  // Some operations may be fine to execute "speculatively" more times than the
  // original number of iterations, in particular side-effect free operations
  // and barriers, even if they cannot be predicated.
  if (isMemoryEffectFree(op) ||
      isa<gpu::BarrierOp, nvgpu::DeviceAsyncCreateGroupOp,
          nvgpu::DeviceAsyncWaitOp>(op)) {
    return op;
  }

  // Otherwise, only async copies can currently be predicated.
  auto asyncCopyOp = dyn_cast<nvgpu::DeviceAsyncCopyOp>(op);
  if (!asyncCopyOp)
    return nullptr;

  // Create srcElement Value based on `predicate`. The next lines generate
  // the following code:
  //
  //   srcElement = (pred) ?  prevSrcElements : 0;
  //
  Location loc = asyncCopyOp->getLoc();
  Value dstElements =
      rewriter.create<arith::ConstantOp>(loc, asyncCopyOp.getDstElementsAttr());
  Value originalSrcElement =
      asyncCopyOp.getSrcElements() ? asyncCopyOp.getSrcElements() : dstElements;
  Value c0Index = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto srcElements = rewriter.create<arith::SelectOp>(
      loc, predicate, originalSrcElement, c0Index);
  auto asyncCopyZeroFillOp = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
      loc, nvgpu::DeviceAsyncTokenType::get(asyncCopyOp.getContext()),
      asyncCopyOp.getDst(), asyncCopyOp.getDstIndices(), asyncCopyOp.getSrc(),
      asyncCopyOp.getSrcIndices(), asyncCopyOp.getDstElements(), srcElements,
      UnitAttr());
  rewriter.replaceOp(asyncCopyOp, asyncCopyZeroFillOp);
  return asyncCopyZeroFillOp;
}

/// Applies loop pipelining with the given depth to the given loop so that
/// copies into the shared memory are pipelined. Doesn't affect other loops.
/// Returns a pair containing the error state and the pipelined op, the latter
/// being null in case of any failure. The error state contains a definite error
/// if the IR has been modified and a silenceable error otherwise.
static std::tuple<DiagnosedSilenceableFailure, scf::ForOp>
pipelineForSharedCopies(RewriterBase &rewriter, scf::ForOp forOp, int64_t depth,
                        bool epiloguePeeling) {
  llvm::SmallPtrSet<Operation *, 16> stage0Ops;
  if (failed(collectStage0PipeliningOps(forOp, stage0Ops))) {
    return std::make_tuple(
        emitSilenceableFailure(forOp, "cannot find stage 0 ops for pipelining"),
        scf::ForOp());
  }
  if (stage0Ops.empty()) {
    return std::make_tuple(
        emitSilenceableFailure(forOp, "no shared memory copy"), scf::ForOp());
  }

  scf::PipeliningOption options;
  unsigned maxDepth = depth;
  auto setAnnotation = [&](Operation *op,
                           scf::PipeliningOption::PipelinerPart part,
                           unsigned iteration) {
    return setAsyncWaitGroupsInFlight(rewriter, op, part, iteration, maxDepth);
  };
  options.getScheduleFn =
      [&](scf::ForOp schedulingFor,
          std::vector<std::pair<Operation *, unsigned>> &ops) {
        if (schedulingFor != forOp)
          return;
        return getPipelineStages(forOp, ops, maxDepth, stage0Ops);
      };
  options.annotateFn = setAnnotation;
  if (!epiloguePeeling) {
    options.peelEpilogue = false;
    options.predicateFn = replaceOpWithPredicatedOp;
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  bool modifiedIR;
  FailureOr<scf::ForOp> maybePipelined =
      pipelineForLoop(rewriter, forOp, options, &modifiedIR);
  if (succeeded(maybePipelined)) {
    return std::make_tuple(DiagnosedSilenceableFailure::success(),
                           *maybePipelined);
  }
  return std::make_tuple(
      modifiedIR
          ? DiagnosedSilenceableFailure::definiteFailure()
          : emitSilenceableFailure(forOp, "pipelining preconditions failed"),
      scf::ForOp());
}

DiagnosedSilenceableFailure PipelineSharedMemoryCopiesOp::applyToOne(
    TransformRewriter &rewriter, scf::ForOp forOp,
    ApplyToEachResultList &results, TransformState &state) {
  auto [diag, pipelined] = pipelineForSharedCopies(
      rewriter, forOp, static_cast<int64_t>(getDepth()), getPeelEpilogue());
  if (diag.succeeded()) {
    results.push_back(pipelined);
    return DiagnosedSilenceableFailure::success();
  }
  if (diag.isDefiniteFailure()) {
    auto diag = emitDefiniteFailure("irreversible pipelining failure");
    if (!getPeelEpilogue()) {
      diag.attachNote(forOp->getLoc()) << "couldn't predicate?";
      diag.attachNote(getLoc()) << "try setting " << getPeelEpilogueAttrName();
    }
    return diag;
  }

  return std::move(diag);
}

//===----------------------------------------------------------------------===//
// RewriteMatmulAsMmaSyncOp
//===----------------------------------------------------------------------===//

/// Helper struct to encode a pair of row/column indexings in the form of
/// affine expressions.
struct RowColIndexing : private std::pair<AffineExpr, AffineExpr> {
  RowColIndexing(AffineExpr row, AffineExpr col)
      : std::pair<AffineExpr, AffineExpr>(row, col) {}

  AffineExpr row() const { return first; };
  AffineExpr col() const { return second; };

  void print(llvm::raw_ostream &os) const {
    os << "- indexing: " << first << ", " << second;
  }
};

/// Helper struct to provide a simple mapping from matmul operations to the
/// corresponding mma.sync operation. This is constrained to the case where the
/// matmul matches the mma.sync operation 1-1.
struct MmaSyncBuilder {
  MmaSyncBuilder(OpBuilder &b, Location loc, OpFoldResult laneId)
      : b(b), loc(loc), laneId(laneId) {}

  using IndexCalculator =
      std::function<SmallVector<RowColIndexing>(MLIRContext *)>;

  /// Create the mma.sync operation corresponding to `linalgOp` along with all
  /// the supporting load/store and vector operations.
  FailureOr<Operation *> buildMmaSync(LinalgOp linalgOp);

private:
  struct MmaSyncInfo {
    std::tuple<IndexCalculator, IndexCalculator, IndexCalculator> indexFns;
    std::tuple<SmallVector<int64_t>, SmallVector<int64_t>, SmallVector<int64_t>>
        vectorShapes;
    SmallVector<int64_t> mmaShape;
    bool tf32Enabled;
  };

  /// Return the specific index calculator for the given `linalgOp` or failure
  /// if the op is not supported. This is the toplevel switch that should just
  /// be Tablegen'd in the future.
  FailureOr<MmaSyncInfo> getIndexCalculators(ArrayRef<int64_t> opShape,
                                             TypeRange elementalTypes);

  //===--------------------------------------------------------------------===//
  // Instruction-specific row, column indexing expression builders.
  // These should all be declaratively specified via Tablegen in the future.
  // The Tablegen specification should be as straightforward as possible to
  // only model the existing size and type combinations.
  //===--------------------------------------------------------------------===//
  //
  // TODO: Tablegen all this.
  //===--------------------------------------------------------------------===//
  // m16n8k4 tf32 case.
  //===--------------------------------------------------------------------===//
  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  /// row =      groupID            for a0
  ///            groupID + 8        for a1
  /// col =  threadIDInGroup
  static SmallVector<RowColIndexing> m16n8k4tf32Lhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    return {RowColIndexing{groupID, threadIDInGroup},
            RowColIndexing{groupID + 8, threadIDInGroup}};
  }

  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  /// row =  threadIDInGroup
  /// col =  groupID
  static SmallVector<RowColIndexing> m16n8k4tf32Rhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    return {RowColIndexing{threadIDInGroup, groupID}};
  }

  /// From the NVIDIA doc:
  /// groupID          = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  /// row =      groupID                            for c0 and c1
  ///          groupID + 8                          for c2 and c3
  /// col =  (threadIDInGroup * 2) + (i & 0x1)    for ci   where i = {0,..,3}
  static SmallVector<RowColIndexing> m16n8k4tf32Res(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    return {RowColIndexing{groupID, threadIDInGroup * 2 + 0},
            RowColIndexing{groupID, threadIDInGroup * 2 + 1},
            RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0},
            RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1}};
  }

  //===--------------------------------------------------------------------===//
  // m16n8k16 f16 case.
  //===--------------------------------------------------------------------===//
  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  ///
  /// row =      groupID            for ai where  0 <= i < 2 || 4 <= i < 6
  ///           groupID + 8         Otherwise
  ///
  /// col =  (threadIDInGroup * 2) + (i & 0x1)          for ai where i <  4
  ///        (threadIDInGroup * 2) + (i & 0x1) + 8      for ai where i >= 4
  static SmallVector<RowColIndexing> m16n8k16f16Lhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    // clang-format off
    return {
      RowColIndexing{groupID, threadIDInGroup * 2 + 0},         // i == 0
      RowColIndexing{groupID, threadIDInGroup * 2 + 1},         // i == 1
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0},     // i == 2
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1},     // i == 3
      RowColIndexing{groupID, threadIDInGroup * 2 + 0 + 8},     // i == 4
      RowColIndexing{groupID, threadIDInGroup * 2 + 1 + 8},     // i == 5
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0 + 8}, // i == 6
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1 + 8}  // i == 7
    };
    // clang-format on
  }

  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  ///
  /// row =  (threadIDInGroup * 2) + (i & 0x1)           for bi where i <  2
  ///        (threadIDInGroup * 2) + (i & 0x1) + 8       for bi where i >= 2
  ///
  /// col = groupID
  static SmallVector<RowColIndexing> m16n8k16f16Rhs(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    // clang-format off
    return {
      RowColIndexing{threadIDInGroup * 2 + 0, groupID},        // i == 0
      RowColIndexing{threadIDInGroup * 2 + 1, groupID},        // i == 1
      RowColIndexing{threadIDInGroup * 2 + 0 + 8, groupID},    // i == 2
      RowColIndexing{threadIDInGroup * 2 + 1 + 8, groupID}     // i == 3
    };
    // clang-format on
  }

  /// From the NVIDIA doc:
  /// groupID           = %laneid >> 2
  /// threadIDInGroup = %laneid % 4
  ///
  /// row =      groupID                               for ci where i <  2
  ///          groupID + 8                             for ci where i >= 2
  ///
  /// col =  (threadIDInGroup * 2) + (i & 0x1)      for ci where i = {0,..,3}
  static SmallVector<RowColIndexing> m16n8k16f16Res(MLIRContext *ctx) {
    auto dim = getAffineDimExpr(0, ctx);
    AffineExpr groupID = dim.floorDiv(4);
    AffineExpr threadIDInGroup = dim % 4;
    // clang-format off
    return {
      RowColIndexing{groupID, threadIDInGroup * 2 + 0},        // i == 0
      RowColIndexing{groupID, threadIDInGroup * 2 + 1},        // i == 1
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 0},    // i == 2
      RowColIndexing{groupID + 8, threadIDInGroup * 2 + 1}     // i == 3
    };
    // clang-format on
  }

  //===--------------------------------------------------------------------===//
  /// Helper functions to create customizable load and stores operations. The
  /// specific shapes of each MMA instruction are passed via the
  /// IndexCalculator callback.
  //===--------------------------------------------------------------------===//
  /// Build a list of memref.load operations indexed at `(row, col)` indices
  /// that make sense for a particular MMA instruction and specified via the
  /// IndexCalculator callback.
  SmallVector<Value> buildMemRefLoads(OpBuilder &b, Location loc,
                                      OpFoldResult laneId, Value memref,
                                      const IndexCalculator &indexFn);

  /// Perform a distributed load of a vector operand of `vectorShape` for a
  /// particular MMA instruction whose `(row, col)` indices are specified via
  /// the IndexCalculator callback. Each `laneId` loads the subportion of the
  /// data that makes sense for the particular MMA operation.
  /// The `vectorShape` matches existing NVGPU dialect op specification but
  /// could also be flattened in the future if needed for simplification.
  Value buildMmaSyncMemRefLoadOperand(OpBuilder &b, Location loc,
                                      OpFoldResult laneId, Value memref,
                                      IndexCalculator indexFn,
                                      ArrayRef<int64_t> vectorShape);

  /// Build a list of memref.store operations indexed at `(row, col)` indices
  /// that make sense for a particular MMA instruction and specified via the
  /// IndexCalculator callback.
  SmallVector<Operation *> buildMemRefStores(OpBuilder &b, Location loc,
                                             ValueRange toStore,
                                             OpFoldResult laneId, Value memref,
                                             const IndexCalculator &indexFn);

  /// Perform a distributed store of a vector operand of `vectorShape` for a
  /// particular MMA instruction whose `(row, col)` indices are specified via
  /// the IndexCalculator callback. Each `laneId` loads the subportion of the
  /// data that makes sense for the particular MMA operation.
  /// The `vectorShape` matches existing NVGPU dialect op specification but
  /// could also be flattened in the future if needed for simplification.
  SmallVector<Operation *> buildMmaSyncMemRefStoreOperand(
      OpBuilder &b, Location loc, Value vectorToStore, OpFoldResult laneId,
      Value memref, IndexCalculator indexFn, ArrayRef<int64_t> vectorShape);

  OpBuilder &b;
  Location loc;
  OpFoldResult laneId;
};

//===--------------------------------------------------------------------===//
/// Helper functions to create customizable load and stores operations. The
/// specific shapes of each MMA instruction are passed via the
/// IndexCalculator callback.
//===--------------------------------------------------------------------===//

template <typename ApplyFn, typename ReduceFn>
static void foreachIndividualVectorElement(Value vector, ApplyFn applyFn,
                                           ReduceFn reduceFn) {
  VectorType vectorType = cast<VectorType>(vector.getType());
  auto vectorShape = vectorType.getShape();
  auto strides = computeStrides(vectorShape);
  for (int64_t idx = 0, e = vectorShape[0] * strides[0]; idx < e; ++idx) {
    auto indices = delinearize(idx, strides);
    reduceFn(applyFn(vector, idx, indices), idx, indices);
  }
}

SmallVector<Value>
MmaSyncBuilder::buildMemRefLoads(OpBuilder &b, Location loc,
                                 OpFoldResult laneId, Value memref,
                                 const IndexCalculator &indexFn) {
  auto aff = [&](AffineExpr e) {
    return affine::makeComposedFoldedAffineApply(b, loc, e, laneId);
  };
  SmallVector<Value> res;
  SmallVector<RowColIndexing> indexings = indexFn(b.getContext());
  for (auto indexing : indexings) {
    Value row = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.row()));
    Value col = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.col()));
    auto load = b.create<memref::LoadOp>(loc, memref, ValueRange{row, col});
    res.push_back(load);
  }
  return res;
}

Value MmaSyncBuilder::buildMmaSyncMemRefLoadOperand(
    OpBuilder &b, Location loc, OpFoldResult laneId, Value memref,
    IndexCalculator indexFn, ArrayRef<int64_t> vectorShape) {
  auto loads = buildMemRefLoads(b, loc, laneId, memref, std::move(indexFn));

  Type elementType = getElementTypeOrSelf(memref.getType());
  auto vt = VectorType::get(vectorShape, elementType);
  Value res = b.create<vector::SplatOp>(loc, vt, loads[0]);
  foreachIndividualVectorElement(
      res,
      /*applyFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        return loads[linearIdx];
      },
      /*reduceFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        res = b.create<vector::InsertOp>(loc, v, res, indices);
      });

  return res;
}

SmallVector<Operation *> MmaSyncBuilder::buildMemRefStores(
    OpBuilder &b, Location loc, ValueRange toStore, OpFoldResult laneId,
    Value memref, const IndexCalculator &indexFn) {
  auto aff = [&](AffineExpr e) {
    return affine::makeComposedFoldedAffineApply(b, loc, e, laneId);
  };
  SmallVector<Operation *> res;
  for (auto [indexing, val] :
       llvm::zip_equal(indexFn(b.getContext()), toStore)) {
    Value row = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.row()));
    Value col = getValueOrCreateConstantIndexOp(b, loc, aff(indexing.col()));
    Operation *store =
        b.create<memref::StoreOp>(loc, val, memref, ValueRange{row, col});
    res.push_back(store);
  }
  return res;
}

SmallVector<Operation *> MmaSyncBuilder::buildMmaSyncMemRefStoreOperand(
    OpBuilder &b, Location loc, Value vectorToStore, OpFoldResult laneId,
    Value memref, IndexCalculator indexFn, ArrayRef<int64_t> vectorShape) {
  SmallVector<Value> toStore;
  toStore.reserve(32);
  foreachIndividualVectorElement(
      vectorToStore,
      /*applyFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        return b.create<vector::ExtractOp>(loc, vectorToStore, indices);
      },
      /*reduceFn=*/
      [&](Value v, int64_t linearIdx, ArrayRef<int64_t> indices) {
        toStore.push_back(v);
      });
  return buildMemRefStores(b, loc, toStore, laneId, memref, std::move(indexFn));
}

static std::tuple<SmallVector<int64_t>, SmallVector<int64_t>,
                  SmallVector<int64_t>>
makeVectorShapes(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs,
                 ArrayRef<int64_t> res) {
  SmallVector<int64_t> vlhs(lhs);
  SmallVector<int64_t> vrhs(rhs);
  SmallVector<int64_t> vres(res);
  return std::make_tuple(vlhs, vrhs, vres);
}

FailureOr<MmaSyncBuilder::MmaSyncInfo>
MmaSyncBuilder::getIndexCalculators(ArrayRef<int64_t> opShape,
                                    TypeRange elementalTypes) {
  // TODO: Tablegen all this.
  Type f16 = b.getF16Type();
  Type f32 = b.getF32Type();
  if (opShape == ArrayRef<int64_t>{16, 8, 4} &&
      elementalTypes == TypeRange{f32, f32, f32}) {
    return MmaSyncInfo{std::make_tuple(&MmaSyncBuilder::m16n8k4tf32Lhs,
                                       &MmaSyncBuilder::m16n8k4tf32Rhs,
                                       &MmaSyncBuilder::m16n8k4tf32Res),
                       makeVectorShapes({2, 1}, {1, 1}, {2, 2}),
                       SmallVector<int64_t>{opShape},
                       /*tf32Enabled=*/true};
  }
  // This is the version with f16 accumulation.
  // TODO: version with f32 accumulation.
  if (opShape == ArrayRef<int64_t>{16, 8, 16} &&
      elementalTypes == TypeRange{f16, f16, f16}) {
    return MmaSyncInfo{std::make_tuple(&MmaSyncBuilder::m16n8k16f16Lhs,
                                       &MmaSyncBuilder::m16n8k16f16Rhs,
                                       &MmaSyncBuilder::m16n8k16f16Res),
                       makeVectorShapes({4, 2}, {2, 2}, {2, 2}),
                       SmallVector<int64_t>{opShape},
                       /*tf32Enabled=*/false};
  }
  return failure();
}

FailureOr<Operation *> MmaSyncBuilder::buildMmaSync(LinalgOp linalgOp) {
  Value lhsMemRef = linalgOp.getDpsInputOperand(0)->get();
  Value rhsMemRef = linalgOp.getDpsInputOperand(1)->get();
  Value resMemRef = linalgOp.getDpsInitOperand(0)->get();
  assert(cast<MemRefType>(lhsMemRef.getType()).getRank() == 2 &&
         "expected lhs to be a 2D memref");
  assert(cast<MemRefType>(rhsMemRef.getType()).getRank() == 2 &&
         "expected rhs to be a 2D memref");
  assert(cast<MemRefType>(resMemRef.getType()).getRank() == 2 &&
         "expected res to be a 2D memref");

  int64_t m = cast<MemRefType>(lhsMemRef.getType()).getShape()[0];
  int64_t n = cast<MemRefType>(rhsMemRef.getType()).getShape()[1];
  int64_t k = cast<MemRefType>(lhsMemRef.getType()).getShape()[1];
  Type lhsType = getElementTypeOrSelf(lhsMemRef.getType());
  Type rhsType = getElementTypeOrSelf(rhsMemRef.getType());
  Type resType = getElementTypeOrSelf(resMemRef.getType());

  FailureOr<MmaSyncInfo> maybeInfo =
      getIndexCalculators({m, n, k}, {lhsType, rhsType, resType});
  if (failed(maybeInfo))
    return failure();

  MmaSyncInfo info = *maybeInfo;
  auto [lhsIndexFn, rhsIndexFn, resIndexFn] = info.indexFns;
  auto [lhsShape, rhsShape, resShape] = info.vectorShapes;
  Value lhs = buildMmaSyncMemRefLoadOperand(b, loc, laneId, lhsMemRef,
                                            lhsIndexFn, lhsShape);
  Value rhs = buildMmaSyncMemRefLoadOperand(b, loc, laneId, rhsMemRef,
                                            rhsIndexFn, rhsShape);
  Value res = buildMmaSyncMemRefLoadOperand(b, loc, laneId, resMemRef,
                                            resIndexFn, resShape);
  res = b.create<nvgpu::MmaSyncOp>(loc, lhs, rhs, res, info.mmaShape,
                                   info.tf32Enabled);
  buildMmaSyncMemRefStoreOperand(b, loc, res, laneId, resMemRef, resIndexFn,
                                 resShape);
  return res.getDefiningOp();
}

DiagnosedSilenceableFailure transform::RewriteMatmulAsMmaSyncOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp linalgOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  bool fail = true;
  // TODO: more robust detection of matmulOp, with transposes etc.
  if (isa_and_nonnull<linalg::MatmulOp>(linalgOp.getOperation())) {
    // Check to not let go the matmul with extended semantic, through this
    // transform.
    if (linalgOp.hasUserDefinedMaps()) {
      return emitSilenceableError()
             << "only matmul ops with non-extended semantics are supported";
    }
    Location loc = linalgOp.getLoc();
    // TODO: more robust computation of laneId, for now assume a single warp.
    Value laneId = rewriter.create<gpu::ThreadIdOp>(
        loc, rewriter.getIndexType(), gpu::Dimension::x);
    if (succeeded(MmaSyncBuilder(rewriter, loc, laneId).buildMmaSync(linalgOp)))
      fail = false;
  }

  if (fail) {
    DiagnosedSilenceableFailure diag = emitSilenceableError()
                                       << "unsupported target op: " << linalgOp;
    diag.attachNote(linalgOp->getLoc()) << "target op";
    return diag;
  }

  rewriter.eraseOp(linalgOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Hopper builders.
//===----------------------------------------------------------------------===//

/// Helper to create the base Hopper-specific operations that are reused in
/// various other places.
struct HopperBuilder {
  HopperBuilder(RewriterBase &rewriter, Location loc)
      : rewriter(rewriter), loc(loc) {}

  TypedValue<nvgpu::MBarrierGroupType>
  buildAndInitBarrierInSharedMemory(OpFoldResult numThreads);

  /// Create tma descriptor op to initiate transfer from global to shared
  /// memory. This must be done before the launch op, on the host.
  TypedValue<nvgpu::TensorMapDescriptorType>
  buildGlobalMemRefDescriptor(TypedValue<MemRefType> memref,
                              gpu::LaunchOp launchOp);

  /// Build a tma load from global memory to shared memory using `barrier` to
  /// synchronize. Return the number of bytes that will be transferred.
  OpFoldResult
  buildTmaAsyncLoad(TypedValue<nvgpu::TensorMapDescriptorType> globalDesc,
                    TypedValue<MemRefType> sharedMemref,
                    TypedValue<nvgpu::MBarrierGroupType> barrier,
                    SmallVectorImpl<Operation *> &loadOps);
  void buildBarrierArriveTx(TypedValue<nvgpu::MBarrierGroupType> barrier,
                            ArrayRef<OpFoldResult> sizes);

  /// If threadIdx.x == 0 does TMA request + wait, else just wait.
  /// Return the operation that performs the transfer on thread0.
  // TODO: In the future, don't hardcode to thread 0 but elect a leader.
  SmallVector<Operation *> buildPredicateLoadsOnThread0(
      ArrayRef<TypedValue<nvgpu::TensorMapDescriptorType>> globalDescriptors,
      ArrayRef<TypedValue<MemRefType>> sharedMemBuffers,
      TypedValue<nvgpu::MBarrierGroupType> barrier);

  void buildTryWaitParity(TypedValue<nvgpu::MBarrierGroupType> barrier);

  RewriterBase &rewriter;
  Location loc;
};

SmallVector<Operation *> HopperBuilder::buildPredicateLoadsOnThread0(
    ArrayRef<TypedValue<nvgpu::TensorMapDescriptorType>> globalDescriptors,
    ArrayRef<TypedValue<MemRefType>> sharedMemBuffers,
    TypedValue<nvgpu::MBarrierGroupType> barrier) {
  SmallVector<Operation *> loadOps;
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value tidx = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  Value cond =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, tidx, zero);
  // clang-format off
  rewriter.create<scf::IfOp>(
    /*location=*/loc,
    /*conditional=*/cond,
    /*thenBuilder=*/
    [&](OpBuilder &lb, Location loc) {
      SmallVector<OpFoldResult> sizes;
      sizes.reserve(globalDescriptors.size());
      for (auto [desc, shmem] : llvm::zip_equal(
              globalDescriptors, sharedMemBuffers)) {
        OpFoldResult sz = buildTmaAsyncLoad(desc, shmem, barrier, loadOps);
        sizes.push_back(sz);
      }
      // TODO: Note that cutlass predeclares the barrier arrive tx before the tma.async.load.
      // This may or may not have perf implications.
      buildBarrierArriveTx(barrier, sizes);
      rewriter.create<scf::YieldOp>(loc);
    },
    /*elseBuilder=*/
    [&](OpBuilder &lb, Location loc) {
      // TODO: is this for no-thread divergence?
      // Should we just yield the size and hoist?
      buildBarrierArriveTx(barrier, getAsIndexOpFoldResult(rewriter.getContext(), 0));
      rewriter.create<scf::YieldOp>(loc);
    });
  // clang-format on
  return loadOps;
}

static Attribute getSharedAddressSpaceAttribute(OpBuilder &b) {
  return gpu::AddressSpaceAttr::get(
      b.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  // return b.getI64IntegerAttr(static_cast<int64_t>(kSharedMemorySpace));
}

TypedValue<nvgpu::MBarrierGroupType>
HopperBuilder::buildAndInitBarrierInSharedMemory(OpFoldResult numThreads) {
  auto sharedMemorySpace = getSharedAddressSpaceAttribute(rewriter);
  Value barrier = rewriter.create<nvgpu::MBarrierCreateOp>(
      loc,
      nvgpu::MBarrierGroupType::get(rewriter.getContext(), sharedMemorySpace));
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<nvgpu::MBarrierInitOp>(
      loc, barrier, getValueOrCreateConstantIndexOp(rewriter, loc, numThreads),
      zero, Value());
  rewriter.create<gpu::BarrierOp>(loc);
  return cast<TypedValue<nvgpu::MBarrierGroupType>>(barrier);
}

TypedValue<nvgpu::TensorMapDescriptorType>
HopperBuilder::buildGlobalMemRefDescriptor(TypedValue<MemRefType> memref,
                                           gpu::LaunchOp launchOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(launchOp);
  Value unrankedMemRef = rewriter.create<memref::CastOp>(
      loc,
      UnrankedMemRefType::get(memref.getType().getElementType(),
                              memref.getType().getMemorySpace()),
      memref);
  SmallVector<OpFoldResult> mixedSizes =
      memref::getMixedSizes(rewriter, loc, memref);
  SmallVector<Value> sizes =
      getValueOrCreateConstantIndexOp(rewriter, loc, mixedSizes);

  auto sharedMemorySpace = getSharedAddressSpaceAttribute(rewriter);
  Value desc = rewriter.create<nvgpu::TmaCreateDescriptorOp>(
      loc,
      nvgpu::TensorMapDescriptorType::get(
          rewriter.getContext(),
          MemRefType::Builder(memref.getType())
              .setMemorySpace(sharedMemorySpace),
          TensorMapSwizzleKind::SWIZZLE_NONE,
          TensorMapL2PromoKind::L2PROMO_NONE, TensorMapOOBKind::OOB_ZERO,
          TensorMapInterleaveKind::INTERLEAVE_NONE),
      unrankedMemRef, sizes);
  return cast<TypedValue<nvgpu::TensorMapDescriptorType>>(desc);
}

OpFoldResult HopperBuilder::buildTmaAsyncLoad(
    TypedValue<nvgpu::TensorMapDescriptorType> globalDesc,
    TypedValue<MemRefType> sharedMemref,
    TypedValue<nvgpu::MBarrierGroupType> barrier,
    SmallVectorImpl<Operation *> &loadOps) {
  MLIRContext *ctx = rewriter.getContext();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Operation *loadOp = rewriter.create<nvgpu::TmaAsyncLoadOp>(
      loc, sharedMemref, barrier, globalDesc, ValueRange{zero, zero}, zero,
      Value(), Value());
  loadOps.push_back(loadOp);
  auto mixedSizes = memref::getMixedSizes(rewriter, loc, sharedMemref);
  SmallVector<AffineExpr> symbols(mixedSizes.size());
  bindSymbolsList(ctx, llvm::MutableArrayRef{symbols});
  AffineExpr prodExprInBytes =
      computeProduct(ctx, symbols) *
      (sharedMemref.getType().getElementTypeBitWidth() / 8);
  auto res = affine::makeComposedFoldedAffineApply(rewriter, loc,
                                                   prodExprInBytes, mixedSizes);
  return res;
}

void HopperBuilder::buildBarrierArriveTx(
    TypedValue<nvgpu::MBarrierGroupType> barrier,
    ArrayRef<OpFoldResult> mixedSizes) {
  assert(!mixedSizes.empty() && "expecte non-empty sizes");
  MLIRContext *ctx = rewriter.getContext();
  SmallVector<AffineExpr> symbols(mixedSizes.size());
  bindSymbolsList(ctx, llvm::MutableArrayRef{symbols});
  AffineExpr sumExpr = computeSum(ctx, symbols);
  OpFoldResult size =
      affine::makeComposedFoldedAffineApply(rewriter, loc, sumExpr, mixedSizes);
  Value sizeVal = getValueOrCreateConstantIndexOp(rewriter, loc, size);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<nvgpu::MBarrierArriveExpectTxOp>(loc, barrier, sizeVal, zero,
                                                   Value());
}

void HopperBuilder::buildTryWaitParity(
    TypedValue<nvgpu::MBarrierGroupType> barrier) {
  Type i1 = rewriter.getI1Type();
  Value parity = rewriter.create<LLVM::ConstantOp>(loc, i1, 0);
  // 10M is an arbitrary, not too small or too big number to specify the number
  // of ticks before retry.
  // TODO: hoist this in a default dialect constant.
  Value ticksBeforeRetry =
      rewriter.create<arith::ConstantIndexOp>(loc, 10000000);
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<nvgpu::MBarrierTryWaitParityOp>(loc, barrier, parity,
                                                  ticksBeforeRetry, zero);
}

//===----------------------------------------------------------------------===//
// RewriteCopyAsTmaOp
//===----------------------------------------------------------------------===//

/// Helper to create the tma operations corresponding to `linalg::CopyOp`.
struct CopyBuilder : public HopperBuilder {
  CopyBuilder(RewriterBase &rewriter, Location loc)
      : HopperBuilder(rewriter, loc) {}

  SmallVector<Operation *> rewrite(ArrayRef<Operation *> copyOps);
};

SmallVector<Operation *> CopyBuilder::rewrite(ArrayRef<Operation *> copyOps) {
  MLIRContext *ctx = rewriter.getContext();
  if (copyOps.empty())
    return SmallVector<Operation *>();

  auto launchOp = copyOps.front()->getParentOfType<gpu::LaunchOp>();
  assert(launchOp && "expected launch op");

  // 1. Init a barrier object in shared memory.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(copyOps.front());
  AffineExpr bx, by, bz;
  bindSymbols(ctx, bx, by, bz);
  AffineExpr prod = computeProduct(ctx, ArrayRef<AffineExpr>{bx, by, bz});
  OpFoldResult numThreads = affine::makeComposedFoldedAffineApply(
      rewriter, loc, prod,
      ArrayRef<OpFoldResult>{launchOp.getBlockSizeX(), launchOp.getBlockSizeY(),
                             launchOp.getBlockSizeZ()});

  TypedValue<nvgpu::MBarrierGroupType> barrier =
      buildAndInitBarrierInSharedMemory(numThreads);

  SmallVector<TypedValue<MemRefType>> shmems;
  SmallVector<TypedValue<nvgpu::TensorMapDescriptorType>> globalDescs;
  for (Operation *op : copyOps) {
    auto copyOp = cast<linalg::CopyOp>(op);
    auto inMemRef =
        cast<TypedValue<MemRefType>>(copyOp.getDpsInputOperand(0)->get());
    assert(inMemRef.getType().getRank() == 2 &&
           "expected in to be a 2D memref");

    // 2. Build global memory descriptor.
    TypedValue<nvgpu::TensorMapDescriptorType> globalDesc =
        buildGlobalMemRefDescriptor(inMemRef, launchOp);
    globalDescs.push_back(globalDesc);

    // 3. Shared memory and descriptor for the tmp array.
    auto shmem =
        cast<TypedValue<MemRefType>>(copyOp.getDpsInitOperand(0)->get());
    shmems.push_back(shmem);
  }

  // 4. Load in from global memory to shared memory using tma.
  OpBuilder::InsertionGuard g2(rewriter);
  rewriter.setInsertionPoint(copyOps.front());
  SmallVector<Operation *> results =
      buildPredicateLoadsOnThread0(globalDescs, shmems, barrier);

  // 5. Spin-loop until data is ready.
  buildTryWaitParity(barrier);

  // 6. Erase the ops that have now been rewritten.
  for (Operation *op : copyOps)
    rewriter.eraseOp(op);

  return results;
}

DiagnosedSilenceableFailure
transform::RewriteCopyAsTmaOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  auto payloadOps = state.getPayloadOps(getTarget());
  gpu::LaunchOp commonLaunchOp;
  Operation *firstOp, *failingOp;
  if (llvm::any_of(payloadOps, [&](Operation *op) {
        if (!commonLaunchOp) {
          commonLaunchOp = op->getParentOfType<gpu::LaunchOp>();
          firstOp = op;
        }
        auto fail = !op->getParentOfType<gpu::LaunchOp>() ||
                    commonLaunchOp != op->getParentOfType<gpu::LaunchOp>() ||
                    !isa<linalg::CopyOp>(op);
        if (fail)
          failingOp = op;
        return fail;
      })) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "target ops must be linalg::CopyOp nested under a common "
           "gpu.LaunchOp to be rewritten because the tma descriptors need to "
           "be created on the host.\nBut got: "
        << *firstOp << "\nand " << *failingOp;
    return diag;
  }

  // TODO: more robust detection of copy, with transposes etc.
  CopyBuilder(rewriter, getLoc()).rewrite(llvm::to_vector(payloadOps));

  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class NVGPUTransformDialectExtension
    : public transform::TransformDialectExtension<
          NVGPUTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NVGPUTransformDialectExtension)

  NVGPUTransformDialectExtension() {
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<affine::AffineDialect>();
    declareGeneratedDialect<nvgpu::NVGPUDialect>();
    declareGeneratedDialect<NVVM::NVVMDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.cpp.inc"

void mlir::nvgpu::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<NVGPUTransformDialectExtension>();
}
