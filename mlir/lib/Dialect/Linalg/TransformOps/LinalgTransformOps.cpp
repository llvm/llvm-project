//===- LinalgTransformOps.cpp - Implementation of Linalg transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "mlir/AsmParser/AsmParser.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/GPUHeuristics.h"
#include "mlir/Dialect/Linalg/TransformOps/Syntax.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <type_traits>

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::transform;

#define DEBUG_TYPE "linalg-transforms"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

/// Attempts to apply the pattern specified as template argument to the given
/// operation. The pattern is expected to have a `returningMatchAndRewrite`
/// function that returns the "main" result or failure. Returns failure if the
/// pattern failed to apply. Extra arguments are forwarded to the pattern
/// constructor.
template <typename PatternTy, typename... Args>
static FailureOr<LinalgOp> tryApply(Operation *operation, Args &&...args) {
  // Check if the given operation has the type expected by the pattern.
  using OpTy = typename llvm::function_traits<
      decltype(&PatternTy::returningMatchAndRewrite)>::template arg_t<0>;
  auto op = dyn_cast<OpTy>(operation);
  if (!op)
    return failure();

  // Apply the pattern directly to the op.
  PatternTy pattern(operation->getContext(), std::forward<Args>(args)...);
  // We want to discourage direct use of PatternRewriter in APIs but In this
  // very specific case, an IRRewriter is not enough.
  struct TrivialPatternRewriter : public PatternRewriter {
  public:
    explicit TrivialPatternRewriter(MLIRContext *context)
        : PatternRewriter(context) {}
  };
  TrivialPatternRewriter rewriter(operation->getContext());
  rewriter.setInsertionPoint(operation);
  auto result = pattern.returningMatchAndRewrite(op, rewriter);
  if (failed(result))
    return failure();
  return cast<LinalgOp>(result->getOperation());
}

/// Assuming that `ofr` is an index attr or a param of index type
/// or a transform dialect handle mapped to exactly one op
/// with one index result, return that value.
static DiagnosedSilenceableFailure unpackSingleIndexResultPayloadOperations(
    transform::TransformState &state, TransformOpInterface transformOp,
    SmallVector<OpFoldResult> &result, ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    if (ofr.is<Attribute>()) {
      if (!isa<IntegerAttr>(ofr.get<Attribute>()))
        return transformOp.emitDefiniteFailure() << "expected IntegerAttr";
      result.push_back(ofr);
      continue;
    }

    Value transformValue = ofr.get<Value>();
    if (isa<TransformParamTypeInterface>(transformValue.getType())) {
      ArrayRef<Attribute> params = state.getParams(transformValue);
      if (params.size() != 1)
        return transformOp.emitDefiniteFailure()
               << "requires exactly one parameter associated";
      result.push_back(params[0]);
      continue;
    }

    auto payloadOps = state.getPayloadOps(transformValue);
    if (!llvm::hasSingleElement(payloadOps)) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "handle must be mapped to exactly one payload op";
      diag.attachNote(transformValue.getLoc())
          << "mapped to " << llvm::range_size(payloadOps) << " payload ops";
      return diag;
    }

    Operation *op = *payloadOps.begin();
    if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex()) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "payload op must have exactly 1 index result";
      diag.attachNote(op->getLoc())
          << "has " << op->getNumResults() << " results";
      return diag;
    }
    result.push_back(op->getResult(0));
  }

  return DiagnosedSilenceableFailure::success();
}

// Given a list of params that are index attrs or a list of OpFoldResults
// that are either index attrs or op handles, return a list of OpFoldResults
// of index attrs or a list of OpFoldResults where all op handles are
// replaced with the first (and only) OpResult of that payload op.
// (There must be exactly one parameter associated with the AnyParamType or
// one mapped payload op which must have exactly one index result.)
static DiagnosedSilenceableFailure unpackSingleIndexResultPayloadOperations(
    transform::TransformState &state, TransformOpInterface transformOp,
    SmallVector<OpFoldResult> &result, Value packedHandle) {
  if (isa<TransformParamTypeInterface>(packedHandle.getType())) {
    ArrayRef<Attribute> params = state.getParams(packedHandle);
    for (auto param : params) {
      if (!isa<IntegerAttr>(param))
        return transformOp.emitDefiniteFailure()
               << "expected the parameter to be associated with an integer "
                  "attribute";
      result.push_back(param);
    }
    return DiagnosedSilenceableFailure::success();
  }

  for (Operation *op : state.getPayloadOps(packedHandle)) {
    if (op->getNumResults() != 1 || !op->getResult(0).getType().isIndex()) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "payload op must have exactly 1 index result";
      diag.attachNote(op->getLoc())
          << "has " << op->getNumResults() << " results";
      return diag;
    }
    result.push_back(op->getResult(0));
  }

  return DiagnosedSilenceableFailure::success();
}

/// When possible, converts each `OpFoldResult` in `mixedResult` to
/// an integer if the value can be statically inferred.  If a result
/// is a `Value` then it must be either a `ParamType` or a handle
/// to an a constant like op.
static DiagnosedSilenceableFailure reifyMixedParamAndHandleResults(
    TransformState &state, TransformOpInterface &transformOp,
    ArrayRef<OpFoldResult> mixedResults, SmallVectorImpl<int64_t> &reified) {
  for (OpFoldResult paramOrHandle : mixedResults) {
    if (isa<Attribute>(paramOrHandle)) {
      reified.push_back(
          cast<IntegerAttr>(paramOrHandle.get<Attribute>()).getInt());
      continue;
    } else if (isa<ParamType>(paramOrHandle.get<Value>().getType())) {
      ArrayRef<Attribute> params = state.getParams(paramOrHandle.get<Value>());
      if (params.size() != 1)
        return transformOp.emitSilenceableError() << "expected a single param";
      reified.push_back(
          cast<IntegerAttr>(params.front()).getValue().getSExtValue());
      continue;
    }

    Value handle = paramOrHandle.get<Value>();
    if (!isa<TransformHandleTypeInterface>(handle.getType()))
      return transformOp.emitSilenceableError() << "unexpected value handle";
    auto payload = state.getPayloadOps(handle);
    if (!llvm::hasSingleElement(payload))
      return transformOp.emitSilenceableError()
             << "requires param or handle that is mapped to 1 payload op";

    Operation *paramOrHandlePayloadOp = *payload.begin();
    if (paramOrHandlePayloadOp->getNumResults() != 1 ||
        !paramOrHandlePayloadOp->getResult(0).getType().isIndex()) {
      return transformOp.emitSilenceableError()
             << "requires param or handle to be result of op with 1 index "
                "result";
    }

    IntegerAttr attr;
    if (!matchPattern(paramOrHandlePayloadOp->getResult(0), m_Constant(&attr)))
      return transformOp.emitSilenceableError()
             << "requires param or handle to be the result of a constant like "
                "op";

    reified.push_back(attr.getInt());
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// Apply...PatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyEraseUnnecessaryInputsPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::populateEraseUnnecessaryInputsPatterns(patterns);
}

void transform::ApplyFoldUnitExtentDimsViaReshapesPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::ControlDropUnitDims options;
  linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
}

void transform::ApplyFoldUnitExtentDimsViaSlicesPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::ControlDropUnitDims options;
  options.rankReductionStrategy =
      linalg::ControlDropUnitDims::RankReductionStrategy::ExtractInsertSlice;
  linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
}

void transform::ApplyTilingCanonicalizationPatternsOp::populatePatterns(
    RewritePatternSet &patterns) {
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
}

//===----------------------------------------------------------------------===//
// BufferizeToAllocationOp
//===----------------------------------------------------------------------===//

void transform::BufferizeToAllocationOp::build(OpBuilder &b,
                                               OperationState &result,
                                               Value target,
                                               Attribute memorySpace) {
  SmallVector<Type> resultTypes;
  resultTypes.push_back(b.getType<transform::AnyValueType>());
  resultTypes.push_back(b.getType<transform::AnyOpType>());
  return build(b, result,
               /*resultTypes=*/resultTypes,
               /*target=*/target,
               /*memorySpace=*/memorySpace);
}

void transform::BufferizeToAllocationOp::build(OpBuilder &b,
                                               OperationState &result,
                                               Value target,
                                               int64_t memorySpace) {
  SmallVector<Type> resultTypes;
  resultTypes.push_back(b.getType<transform::AnyValueType>());
  resultTypes.push_back(b.getType<transform::AnyOpType>());
  return build(b, result,
               /*resultTypes=*/resultTypes,
               /*target=*/target,
               /*memorySpace=*/b.getI64IntegerAttr(memorySpace));
}

namespace {
class NewOpsListener : public RewriterBase::ForwardingListener {
public:
  using RewriterBase::ForwardingListener::ForwardingListener;

  SmallVector<Operation *> getNewOps() const {
    return SmallVector<Operation *>(newOps.begin(), newOps.end());
  }

private:
  void notifyOperationInserted(Operation *op,
                               OpBuilder::InsertPoint previous) override {
    ForwardingListener::notifyOperationInserted(op, previous);
    // We only care about newly created ops.
    if (previous.isSet())
      return;
    auto inserted = newOps.insert(op);
    (void)inserted;
    assert(inserted.second && "expected newly created op");
  }

  void notifyOperationErased(Operation *op) override {
    ForwardingListener::notifyOperationErased(op);
    op->walk([&](Operation *op) { newOps.erase(op); });
  }

  DenseSet<Operation *> newOps;
};
} // namespace

DiagnosedSilenceableFailure transform::BufferizeToAllocationOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  // Attach listener to keep track of newly created ops.
  OpBuilder::Listener *previousListener = rewriter.getListener();
  auto resetListener =
      llvm::make_scope_exit([&]() { rewriter.setListener(previousListener); });
  NewOpsListener newOpsListener(previousListener);
  rewriter.setListener(&newOpsListener);

  linalg::BufferizeToAllocationOptions options;
  if (getMemcpyOp() == "bufferization.materialize_in_destination") {
    options.memcpyOp = linalg::BufferizeToAllocationOptions::MemcpyOp::
        MaterializeInDestination;
  } else if (getMemcpyOp() == "memref.copy") {
    options.memcpyOp =
        linalg::BufferizeToAllocationOptions::MemcpyOp::MemrefCopy;
  } else if (getMemcpyOp() == "linalg.copy") {
    options.memcpyOp =
        linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy;
  } else {
    llvm_unreachable("invalid memcpy op");
  }
  if (getAllocOp() == "memref.alloc") {
    options.allocOp =
        linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloc;
  } else if (getAllocOp() == "memref.alloca") {
    options.allocOp =
        linalg::BufferizeToAllocationOptions::AllocOp::MemrefAlloca;
  } else {
    llvm_unreachable("invalid alloc op");
  }
  options.bufferizeDestinationOnly = getBufferizeDestinationOnly();
  options.emitDealloc = getEmitDealloc();

  // Bufferize ops.
  Attribute memorySpace =
      getMemorySpace().has_value() ? getMemorySpace().value() : Attribute();
  SmallVector<Value> allocatedBuffers;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    Value buffer =
        linalg::bufferizeToAllocation(rewriter, options, op, memorySpace);
    if (!buffer) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "failed to bufferize operation";
      diag.attachNote(op->getLoc()) << "target payload op";
      return diag;
    }
    allocatedBuffers.push_back(buffer);
  }

  // Set results.
  results.setValues(cast<OpResult>(getAllocatedBuffer()), allocatedBuffers);
  results.set(cast<OpResult>(getNewOps()), newOpsListener.getNewOps());
  return DiagnosedSilenceableFailure::success();
}

void transform::BufferizeToAllocationOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (getBufferizeDestinationOnly()) {
    // The destination is replaced with a newly allocated buffer, but the op
    // itself remains in place.
    onlyReadsHandle(getTargetMutable(), effects);
  } else {
    consumesHandle(getTargetMutable(), effects);
  }
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult transform::BufferizeToAllocationOp::verify() {
  if (getMemcpyOp() != "bufferization.materialize_in_destination" &&
      getMemcpyOp() != "memref.copy" && getMemcpyOp() != "linalg.copy")
    return emitOpError() << "unsupported memcpy op";
  if (getAllocOp() != "memref.alloc" && getAllocOp() != "memref.alloca")
    return emitOpError() << "unsupported alloc op";
  return success();
}

//===----------------------------------------------------------------------===//
// DecomposeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::DecomposeOp::applyToOne(transform::TransformRewriter &rewriter,
                                   LinalgOp target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
#define DOWNSCALE(trans)                                                       \
  {                                                                            \
    FailureOr<LinalgOp> res = tryApply<trans>(target);                         \
    if (succeeded(res)) {                                                      \
      results.push_back(*res);                                                 \
      return DiagnosedSilenceableFailure::success();                           \
    }                                                                          \
  }

#define DOWNSCALE_CALL(a, b) DownscaleSizeOneWindowed2DConvolution<a, b>
#define DOWNSCALE_NORMAL(a, b) DOWNSCALE(DOWNSCALE_CALL(a, b))

  DOWNSCALE_NORMAL(Conv2DNhwcHwcfOp, Conv1DNwcWcfOp)
  DOWNSCALE_NORMAL(Conv2DNchwFchwOp, Conv1DNcwFcwOp)
  DOWNSCALE_NORMAL(PoolingNhwcSumOp, PoolingNwcSumOp)
  DOWNSCALE_NORMAL(PoolingNchwSumOp, PoolingNcwSumOp)
  DOWNSCALE_NORMAL(PoolingNhwcMaxOp, PoolingNwcMaxOp)
  DOWNSCALE_NORMAL(PoolingNhwcMaxUnsignedOp, PoolingNwcMaxUnsignedOp)
  DOWNSCALE_NORMAL(PoolingNhwcMinOp, PoolingNwcMinOp)
  DOWNSCALE_NORMAL(PoolingNhwcMinUnsignedOp, PoolingNwcMinUnsignedOp)
  DOWNSCALE_NORMAL(PoolingNchwMaxOp, PoolingNcwMaxOp)
  DOWNSCALE(DownscaleDepthwiseConv2DNhwcHwcOp)
  DOWNSCALE(DownscaleConv2DOp)
#undef DOWNSCALE_NORMAL
#undef DOWNSCALE_CALL
#undef DOWNSCALE
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// DecomposeInterfaceOp
//===----------------------------------------------------------------------===//

// Decompose the target operation if it implements the AggregatedOpInterface.
// Push the decomposed operations (the ones that replaces the values produced by
// \p target) in the `results`.
DiagnosedSilenceableFailure transform::DecomposeInterfaceOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto decomposableOp = dyn_cast<AggregatedOpInterface>(target);
  if (!decomposableOp) {
    failed(rewriter.notifyMatchFailure(target,
                                       "payload is not a decomposable op"));
    return emitDefaultSilenceableFailure(target);
  }

  FailureOr<SmallVector<Value>> maybeNewResults =
      decomposableOp.decomposeOperation(rewriter);
  if (failed(maybeNewResults))
    return emitDefaultSilenceableFailure(target);

  rewriter.replaceOp(decomposableOp, *maybeNewResults);
  for (Value val : *maybeNewResults) {
    Operation *definition = val.getDefiningOp();
    if (definition)
      results.push_back(definition);
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// EliminateLinalgOpAnchoredEmptyTensorsOp
//===----------------------------------------------------------------------===//

void transform::EliminateLinalgOpAnchoredEmptyTensorsOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::EliminateLinalgOpAnchoredEmptyTensorsOp::apply(
    transform::TransformRewriter &rewriter, TransformResults &transformResults,
    TransformState &state) {
  bufferization::OneShotBufferizationOptions options;
  options.allowReturnAllocsFromLoops = true;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    bufferization::OneShotAnalysisState state(target, options);
    if (failed(analyzeOp(target, state)))
      return mlir::emitSilenceableFailure(target->getLoc())
             << "failed to analyze op";
    if (failed(linalg::linalgOpAnchoredEmptyTensorEliminationStep(
            rewriter, target, state)))
      return mlir::emitSilenceableFailure(target->getLoc())
             << "failed to eliminate LinalgOp anchored tensor.empty ops";
  }
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FuseOp
//===----------------------------------------------------------------------===//

/// Apply a tiling transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
template <typename Range>
static LogicalResult applyTilingToAll(
    RewriterBase &rewriter, Operation *transformOp, Range &&payloadOps,
    unsigned numLoops, transform::TransformResults &transformResults,
    function_ref<FailureOr<scf::SCFTileAndFuseResult>(TilingInterface)>
        applyFn) {
  SmallVector<Operation *> tiledLinalgOps;
  SmallVector<SmallVector<Operation *>> loopOps(numLoops);

  for (Operation *target : payloadOps) {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
    if (!tilingInterfaceOp)
      return transformOp->emitError("only TilingInterface ops are supported");

    rewriter.setInsertionPoint(target);
    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        applyFn(tilingInterfaceOp);
    if (failed(tiledResults))
      return failure();

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{target};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res))
          rewriter.replaceAllUsesWith(res, replacement);
      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }

    // Report back the relevant handles to the transform op.
    tiledLinalgOps.push_back(tiledResults->tiledAndFusedOps.front());
    assert(tiledResults->loops.size() == numLoops &&
           "Mismatched number of loops, tile and fuse transform should have "
           "failed");
    for (unsigned int i = 0; i < numLoops; ++i)
      loopOps[i].push_back(tiledResults->loops[i]);
  }

  transformResults.set(transformOp->getOpResult(0), tiledLinalgOps);
  for (unsigned int i = 0; i < numLoops; ++i)
    transformResults.set(transformOp->getOpResult(i + 1), loopOps[i]);

  return success();
}

DiagnosedSilenceableFailure
transform::FuseOp::apply(transform::TransformRewriter &rewriter,
                         mlir::transform::TransformResults &transformResults,
                         mlir::transform::TransformState &state) {
  SmallVector<int64_t> tileSizes =
      extractFromIntegerArrayAttr<int64_t>(getTileSizes());
  SmallVector<int64_t> tileInterchange =
      extractFromIntegerArrayAttr<int64_t>(getTileInterchange());

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.interchangeVector = tileInterchange;
  SmallVector<OpFoldResult> tileSizesOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);
  tilingOptions = tilingOptions.setTileSizes(tileSizesOfr);
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.tilingOptions = tilingOptions;
  LogicalResult result = applyTilingToAll(
      rewriter, getOperation(), state.getPayloadOps(getTarget()),
      tileSizes.size() - llvm::count(tileSizes, 0), transformResults,
      [&](TilingInterface tilingInterfaceOp)
          -> FailureOr<scf::SCFTileAndFuseResult> {
        return tileConsumerAndFuseProducersUsingSCF(rewriter, tilingInterfaceOp,
                                                    tileAndFuseOptions);
      });
  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
}

LogicalResult transform::FuseOp::verify() {
  SmallVector<int64_t> permutation =
      extractFromIntegerArrayAttr<int64_t>(getTileInterchange());
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError() << "expects interchange to be a permutation, found "
                         << getTileInterchange();
  }

  SmallVector<int64_t> sizes =
      extractFromIntegerArrayAttr<int64_t>(getTileSizes());
  size_t numExpectedLoops = sizes.size() - llvm::count(sizes, 0);
  if (numExpectedLoops != getNumResults() - 1)
    return emitOpError() << "expects " << numExpectedLoops << " loop results";

  return success();
}

//===----------------------------------------------------------------------===//
// FuseIntoContainingOp
//===----------------------------------------------------------------------===//

void transform::FuseIntoContainingOp::build(OpBuilder &builder,
                                            OperationState &result,
                                            Value producerOp,
                                            Value containingOp) {
  result.addOperands({producerOp, containingOp});
  auto resultType = transform::AnyOpType::get(builder.getContext());
  result.addTypes({resultType, resultType});
}

/// Add new operands to the forall op for users of the producerOp
/// that are dominated by the containing scf.forall op.
static Operation *replaceForAllWithNewSignature(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp, TilingResult &tileAndFuseResult,
    int64_t resultNumber, SmallVector<OpFoldResult> &offsets,
    SmallVector<OpFoldResult> &sizes) {

  // Count number of users not including the containing op
  SetVector<Operation *> dominatedUsers;
  DominanceInfo domInfo(containingOp);
  for (Operation *user : producerOp->getResult(resultNumber).getUsers()) {
    if (!containingOp->isAncestor(user) &&
        (domInfo.dominates(containingOp, user))) {
      dominatedUsers.insert(user);
    }
  }
  if (dominatedUsers.empty())
    return nullptr;

  // Create new scf.forall op
  auto forallOp = cast<scf::ForallOp>(containingOp);
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forallOp);

  // Get new output
  Location loc = forallOp.getLoc();
  auto genericOp = dyn_cast<linalg::GenericOp>(producerOp);
  if (!genericOp)
    return nullptr;
  SmallVector<Value> outputs = genericOp.getOutputs();
  SmallVector<Value> newOuts(forallOp.getOutputs());
  newOuts.push_back(outputs[resultNumber]);

  // Create new scf.forall op
  auto newforallOp = rewriter.create<scf::ForallOp>(
      loc, forallOp.getMixedLowerBound(), forallOp.getMixedUpperBound(),
      forallOp.getMixedStep(), newOuts, forallOp.getMapping());
  rewriter.eraseBlock(newforallOp.getBody());
  newforallOp.getRegion().takeBody(forallOp.getRegion());

  // Add additional block argument for new value being returned
  // and replaces all uses of the new output with corresponding bbArg
  // inside the scf.forall to enable fusion into this new scf.forall.
  newforallOp.getBody()->addArgument(newOuts.back().getType(),
                                     newOuts.back().getLoc());
  auto bbArgs = newforallOp.getBody()->getArguments();
  rewriter.replaceUsesWithIf(newOuts.back(), bbArgs.back(),
                             [&](OpOperand &use) {
                               Operation *op = use.getOwner();
                               return newforallOp->isProperAncestor(op);
                             });

  // Fix terminator
  scf::InParallelOp terminatorOp = newforallOp.getTerminator();
  SmallVector<Operation *> yieldingOps = llvm::to_vector<4>(llvm::map_range(
      terminatorOp.getYieldingOps(), [](Operation &op) { return &op; }));
  Operation *firstYieldOp = yieldingOps.front();
  rewriter.setInsertionPoint(firstYieldOp);
  Value src = tileAndFuseResult.tiledValues[0];
  Value dst = newforallOp.getRegionIterArgs().back();
  SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
  rewriter.create<tensor::ParallelInsertSliceOp>(firstYieldOp->getLoc(), src,
                                                 dst, offsets, sizes, strides);

  for (auto result : llvm::enumerate(forallOp.getResults())) {
    rewriter.replaceAllUsesWith(result.value(),
                                newforallOp->getResult(result.index()));
  }
  rewriter.replaceUsesWithIf(producerOp->getResult(resultNumber),
                             newforallOp->getResults().back(),
                             [&](OpOperand &use) {
                               Operation *user = use.getOwner();
                               return dominatedUsers.contains(user);
                             });
  return newforallOp;
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
/// If tiled op has uses that are dominated by `containingOp`, return
/// a new `containingOp` with results of the fused op appended to
/// results of the `containingOp` or nullptr if there are no dominated uses.
static std::tuple<SmallVector<Operation *>, Operation *>
tileAndFuseFirstExtractUse(RewriterBase &rewriter, Diagnostic &diag,
                           Operation *producerOp, Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return {};
  }

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto it = llvm::find_if(tileableProducer->getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (it == tileableProducer->getUsers().end()) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find fusion opportunity for: " << *tileableProducer;
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  int64_t resultNumber =
      cast<OpResult>(sliceOpToTile.getSource()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  SmallVector<OpFoldResult> offsets = sliceOpToTile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOpToTile.getMixedSizes();

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducer.generateResultTileValue(rewriter, resultNumber, offsets,
                                               sizes);

  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

#ifndef NDEBUG
  for (auto *tiledOp : tileAndFuseResult->tiledOps) {
    LLVM_DEBUG(DBGS() << "tiledProducer: " << *tiledOp << "\n");
  }
#endif

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  if (failed(maybeRankReduced)) {
    diag.attachNote(producerOp->getLoc())
        << "shape types don't match (missing canonicalization?):\nTiledOp: "
        << tileAndFuseResult->tiledValues[0]
        << "\nSliceOp: " << sliceOpToTile.getOperation() << '\n';
    return {};
  }
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Add new outputs to containing op, if required
  Operation *newContainingOp = replaceForAllWithNewSignature(
      rewriter, diag, producerOp, containingOp, *tileAndFuseResult,
      resultNumber, offsets, sizes);

  return std::make_tuple(tileAndFuseResult->tiledOps, newContainingOp);
}

/// First, find the first "scf::ForallOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static SmallVector<Operation *>
tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return {};
  }

  // Search the first use by a "scf::ForallOp" user.
  scf::ForallOp forallOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        forallOp = dyn_cast<scf::ForallOp>(use.getOwner());
        return forallOp;
      });
  // If it's not from the containing op, return.
  if (!forallOp || forallOp != containingOp) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find a use by the containing op: " << *tileableProducer;
    return {};
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = forallOp.getTiedBlockArgument(pUse);

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples, maybe
  // evolve into an interface.
  auto itBBArgUsers = llvm::find_if(bbArg.getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (itBBArgUsers == bbArg.getUsers().end()) {
    diag.attachNote(containingOp->getLoc())
        << "could not find fusion opportunity for bbArg: " << bbArg;
    return {};
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = cast<OpResult>(pUse->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to get destination tensors for: " << *tileableProducer;
    return {};
  }

  IRMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }

  // Replace the extract op.
  auto maybeRankReduced = tensor::ExtractSliceOp::rankReduceIfNeeded(
      rewriter, sliceOpToTile->getLoc(), tileAndFuseResult->tiledValues[0],
      cast<RankedTensorType>(sliceOpToTile->getResult(0).getType()).getShape());
  assert(succeeded(maybeRankReduced) && "unexpected shape");
  rewriter.replaceOp(sliceOpToTile, *maybeRankReduced);

  // Replace the use in containingOp.
  rewriter.modifyOpInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return tileAndFuseResult->tiledOps;
}

static Operation *cloneAndFuseFirstUse(RewriterBase &rewriter, Diagnostic &diag,
                                       Operation *producerOp,
                                       Operation *containingOp) {
  LLVM_DEBUG(DBGS() << "Try to fuse an use by cloning\n");

  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      if (containingOp->isProperAncestor(use.getOwner())) {
        uses.push_back(&use);
        continue;
      }
      // Cannot clone and fuse if the use is by the containing op itself: fail
      // immediately.
      if (containingOp == use.getOwner()) {
        diag.attachNote(producerOp->getLoc())
            << "producer op use by containing op cannot be fused by cloning";
        return nullptr;
      }
    }
  }

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty()) {
    diag.attachNote(producerOp->getLoc()) << "no fusion opportunity by cloning";
    return nullptr;
  }

  // Clone and fuse inside the containing op.
  Operation *fusedOp = nullptr;
  OpOperand *use = uses.front();
  // Parallel insert slice is not a valid clone destination.
  // TODO: Generalize to other type of ops.
  assert(!isa<tensor::ParallelInsertSliceOp>(use->getOwner()) &&
         "Parallel insert slice is not a valid clone destination");
  unsigned resultNumber = cast<OpResult>(use->get()).getResultNumber();
  LLVM_DEBUG(DBGS() << "resultNumber: " << resultNumber << "\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use->getOwner());
  fusedOp = rewriter.clone(*producerOp);
  rewriter.modifyOpInPlace(
      use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });

  return fusedOp;
}

bool transform::FuseIntoContainingOp::allowsRepeatedHandleOperands() {
  // Allow repeated handles since we are fusing everything anyway.
  return true;
}

DiagnosedSilenceableFailure
transform::FuseIntoContainingOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  auto producerOps = state.getPayloadOps(getProducerOp());
  auto containingOps = state.getPayloadOps(getContainingOp());
  if (!llvm::hasSingleElement(containingOps)) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << llvm::range_size(containingOps) << ")";
  }
  Operation *containingOp = *containingOps.begin();

  // If nothing to fuse, propagate success.
  if (std::empty(producerOps)) {
    results.set(cast<OpResult>(getFusedOp()), SmallVector<mlir::Operation *>{});
    results.set(cast<OpResult>(getNewContainingOp()), {containingOp});
    return DiagnosedSilenceableFailure::success();
  }

  // Helper function to find the next producer that should be fused. Take any
  // producer that has a use inside the containing op.
  SetVector<Operation *> remainingProducers(producerOps.begin(),
                                            producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<Operation *> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      // The containing op may be a user of producerOp: use isAncestor.
      int64_t numUsesInContainingOp =
          llvm::count_if(producerOp->getUsers(), [&](Operation *op) {
            return containingOp->isAncestor(op);
          });
      // TODO: When resolving the TODO below (no duplicate ops), take an op
      // that has no use among the remaining producers. This is a topological
      // sorting.
      if (numUsesInContainingOp > 0) {
        if (numUsesInContainingOp == 1)
          remainingProducers.erase(remainingProducers.begin() + it.index());
        return producerOp;
      }
    }
    return failure();
  };

  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      auto diag = mlir::emitSilenceableFailure(getLoc())
                  << "could not find next producer to fuse into container";
      diag.attachNote(containingOp->getLoc()) << "containing op";
      return diag;
    }

    Operation *producerOp = *nextProducer;

    // Default diagnostic, to be complemented with more failure information.
    Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not fuse " << *producerOp << " into " << *containingOp;

    // TODO: If there are multiple uses of the producer in the containing op,
    // we currently tile/clone the op multiple times (once per use). In some
    // cases, we can tile/clone once and reuse the value for each use.
    // Futhermore, producers should then be traversed according to a
    // topological sorting.
    auto [tiledOps, newContainingOp] =
        tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp);
    if (!tiledOps.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused a direct extract use\n" << *containingOp);
      fusedOps.append(tiledOps);
      if (newContainingOp) {
        // Update handles associated with the containing op so we don't need to
        // invalidate them. This is a hack to support better composability
        // between tiling and fusion while a proper mechanism is being
        // investigated.
        //
        // DO NOT replicate this elsewhere unless you understand what you are
        // doing.
        LogicalResult replacementStatus =
            rewriter.notifyPayloadOperationReplaced(containingOp,
                                                    newContainingOp);
        (void)replacementStatus;
        assert(succeeded(replacementStatus) &&
               "unable to update transform state mapping");
        rewriter.eraseOp(containingOp);
        containingOp = newContainingOp;
      }
      continue;
    }

    SmallVector<Operation *> tiledContainingOpOperand =
        tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, diag, producerOp, containingOp);
    if (!tiledContainingOpOperand.empty()) {
      LLVM_DEBUG(DBGS() << "\nFused an extract use through block argument\n"
                        << *containingOp);
      fusedOps.append(tiledContainingOpOperand);
      continue;
    }

    Operation *cloned =
        cloneAndFuseFirstUse(rewriter, diag, producerOp, containingOp);
    if (cloned) {
      LLVM_DEBUG(DBGS() << "\nFused an use by cloning\n" << *containingOp);
      fusedOps.push_back(cloned);
      continue;
    }
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  results.set(cast<OpResult>(getFusedOp()), fusedOps);
  results.set(cast<OpResult>(getNewContainingOp()), {containingOp});
  return DiagnosedSilenceableFailure::success();
}

void transform::FuseIntoContainingOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// GeneralizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GeneralizeOp::applyToOne(transform::TransformRewriter &rewriter,
                                    LinalgOp target,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  // Exit early if no transformation is needed.
  if (isa<GenericOp>(target)) {
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }
  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> generic = generalizeNamedOp(rewriter, target);
  if (succeeded(generic)) {
    results.push_back(generic->getOperation());
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// SpecializeOp
//===----------------------------------------------------------------------===/

DiagnosedSilenceableFailure
transform::SpecializeOp::applyToOne(transform::TransformRewriter &rewriter,
                                    LinalgOp target,
                                    transform::ApplyToEachResultList &results,
                                    transform::TransformState &state) {
  // Exit early if the operation is not a generic.
  if (!isa<GenericOp>(target)) {
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }
  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> named =
      specializeGenericOp(rewriter, cast<GenericOp>(target));
  if (succeeded(named)) {
    results.push_back(named->getOperation());
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// InterchangeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::InterchangeOp::applyToOne(transform::TransformRewriter &rewriter,
                                     GenericOp target,
                                     transform::ApplyToEachResultList &results,
                                     transform::TransformState &state) {
  ArrayRef<int64_t> interchangeVector = getIteratorInterchange();
  // Exit early if no transformation is needed.
  if (interchangeVector.empty()) {
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }

  unsigned numLoops = cast<LinalgOp>(target.getOperation()).getNumLoops();
  if (interchangeVector.size() != numLoops) {
    return emitSilenceableError()
           << getIteratorInterchangeAttrName() << " has length ("
           << interchangeVector.size()
           << ") different from the number of loops in the target operation ("
           << numLoops << ")";
  }
  FailureOr<GenericOp> res =
      interchangeGenericOp(rewriter, target,
                           SmallVector<unsigned>(interchangeVector.begin(),
                                                 interchangeVector.end()));
  if (failed(res))
    return emitDefiniteFailure() << "failed to apply";
  results.push_back(res->getOperation());
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::InterchangeOp::verify() {
  ArrayRef<int64_t> permutation = getIteratorInterchange();
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError()
           << "expects iterator_interchange to be a permutation, found "
           << getIteratorInterchange();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// LowerPackOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerPackOp::applyToOne(
    transform::TransformRewriter &rewriter, tensor::PackOp target,
    transform::ApplyToEachResultList &transformResults,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  FailureOr<LowerPackResult> res = lowerPack(rewriter, target);
  if (failed(res)) {
    return mlir::emitSilenceableFailure(target->getLoc())
           << "cannot lower to pad + expand + transpose";
  }
  transformResults.push_back(res->padOp);
  transformResults.push_back(res->expandShapeOp);
  transformResults.push_back(res->transposeOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LowerUnPackOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::LowerUnPackOp::applyToOne(
    transform::TransformRewriter &rewriter, tensor::UnPackOp target,
    transform::ApplyToEachResultList &transformResults,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  FailureOr<LowerUnPackOpResult> res = lowerUnPack(rewriter, target);
  if (failed(res)) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "cannot lower to transpose + collapse + extract";
    diag.attachNote(target->getLoc()) << "target payload op";
    return diag;
  }
  transformResults.push_back(res->emptyOp);
  transformResults.push_back(res->transposeOp);
  transformResults.push_back(res->collapseShapeOp);
  transformResults.push_back(res->extractSliceOp);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// MatchOp
//===---------------------------------------------------------------------===//

void transform::MatchOp::build(OpBuilder &builder, OperationState &result,
                               Value target, ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::MatchOp::build(OpBuilder &builder, OperationState &result,
                               TypeRange resultTypes, Value target,
                               ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(resultTypes);
}

DiagnosedSilenceableFailure
transform::MatchOp::apply(transform::TransformRewriter &rewriter,
                          transform::TransformResults &results,
                          transform::TransformState &state) {
  llvm::StringSet<> strs;
  if (getOps().has_value())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  auto payloadOps = state.getPayloadOps(getTarget());
  if (!llvm::hasSingleElement(payloadOps)) {
    return emitDefiniteFailure("requires exactly one target handle");
  }

  SmallVector<Operation *> res;
  bool incorrectNumOperandTypes = false;
  auto matchFun = [&](Operation *op) {
    if (getOps().has_value() && !strs.contains(op->getName().getStringRef()))
      return;

    // Interfaces cannot be matched by name, just by ID.
    // So we specifically encode the interfaces we care about for this op.
    if (getInterface().has_value()) {
      auto iface = getInterface().value();
      if (iface == transform::MatchInterfaceEnum::LinalgOp &&
          !isa<LinalgOp>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::TilingInterface &&
          !isa<TilingInterface>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::LoopLikeInterface &&
          !isa<LoopLikeOpInterface>(op))
        return;
    }

    // Check if all specified attributes match.
    if (getOpAttrs().has_value()) {
      DictionaryAttr opAttrs = getOpAttrs().value();
      for (NamedAttribute attr : opAttrs) {
        if (attr.getName() == getInterfaceAttrName() ||
            attr.getName() == getOpsAttrName())
          continue;
        if (!op->hasAttr(attr.getName()))
          return;
        if (op->getAttr(attr.getName()) != attr.getValue())
          return;
      }
    }

    if (getFilterResultType().has_value()) {
      Type t = getFilterResultType().value();
      if (op->getNumResults() != 1 || op->getResultTypes().front() != t)
        return;
    }

    if (getFilterOperandTypes().has_value()) {
      mlir::ArrayAttr types = getFilterOperandTypes().value();
      auto operandTypes = op->getOperandTypes();

      if (types.size() == 1) {
        // All the operands must must be equal to the specified type
        auto typeattr =
            dyn_cast<mlir::TypeAttr>(getFilterOperandTypes().value()[0]);
        Type t = cast<::mlir::Type>(typeattr.getValue());
        if (!llvm::all_of(op->getOperandTypes(),
                          [&](Type operandType) { return operandType == t; }))
          return;
      } else {
        // The operand types must match all the types in the list (in the same
        // order in with they are specified)
        if (types.size() != operandTypes.size()) {
          incorrectNumOperandTypes = true;
          return;
        }

        for (auto [attr, operandType] :
             llvm::zip_equal(getFilterOperandTypes().value(), operandTypes)) {
          auto typeattr = cast<mlir::TypeAttr>(attr);
          Type type = cast<::mlir::Type>(typeattr.getValue());

          if (type != operandType)
            return;
        }
      }
    }

    // All constraints are satisfied.
    res.push_back(op);
    return;
  };

  (*payloadOps.begin())->walk(matchFun);
  if (incorrectNumOperandTypes)
    return emitDefiniteFailure("If filter_operand_types contains more than a "
                               "type, then it must contain as much types as "
                               "the number of operands in the target ops");
  results.set(cast<OpResult>(getResult()), res);
  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// MultiTileSizesOp
//===---------------------------------------------------------------------===//

static void printMultitileSizesTypes(OpAsmPrinter &printer, Operation *op,
                                     Type targetType, Type lowSizeType, Type,
                                     Type) {
  printer.printFunctionalType(TypeRange{targetType}, TypeRange{lowSizeType});
}

static ParseResult parseMultitileSizesTypes(OpAsmParser &parser,
                                            Type &targetType, Type &lowSizeType,
                                            Type &highSizeType,
                                            Type &splitPointType) {
  FunctionType funcType;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseType<FunctionType>(funcType)))
    return failure();

  if (funcType.getNumInputs() != 1 || funcType.getNumResults() != 1) {
    parser.emitError(typeLoc) << "expects a trailing functional type with one "
                                 "argument and one result";
  }
  targetType = funcType.getInput(0);
  lowSizeType = highSizeType = splitPointType = funcType.getResult(0);

  return success();
}

DiagnosedSilenceableFailure transform::MultiTileSizesOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp target,
    transform::ApplyToEachResultList &results, TransformState &state) {
  if (isa<TransformParamTypeInterface>(getLowSize().getType())) {
    if (target.hasDynamicShape()) {
      auto diag = emitSilenceableError()
                  << "cannot compute parametric tile sizes for dynamically "
                     "shaped payload op";
      diag.attachNote(target->getLoc()) << "payload op";
      return diag;
    }

    FailureOr<StaticMultiSizeSpecification> spec = computeStaticMultiTileSizes(
        target, getDimension(), getTargetSize(), getDivisor());
    if (failed(spec)) {
      return emitSilenceableError()
             << "failed to compute multi-size tiling sizes";
    }

    Builder builder(target.getContext());
    results.assign(llvm::map_range(
        ArrayRef<int64_t>({spec->lowTileSize, spec->highTileSize,
                           spec->lowTileSize * spec->lowTripCount}),
        [&builder, this](int64_t value) {
          return builder.getIntegerAttr(
              cast<ParamType>(getLowSize().getType()).getType(), value);
        }));
    return DiagnosedSilenceableFailure::success();
  }

  OpBuilder builder(target.getContext());
  builder.setInsertionPoint(target);
  OpFoldResult targetSize = builder.getIndexAttr(getTargetSize());
  OpFoldResult divisor = builder.getIndexAttr(getDivisor());
  FailureOr<MultiSizeSpecification> spec = computeMultiTileSizes(
      builder, target, getDimension(), targetSize, divisor);
  if (failed(spec)) {
    return emitSilenceableError() << "could not generate tile size computation";
  }

  AffineExpr s0 = builder.getAffineSymbolExpr(0);
  AffineExpr s1 = builder.getAffineSymbolExpr(1);
  Operation *splitPoint =
      affine::makeComposedAffineApply(builder, target.getLoc(), s0 * s1,
                                      {spec->lowTileSize, spec->lowTripCount});
  Operation *lowTileSize = spec->lowTileSize.getDefiningOp();
  Operation *highTileSize = spec->highTileSize.getDefiningOp();
  assert(lowTileSize && highTileSize && splitPoint &&
         "tile sizes are not produced by operations");
  results.reserve(results.size() + 3);
  results.push_back(lowTileSize);
  results.push_back(highTileSize);
  results.push_back(splitPoint);
  return DiagnosedSilenceableFailure::success();
}

void transform::MultiTileSizesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  if (isa<TransformParamTypeInterface>(getLowSize().getType()))
    onlyReadsPayload(effects);
  else
    modifiesPayload(effects);
}

LogicalResult transform::MultiTileSizesOp::verify() {
  if (getLowSize().getType() != getHighSize().getType() ||
      getLowSize().getType() != getSplitPoint().getType()) {
    return emitOpError() << "expects all results type to be the same";
  }
  return success();
}

//===---------------------------------------------------------------------===//
// PackOp
//===---------------------------------------------------------------------===//

void transform::PackOp::build(OpBuilder &builder, OperationState &result,
                              Value target,
                              ArrayRef<OpFoldResult> mixedPackedSizes) {
  SmallVector<int64_t> staticPackedSizes;
  SmallVector<Value> dynamicPackedSizes;
  dispatchIndexOpFoldResults(mixedPackedSizes, dynamicPackedSizes,
                             staticPackedSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  Type linalgOpHType = transform::OperationType::get(
      builder.getContext(), GenericOp::getOperationName());
  build(builder, result,
        /*resultType=*/linalgOpHType,
        /*target=*/target,
        /*dynamic_sizes=*/dynamicPackedSizes,
        /*static_sizes=*/builder.getDenseI64ArrayAttr(staticPackedSizes));
}

SmallVector<OpFoldResult> transform::PackOp::getMixedPackedSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticPackedSizes(), getPackedSizes(), b);
}

DiagnosedSilenceableFailure
transform::PackOp::apply(transform::TransformRewriter &rewriter,
                         transform::TransformResults &transformResults,
                         transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  // If nothing to pack, propagate success.
  if (std::empty(targetOps)) {
    transformResults.set(cast<OpResult>(getPackedOp()),
                         ArrayRef<Operation *>({}));
    return DiagnosedSilenceableFailure::success();
  }
  // Fail on multi-op handles.
  auto linalgOp = dyn_cast<LinalgOp>(*targetOps.begin());
  if (!llvm::hasSingleElement(targetOps) || !linalgOp) {
    return emitSilenceableError()
           << "requires target to map to exactly 1 LinalgOp (got "
           << llvm::range_size(targetOps) << ")";
  }
  // Fail on mismatched number of pack sizes.
  if (getMixedPackedSizes().size() != linalgOp.getNumLoops()) {
    return emitSilenceableError()
           << "requires number of packed sizes match the number of loops ("
           << getMixedPackedSizes().size() << " vs " << linalgOp.getNumLoops()
           << ")";
  }

  // Unpack handles to constants or actual SSA index values.
  SmallVector<OpFoldResult> packedSizes;
  DiagnosedSilenceableFailure status = unpackSingleIndexResultPayloadOperations(
      state, *this, packedSizes, getMixedPackedSizes());

  rewriter.setInsertionPoint(linalgOp);
  FailureOr<PackResult> maybeResult = pack(rewriter, linalgOp, packedSizes);
  if (failed(maybeResult))
    return emitDefiniteFailure("data tiling failed");

  transformResults.set(cast<OpResult>(getPackedOp()),
                       {maybeResult->packedLinalgOp.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

void transform::PackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::onlyReadsHandle(getPackedSizesMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// PackGreedilyOp.
//===---------------------------------------------------------------------===//

LogicalResult transform::PackGreedilyOp::verify() {
  if (!isPermutationVector(getMatmulInnerDimsOrder())) {
    return emitOpError() << getMatmulInnerDimsOrderAttrName()
                         << " is not a valid permutation";
  }
  // TODO: relax to allow empty once we have another strategy than just matmul.
  if (!getMatmulPaddedSizesNextMultipleOf().empty()) {
    for (auto [s, nmo] :
         llvm::zip_equal(getMixedMatmulPackedSizes(),
                         getMatmulPaddedSizesNextMultipleOf())) {
      std::optional<int64_t> maybeStaticPackedSize = getConstantIntValue(s);
      if (nmo != 0 &&
          (!maybeStaticPackedSize.has_value() || *maybeStaticPackedSize != 0)) {
        return emitOpError() << "at most one of the packed_size and the "
                                "padded_sizes_next_multiple_of can be nonzero "
                                "for the matmul strategy";
      }
    }
  }
  return success();
}

DiagnosedSilenceableFailure
PackGreedilyOp::apply(transform::TransformRewriter &rewriter,
                      transform::TransformResults &transformResults,
                      transform::TransformState &state) {
  SmallVector<Operation *> results;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    auto linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      continue;
    // linalgOp will be replaced and the insertion point may be invalidated if
    // we set it before -> set it after.
    rewriter.setInsertionPointAfter(linalgOp);
    // Failing to pack greedily is perfectly fine.
    // In the future we will want to order packings according to some metric.
    FailureOr<PackResult> packResult = packMatmulGreedily(
        /*rewriter=*/rewriter,
        /*linalgOp=*/linalgOp,
        /*mnkPackedSizes=*/getMixedMatmulPackedSizes(),
        /*mnkPaddedSizesNextMultipleOf=*/
        getMatmulPaddedSizesNextMultipleOf(),
        /*mnkOrder=*/getMatmulInnerDimsOrder());
    if (succeeded(packResult)) {
      results.push_back(packResult->packedLinalgOp);
      continue;
    }
    results.push_back(linalgOp);
  }
  transformResults.set(cast<OpResult>(getPackedOp()), results);
  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> PackGreedilyOp::getMixedMatmulPackedSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticMatmulPackedSizes(), getMatmulPackedSizes(),
                        b);
}

void transform::PackGreedilyOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(getTargetMutable(), effects);
  transform::onlyReadsHandle(getMatmulPackedSizesMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// PackTransposeOp
//===---------------------------------------------------------------------===//

LogicalResult transform::PackTransposeOp::verify() {
  if (!isPermutationVector(getInnerPerm())) {
    return emitOpError() << getInnerPermAttrName()
                         << " is not a valid permutation";
  }
  if (!isPermutationVector(getOuterPerm())) {
    return emitOpError() << getOuterPermAttrName()
                         << " is not a valid permutation";
  }
  if (getInnerPerm().empty() && getOuterPerm().empty()) {
    return emitOpError() << " at least one of " << getInnerPermAttrName()
                         << " or " << getOuterPermAttrName()
                         << " must be specified";
  }
  return success();
}

namespace {
enum class OuterOrInnerPerm { Outer = 0, Inner = 1 };
} // namespace

/// Return true if `permutation` is a valid permutation of the
/// `outer_dims_perm` (case OuterOrInnerPerm::Outer) or `inner_dims_pos`
/// (OuterOrInnerPerm::Inner) of the `tensor.pack` or `tensor.unpack` `op.
/// This is the case when the `permutation` rank matches the rank expected by
/// `op` and `permutation` is itself a permutation vector.
/// Return true if either `op` or `permutation` are empty to allow a simpler
/// polymorphic implementation.
template <typename RelayoutOpTy>
bool isValidPackingPermutation(
    RelayoutOpTy op, ArrayRef<int64_t> permutation,
    OuterOrInnerPerm outerOrInnerPerm = OuterOrInnerPerm::Outer) {
  static_assert(
      llvm::is_one_of<RelayoutOpTy, tensor::PackOp, tensor::UnPackOp>::value,
      "applies to only pack or unpack operations");
  if (!op || permutation.empty())
    return true;
  size_t innerRank = op.getInnerDimsPos().size();
  if (outerOrInnerPerm == OuterOrInnerPerm::Inner)
    return permutation.size() == innerRank && isPermutationVector(permutation);
  // op.getOuterDimsPerm() may be empty, in which case it is identity.
  // Don't rely on it.
  if (std::is_same<RelayoutOpTy, tensor::PackOp>::value) {
    return permutation.size() == op.getSourceRank() &&
           isPermutationVector(permutation);
  }
  return permutation.size() == op.getDestRank() &&
         isPermutationVector(permutation);
}

DiagnosedSilenceableFailure
transform::PackTransposeOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &transformResults,
                                  transform::TransformState &state) {
  auto packOrUnpackOps = state.getPayloadOps(getTargetPackOrUnPackOp());
  auto linalgOps = state.getPayloadOps(getTargetLinalgOp());
  // Step 1. If nothing to pack, propagate success.
  if (std::empty(packOrUnpackOps)) {
    transformResults.set(cast<OpResult>(getPackedOp()), {});
    transformResults.set(cast<OpResult>(getPackOp()), {});
    transformResults.set(cast<OpResult>(getUnPackOp()), {});
    return DiagnosedSilenceableFailure::success();
  }

  // Step 2. Bunch of runtime sanity check and error messages.
  // Step 2.1. Fail on multi-op handles.
  if (!llvm::hasSingleElement(packOrUnpackOps) ||
      !llvm::hasSingleElement(linalgOps)) {
    return emitSilenceableError()
           << "requires target to map to exactly 1 "
              "packing op and 1 packed op ("
           << "got " << llvm::range_size(packOrUnpackOps) << " and "
           << llvm::range_size(linalgOps) << ")";
  }

  // Step 2.2. Fail on wrong type.
  auto packOp = dyn_cast<tensor::PackOp>(*packOrUnpackOps.begin());
  auto unPackOp = dyn_cast<tensor::UnPackOp>(*packOrUnpackOps.begin());
  if ((!packOp && !unPackOp)) {
    return emitSilenceableError() << "requires target to map to a "
                                     "tensor.pack or tensor.unpack";
  }
  LinalgOp linalgOpTarget = dyn_cast<LinalgOp>(*linalgOps.begin());
  if (!linalgOpTarget)
    return emitSilenceableError() << "requires a LinalgOp target";

  // Step 2.3. Fail if we can't get the producer / consumer Linalg op.
  LinalgOp linalgOp;
  if (packOp && packOp.getResult().hasOneUse())
    linalgOp = dyn_cast<LinalgOp>(*(packOp.getResult().getUsers().begin()));
  else if (unPackOp)
    linalgOp = unPackOp.getSource().getDefiningOp<LinalgOp>();
  if (linalgOp != linalgOpTarget) {
    auto errorMsg =
        packOp ? StringLiteral{"not a single use by the LinalgOp target"}
               : StringLiteral{"not produced by the LinalgOp target"};
    return emitSilenceableError() << errorMsg;
  }

  // Step 2.4. If we have an UnPackOp, we need to fetch the symmetrical
  // PackOp.
  if (unPackOp) {
    assert(!packOp && "packOp must be null on entry when unPackOp is not null");
    OpOperand *packUse = linalgOp.getDpsInitOperand(
        cast<OpResult>(unPackOp.getSource()).getResultNumber());
    packOp = dyn_cast_or_null<tensor::PackOp>(packUse->get().getDefiningOp());
    if (!packOp || !packOp.getResult().hasOneUse())
      return emitSilenceableError() << "could not find matching pack op";
  }

  // Step 2.5. Fail if any permutation does not validate.
  for (auto permType : {OuterOrInnerPerm::Outer, OuterOrInnerPerm::Inner}) {
    ArrayRef<int64_t> perm =
        (permType == OuterOrInnerPerm::Outer) ? getOuterPerm() : getInnerPerm();
    auto errorMsg = (permType == OuterOrInnerPerm::Outer)
                        ? StringLiteral{"invalid outer_perm"}
                        : StringLiteral{"invalid inner_perm"};
    if (!isValidPackingPermutation(packOp, perm, permType) ||
        !isValidPackingPermutation(unPackOp, perm, permType)) {
      Operation *packOrUnpackOp =
          unPackOp ? unPackOp.getOperation() : packOp.getOperation();
      return emitSilenceableError() << errorMsg << ": " << *packOrUnpackOp;
    }
  }

  // From here on, packOp and linalgOp are always present, unPackOp may or may
  // not be present.
  assert(packOp && linalgOp && "unexpected null op");

  // Step 3. Actually transpose the ops.
  FailureOr<PackTransposeResult> res = packTranspose(
      rewriter, packOp, linalgOp, unPackOp, getOuterPerm(), getInnerPerm());
  // Preconditions have been checked, it is an error to fail here.
  assert(succeeded(res) && "unexpected packTranspose failure");

  // Step 4. Return results.
  transformResults.set(cast<OpResult>(getPackOp()), {res->transposedPackOp});
  transformResults.set(cast<OpResult>(getPackedOp()),
                       {res->transposedLinalgOp});
  if (unPackOp) {
    transformResults.set(cast<OpResult>(getUnPackOp()),
                         {res->transposedUnPackOp});
  } else {
    transformResults.set(cast<OpResult>(getUnPackOp()), {});
  }

  return DiagnosedSilenceableFailure::success();
}

//===---------------------------------------------------------------------===//
// PadOp
//===---------------------------------------------------------------------===//

void transform::PadOp::build(OpBuilder &b, OperationState &result, Value target,
                             ArrayRef<int64_t> paddingDimensions,
                             ArrayRef<int64_t> padToMultipleOf,
                             ArrayRef<int64_t> packPaddings,
                             ArrayRef<Attribute> transposePaddings,
                             StringRef copyBackOp) {
  auto resultType = transform::AnyOpType::get(b.getContext());
  return build(/*builder=*/b,
               /*result=*/result,
               /*types=*/TypeRange{resultType, resultType},
               /*target=*/target,
               /*paddingValues=*/ArrayAttr(), // let inference handle this
               /*paddingDimensions=*/b.getI64ArrayAttr(paddingDimensions),
               /*padToMultipleOf=*/ValueRange{},
               /*padToMultipleOf=*/
               (padToMultipleOf.empty()
                    ? DenseI64ArrayAttr()
                    : b.getDenseI64ArrayAttr(padToMultipleOf)),
               /*packPaddings=*/b.getI64ArrayAttr(packPaddings),
               /*transposePaddings=*/b.getArrayAttr(transposePaddings),
               /*copyBackOp=*/b.getStringAttr(copyBackOp));
}

void transform::PadOp::build(OpBuilder &b, OperationState &result, Value target,
                             ArrayRef<int64_t> paddingDimensions,
                             ArrayRef<OpFoldResult> mixedPadToMultipleOf,
                             ArrayRef<int64_t> packPaddings,
                             ArrayRef<Attribute> transposePaddings,
                             StringRef copyBackOp) {
  auto resultType = transform::AnyOpType::get(b.getContext());
  SmallVector<int64_t> staticPadToMultipleOf;
  SmallVector<Value> dynamicPadToMultipleOf;
  dispatchIndexOpFoldResults(mixedPadToMultipleOf, dynamicPadToMultipleOf,
                             staticPadToMultipleOf);
  return build(/*builder=*/b,
               /*result=*/result,
               /*types=*/TypeRange{resultType, resultType},
               /*target=*/target,
               /*paddingValues=*/ArrayAttr(), // let inference handle this
               /*paddingDimensions=*/b.getI64ArrayAttr(paddingDimensions),
               /*padToMultipleOf=*/dynamicPadToMultipleOf,
               /*padToMultipleOf=*/staticPadToMultipleOf,
               /*packPaddings=*/b.getI64ArrayAttr(packPaddings),
               /*transposePaddings=*/b.getArrayAttr(transposePaddings),
               /*copyBackOp=*/b.getStringAttr(copyBackOp));
}

void PadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getPadToMultipleOfMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

SmallVector<OpFoldResult> PadOp::getMixedPadToMultipleOf() {
  Builder b(getContext());
  return getMixedValues(getStaticPadToMultipleOf(), getPadToMultipleOf(), b);
}

DiagnosedSilenceableFailure
transform::PadOp::apply(transform::TransformRewriter &rewriter,
                        transform::TransformResults &results,
                        transform::TransformState &state) {
  auto transformOp = cast<TransformOpInterface>(getOperation());
  SmallVector<Operation *> paddedOps, padOps, copyBackOps;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto linalgTarget = dyn_cast<LinalgOp>(target);
    if (!linalgTarget) {
      auto diag = emitSilenceableError() << "expected LinalgOp target";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }

    // Convert the integer packing flags to booleans.
    SmallVector<bool> packPaddings;
    for (int64_t packPadding :
         extractFromIntegerArrayAttr<int64_t>(getPackPaddings()))
      packPaddings.push_back(static_cast<bool>(packPadding));

    // Convert the padding values to attributes.
    SmallVector<Attribute> paddingValues;
    for (auto const &it :
         llvm::zip(getPaddingValues(), linalgTarget->getOperandTypes())) {
      auto attr = dyn_cast<TypedAttr>(std::get<0>(it));
      if (!attr) {
        emitOpError("expects padding values to be typed attributes");
        return DiagnosedSilenceableFailure::definiteFailure();
      }
      Type elementType = getElementTypeOrSelf(std::get<1>(it));
      // Try to parse string attributes to obtain an attribute of element type.
      if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
        auto parsedAttr = dyn_cast_if_present<TypedAttr>(parseAttribute(
            stringAttr, getContext(), elementType,
            /*numRead=*/nullptr, /*isKnownNullTerminated=*/true));
        if (!parsedAttr || parsedAttr.getType() != elementType) {
          auto diag = this->emitOpError("expects a padding that parses to ")
                      << elementType << ", got " << std::get<0>(it);
          diag.attachNote(linalgTarget.getLoc()) << "when applied to this op";
          return DiagnosedSilenceableFailure::definiteFailure();
        }
        paddingValues.push_back(parsedAttr);
        continue;
      }
      // Otherwise, add the attribute directly.
      if (attr.getType() != elementType) {
        auto diag = this->emitOpError("expects a padding value of type ")
                    << elementType << ", got " << attr;
        diag.attachNote(linalgTarget.getLoc()) << "when applied to this op";
        return DiagnosedSilenceableFailure::definiteFailure();
      }
      paddingValues.push_back(attr);
    }

    // Extract the transpose vectors.
    SmallVector<SmallVector<int64_t>> transposePaddings;
    for (Attribute transposeVector : cast<ArrayAttr>(getTransposePaddings()))
      transposePaddings.push_back(extractFromIntegerArrayAttr<int64_t>(
          cast<ArrayAttr>(transposeVector)));

    LinalgOp paddedOp;
    LinalgPaddingOptions options;
    options.paddingDimensions =
        extractFromIntegerArrayAttr<int64_t>(getPaddingDimensions());

    SmallVector<int64_t> padToMultipleOf;
    DiagnosedSilenceableFailure status = reifyMixedParamAndHandleResults(
        state, transformOp, getMixedPadToMultipleOf(), padToMultipleOf);
    if (!status.succeeded())
      return status;
    if (padToMultipleOf.empty())
      padToMultipleOf =
          SmallVector<int64_t>(options.paddingDimensions.size(), 1);

    options.padToMultipleOf = padToMultipleOf;
    options.paddingValues = paddingValues;
    options.packPaddings = packPaddings;
    if (getCopyBackOp() ==
        bufferization::MaterializeInDestinationOp::getOperationName()) {
      options.copyBackOp = LinalgPaddingOptions::CopyBackOp::
          BufferizationMaterializeInDestination;
    } else if (getCopyBackOp() == linalg::CopyOp::getOperationName()) {
      options.copyBackOp = LinalgPaddingOptions::CopyBackOp::LinalgCopy;
    } else if (getCopyBackOp() == kCopyOpNone) {
      options.copyBackOp = LinalgPaddingOptions::CopyBackOp::None;
    } else {
      llvm_unreachable("unsupported copy_back op");
    }

    SmallVector<Value> replacements;
    SmallVector<tensor::PadOp> newPadOps;
    if (failed(rewriteAsPaddedOp(rewriter, linalgTarget, options, paddedOp,
                                 replacements, newPadOps))) {
      auto diag = emitSilenceableError() << "failed to pad op";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }

    // We need to perform our own replacement here because this API is still
    // used in patterns that "pad and hoist", for which the replacement values
    // need to be different.
    // TODO: clean this up and stop "pad and hoist" behavior more globally now
    // that we have more composable abstractions.
    rewriter.replaceOp(linalgTarget, replacements);
    paddedOps.push_back(paddedOp);
    padOps.append(newPadOps.begin(), newPadOps.end());
    if (options.copyBackOp != LinalgPaddingOptions::CopyBackOp::None) {
      for (Value v : replacements) {
        Operation *copyBackOp = v.getDefiningOp();
        if (!llvm::is_contained(copyBackOps, copyBackOp))
          copyBackOps.push_back(copyBackOp);
      }
    }
  }

  results.set(cast<OpResult>(getPadded()), paddedOps);
  results.set(cast<OpResult>(getPad()), padOps);
  results.set(cast<OpResult>(getCopy()), copyBackOps);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::PadOp::verify() {
  SmallVector<int64_t> packPaddings =
      extractFromIntegerArrayAttr<int64_t>(getPackPaddings());
  if (any_of(packPaddings, [](int64_t packPadding) {
        return packPadding != 0 && packPadding != 1;
      })) {
    return emitOpError()
           << "expects pack_paddings to contain booleans (0/1), found "
           << getPackPaddings();
  }

  SmallVector<int64_t> paddingDimensions =
      extractFromIntegerArrayAttr<int64_t>(getPaddingDimensions());
  if (any_of(paddingDimensions,
             [](int64_t paddingDimension) { return paddingDimension < 0; })) {
    return emitOpError() << "expects padding_dimensions to contain positive "
                            "integers, found "
                         << getPaddingDimensions();
  }
  if (!getMixedPadToMultipleOf().empty()) {
    if (getMixedPadToMultipleOf().size() != paddingDimensions.size()) {
      return emitOpError() << "expects as many multiples as padding_dimensions";
    }
  }
  ArrayAttr transposes = getTransposePaddings();
  for (Attribute attr : transposes) {
    SmallVector<int64_t> transpose = extractFromIntegerArrayAttr<int64_t>(attr);
    auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, transpose.size()));
    if (!std::is_permutation(sequence.begin(), sequence.end(),
                             transpose.begin(), transpose.end())) {
      return emitOpError()
             << "expects transpose_paddings to be a permutation, found "
             << attr;
    }
  }
  if (getCopyBackOp() !=
          bufferization::MaterializeInDestinationOp::getOperationName() &&
      getCopyBackOp() != linalg::CopyOp::getOperationName() &&
      getCopyBackOp() != kCopyOpNone)
    return emitOpError() << "invalid copy_back_op";
  return success();
}

//===---------------------------------------------------------------------===//
// HoistPadOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::HoistPadBuildPackingLoopNestOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  auto targetOps = state.getPayloadOps(getTarget());
  auto loopOps = state.getPayloadOps(getLoop());
  if (!llvm::hasSingleElement(targetOps) || !llvm::hasSingleElement(loopOps)) {
    return emitDefiniteFailure()
           << "requires exactly one target and one loop handle (got "
           << llvm::range_size(targetOps) << " and "
           << llvm::range_size(loopOps) << ")";
  }

  auto padOp = dyn_cast_or_null<tensor::PadOp>(*targetOps.begin());
  auto loopOp = dyn_cast_or_null<scf::ForOp>(*loopOps.begin());
  if (!padOp || !loopOp)
    return emitDefiniteFailure() << "requires exactly 2 non-null handles";

  FailureOr<linalg::detail::PackingResult> result =
      linalg::detail::buildPackingLoopNest(rewriter, padOp, loopOp,
                                           getTranspose());
  if (failed(result))
    return emitDefiniteFailure() << "could not build packing loop nest";

  if (result->clonedLoopIvs.empty()) {
    transformResults.set(cast<OpResult>(getPackingLoop()),
                         {result->hoistedPadOp.getOperation()});
    return DiagnosedSilenceableFailure::success();
  }
  auto outerPackedLoop =
      scf::getForInductionVarOwner(result->clonedLoopIvs.front());
  transformResults.set(cast<OpResult>(getPackingLoop()),
                       {outerPackedLoop.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::HoistPadBuildPackingLoopNestOp::verify() {
  ArrayRef<int64_t> transpose = getTranspose();
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, transpose.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(), transpose.begin(),
                           transpose.end())) {
    return emitOpError() << "expects transpose to be a permutation, found "
                         << getTranspose();
  }
  return success();
}

void transform::HoistPadBuildPackingLoopNestOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::onlyReadsHandle(getLoopMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure
transform::HoistPadOp::applyToOne(transform::TransformRewriter &rewriter,
                                  tensor::PadOp target,
                                  transform::ApplyToEachResultList &results,
                                  transform::TransformState &state) {
  tensor::PadOp hoistedPadOp;
  SmallVector<GenericOp> transposeOps;
  FailureOr<Value> result =
      hoistPaddingOnTensors(rewriter, target, getNumLoops(), getTranspose(),
                            hoistedPadOp, transposeOps);
  if (succeeded(result)) {
    // We need to perform our own replacement here because this API is still
    // used in patterns that "pad and hoist", for which the replacement values
    // need to be different.
    // TODO: clean this up and stop "pad and hoist" behavior more globally now
    // that we have more composable abstractions.
    rewriter.replaceOp(target, *result);
    results.push_back(hoistedPadOp);
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefaultSilenceableFailure(target);
}

LogicalResult transform::HoistPadOp::verify() {
  ArrayRef<int64_t> transpose = getTranspose();
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, transpose.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(), transpose.begin(),
                           transpose.end())) {
    return emitOpError() << "expects transpose to be a permutation, found "
                         << getTranspose();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PromoteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PromoteOp::applyToOne(transform::TransformRewriter &rewriter,
                                 LinalgOp target,
                                 transform::ApplyToEachResultList &results,
                                 transform::TransformState &state) {
  LinalgPromotionOptions promotionOptions;
  if (!getOperandsToPromote().empty())
    promotionOptions = promotionOptions.setOperandsToPromote(
        extractFromIntegerArrayAttr<int64_t>(getOperandsToPromote()));
  if (getUseFullTilesByDefault())
    promotionOptions = promotionOptions.setUseFullTileBuffersByDefault(
        getUseFullTilesByDefault());
  if (getUseAlloca())
    promotionOptions = promotionOptions.setUseAlloca(getUseAlloca());
  if (!getUseFullTileBuffers().empty())
    promotionOptions = promotionOptions.setUseFullTileBuffers(
        llvm::to_vector(getUseFullTileBuffers().getAsValueRange<BoolAttr>()));
  if (getAlignment().has_value())
    promotionOptions = promotionOptions.setAlignment(*getAlignment());
  if (getMemorySpace().has_value())
    promotionOptions = promotionOptions.setMemorySpace(*getMemorySpace());

  if (getMapping().has_value()) {
    // The mapping should only contain an element
    auto mapping = *getMapping();
    if (mapping.size() > 1)
      return emitDefaultDefiniteFailure(target);

    auto addressSpace = cast<mlir::gpu::GPUMemorySpaceMappingAttr>(mapping[0]);

    if (addressSpace.getAddressSpace() ==
        mlir::gpu::GPUDialect::getWorkgroupAddressSpace()) {
      promotionOptions =
          promotionOptions
              .setAllocationDeallocationFns(allocateWorkgroupMemory,
                                            deallocateWorkgroupMemory)
              .setCopyInOutFns(copyToWorkgroupMemory, copyToWorkgroupMemory)
              .setUseFullTileBuffers({false, false});
    } else if (addressSpace.getAddressSpace() ==
               mlir::gpu::GPUDialect::getPrivateAddressSpace()) {
      promotionOptions =
          promotionOptions
              .setAllocationDeallocationFns(allocateGPUPrivateMemory,
                                            deallocateGPUPrivateMemory)
              .setCopyInOutFns(copyToGPUPrivateMemory, copyToGPUPrivateMemory)
              .setUseFullTileBuffers({false, false});
    } else {
      return emitDefaultDefiniteFailure(target);
    }
  }

  if (failed(promoteSubviewsPrecondition(target, promotionOptions)))
    return emitDefaultDefiniteFailure(target);

  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> res = promoteSubViews(rewriter, target, promotionOptions);
  if (failed(res))
    return emitDefaultDefiniteFailure(target);
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ReplaceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReplaceOp::apply(transform::TransformRewriter &rewriter,
                            TransformResults &transformResults,
                            TransformState &state) {
  auto payload = state.getPayloadOps(getTarget());

  // Check for invalid targets.
  for (Operation *target : payload) {
    if (target->getNumOperands() > 0)
      return emitDefiniteFailure() << "expected target without operands";
    if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
        target->getNumRegions() > 0)
      return emitDefiniteFailure()
             << "expected target that is isolated from above";
  }

  // Clone and replace.
  Operation *pattern = &getBodyRegion().front().front();
  SmallVector<Operation *> replacements;
  for (Operation *target : payload) {
    if (getOperation()->isAncestor(target))
      continue;
    rewriter.setInsertionPoint(target);
    Operation *replacement = rewriter.clone(*pattern);
    rewriter.replaceOp(target, replacement->getResults());
    replacements.push_back(replacement);
  }
  transformResults.set(cast<OpResult>(getReplacement()), replacements);
  return DiagnosedSilenceableFailure::success();
}

void transform::ReplaceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

LogicalResult transform::ReplaceOp::verify() {
  if (!getBodyRegion().hasOneBlock())
    return emitOpError() << "expected one block";
  if (std::distance(getBodyRegion().front().begin(),
                    getBodyRegion().front().end()) != 1)
    return emitOpError() << "expected one operation in block";
  Operation *replacement = &getBodyRegion().front().front();
  if (replacement->getNumOperands() > 0)
    return replacement->emitOpError()
           << "expected replacement without operands";
  if (!replacement->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
      replacement->getNumRegions() > 0)
    return replacement->emitOpError()
           << "expect op that is isolated from above";
  return success();
}

//===----------------------------------------------------------------------===//
// ScalarizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ScalarizeOp::applyToOne(transform::TransformRewriter &rewriter,
                                   LinalgOp target,
                                   transform::ApplyToEachResultList &results,
                                   transform::TransformState &state) {
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizeComputationFunction([&](OpBuilder &b, Operation *) {
    SmallVector<OpFoldResult> tileSizes;
    Location loc = target.getLoc();
    SmallVector<OpFoldResult> allShapeSizes =
        target.createFlatListOfOperandDims(b, loc);
    AffineMap map = target.getShapesToLoopsMap();
    if (!map)
      return tileSizes;
    SmallVector<OpFoldResult> shapeSizes =
        affine::makeComposedFoldedMultiResultAffineApply(rewriter, loc, map,
                                                         allShapeSizes);
    // If the shape size is dynamic, tile by 1.
    // Otherwise, do not tile (i.e. tile size 0).
    for (OpFoldResult shapeSize : shapeSizes) {
      tileSizes.push_back(getConstantIntValue(shapeSize) ? b.getIndexAttr(0)
                                                         : b.getIndexAttr(1));
    }
    return tileSizes;
  });
  SmallVector<int64_t> emptyTileSizes;
  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> maybeTilingResult = tileUsingSCF(
      rewriter, cast<TilingInterface>(target.getOperation()), tilingOptions);
  if (failed(maybeTilingResult))
    return emitDefaultDefiniteFailure(target);

  if (target->getNumResults())
    rewriter.replaceOp(target, maybeTilingResult->replacements);
  else
    rewriter.eraseOp(target);

  results.reserve(maybeTilingResult->tiledOps.size());
  for (Operation *tiled : maybeTilingResult->tiledOps)
    results.push_back(tiled);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ConvertToLoopsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ConvertToLoopsOp::apply(transform::TransformRewriter &rewriter,
                                   transform::TransformResults &results,
                                   transform::TransformState &state) {
  SmallVector<Operation *> loops;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto tilingOp = dyn_cast<TilingInterface>(*target);
    if (!target) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected the payload to implement TilingInterface";
      diag.attachNote(target->getLoc()) << "payload op";
      return diag;
    }
    rewriter.setInsertionPoint(target);
    FailureOr<SmallVector<scf::ForOp>> generatedLoops =
        scf::lowerToLoopsUsingSCFForOp(rewriter, tilingOp);
    if (failed(generatedLoops))
      return emitDefaultDefiniteFailure(target);
    for (scf::ForOp &loop : *generatedLoops) {
      loops.push_back(loop.getOperation());
    }
    rewriter.eraseOp(target);
  }
  results.set(cast<OpResult>(getResult()), loops);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// RewriteInDestinationPassingStyleOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::RewriteInDestinationPassingStyleOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  SmallVector<Operation *> res;
  rewriter.setInsertionPoint(target);
  FailureOr<Operation *> maybeResult =
      TypeSwitch<Operation *, FailureOr<Operation *>>(target)
          .Case<tensor::FromElementsOp, tensor::GenerateOp, tensor::PadOp>(
              [&rewriter](auto op) {
                return rewriteInDestinationPassingStyle(rewriter, op);
              });
  if (failed(maybeResult))
    return emitDefaultSilenceableFailure(target);
  results.push_back(*maybeResult);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
SplitOp::apply(transform::TransformRewriter &rewriter,
               TransformResults &results, TransformState &state) {
  // Collect the dynamic split points if provided.
  SmallVector<Operation *> payload =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  bool isMultiwaySplit = getMultiway();

  if (isMultiwaySplit && !llvm::hasSingleElement(payload)) {
    return mlir::emitSilenceableFailure(getLoc())
           << "requires exactly one target when "
              "multiway split is enabled (got "
           << llvm::range_size(payload) << ")";
  }

  SmallVector<OpFoldResult> chunkSizes;

  if (!isMultiwaySplit)
    chunkSizes.reserve(payload.size());

  if (getDynamicChunkSizes()) {
    auto diag = DiagnosedSilenceableFailure::success();
    if (isa<TransformHandleTypeInterface>(getDynamicChunkSizes().getType())) {
      chunkSizes = llvm::to_vector(llvm::map_range(
          state.getPayloadOps(getDynamicChunkSizes()), [&](Operation *op) {
            if (op->getNumResults() != 1 ||
                !op->getResult(0).getType().isIndex()) {
              diag = emitSilenceableError()
                     << "expected dynamic split point handle to point to a "
                        "single-result index-typed op";
              diag.attachNote(op->getLoc()) << "dynamic split point";
            }
            return OpFoldResult(op->getResult(0));
          }));
    } else {
      chunkSizes = llvm::to_vector(
          llvm::map_range(state.getParams(getDynamicChunkSizes()),
                          [](Attribute attr) { return OpFoldResult(attr); }));
    }
    if (diag.isSilenceableFailure())
      return diag;

    // For multiway split, a single payload is expected to have multiple
    // split points.
    if (!isMultiwaySplit && chunkSizes.size() != payload.size()) {
      return emitDefiniteFailure()
             << "expected the dynamic split point handle to point to as "
                "many operations ("
             << chunkSizes.size() << ") as the target handle ("
             << payload.size() << ")";
    }
  } else {
    chunkSizes.resize(payload.size(),
                      rewriter.getIndexAttr(getStaticChunkSizes()));
  }

  auto checkStructuredOpAndDimensions =
      [&](LinalgOp linalgOp, Location loc) -> DiagnosedSilenceableFailure {
    if (!linalgOp) {
      auto diag = emitSilenceableError() << "only applies to structured ops";
      diag.attachNote(loc) << "target op";
      return diag;
    }

    if (getDimension() >= linalgOp.getNumLoops()) {
      auto diag = emitSilenceableError() << "dimension " << getDimension()
                                         << " does not exist in target op";
      diag.attachNote(loc) << "target op";
      return diag;
    }
    return DiagnosedSilenceableFailure::success();
  };

  auto checkFailureInSplitting =
      [&](bool hasFailed, Location loc) -> DiagnosedSilenceableFailure {
    if (hasFailed) {
      auto diag = emitDefiniteFailure() << "internal failure in splitting";
      diag.attachNote(loc) << "target op";
      return diag;
    }
    return DiagnosedSilenceableFailure::success();
  };

  if (isMultiwaySplit) {

    // Split a single target operation at multiple points.
    SmallVector<Operation *> opList;
    TilingInterface head, tail;
    Operation *target = payload.front();

    LinalgOp linalgOp = dyn_cast<LinalgOp>(target);

    // Check that the target is a valid LinalgOp with correct dimensions.
    DiagnosedSilenceableFailure diag =
        checkStructuredOpAndDimensions(linalgOp, target->getLoc());
    if (diag.isSilenceableFailure())
      return diag;

    for (auto &&[idx, chunkSize] : llvm::enumerate(chunkSizes)) {

      if (idx > 0)
        target = tail.getOperation();

      if (!target)
        break;

      linalgOp = cast<LinalgOp>(target);
      Location loc = target->getLoc();

      rewriter.setInsertionPoint(linalgOp);
      std::tie(head, tail) = linalg::splitOp(
          rewriter, cast<TilingInterface>(linalgOp.getOperation()),
          getDimension(), chunkSize);

      // Propagate errors.
      DiagnosedSilenceableFailure diag =
          checkFailureInSplitting(!head && !tail, loc);
      if (diag.isDefiniteFailure())
        return diag;

      opList.push_back(head.getOperation());
    }

    // Append any leftover parts to the end of the result list.
    if (tail)
      opList.push_back(tail.getOperation());
    results.set(cast<OpResult>(getFirst()), opList);
    results.set(cast<OpResult>(getSecond()), {});

  } else {
    // Split each target operation.
    SmallVector<Operation *> first, second;
    Operation *noSecondPart = nullptr;
    for (const auto &pair : llvm::zip(payload, chunkSizes)) {
      Operation *target = std::get<0>(pair);
      Location loc = target->getLoc();
      LinalgOp linalgOp = dyn_cast<LinalgOp>(target);
      DiagnosedSilenceableFailure diag =
          checkStructuredOpAndDimensions(linalgOp, target->getLoc());

      if (diag.isSilenceableFailure())
        return diag;

      rewriter.setInsertionPoint(linalgOp);
      std::tie(first.emplace_back(), second.emplace_back()) = linalg::splitOp(
          rewriter, cast<TilingInterface>(linalgOp.getOperation()),
          getDimension(), std::get<1>(pair));

      // Propagate errors.
      DiagnosedSilenceableFailure diagSplit =
          checkFailureInSplitting(!first.back() && !second.back(), loc);
      if (diagSplit.isDefiniteFailure())
        return diag;

      // Do not add null second parts.
      if (!second.back()) {
        noSecondPart = target;
        second.pop_back();
      }
    }

    if (second.size() != first.size() && !second.empty()) {
      auto diag = emitSilenceableError()
                  << "splitting does not produce the second part for a subset "
                     "of targets";
      diag.attachNote()
          << "expected splitting to produce the second part of all "
             "or none of the targets";
      diag.attachNote(noSecondPart->getLoc())
          << "first target with no second part";
      return diag;
    }

    results.set(cast<OpResult>(getFirst()), first);
    results.set(cast<OpResult>(getSecond()), second);
  }
  return DiagnosedSilenceableFailure::success();
}

void SplitOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  if (getDynamicChunkSizes())
    onlyReadsHandle(getDynamicChunkSizesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

ParseResult SplitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand target, dynamicChunkSizes;
  IntegerAttr staticChunkSizes;
  if (parser.parseOperand(target) || parser.parseKeyword("after"))
    return failure();

  OptionalParseResult dynamicPointParseResult =
      parser.parseOptionalOperand(dynamicChunkSizes);
  if (!dynamicPointParseResult.has_value()) {
    int64_t staticChunkSizesValue;
    if (failed(parser.parseInteger(staticChunkSizesValue)))
      return failure();

    staticChunkSizes =
        parser.getBuilder().getI64IntegerAttr(staticChunkSizesValue);
  }

  Type targetType;
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(targetType) ||
      parser.resolveOperand(target, targetType, result.operands)) {
    return failure();
  }
  if (dynamicPointParseResult.has_value()) {
    Type ChunkSizesType;
    if (failed(*dynamicPointParseResult) || parser.parseComma() ||
        parser.parseType(ChunkSizesType) ||
        parser.resolveOperand(dynamicChunkSizes, ChunkSizesType,
                              result.operands)) {
      return failure();
    }

    staticChunkSizes =
        parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
  }

  result.addAttribute(
      SplitOp::getStaticChunkSizesAttrName(result.name).getValue(),
      staticChunkSizes);
  result.addTypes({targetType, targetType});
  return success();
}

void SplitOp::print(OpAsmPrinter &printer) {
  printer << " " << getTarget() << " after ";
  int64_t staticChunkSize = static_cast<int64_t>(getStaticChunkSizes());
  if (staticChunkSize != ShapedType::kDynamic)
    printer << staticChunkSize;
  else
    printer << getDynamicChunkSizes();
  printer << " ";
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {getStaticChunkSizesAttrName()});
  printer << " : " << getTarget().getType();
  if (staticChunkSize == ShapedType::kDynamic)
    printer << ", " << getDynamicChunkSizes().getType();
}

LogicalResult SplitOp::verify() {
  if ((static_cast<int64_t>(getStaticChunkSizes()) != ShapedType::kDynamic) ^
      (getDynamicChunkSizes() == nullptr)) {
    return emitOpError() << "expects either a dynamic or a static split "
                            "point to be provided";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SplitReductionOp
//===----------------------------------------------------------------------===//

void transform::SplitReductionOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    int64_t splitFactor, int64_t insertSplitDimension, bool innerParallel,
    bool useScalingAlgorithm, bool useAlloc) {
  MLIRContext *ctx = builder.getContext();
  result.addOperands(target);
  result.addAttribute(SplitReductionOp::getSplitFactorAttrName(result.name),
                      builder.getI64IntegerAttr(splitFactor));
  result.addAttribute(
      SplitReductionOp::getInsertSplitDimensionAttrName(result.name),
      builder.getI64IntegerAttr(insertSplitDimension));
  if (innerParallel) {
    result.addAttribute(SplitReductionOp::getInnerParallelAttrName(result.name),
                        builder.getUnitAttr());
  }
  if (useScalingAlgorithm) {
    result.addAttribute(
        SplitReductionOp::getUseScalingAlgorithmAttrName(result.name),
        builder.getUnitAttr());
  }
  if (useAlloc) {
    result.addAttribute(SplitReductionOp::getUseAllocAttrName(result.name),
                        builder.getUnitAttr());
  }
  auto resultType = transform::AnyOpType::get(ctx);
  result.addTypes({resultType, resultType, resultType, resultType});
}

DiagnosedSilenceableFailure transform::SplitReductionOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  ControlSplitReductionFn splitFn = [&](LinalgOp) {
    return linalg::SplitReductionOptions{int64_t(getSplitFactor()),
                                         unsigned(getInsertSplitDimension()),
                                         bool(getInnerParallel())};
  };
  rewriter.setInsertionPoint(target);
  FailureOr<SplitReductionResult> splitResult =
      (getUseScalingAlgorithm())
          ? splitReductionByScaling(rewriter, target, splitFn, getUseAlloc())
          : splitReduction(rewriter, target, splitFn, getUseAlloc());
  if (failed(splitResult))
    return emitDefaultDefiniteFailure(target);

  results.push_back(splitResult->initOrAlloc);
  results.push_back(splitResult->fillOp);
  results.push_back(splitResult->splitLinalgOp);
  results.push_back(splitResult->resultCombiningLinalgOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TileReductionUsingForOp
//===----------------------------------------------------------------------===//

void transform::TileReductionUsingForOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticTileSizes) {
  // Call the default builder.
  // This is future-proof re mixed static-dynamic and setting up the proper
  // operands segment sizes attributes for multiple variadic operands.
  // In the absence of this, horrible bugs ensue.
  // TODO: support mixed static-dynamic (see TileUsingForallOp).
  MLIRContext *ctx = builder.getContext();
  auto opTy = transform::AnyOpType::get(ctx);
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  build(builder, result,
        /*resultTypes=*/TypeRange{opTy, opTy, opTy, opTy},
        /*target=*/target,
        /*tile_sizes=*/staticTileSizesAttr);
}

DiagnosedSilenceableFailure transform::TileReductionUsingForOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFReductionTilingResult> result = scf::tileReductionUsingScf(
      rewriter, cast<PartialReductionOpInterface>(target.getOperation()),
      getAsOpFoldResult(rewriter.getI64ArrayAttr(getTileSizes())));

  if (failed(result))
    return emitDefaultSilenceableFailure(target);
  for (Value initValue : result->initialValues)
    results.push_back(initValue.getDefiningOp());
  for (auto parallelTiledOp : result->parallelTiledOps)
    results.push_back(parallelTiledOp);
  for (auto mergeOp : result->mergeOps)
    results.push_back(mergeOp);
  results.push_back(result->loops.front());
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TileReductionUsingForallOp
//===----------------------------------------------------------------------===//

void transform::TileReductionUsingForallOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticNumThreads, ArrayRef<int64_t> staticTileSizes,
    ArrayAttr mapping) {
  // Call the default builder.
  // This is future-proof re mixed static-dynamic and setting up the proper
  // operands segment sizes attributes for multiple variadic operands.
  // In the absence of this, horrible bugs ensue.
  // TODO: support mixed static-dynamic (see TileUsingForallOp).
  MLIRContext *ctx = builder.getContext();
  auto opTy = transform::AnyOpType::get(ctx);
  auto staticNumThreadsAttr = builder.getDenseI64ArrayAttr(staticNumThreads);
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  build(builder, result,
        /*resultTypes=*/TypeRange{opTy, opTy, opTy, opTy},
        /*target=*/target,
        /*num_threads=*/staticNumThreadsAttr,
        /*tile_sizes=*/staticTileSizesAttr,
        /*mapping=*/mapping);
}

DiagnosedSilenceableFailure transform::TileReductionUsingForallOp::applyToOne(
    transform::TransformRewriter &rewriter, LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  SmallVector<OpFoldResult> numThreads =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(getNumThreads()));
  SmallVector<OpFoldResult> tileSizes =
      getAsOpFoldResult(rewriter.getI64ArrayAttr(getTileSizes()));
  FailureOr<linalg::ForallReductionTilingResult> result =
      linalg::tileReductionUsingForall(
          rewriter, cast<PartialReductionOpInterface>(target.getOperation()),
          numThreads, tileSizes, getMapping());

  if (failed(result)) {
    auto diag = emitSilenceableError() << "could not tile reduction";
    diag.attachNote(target.getLoc()) << "target operation";
    return diag;
  }
  for (Value initValue : result->initialValues)
    results.push_back(initValue.getDefiningOp());
  for (auto parallelTiledOp : result->parallelTiledOps)
    results.push_back(parallelTiledOp);
  for (auto mergeOp : result->mergeOps)
    results.push_back(mergeOp);
  results.push_back(result->loops);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ContinuousTileSizesOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ContinuousTileSizesOp::apply(transform::TransformRewriter &rewriter,
                                        TransformResults &transformResults,
                                        TransformState &state) {

  SmallVector<Operation *> targetOps =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (!llvm::hasSingleElement(targetOps)) {
    return mlir::emitSilenceableFailure(getLoc())
           << "requires exactly one target (got " << llvm::range_size(targetOps)
           << ")";
  }

  Operation *target = *targetOps.begin();
  auto linalgOp = dyn_cast<LinalgOp>(target);
  auto tileableOp = dyn_cast<TilingInterface>(target);

  if (!linalgOp)
    return emitDefiniteFailure() << "expected Linalg Op";

  OpBuilder builder(linalgOp.getContext());

  if (isa<TransformParamTypeInterface>(getChunkSizes().getType())) {
    if (linalgOp.hasDynamicShape()) {
      auto diag = emitSilenceableError()
                  << "cannot compute parametric tile sizes for dynamically "
                     "shaped payload op";
      diag.attachNote(linalgOp->getLoc()) << "payload op";
      return diag;
    }

    FailureOr<StaticContinuousTileSizeSpecification> spec =
        computeStaticContinuousTileSizes(linalgOp, getDimension(),
                                         getTargetSize());
    if (failed(spec)) {
      return emitSilenceableError()
             << "failed to compute multi-size tiling sizes";
    }

    SmallVector<int64_t> chunkSizes;

    for (auto &&[tileSize, tripCount] :
         llvm::zip_equal(spec->tileSizes, spec->tripCounts))
      chunkSizes.push_back(tileSize * tripCount);

    auto getI64AttrsFromI64 = [&](ArrayRef<int64_t> values) {
      return llvm::map_to_vector(values, [&](int64_t value) -> Attribute {
        return builder.getI64IntegerAttr(value);
      });
    };
    transformResults.setParams(cast<OpResult>(getTileSizes()),
                               getI64AttrsFromI64(spec->tileSizes));
    transformResults.setParams(cast<OpResult>(getChunkSizes()),
                               getI64AttrsFromI64(chunkSizes));

    return DiagnosedSilenceableFailure::success();
  }

  builder.setInsertionPoint(linalgOp);

  OpFoldResult targetSize = builder.getIndexAttr(getTargetSize());
  unsigned dimension = getDimension();

  FailureOr<ContinuousTileSizeSpecification> spec = computeContinuousTileSizes(
      builder, tileableOp, dimension, targetSize, true);
  if (failed(spec)) {
    return emitSilenceableError() << "could not generate tile size computation";
  }

  AffineExpr s0 = builder.getAffineSymbolExpr(0);
  AffineExpr s1 = builder.getAffineSymbolExpr(1);
  auto apply = [&](AffineExpr expr, ArrayRef<OpFoldResult> ofrs) -> Value {
    return affine::makeComposedAffineApply(builder, linalgOp->getLoc(), expr,
                                           ofrs);
  };

  SmallVector<Value> chunkSizes;
  Value splitPoint;
  for (auto &&[tileSize, tripCount] :
       llvm::zip_equal(spec->tileSizes, spec->tripCounts)) {
    splitPoint = apply(s0 * s1, {tileSize, tripCount});
    chunkSizes.push_back(splitPoint);
  }

  auto getDefiningOps = [&](ArrayRef<Value> values) {
    return llvm::map_to_vector(values, [&](Value value) -> Operation * {
      return value.getDefiningOp();
    });
  };

  transformResults.set(cast<OpResult>(getTileSizes()),
                       getDefiningOps(spec->tileSizes));
  transformResults.set(cast<OpResult>(getChunkSizes()),
                       getDefiningOps(chunkSizes));

  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::ContinuousTileSizesOp::verify() {

  if (getTileSizes().getType() != getChunkSizes().getType()) {
    return emitOpError() << "expects all results type to be the same";
  }

  return success();
}

void transform::ContinuousTileSizesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  if (isa<TransformParamTypeInterface>(getTileSizes().getType()))
    onlyReadsPayload(effects);
  else
    modifiesPayload(effects);
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
}

static void printContinuousTileSizeTypes(OpAsmPrinter &printer, Operation *op,
                                         Type targetType, Type tile_sizes,
                                         Type) {
  printer.printFunctionalType(TypeRange{targetType}, TypeRange{tile_sizes});
}

static ParseResult parseContinuousTileSizeTypes(OpAsmParser &parser,
                                                Type &targetType,
                                                Type &tileSizesType,
                                                Type &chunkSizesType) {
  FunctionType funcType;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (failed(parser.parseType<FunctionType>(funcType)))
    return failure();

  if (funcType.getNumInputs() != 1 || funcType.getNumResults() != 1) {
    parser.emitError(typeLoc) << "expects a trailing functional type with one "
                                 "argument and one result";
  }
  targetType = funcType.getInput(0);
  tileSizesType = chunkSizesType = funcType.getResult(0);

  return success();
}

//===----------------------------------------------------------------------===//
// TileUsingForOp
//===----------------------------------------------------------------------===//

void transform::TileUsingForOp::build(
    OpBuilder &builder, OperationState &result, TypeRange loopTypes,
    Value target, ArrayRef<int64_t> staticTileSizes,
    ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  return build(builder, result, loopTypes,
               /*target=*/target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               interchange, scalableSizes);
}

void transform::TileUsingForOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<int64_t> staticTileSizes, ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  build(builder, result, target,
        getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
        interchange, scalableSizes);
}

void transform::TileUsingForOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedTileSizes, ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  // Loop types are automaticaly splat by the callee, setting up one is
  // enough.
  SmallVector<Type> loopTypes(1, builder.getType<transform::AnyOpType>());
  build(builder, result, loopTypes, target, mixedTileSizes, interchange,
        scalableSizes);
}

void transform::TileUsingForOp::build(
    OpBuilder &builder, OperationState &result, TypeRange loopTypes,
    Value target, ArrayRef<OpFoldResult> mixedTileSizes,
    ArrayRef<int64_t> interchange,
    std::optional<ArrayRef<bool>> scalableSizes) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this,
  // horrible bugs ensue.
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  unsigned numExpectedLoops =
      staticTileSizes.size() - llvm::count(staticTileSizes, 0);
  SmallVector<Type> resultTypes;
  resultTypes.reserve(numExpectedLoops);
  assert((loopTypes.size() == 1 || loopTypes.size() == numExpectedLoops) &&
         "expected one loop type or as many as loops");
  if (loopTypes.size() == 1)
    resultTypes.append(numExpectedLoops, loopTypes[0]);
  else
    llvm::append_range(resultTypes, loopTypes);
  SmallVector<bool> expandedScalableSizes(mixedTileSizes.size(), false);
  if (scalableSizes.has_value())
    expandedScalableSizes.assign(scalableSizes->begin(), scalableSizes->end());
  build(builder, result, /*tiled_linalg_op=*/target.getType(),
        /*loops=*/resultTypes,
        /*target=*/target,
        /*dynamic_sizes=*/dynamicTileSizes,
        /*static_sizes=*/staticTileSizesAttr,
        /*interchange=*/builder.getDenseI64ArrayAttr(interchange),
        /*scalable_sizes=*/expandedScalableSizes);
}

LogicalResult transform::TileUsingForOp::verify() {
  if (getMixedSizes().size() != getScalableSizes().size())
    return emitOpError("expected same number of sizes (")
           << getMixedSizes().size() << ") and scalable sizes ("
           << getScalableSizes().size() << ")";
  ArrayRef<int64_t> staticSizes = getStaticSizes();
  unsigned numExpectedLoops = staticSizes.size() - llvm::count(staticSizes, 0);
  if (getLoops().size() != numExpectedLoops)
    return emitOpError("expected number of loops to tile (")
           << numExpectedLoops << ") to match number of `loops` results ("
           << getLoops().size() << ")";
  return success();
}

DiagnosedSilenceableFailure
transform::TileUsingForOp::apply(transform::TransformRewriter &rewriter,
                                 TransformResults &transformResults,
                                 TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  SmallVector<SmallVector<Operation *>> dynamicSizeProducers;
  SmallVector<SmallVector<int64_t>> paramSizes;
  dynamicSizeProducers.reserve(getDynamicSizes().size());
  paramSizes.reserve(getDynamicSizes().size());
  for (Value transformValue : getDynamicSizes()) {
    if (isa<ParamType>(transformValue.getType())) {
      dynamicSizeProducers.push_back({});
      ArrayRef<Attribute> params = state.getParams(transformValue);
      paramSizes.push_back(
          llvm::to_vector(llvm::map_range(params, [](Attribute attr) {
            return cast<IntegerAttr>(attr).getValue().getSExtValue();
          })));

      if (paramSizes.back().size() != targets.size()) {
        DiagnosedSilenceableFailure diag =
            emitSilenceableError()
            << "expected as many parameter values ("
            << dynamicSizeProducers.back().size() << ") as target ops ("
            << targets.size() << ")";
        diag.attachNote(transformValue.getLoc()) << "for this parameter";
        return diag;
      }

      continue;
    }
    paramSizes.push_back({});
    dynamicSizeProducers.push_back(
        llvm::to_vector(state.getPayloadOps(transformValue)));

    if (dynamicSizeProducers.back().size() != targets.size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected as many dynamic size-producing operations ("
          << dynamicSizeProducers.back().size() << ") as target ops ("
          << targets.size() << ")";
      diag.attachNote(transformValue.getLoc()) << "for this handle";
      return diag;
    }

    for (Operation *op : dynamicSizeProducers.back()) {
      if (op->getNumResults() == 1 &&
          isa<IndexType>(op->getResult(0).getType())) {
        continue;
      }

      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected sizes to be produced by ops "
                                    "with a single index-type result";
      diag.attachNote(op->getLoc()) << "size producer op";
      diag.attachNote(transformValue.getLoc()) << "for this handle";
      return diag;
    }
  }

  SmallVector<Operation *> tiled;
  SmallVector<SmallVector<Operation *, 4>, 4> loops;
  loops.resize(getLoops().size());
  auto scalableSizes = getScalableSizes();
  for (auto [i, op] : llvm::enumerate(targets)) {
    auto tilingInterface = dyn_cast<TilingInterface>(op);
    if (!tilingInterface) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "only ops implementing TilingInterface are supported";
      diag.attachNote(op->getLoc()) << "target op";
      return diag;
    }
    if (tileSizes.size() > tilingInterface.getLoopIteratorTypes().size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "too many tiles provided, expected at most "
          << tilingInterface.getLoopIteratorTypes().size() << " found "
          << tileSizes.size();
      diag.attachNote(op->getLoc()) << "target op";
      return diag;
    }

    scf::SCFTilingOptions tilingOptions;
    if (tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [](OpBuilder &, Operation *) -> SmallVector<OpFoldResult> {
            return {};
          });
    } else {
      tilingOptions.setTileSizeComputationFunction([&, index = i](OpBuilder &b,
                                                                  Operation *) {
        SmallVector<OpFoldResult> sizes;
        sizes.reserve(tileSizes.size());
        unsigned dynamicIdx = 0;

        for (auto [ofrIdx, ofr] : llvm::enumerate(getMixedSizes())) {
          if (auto attr = llvm::dyn_cast_if_present<Attribute>(ofr)) {
            if (scalableSizes[ofrIdx]) {
              auto val = b.create<arith::ConstantIndexOp>(
                  getLoc(), cast<IntegerAttr>(attr).getInt());
              Value vscale =
                  b.create<vector::VectorScaleOp>(getLoc(), b.getIndexType());
              sizes.push_back(
                  b.create<arith::MulIOp>(getLoc(), val, vscale).getResult());
            } else {
              sizes.push_back(attr);
            }
            continue;
          }
          ArrayRef<Operation *> dynamicSizes = dynamicSizeProducers[dynamicIdx];
          ArrayRef<int64_t> params = paramSizes[dynamicIdx];
          ++dynamicIdx;
          assert((dynamicSizes.empty() ^ params.empty()) &&
                 "expected either dynamic sizes or parameters");
          if (!params.empty()) {
            sizes.push_back(b.getIndexAttr(params[index]));
          } else {
            sizes.push_back(dynamicSizes[index]->getResult(0));
          }
        }
        return sizes;
      });
    }

    tilingOptions.setInterchange(getInterchange());
    FailureOr<scf::SCFTilingResult> maybeTilingResult =
        tileUsingSCF(rewriter, tilingInterface, tilingOptions);
    if (failed(maybeTilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    rewriter.replaceOp(op, maybeTilingResult->replacements);

    tiled.append(maybeTilingResult->tiledOps);
    for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(cast<OpResult>(getTiledLinalgOp()), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(cast<OpResult>(getLoops()[en.index()]), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::TileUsingForOp::getMixedSizes() {
  ValueRange dynamic = getDynamicSizes();
  ArrayRef<int64_t> tileSizes = getStaticSizes();
  SmallVector<OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  Builder builder(getContext());
  for (int64_t size : tileSizes) {
    if (size == ShapedType::kDynamic) {
      results.push_back(dynamic[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

void transform::TileUsingForOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getDynamicSizesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// TileUsingForallOp
//===----------------------------------------------------------------------===//

void transform::TileUsingForallOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         ArrayRef<int64_t> staticTileSizes,
                                         transform::TileSizesSpec,
                                         ArrayAttr mapping) {
  return build(builder, result,
               /*target=*/target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               /*_=*/TileSizesSpec(),
               /*mapping=*/mapping);
}

void transform::TileUsingForallOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         ArrayRef<OpFoldResult> mixedTileSizes,
                                         transform::TileSizesSpec,
                                         ArrayAttr mapping) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this,
  // horrible bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = transform::AnyOpType::get(ctx);
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*num_threads=*/ValueRange{},
        /*tile_sizes=*/dynamicTileSizes,
        /*packed_num_threads=*/Value(),
        /*packed_tile_sizes=*/Value(),
        /*static_num_threads=*/builder.getDenseI64ArrayAttr({}),
        /*static_tile_sizes=*/staticTileSizesAttr,
        /*mapping=*/mapping);
}

void transform::TileUsingForallOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         ArrayRef<int64_t> staticNumThreads,
                                         transform::NumThreadsSpec,
                                         ArrayAttr mapping) {
  return build(builder, result, target,
               getAsOpFoldResult(builder.getI64ArrayAttr(staticNumThreads)),
               NumThreadsSpec(), mapping);
}

void transform::TileUsingForallOp::build(OpBuilder &builder,
                                         OperationState &result, Value target,
                                         ArrayRef<OpFoldResult> mixedNumThreads,
                                         transform::NumThreadsSpec,
                                         ArrayAttr mapping) {
  SmallVector<int64_t> staticNumThreads;
  SmallVector<Value> dynamicNumThreads;
  dispatchIndexOpFoldResults(mixedNumThreads, dynamicNumThreads,
                             staticNumThreads);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this,
  // horrible bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = transform::AnyOpType::get(ctx);
  auto staticNumThreadsAttr = builder.getDenseI64ArrayAttr(staticNumThreads);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*num_threads=*/dynamicNumThreads,
        /*tile_sizes=*/ValueRange{},
        /*packed_num_threads=*/Value(),
        /*packed_tile_sizes=*/Value(),
        /*static_num_threads=*/staticNumThreadsAttr,
        /*static_tile_sizes=*/builder.getDenseI64ArrayAttr({}),
        /*mapping=*/mapping);
}

DiagnosedSilenceableFailure transform::tileToForallOpImpl(
    RewriterBase &rewriter, transform::TransformState &state,
    TransformOpInterface transformOp, Operation *target,
    ArrayRef<OpFoldResult> mixedNumThreads,
    ArrayRef<OpFoldResult> mixedTileSizes, std::optional<ArrayAttr> mapping,
    scf::SCFTilingResult &tilingResult) {
  // Transform all targets one by one.
  auto tileableOp = dyn_cast<TilingInterface>(target);
  if (!tileableOp) {
    DiagnosedSilenceableFailure diag =
        transformOp.emitSilenceableError()
        << "only TilingInterface ops are supported";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  rewriter.setInsertionPoint(tileableOp);
  scf::SCFTilingOptions options;
  options.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  if (!mixedNumThreads.empty()) {
    options.setNumThreads(mixedNumThreads);
  } else {
    SmallVector<Range> loopRanges = tileableOp.getIterationDomain(rewriter);
    unsigned nLoops = loopRanges.size();
    SmallVector<OpFoldResult> numThreads;
    numThreads.reserve(nLoops);
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    AffineExpr divExpr = s0.ceilDiv(s1);
    for (int i = 0, e = std::min(mixedTileSizes.size(), loopRanges.size());
         i < e; ++i) {
      OpFoldResult numTiles = mixedTileSizes[i];
      if (!isConstantIntValue(numTiles, 0))
        numTiles = affine::makeComposedFoldedAffineApply(
            rewriter, tileableOp.getLoc(), divExpr,
            {loopRanges[i].size, numTiles});
      numThreads.push_back(numTiles);
    }
    options.setNumThreads(numThreads);
    options.setTileSizes(mixedTileSizes);
  }
  if (mapping) {
    options.setMapping(mapping.value().getValue());
  }
  FailureOr<scf::SCFTilingResult> maybeTilingResult =
      scf::tileUsingSCF(rewriter, tileableOp, options);

  if (failed(maybeTilingResult))
    return transformOp.emitDefaultSilenceableFailure(tileableOp);
  rewriter.replaceOp(tileableOp, maybeTilingResult->replacements);

  tilingResult = *maybeTilingResult;
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::TileUsingForallOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  // Unpack handles.
  SmallVector<OpFoldResult> mixedNumThreads;
  DiagnosedSilenceableFailure status =
      getPackedNumThreads()
          ? unpackSingleIndexResultPayloadOperations(
                state, transformOp, mixedNumThreads, getPackedNumThreads())
          : unpackSingleIndexResultPayloadOperations(
                state, transformOp, mixedNumThreads, getMixedNumThreads());
  if (!status.succeeded())
    return status;
  SmallVector<OpFoldResult> mixedTileSizes;
  status = getPackedTileSizes()
               ? unpackSingleIndexResultPayloadOperations(
                     state, transformOp, mixedTileSizes, getPackedTileSizes())
               : unpackSingleIndexResultPayloadOperations(
                     state, transformOp, mixedTileSizes, getMixedTileSizes());
  if (!status.succeeded())
    return status;

  for (Operation *target : state.getPayloadOps(getTarget())) {
    scf::SCFTilingResult tilingResult;
    DiagnosedSilenceableFailure diag = tileToForallOpImpl(
        rewriter, state, transformOp, target, mixedNumThreads, mixedTileSizes,
        getMapping(), tilingResult);
    if (!diag.succeeded())
      return diag;
    tileOps.push_back(tilingResult.loops.front());
    tiledOps.append(tilingResult.tiledOps);
  }

  transformResults.set(cast<OpResult>(getForallOp()), tileOps);
  transformResults.set(cast<OpResult>(getTiledOp()), tiledOps);

  return DiagnosedSilenceableFailure::success();
}

void transform::TileUsingForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getTileSizesMutable(), effects);
  onlyReadsHandle(getNumThreadsMutable(), effects);
  onlyReadsHandle(getPackedNumThreadsMutable(), effects);
  onlyReadsHandle(getPackedTileSizesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

SmallVector<OpFoldResult> TileUsingForallOp::getMixedNumThreads() {
  Builder b(getContext());
  return getMixedValues(getStaticNumThreads(), getNumThreads(), b);
}

SmallVector<OpFoldResult> TileUsingForallOp::getMixedTileSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticTileSizes(), getTileSizes(), b);
}

LogicalResult TileUsingForallOp::verify() {
  int numThreadsSpec = static_cast<int>(!getMixedNumThreads().empty()) +
                       static_cast<int>(getPackedNumThreads() != Value());
  if (numThreadsSpec > 1)
    return emitOpError(
        "num_threads and packed_num_threads are mutually exclusive");
  int tileSizesSpec = static_cast<int>(!getMixedTileSizes().empty()) +
                      static_cast<int>(getPackedTileSizes() != Value());
  if (tileSizesSpec > 1)
    return emitOpError(
        "tile_sizes and packed_tile_sizes are mutually exclusive");
  if (numThreadsSpec == 0 && tileSizesSpec == 0)
    return emitOpError("either (packed_)num_threads or (packed_)tile_sizes "
                       "must be specified");
  return success();
}

//===----------------------------------------------------------------------===//
// VectorizeChildrenAndApplyPatternsOp
//===----------------------------------------------------------------------===//

void transform::VectorizeChildrenAndApplyPatternsOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    bool vectorizePadding, bool vectorizeExtract, bool flatten1DDepthwiseConv) {
  result.addOperands(target);
  if (vectorizePadding) {
    result.addAttribute(
        VectorizeChildrenAndApplyPatternsOp::getVectorizePaddingAttrName(
            result.name),
        builder.getUnitAttr());
  }
  if (vectorizeExtract) {
    result.addAttribute(
        VectorizeChildrenAndApplyPatternsOp::getVectorizeNdExtractAttrName(
            result.name),
        builder.getUnitAttr());
  }
  if (flatten1DDepthwiseConv) {
    result.addAttribute(
        VectorizeChildrenAndApplyPatternsOp::getFlatten_1dDepthwiseConvAttrName(
            result.name),
        builder.getUnitAttr());
  }
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

namespace {
/// This is an helper only to call vectorize via a pattern inside of
/// VectorizeChildrenAndApplyPatternsOp::applyToOne.
struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context,
                                bool vectorizeExtract = false,
                                bool flattenConv = false)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorizeNDExtract(vectorizeExtract),
        flatten1DDepthwiseConv(flattenConv) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return rewriter.notifyMatchFailure(op, "expected Linalg Op");
    return vectorize(rewriter, linalgOp, /*inputVectorSizes=*/{},
                     /*scalableVecDims=*/{}, vectorizeNDExtract,
                     flatten1DDepthwiseConv);
  }

private:
  /// Controls whether to vectorize `tensor.extract` when the input tensor is
  /// rank >= 2.
  bool vectorizeNDExtract = false;
  /// Controls whether to "flatten" the channel dimension when vectorising 1D
  /// depthwise convolutions. This should lead to bette vectorization for
  /// tensors with a low number of channel dimensions.
  bool flatten1DDepthwiseConv = false;
};
} // namespace

DiagnosedSilenceableFailure
transform::VectorizeChildrenAndApplyPatternsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<VectorizationPattern>(ctx, getVectorizeNdExtract(),
                                     getFlatten_1dDepthwiseConv());

  if (!getDisableTransferPermutationMapLoweringPatterns())
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  if (!getDisableMultiReductionToContractPatterns())
    vector::populateVectorReductionToContractPatterns(patterns);

  vector::populateSinkVectorBroadcastPatterns(patterns);

  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

  patterns.add<CopyVectorizationPattern>(ctx);

  if (getVectorizePadding())
    linalg::populatePadOpVectorizationPatterns(patterns);

  TrackingListener listener(state, *this);
  GreedyRewriteConfig config;
  config.listener = &listener;
  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns), config)))
    return emitDefaultDefiniteFailure(target);

  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::VectorizeOp::apply(
    transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &transformResults,
    mlir::transform::TransformState &state) {
  auto targets = state.getPayloadOps(getTarget());
  if (std::empty(targets))
    return DiagnosedSilenceableFailure::success();
  auto transformOp = cast<TransformOpInterface>(getOperation());
  SmallVector<int64_t> vectorSizes;
  DiagnosedSilenceableFailure status = reifyMixedParamAndHandleResults(
      state, transformOp, getMixedVectorSizes(), vectorSizes);
  if (!status.succeeded())
    return status;

  // TODO: Check that the correct number of vectorSizes was provided.
  for (Operation *target : targets) {
    if (!isa<linalg::LinalgOp, tensor::PadOp, tensor::PackOp, tensor::UnPackOp>(
            target)) {
      return mlir::emitSilenceableFailure(target->getLoc())
             << "Unsupported Op, cannot vectorize";
    }

    if (failed(linalg::vectorize(rewriter, target, vectorSizes,
                                 getScalableSizes(),
                                 getVectorizeNdExtract().has_value()
                                     ? getVectorizeNdExtract().value()
                                     : false))) {
      return mlir::emitSilenceableFailure(target->getLoc())
             << "Attempted to vectorize, but failed";
    }
  }

  return DiagnosedSilenceableFailure::success();
}

void transform::VectorizeOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getVectorSizesMutable(), effects);
  modifiesPayload(effects);
}

SmallVector<OpFoldResult> VectorizeOp::getMixedVectorSizes() {
  OpBuilder b(getContext());
  return getMixedValues(getStaticVectorSizes(), getVectorSizes(), b);
}

LogicalResult transform::VectorizeOp::verify() {
  if (getStaticVectorSizes().size() != getScalableSizes().size())
    return emitOpError("expected same number of vector sizes (")
           << getStaticVectorSizes().size() << ") and scalable sizes ("
           << getScalableSizes().size() << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// HoistRedundantVectorTransfersOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HoistRedundantVectorTransfersOp::applyToOne(
    transform::TransformRewriter &rewriter, func::FuncOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // WARNING: This hoisting does not model parallelism and is generally
  // incorrect when used on distributed loops with memref semantics!
  // TODO: obsolete and should be retired.
  linalg::hoistRedundantVectorTransfers(target);
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HoistRedundantVectorBroadcastsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HoistRedundantVectorBroadcastsOp::applyToOne(
    transform::TransformRewriter &rewriter, mlir::Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  linalg::hoistRedundantVectorBroadcasts(rewriter, target);
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ConvertConv2DToImg2ColOp.
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ConvertConv2DToImg2ColOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  auto maybeTransformed =
      TypeSwitch<Operation *, FailureOr<std::pair<Operation *, Operation *>>>(
          target)
          .Case([&](linalg::Conv2DNhwcHwcfOp op) {
            return rewriteInIm2Col(rewriter, op);
          })
          .Case([&](linalg::Conv2DNhwcFhwcOp op) {
            return rewriteInIm2Col(rewriter, op);
          })
          .Case([&](linalg::DepthwiseConv2DNhwcHwcOp op) {
            return rewriteInIm2Col(rewriter, op);
          })
          .Case([&](linalg::Conv2DNchwFchwOp op) {
            return rewriteInIm2Col(rewriter, op);
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(op, "not supported");
          });
  if (failed(maybeTransformed))
    return emitDefaultSilenceableFailure(target);
  // Handle to the operation producing the img2col tensor.
  results.push_back(maybeTransformed->first);
  // Handle to the operation that replaces the original convolution.
  results.push_back(maybeTransformed->second);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FlattenElementwiseLinalgOp.
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::FlattenElementwiseLinalgOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  if (!isElementwise(target))
    return mlir::emitSilenceableFailure(target->getLoc())
           << "only elementwise flattening is supported";

  // If rank <= 1, do nothing
  if (target.getNumLoops() <= 1) {
    results.push_back(target);
    return DiagnosedSilenceableFailure::success();
  }

  // Attempt to flatten all dims to one.
  ReassociationIndices reassociation(target.getNumLoops());
  std::iota(reassociation.begin(), reassociation.end(), 0);
  auto maybeFlattened =
      collapseOpIterationDims(target, reassociation, rewriter);
  if (failed(maybeFlattened))
    return mlir::emitSilenceableFailure(target->getLoc())
           << "attempted to flatten, but failed";
  results.push_back(maybeFlattened->collapsedOp);
  rewriter.replaceOp(target, maybeFlattened->results);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TransposeConv2DOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::TransposeConv2DOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  auto maybeTransformed =
      TypeSwitch<Operation *, FailureOr<Operation *>>(target)
          .Case([&](linalg::Conv2DNhwcFhwcOp op) {
            return transposeConv2D(rewriter, op);
          })
          .Case([&](linalg::Conv2DNhwcFhwcQOp op) {
            return transposeConv2D(rewriter, op);
          })
          .Default([&](Operation *op) {
            return rewriter.notifyMatchFailure(op, "not supported");
          });
  if (failed(maybeTransformed))
    return emitDefaultSilenceableFailure(target);
  // Handle to the new Conv2D operation with transposed filters
  results.push_back(*maybeTransformed);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TransposeMatmulOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::TransposeMatmulOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  bool transposeLHS = getInputToTranspose() == TransposeMatmulInput::lhs;
  auto maybeTransformed =
      TypeSwitch<Operation *, FailureOr<Operation *>>(target)
          .Case([&](linalg::MatmulOp op) {
            return transposeMatmul(rewriter, op, transposeLHS);
          })
          .Case([&](linalg::BatchMatmulOp op) {
            return transposeBatchMatmul(rewriter, op, transposeLHS);
          })
          .Default([&](Operation *op) { return failure(); });
  if (failed(maybeTransformed))
    return emitSilenceableFailure(target->getLoc()) << "not supported";
  // Handle to the new Matmul operation with transposed filters
  results.push_back(*maybeTransformed);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// InsertSliceToCopyOp
//===----------------------------------------------------------------------===//
template <typename OpTy>
DiagnosedSilenceableFailure doit(RewriterBase &rewriter, OpTy target,
                                 transform::ApplyToEachResultList &results,
                                 transform::TransformState &state) {
  static_assert(llvm::is_one_of<OpTy, tensor::InsertSliceOp,
                                tensor::ParallelInsertSliceOp>() &&
                "wrong op type");

  if (auto copySource =
          target.getSource().template getDefiningOp<linalg::CopyOp>()) {
    results.push_back(copySource);
    return DiagnosedSilenceableFailure::success();
  }

  // If we are inside an InParallel region, temporarily set the insertion point
  // outside: only tensor.parallel_insert_slice ops are allowed in there.
  if constexpr (std::is_same_v<OpTy, tensor::ParallelInsertSliceOp>) {
    rewriter.setInsertionPoint(
        target->template getParentOfType<scf::InParallelOp>());
  }

  Value extracted = rewriter.create<tensor::ExtractSliceOp>(
      target.getLoc(), target.getDest(), target.getMixedOffsets(),
      target.getMixedSizes(), target.getMixedStrides());
  Value copied = rewriter
                     .create<linalg::CopyOp>(target.getLoc(),
                                             target.getSource(), extracted)
                     .getResult(0);
  // Reset the insertion point.
  rewriter.setInsertionPoint(target);
  rewriter.replaceOpWithNewOp<OpTy>(
      target, copied, target.getDest(), target.getMixedOffsets(),
      target.getMixedSizes(), target.getMixedStrides());

  results.push_back(copied.getDefiningOp());
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::InsertSliceToCopyOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *targetOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {

  rewriter.setInsertionPoint(targetOp);
  if (auto target = dyn_cast<tensor::InsertSliceOp>(targetOp))
    return doit(rewriter, target, results, state);
  if (auto target = dyn_cast<tensor::ParallelInsertSliceOp>(targetOp))
    return doit(rewriter, target, results, state);

  DiagnosedSilenceableFailure diag =
      emitSilenceableError()
      << "only InsertSliceOp and ParallelInsertSliceOp ops are supported";
  diag.attachNote(targetOp->getLoc()) << "target op";
  return diag;
}

//===----------------------------------------------------------------------===//
// MapCopyToThreadsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MapCopyToThreadsOp::applyToOne(
    transform::TransformRewriter &rewriter, Operation *target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // Check if the op is supported.
  if (!isa<linalg::CopyOp, tensor::PadOp>(target)) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "only linalg.copy and tensor.pad target ops are supported";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }
  assert(target->getNumResults() == 1 && "expected single result");
  auto resultShapedType = cast<ShapedType>(target->getResult(0).getType());
  if (!resultShapedType.hasStaticShape()) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "only statically sized ops of rank <= 3 are supported";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  // Conservatively set the minimum viable desired bitwidth alignment.
  int64_t desiredBitAlignment = getDesiredBitAlignment();
  int64_t eltBitwidth =
      resultShapedType.getElementType().getIntOrFloatBitWidth();
  if (desiredBitAlignment % eltBitwidth != 0) {
    desiredBitAlignment = eltBitwidth;
  }

  gpu::CopyMappingInfo mapping(
      /*ctx=*/getContext(),
      /*totalNumThreads=*/getTotalNumThreads(),
      /*alignment=*/desiredBitAlignment,
      /*sizes=*/resultShapedType.getShape(),
      /*favorPredication=*/false,
      /*elementalBitwidth=*/
      resultShapedType.getElementType().getIntOrFloatBitWidth());
  if (mapping.status == gpu::CopyMappingInfo::Status::Invalid) {
    DiagnosedSilenceableFailure diag =
        emitSilenceableError()
        << "too few threads to map copy op to threads on the most minor "
           "dimension, given alignment and vector size constraints, try "
           "smaller tile size of mapping to more threads";
    diag.attachNote(target->getLoc()) << "target op";
    return diag;
  }

  // OpBuilder only used to compute attributes.
  OpBuilder b(getContext());
  scf::SCFTilingResult tilingResult;
  DiagnosedSilenceableFailure diag = tileToForallOpImpl(
      /*rewriter=*/rewriter,
      /*state=*/state,
      /*transformOp=*/*this,
      /*target=*/target,
      /*mixedNumThreads=*/getMixedValues(mapping.numThreads, {}, b),
      /*mixedTileSizes=*/ArrayRef<OpFoldResult>{},
      /*mapping=*/b.getArrayAttr(mapping.threadMapping),
      /*tilingResult=*/tilingResult);
  if (!diag.succeeded())
    return diag;

  results.push_back(tilingResult.loops.front());
  for (auto op : tilingResult.tiledOps)
    results.push_back(op);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// WinogradConv2DOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::WinogradConv2DOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  rewriter.setInsertionPoint(target);
  FailureOr<Operation *> maybeTransformed = failure();
  bool supported = TypeSwitch<Operation *, bool>(target)
                       .Case([&](linalg::Conv2DNhwcFhwcOp op) {
                         maybeTransformed =
                             winogradConv2D(rewriter, op, getM(), getR());
                         return true;
                       })
                       .Default([&](Operation *op) { return false; });

  if (!supported) {
    return emitSilenceableError()
           << "this operation is not supported to convert to Winograd Conv2D";
  }

  if (supported && failed(maybeTransformed)) {
    return emitSilenceableError() << "apply Winograd Conv2D failed";
  }

  results.push_back(*maybeTransformed);
  return DiagnosedSilenceableFailure::success();
}

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOpsEnums.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"
