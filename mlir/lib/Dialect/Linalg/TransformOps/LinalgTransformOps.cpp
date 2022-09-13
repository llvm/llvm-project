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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::transform;

/// Extracts a vector of unsigned from an array attribute. Asserts if the
/// attribute contains values other than intergers. May truncate.
static SmallVector<unsigned> extractUIntArray(ArrayAttr attr) {
  SmallVector<unsigned> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getZExtValue());
  return result;
}

namespace {
/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};
} // namespace

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
  SimpleRewriter rewriter(operation->getContext());
  rewriter.setInsertionPoint(operation);
  auto result = pattern.returningMatchAndRewrite(op, rewriter);
  if (failed(result))
    return failure();
  return cast<LinalgOp>(result->getOperation());
}

//===----------------------------------------------------------------------===//
// DecomposeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::DecomposeOp::applyToOne(linalg::LinalgOp target,
                                   SmallVectorImpl<Operation *> &results,
                                   transform::TransformState &state) {
  FailureOr<LinalgOp> windowedNhwc =
      tryApply<DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNhwcHwcfOp,
                                                     Conv1DNwcWcfOp>>(target);
  if (succeeded(windowedNhwc)) {
    results.push_back(*windowedNhwc);
    return DiagnosedSilenceableFailure(success());
  }
  FailureOr<LinalgOp> windowedNchw =
      tryApply<DownscaleSizeOneWindowed2DConvolution<linalg::Conv2DNchwFchwOp,
                                                     Conv1DNcwFcwOp>>(target);
  if (succeeded(windowedNchw)) {
    results.push_back(*windowedNchw);
    return DiagnosedSilenceableFailure(success());
  }
  FailureOr<LinalgOp> depthwise =
      tryApply<DownscaleDepthwiseConv2DNhwcHwcOp>(target);
  if (succeeded(depthwise)) {
    results.push_back(*depthwise);
    return DiagnosedSilenceableFailure(success());
  }
  results.assign(1, nullptr);
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// FuseOp
//===----------------------------------------------------------------------===//

/// Apply a tiling transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTilingToAll(Operation *transformOp, ArrayRef<Operation *> payloadOps,
                 unsigned numLoops,
                 transform::TransformResults &transformResults,
                 function_ref<FailureOr<TiledLinalgOp>(LinalgOp)> applyFn) {
  SmallVector<Operation *> tiledLinalgOps;
  SmallVector<SmallVector<Operation *>> loopOps(numLoops);
  for (unsigned int i = 0; i < numLoops; ++i)
    loopOps[i].reserve(payloadOps.size());

  for (Operation *target : payloadOps) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
    if (!linalgOp)
      return transformOp->emitError("only LinalgOps are supported");

    FailureOr<TiledLinalgOp> tiled = applyFn(linalgOp);
    if (failed(tiled))
      return failure();

    tiledLinalgOps.push_back(tiled->op);
    if (tiled->loops.size() != numLoops)
      // Not enough loops were generated. This usually means that the input size
      // was smaller than the tiling size.
      // TODO: LinalgTilingPattern should return failure().
      return failure();
    for (unsigned int i = 0; i < numLoops; ++i)
      loopOps[i].push_back(tiled->loops[i]);
  }

  transformResults.set(transformOp->getOpResult(0), tiledLinalgOps);
  for (unsigned int i = 0; i < numLoops; ++i)
    transformResults.set(transformOp->getOpResult(i + 1), loopOps[i]);
  return success();
}

/// Parse a tiling-like operation that returns the tiled op as well as the
/// created tile loops. The function counts the non-zero tile sizes to compute
/// the number of results.
static ParseResult parseTileLikeOp(OpAsmParser &parser, OperationState &result,
                                   StringRef sizesAttrName) {
  OpAsmParser::UnresolvedOperand targetOperand;
  SMLoc opLoc = parser.getCurrentLocation();
  if (parser.parseOperand(targetOperand) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  Attribute sizesAttr = result.attributes.get(sizesAttrName);
  if (!sizesAttr)
    return parser.emitError(opLoc)
           << "expected '" << sizesAttrName << "' attribute";
  auto sizesArrayAttr = sizesAttr.dyn_cast<ArrayAttr>();
  if (!sizesArrayAttr)
    return parser.emitError(opLoc)
           << "'" << sizesAttrName << "' attribute must be an array";
  Type pdlOpType = parser.getBuilder().getType<pdl::OperationType>();
  size_t numExpectedLoops =
      sizesArrayAttr.size() -
      llvm::count(extractFromI64ArrayAttr(sizesArrayAttr), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOpType));
  if (parser.resolveOperand(targetOperand, pdlOpType, result.operands))
    return failure();
  return success();
}

DiagnosedSilenceableFailure
transform::FuseOp::apply(mlir::transform::TransformResults &transformResults,
                         mlir::transform::TransformState &state) {
  LinalgTilingAndFusionOptions fusionOptions;
  fusionOptions.tileSizes = extractFromI64ArrayAttr(getTileSizes());
  fusionOptions.tileInterchange = extractFromI64ArrayAttr(getTileInterchange());

  LogicalResult result = applyTilingToAll(
      getOperation(), state.getPayloadOps(getTarget()),
      fusionOptions.tileSizes.size() - llvm::count(fusionOptions.tileSizes, 0),
      transformResults, [&](LinalgOp linalgOp) -> FailureOr<TiledLinalgOp> {
        LinalgTileAndFuseTensorOpsPattern pattern(getContext(), fusionOptions);
        SimpleRewriter rewriter(getContext());
        rewriter.setInsertionPoint(linalgOp);
        FailureOr<TileLoopNest> tileLoopNest =
            pattern.returningMatchAndRewrite(linalgOp, rewriter);
        if (failed(tileLoopNest))
          return failure();

        TiledLinalgOp tiledLinalgOp;
        tiledLinalgOp.op = tileLoopNest->getRootOp();
        tiledLinalgOp.loops = {tileLoopNest->getLoopOps().begin(),
                               tileLoopNest->getLoopOps().end()};
        return tiledLinalgOp;
      });
  return DiagnosedSilenceableFailure(result);
}

ParseResult transform::FuseOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseTileLikeOp(
      parser, result,
      transform::FuseOp::getTileSizesAttrName(result.name).getValue());
}

void transform::FuseOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult transform::FuseOp::verify() {
  SmallVector<int64_t> permutation =
      extractFromI64ArrayAttr(getTileInterchange());
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError() << "expects interchange to be a permutation, found "
                         << getTileInterchange();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// FuseIntoContainingOp
//===----------------------------------------------------------------------===//

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is now fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation *tileAndFuseFirstExtractUse(Operation *producerOp,
                                             Operation *containingOp,
                                             RewriterBase &rewriter) {
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer)
    return nullptr;

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  auto it = llvm::find_if(tileableProducer->getUsers(), [&](Operation *user) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Check for a non-empty fusion opportunity.
  if (it == tileableProducer->getUsers().end())
    return nullptr;
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  FailureOr<Value> tiledProducer = tileableProducer.generateResultTileValue(
      rewriter, /*resultNumber=*/0, sliceOpToTile.getMixedOffsets(),
      sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer))
    return nullptr;

  // Replace the extract op.
  Operation *fusedOp = tiledProducer->getDefiningOp();
  rewriter.replaceOp(sliceOpToTile, fusedOp->getResult(0));
  return fusedOp;
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is now fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation *tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    Operation *producerOp, Operation *containingOp, RewriterBase &rewriter) {

  auto foreachThreadOp = dyn_cast<scf::ForeachThreadOp>(containingOp);
  if (!foreachThreadOp)
    return nullptr;

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer)
    return nullptr;

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse;
  BlockArgument bbArg;
  tensor::ExtractSliceOp sliceOpToTile;
  // Only consider slices that may come from the containingOp args.
  for (OpOperand &use : tileableProducer->getUses()) {
    if (use.getOwner() != containingOp)
      continue;
    pUse = &use;
    bbArg = foreachThreadOp.getTiedBlockArgument(&use);
    for (Operation *user : bbArg.getUsers()) {
      auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!sliceOp)
        continue;
      if (!containingOp->isAncestor(sliceOp))
        continue;
      sliceOpToTile = sliceOp;
      break;
    }
    if (sliceOpToTile)
      break;
  }

  // Check for a non-empty list of fusion opportunities.
  if (!sliceOpToTile || !pUse)
    return nullptr;

  // Ensure there is exactly one destination operand that we can replace the
  // ForeachThreadOp bbArg with.
  auto destinationOperands = tileableProducer.getDestinationOperands(rewriter);
  if (destinationOperands.size() != 1)
    return nullptr;

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling, replace and then
  // tile.
  BlockAndValueMapping bvm;
  bvm.map(destinationOperands.front(), bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<Value> tiledProducer =
      tileableProducerClone.generateResultTileValue(
          rewriter, /*resultNumber=*/0, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer))
    return nullptr;

  // Replace the extract op.
  Operation *fusedOp = tiledProducer->getDefiningOp();
  rewriter.replaceOp(sliceOpToTile, fusedOp->getResult(0));

  // Replace the use in containingOp.
  rewriter.startRootUpdate(fusedOp);
  containingOp->setOperand(pUse->getOperandNumber(),
                           destinationOperands.front());
  rewriter.finalizeRootUpdate(fusedOp);

  return fusedOp;
}

static Operation *cloneAndFuseFirstUse(Operation *producerOp,
                                       Operation *containingOp,
                                       RewriterBase &rewriter) {
  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults()) {
    for (OpOperand &use : result.getUses()) {
      if (containingOp->isProperAncestor(use.getOwner())) {
        uses.push_back(&use);
        continue;
      }
      // Cannot clone and fuse if the use is fom the containing op itself: fail.
      if (containingOp == use.getOwner())
        return nullptr;
    }
  }

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty())
    return nullptr;

  // Clone and fuse inside the containing op.
  Operation *fusedOp = nullptr;
  for (OpOperand *use : uses) {
    // Parallel insert slice is not a valid clone destination.
    // TODO: Generalize to other type of ops.
    assert(!isa<tensor::ParallelInsertSliceOp>(use->getOwner()) &&
           "Parallel insert slice is not a valid clone destination");
    unsigned resultNumber = use->get().cast<OpResult>().getResultNumber();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(use->getOwner());
    fusedOp = rewriter.clone(*producerOp);
    rewriter.updateRootInPlace(
        use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });
    break;
  }

  return fusedOp;
}

DiagnosedSilenceableFailure
transform::FuseIntoContainingOp::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  ArrayRef<Operation *> producerOps = state.getPayloadOps(getProducerOp());
  // If nothing to fuse, propagate success.
  if (producerOps.empty()) {
    results.set(getResult().cast<OpResult>(), SmallVector<mlir::Operation *>{});
    return DiagnosedSilenceableFailure::success();
  }
  for (Operation *producerOp : producerOps) {
    if (producerOp->getNumResults() != 1) {
      Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
      diag << "op with != 1 results not supported";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  }
  ArrayRef<Operation *> containingOps = state.getPayloadOps(getContainingOp());
  if (containingOps.size() != 1)
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one containing_op handle (got ")
        << containingOps.size() << ")");
  Operation *containingOp = containingOps.front();

  // Helper function to find the next producer that should be fused. Take any
  // producer that has a use inside the containing op.
  SmallVector<Operation *> remainingProducers(producerOps.begin(),
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

  IRRewriter rewriter(getContext());
  while (!remainingProducers.empty()) {
    auto nextProducer = getNextProducer();
    if (failed(nextProducer)) {
      Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
      diag << "could not fuse ops into container";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }

    Operation *producerOp = *nextProducer;
    // TODO: If there are multiple uses of the producer in the containing op,
    // we currently tile/clone the op multiple times (once per use). In some
    // cases, we can tile/clone once and reuse the value for each use.
    // Futhermore, producers should then be traversed according to a
    // topological sorting.
    Operation *tiled =
        tileAndFuseFirstExtractUse(producerOp, containingOp, rewriter);
    if (tiled) {
      fusedOps.push_back(tiled);
      continue;
    }

    Operation *tiledContainingOpOperand =
        tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            producerOp, containingOp, rewriter);
    if (tiledContainingOpOperand) {
      fusedOps.push_back(tiledContainingOpOperand);
      continue;
    }

    Operation *cloned =
        cloneAndFuseFirstUse(producerOp, containingOp, rewriter);
    if (cloned) {
      fusedOps.push_back(cloned);
      continue;
    }

    Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not fuse " << *producerOp << "into " << *containingOp;
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  results.set(getFusedOp().cast<OpResult>(), fusedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// GeneralizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::GeneralizeOp::applyToOne(linalg::LinalgOp target,
                                    SmallVectorImpl<Operation *> &results,
                                    transform::TransformState &state) {
  // Exit early if no transformation is needed.
  if (isa<GenericOp>(target)) {
    results.push_back(target);
    return DiagnosedSilenceableFailure(success());
  }
  FailureOr<LinalgOp> generic = tryApply<LinalgGeneralizationPattern>(target);
  if (succeeded(generic)) {
    results.push_back(generic->getOperation());
    return DiagnosedSilenceableFailure(success());
  }
  results.assign(1, nullptr);
  return emitDefaultSilenceableFailure(target);
}

//===----------------------------------------------------------------------===//
// InterchangeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::InterchangeOp::applyToOne(linalg::GenericOp target,
                                     SmallVectorImpl<Operation *> &results,
                                     transform::TransformState &state) {
  SmallVector<unsigned> interchangeVector =
      extractUIntArray(getIteratorInterchange());
  // Exit early if no transformation is needed.
  if (interchangeVector.empty()) {
    results.push_back(target);
    return DiagnosedSilenceableFailure(success());
  }
  SimpleRewriter rewriter(target->getContext());
  FailureOr<GenericOp> res =
      interchangeGenericOp(rewriter, target, interchangeVector);
  if (failed(res))
    return DiagnosedSilenceableFailure::definiteFailure();
  results.push_back(res->getOperation());
  return DiagnosedSilenceableFailure(success());
}

LogicalResult transform::InterchangeOp::verify() {
  SmallVector<unsigned> permutation =
      extractUIntArray(getIteratorInterchange());
  auto sequence = llvm::to_vector(llvm::seq<unsigned>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError()
           << "expects iterator_interchange to be a permutation, found "
           << getIteratorInterchange();
  }
  return success();
}

//===---------------------------------------------------------------------===//
// MatchOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MatchOp::apply(transform::TransformResults &results,
                          transform::TransformState &state) {
  llvm::StringSet<> strs;
  if (getOps().has_value())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  if (payloadOps.size() != 1)
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one target handle"));

  SmallVector<Operation *> res;
  auto matchFun = [&](Operation *op) {
    if (getOps().has_value() && !strs.contains(op->getName().getStringRef()))
      return;

    // Interfaces cannot be matched by name, just by ID.
    // So we specifically encode the interfaces we care about for this op.
    if (getInterface().has_value()) {
      auto iface = getInterface().value();
      if (iface == transform::MatchInterfaceEnum::LinalgOp &&
          !isa<linalg::LinalgOp>(op))
        return;
      if (iface == transform::MatchInterfaceEnum::TilingInterface &&
          isa<TilingInterface>(op))
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

    // All constraints are satisfied.
    res.push_back(op);
    return;
  };

  payloadOps.front()->walk(matchFun);
  results.set(getResult().cast<OpResult>(), res);
  return DiagnosedSilenceableFailure(success());
}

//===---------------------------------------------------------------------===//
// MultiTileSizesOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::MultiTileSizesOp::applyToOne(
    LinalgOp target, SmallVector<Operation *> &results, TransformState &state) {
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
      makeComposedAffineApply(builder, target.getLoc(), s0 * s1,
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
  onlyReadsHandle(getTarget(), effects);
  producesHandle(getResults(), effects);
  modifiesPayload(effects);
}

//===---------------------------------------------------------------------===//
// PadOp
//===---------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PadOp::applyToOne(linalg::LinalgOp target,
                             SmallVectorImpl<Operation *> &results,
                             transform::TransformState &state) {
  // Convert the integer packing flags to booleans.
  SmallVector<bool> packPaddings;
  for (int64_t packPadding : extractFromI64ArrayAttr(getPackPaddings()))
    packPaddings.push_back(static_cast<bool>(packPadding));

  // Convert the padding values to attributes.
  SmallVector<Attribute> paddingValues;
  for (auto const &it :
       llvm::zip(getPaddingValues(), target->getOperandTypes())) {
    auto attr = std::get<0>(it).dyn_cast<TypedAttr>();
    if (!attr) {
      emitOpError("expects padding values to be typed attributes");
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    Type elementType = getElementTypeOrSelf(std::get<1>(it));
    // Try to parse string attributes to obtain an attribute of element type.
    if (auto stringAttr = attr.dyn_cast<StringAttr>()) {
      paddingValues.push_back(
          parseAttribute(attr.cast<StringAttr>(), elementType));
      if (!paddingValues.back()) {
        auto diag = this->emitOpError("expects a padding that parses to ")
                    << elementType << ", got " << std::get<0>(it);
        diag.attachNote(target.getLoc()) << "when applied to this op";
        return DiagnosedSilenceableFailure::definiteFailure();
      }
      continue;
    }
    // Otherwise, add the attribute directly.
    if (attr.getType() != elementType) {
      auto diag = this->emitOpError("expects a padding value of type ")
                  << elementType << ", got " << attr;
      diag.attachNote(target.getLoc()) << "when applied to this op";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
    paddingValues.push_back(attr);
  }

  // Extract the transpose vectors.
  SmallVector<SmallVector<int64_t>> transposePaddings;
  for (Attribute transposeVector : getTransposePaddings().cast<ArrayAttr>())
    transposePaddings.push_back(
        extractFromI64ArrayAttr(transposeVector.cast<ArrayAttr>()));

  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValues);
  paddingOptions.setPaddingDimensions(
      extractFromI64ArrayAttr(getPaddingDimensions()));
  paddingOptions.setPackPaddings(packPaddings);
  paddingOptions.setHoistPaddings(extractFromI64ArrayAttr(getHoistPaddings()));
  paddingOptions.setTransposePaddings(transposePaddings);

  FailureOr<LinalgOp> result =
      tryApply<LinalgPaddingPattern>(target, paddingOptions);
  if (succeeded(result)) {
    results.push_back(result->getOperation());
    return DiagnosedSilenceableFailure(success());
  }

  results.assign(1, nullptr);
  return emitDefaultSilenceableFailure(target);
}

LogicalResult transform::PadOp::verify() {
  SmallVector<int64_t> packPaddings =
      extractFromI64ArrayAttr(getPackPaddings());
  if (any_of(packPaddings, [](int64_t packPadding) {
        return packPadding != 0 && packPadding != 1;
      })) {
    return emitOpError()
           << "expects pack_paddings to contain booleans (0/1), found "
           << getPackPaddings();
  }

  SmallVector<int64_t> paddingDimensions =
      extractFromI64ArrayAttr(getPaddingDimensions());
  if (any_of(paddingDimensions,
             [](int64_t paddingDimension) { return paddingDimension < 0; })) {
    return emitOpError() << "expects padding_dimensions to contain positive "
                            "integers, found "
                         << getPaddingDimensions();
  }

  SmallVector<int64_t> hoistPaddings =
      extractFromI64ArrayAttr(getHoistPaddings());
  if (any_of(hoistPaddings,
             [](int64_t hoistPadding) { return hoistPadding < 0; })) {
    return emitOpError()
           << "expects hoist_paddings to contain positive integers, found "
           << getHoistPaddings();
  }

  ArrayAttr transposes = getTransposePaddings();
  for (Attribute attr : transposes) {
    SmallVector<int64_t> transpose = extractFromI64ArrayAttr(attr);
    auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, transpose.size()));
    if (!std::is_permutation(sequence.begin(), sequence.end(),
                             transpose.begin(), transpose.end())) {
      return emitOpError()
             << "expects transpose_paddings to be a permutation, found "
             << attr;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PromoteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::PromoteOp::applyToOne(linalg::LinalgOp target,
                                 SmallVectorImpl<Operation *> &results,
                                 transform::TransformState &state) {
  LinalgPromotionOptions promotionOptions;
  if (!getOperandsToPromote().empty())
    promotionOptions = promotionOptions.setOperandsToPromote(
        extractFromI64ArrayAttr(getOperandsToPromote()));
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

  if (failed(promoteSubviewsPrecondition(target, promotionOptions)))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  SimpleRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> res = promoteSubViews(rewriter, target, promotionOptions);
  if (failed(res))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(target);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// ScalarizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ScalarizeOp::applyToOne(linalg::LinalgOp target,
                                   SmallVectorImpl<Operation *> &results,
                                   transform::TransformState &state) {
  LinalgTilingOptions tilingOptions;
  tilingOptions.scalarizeDynamicDims();
  // Tiling with "scalarize_dyn_dims" actually sets the same lambda as the
  // tile sizes and asserts that it is not already set.
  SmallVector<int64_t> emptyTileSizes;
  LinalgTilingPattern pattern(getContext(), tilingOptions);
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<TiledLinalgOp> result =
      pattern.returningMatchAndRewrite(target, rewriter);
  if (failed(result))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.push_back(result->op);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure SplitOp::apply(TransformResults &results,
                                           TransformState &state) {
  // Collect the dynamic split points if provided.
  ArrayRef<Operation *> payload = state.getPayloadOps(getTarget());
  SimpleRewriter rewriter(getContext());
  SmallVector<OpFoldResult> splitPoints;
  splitPoints.reserve(payload.size());
  if (getDynamicSplitPoint()) {
    auto diag = DiagnosedSilenceableFailure::success();
    splitPoints = llvm::to_vector(llvm::map_range(
        state.getPayloadOps(getDynamicSplitPoint()), [&](Operation *op) {
          if (op->getNumResults() != 1 ||
              !op->getResult(0).getType().isIndex()) {
            diag = emitSilenceableError()
                   << "expected dynamic split point handle to point to a "
                      "single-result index-typed op";
            diag.attachNote(op->getLoc()) << "dynamic split point";
          }
          return OpFoldResult(op->getResult(0));
        }));
    if (!diag.succeeded())
      return diag;

    if (splitPoints.size() != payload.size()) {
      emitError() << "expected the dynamic split point handle to point to as "
                     "many operations ("
                  << splitPoints.size() << ") as the target handle ("
                  << payload.size() << ")";
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  } else {
    splitPoints.resize(payload.size(),
                       rewriter.getIndexAttr(getStaticSplitPoint()));
  }

  // Split each target operation.
  SmallVector<Operation *> first, second;
  for (const auto &pair : llvm::zip(payload, splitPoints)) {
    Operation *target = std::get<0>(pair);
    auto linalgOp = dyn_cast<LinalgOp>(target);
    if (!linalgOp) {
      auto diag = emitSilenceableError() << "only applies to structured ops";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }

    if (getDimension() >= linalgOp.getNumLoops()) {
      auto diag = emitSilenceableError() << "dimension " << getDimension()
                                         << " does not exist in target op";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }

    rewriter.setInsertionPoint(linalgOp);
    std::tie(first.emplace_back(), second.emplace_back()) = linalg::splitOp(
        rewriter, cast<TilingInterface>(linalgOp.getOperation()),
        getDimension(), std::get<1>(pair));
  }

  results.set(getFirst().cast<OpResult>(), first);
  results.set(getSecond().cast<OpResult>(), second);
  return DiagnosedSilenceableFailure::success();
}

void SplitOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  if (getDynamicSplitPoint())
    onlyReadsHandle(getDynamicSplitPoint(), effects);
  producesHandle(getResults(), effects);
  modifiesPayload(effects);
}

ParseResult SplitOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand target, dynamicSplitPoint;
  IntegerAttr staticSplitPoint;
  auto pdlOperationType =
      pdl::OperationType::get(parser.getBuilder().getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parser.parseKeyword("after"))
    return failure();

  OptionalParseResult dynamicPointParseResult =
      parser.parseOptionalOperand(dynamicSplitPoint);
  if (!dynamicPointParseResult.has_value()) {
    int64_t staticSplitPointValue;
    if (failed(parser.parseInteger(staticSplitPointValue)))
      return failure();

    staticSplitPoint =
        parser.getBuilder().getI64IntegerAttr(staticSplitPointValue);
  } else {
    if (failed(*dynamicPointParseResult) ||
        parser.resolveOperand(dynamicSplitPoint, pdlOperationType,
                              result.operands)) {
      return failure();
    }

    staticSplitPoint =
        parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamicSize);
  }

  result.addAttribute(
      SplitOp::getStaticSplitPointAttrName(result.name).getValue(),
      staticSplitPoint);
  if (failed(parser.parseOptionalAttrDict(result.attributes)))
    return failure();

  result.addTypes({pdlOperationType, pdlOperationType});
  return success();
}

void SplitOp::print(OpAsmPrinter &printer) {
  printer << " " << getTarget() << " after ";
  int64_t staticSplitSize = static_cast<int64_t>(getStaticSplitPoint());
  if (staticSplitSize != ShapedType::kDynamicSize)
    printer << staticSplitSize;
  else
    printer << getDynamicSplitPoint();
  printer << " ";
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {getStaticSplitPointAttrName()});
}

LogicalResult SplitOp::verify() {
  if ((static_cast<int64_t>(getStaticSplitPoint()) !=
       ShapedType::kDynamicSize) ^
      (getDynamicSplitPoint() == nullptr)) {
    return emitOpError() << "expects either a dynamic or a static split "
                            "point to be provided";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SplitReductionOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::SplitReductionOp::applyToOne(linalg::LinalgOp target,
                                        SmallVectorImpl<Operation *> &results,
                                        transform::TransformState &state) {
  ControlSplitReductionFn splitFn = [&](LinalgOp) {
    return std::pair<int64_t, unsigned>(getSplitFactor(),
                                        getInsertSplitDimension());
  };
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<SplitReductionResult> splitResult =
      (getUseScalingAlgorithm())
          ? splitReductionByScaling(rewriter, target, splitFn, getUseAlloc())
          : splitReduction(rewriter, target, splitFn, getUseAlloc());
  if (failed(splitResult))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.push_back(splitResult->initOrAlloc);
  results.push_back(splitResult->fillOp);
  results.push_back(splitResult->splitLinalgOp);
  results.push_back(splitResult->resultCombiningLinalgOp);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::TileOp::apply(TransformResults &transformResults,
                         TransformState &state) {
  LinalgTilingOptions tilingOptions;
  SmallVector<int64_t> tileSizes = extractFromI64ArrayAttr(getStaticSizes());

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  SmallVector<ArrayRef<Operation *>> dynamicSizeProducers;
  dynamicSizeProducers.reserve(getDynamicSizes().size());
  for (Value dynamicSizeProducerHandle : getDynamicSizes()) {
    dynamicSizeProducers.push_back(
        state.getPayloadOps(dynamicSizeProducerHandle));

    if (dynamicSizeProducers.back().size() != targets.size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected as many dynamic size-producing operations ("
          << dynamicSizeProducers.back().size() << ") as target ops ("
          << targets.size() << ")";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }

    for (Operation *op : dynamicSizeProducers.back()) {
      if (op->getNumResults() == 1 &&
          op->getResult(0).getType().isa<IndexType>())
        continue;
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected sizes to be produced by ops "
                                    "with a single index-type result";
      diag.attachNote(op->getLoc()) << "size producer op";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }
  }

  SmallVector<Operation *> tiled;
  SmallVector<SmallVector<Operation *, 4>, 4> loops;
  loops.resize(getLoops().size());
  for (auto &en : llvm::enumerate(targets)) {
    auto linalgOp = dyn_cast<LinalgOp>(en.value());
    if (!linalgOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "only linalg ops are supported";
      diag.attachNote(en.value()->getLoc()) << "target op";
      return diag;
    }

    unsigned index = en.index();
    if (!tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [&, index](OpBuilder &b, Operation *) {
            SmallVector<Value, 4> sizes;
            sizes.reserve(tileSizes.size());
            unsigned dynamicIdx = 0;
            for (OpFoldResult ofr : getMixedSizes()) {
              if (auto attr = ofr.dyn_cast<Attribute>()) {
                sizes.push_back(b.create<arith::ConstantIndexOp>(
                    getLoc(), attr.cast<IntegerAttr>().getInt()));
              } else {
                sizes.push_back(
                    dynamicSizeProducers[dynamicIdx++][index]->getResult(0));
              }
            }
            return sizes;
          });
    }

    tilingOptions.setInterchange(extractUIntArray(getInterchange()));
    LinalgTilingPattern pattern(getContext(), tilingOptions);
    SimpleRewriter rewriter(linalgOp.getContext());
    FailureOr<TiledLinalgOp> tiledOp =
        pattern.returningMatchAndRewrite(linalgOp, rewriter);
    if (failed(tiledOp))
      return DiagnosedSilenceableFailure::definiteFailure();

    tiled.push_back(tiledOp->op);
    for (const auto &en2 : llvm::enumerate(tiledOp->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(getLoops()[en.index()].cast<OpResult>(), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::TileOp::getMixedSizes() {
  ValueRange dynamic = getDynamicSizes();
  SmallVector<int64_t> tileSizes = extractFromI64ArrayAttr(getStaticSizes());
  SmallVector<OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  Builder builder(getContext());
  for (int64_t size : tileSizes) {
    if (size == ShapedType::kDynamicSize) {
      results.push_back(dynamic[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

ParseResult transform::TileOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  ArrayAttr staticSizes;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes,
                            ShapedType::kDynamicSize) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return ParseResult::failure();

  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  size_t numExpectedLoops =
      staticSizes.size() - llvm::count(extractFromI64ArrayAttr(staticSizes), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOperationType));
  return success();
}

void TileOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes(),
                        ShapedType::kDynamicSize);
  p.printOptionalAttrDict((*this)->getAttrs(), {getStaticSizesAttrName()});
}

void transform::TileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  onlyReadsHandle(getDynamicSizes(), effects);
  producesHandle(getTiledLinalgOp(), effects);
  producesHandle(getLoops(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// TileToForeachThreadOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::TileToForeachThreadOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // If there the target payload ops are empty, there is nothing to do.
  if (targets.empty()) {
    transformResults.set(getForeachThreadOp().cast<OpResult>(), {});
    transformResults.set(getTiledOp().cast<OpResult>(), {});
    return DiagnosedSilenceableFailure(success());
  }

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  // Given a list of OpFoldResults that are either index attrs or op handles,
  // return a list of OpFoldResults where all op handles are replaced with the
  // first (and only) OpResult of that payload op. (There must be exactly one
  // mapped payload op and it must have exactly one index result.)
  auto getOpResultsOrIndexAttrs =
      [&](SmallVector<OpFoldResult> &result,
          ArrayRef<OpFoldResult> opHandlesOrIndexAttrs) {
        for (OpFoldResult ofr : opHandlesOrIndexAttrs) {
          if (ofr.is<Attribute>()) {
            result.push_back(ofr);
            continue;
          }
          ArrayRef<Operation *> dynamicNumThreads =
              state.getPayloadOps(ofr.get<Value>());
          if (dynamicNumThreads.size() != 1) {
            DiagnosedSilenceableFailure diag =
                emitSilenceableError()
                << "handle must be mapped to exactly 1 payload op";
            diag.attachNote(ofr.get<Value>().getLoc())
                << "mapped to " << dynamicNumThreads.size() << " ops";
            return diag;
          }
          Operation *op = dynamicNumThreads[0];
          if (op->getNumResults() != 1 ||
              !op->getResult(0).getType().isIndex()) {
            DiagnosedSilenceableFailure diag =
                emitSilenceableError()
                << "payload op must have exactly 1 index result";
            diag.attachNote(op->getLoc())
                << "has " << op->getNumResults() << " results";
            return diag;
          }
          result.push_back(op->getResult(0));
        }

        return DiagnosedSilenceableFailure(success());
      };

  // getMixedNumThreads are OpFoldResults[index attributes or PDL operation].
  // Convert to OpFoldResults[index attributes or payload op].
  SmallVector<OpFoldResult> numThreads;
  DiagnosedSilenceableFailure status =
      getOpResultsOrIndexAttrs(numThreads, getMixedNumThreads());
  if (!status.succeeded())
    return status;

  // getMixedTileSizes are OpFoldResults[index attributes or PDL operation].
  // Convert to OpFoldResults[index attributes or payload op].
  SmallVector<OpFoldResult> tileSizes;
  status = getOpResultsOrIndexAttrs(tileSizes, getMixedTileSizes());
  if (!status.succeeded())
    return status;

  // Transform all targets one by one.
  for (Operation *target : targets) {
    auto tilableOp = dyn_cast<TilingInterface>(target);
    if (!tilableOp) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "only TilingInterface ops are supported";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    rewriter.setInsertionPoint(tilableOp);
    auto maybeThreadDimMappingAttr = getThreadDimMapping();
    auto dimMapping = llvm::to_vector(
        maybeThreadDimMappingAttr
            ? extractFromI64ArrayAttr(*maybeThreadDimMappingAttr)
            : ArrayRef<int64_t>{});

    FailureOr<ForeachThreadTilingResult> tilingResult = failure();
    if (!getMixedNumThreads().empty()) {
      tilingResult = linalg::tileToForeachThreadOp(rewriter, tilableOp,
                                                   numThreads, dimMapping);
    } else {
      tilingResult = linalg::tileToForeachThreadOpUsingTileSizes(
          rewriter, tilableOp, tileSizes, dimMapping);
    }

    if (failed(tilingResult))
      return emitDefaultSilenceableFailure(tilableOp);
    rewriter.replaceOp(tilableOp, tilingResult->tileOp->getResults());

    tileOps.push_back(tilingResult->tileOp);
    tiledOps.push_back(tilingResult->tiledOp);
  }

  transformResults.set(getForeachThreadOp().cast<OpResult>(), tileOps);
  transformResults.set(getTiledOp().cast<OpResult>(), tiledOps);

  return DiagnosedSilenceableFailure(success());
}

void transform::TileToForeachThreadOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  onlyReadsHandle(getTileSizes(), effects);
  onlyReadsHandle(getNumThreads(), effects);
  producesHandle(getResults(), effects);
}

SmallVector<OpFoldResult> TileToForeachThreadOp::getMixedNumThreads() {
  return getMixedSizes(getStaticNumThreads(), getNumThreads());
}

SmallVector<OpFoldResult> TileToForeachThreadOp::getMixedTileSizes() {
  return getMixedSizes(getStaticTileSizes(), getTileSizes());
}

LogicalResult TileToForeachThreadOp::verify() {
  if (getMixedNumThreads().empty() == getMixedTileSizes().empty())
    return emitOpError("either num_threads or tile_sizes must be specified");
  return success();
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::VectorizeOp::applyToOne(Operation *target,
                                   SmallVectorImpl<Operation *> &results,
                                   transform::TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto diag = this->emitOpError("requires isolated-from-above targets");
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return DiagnosedSilenceableFailure::definiteFailure();
  }

  MLIRContext *ctx = getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<LinalgVectorizationPattern>(ctx);

  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  if (getVectorizePadding())
    linalg::populatePadOpVectorizationPatterns(patterns);

  if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns))))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.push_back(target);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
/// Registers new ops and declares PDL as dependent dialect since the
/// additional ops are using PDL types for operands and results.
class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();

    declareGeneratedDialect<AffineDialect>();
    declareGeneratedDialect<arith::ArithmeticDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOpsEnums.cpp.inc"

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"

void mlir::linalg::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
