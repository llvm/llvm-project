//===- LinalgTransformOps.cpp - Implementation of Linalg transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
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
  FailureOr<LinalgOp> windowed =
      tryApply<DownscaleSizeOneWindowed2DConvolution>(target);
  if (succeeded(windowed)) {
    results.push_back(*windowed);
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

static FailureOr<SmallVector<Operation *>> tileAndFuse(Operation *producerOp,
                                                       Operation *containingOp,
                                                       RewriterBase &rewriter) {
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer)
    return failure();

  // Search the producer slices accessed within the containing operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples. Maybe
  // evolve into an interface.
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  for (Operation *user : tileableProducer->getUsers()) {
    auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(user);
    if (!sliceOp)
      continue;
    if (!containingOp->isProperAncestor(sliceOp))
      continue;
    sliceOps.push_back(sliceOp);
  }

  // Check for a non-empty list of fusion opportunities.
  if (sliceOps.empty())
    return failure();

  SmallVector<Value> destinationOperands =
      tileableProducer.getDestinationOperands(rewriter);

  // Try to fuse the producer in-place.
  SmallVector<Operation *> fusedOps;
  for (tensor::ExtractSliceOp sliceOp : sliceOps) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(sliceOp);

    // Tile the producer.
    FailureOr<Value> tiledProducer = tileableProducer.generateResultTileValue(
        rewriter, /*resultNumber=*/0, destinationOperands,
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(), true);
    if (failed(tiledProducer))
      return failure();
    fusedOps.push_back(tiledProducer->getDefiningOp());
  }

  // Replace the extract op.
  for (const auto &en : enumerate(sliceOps))
    rewriter.replaceOp(en.value(), fusedOps[en.index()]->getResult(0));
  return fusedOps;
}

static FailureOr<SmallVector<Operation *>>
cloneAndFuse(Operation *producerOp, Operation *containingOp,
             RewriterBase &rewriter) {
  // Gather all uses inside the containing op.
  SmallVector<OpOperand *> uses;
  for (OpResult result : producerOp->getOpResults())
    for (OpOperand &use : result.getUses())
      if (containingOp->isProperAncestor(use.getOwner()))
        uses.push_back(&use);

  // Check for a non-empty list of fusion opportunities.
  if (uses.empty())
    return failure();

  // Clone and fuse inside the containing op.
  SmallVector<Operation *> fusedOps;
  for (OpOperand *use : uses) {
    unsigned resultNumber = use->get().cast<OpResult>().getResultNumber();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(use->getOwner());
    Operation *cloned = rewriter.clone(*producerOp);
    rewriter.updateRootInPlace(
        use->getOwner(), [&] { use->set(cloned->getOpResult(resultNumber)); });
    fusedOps.push_back(cloned);
  }

  return fusedOps;
}

DiagnosedSilenceableFailure
transform::FuseIntoContainingOp::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  ArrayRef<Operation *> producerOps = state.getPayloadOps(getProducerOp());
  for (Operation *producerOp : producerOps) {
    if (producerOp->getNumResults() != 1) {
      Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Note);
      diag << "op with != 1 results not supported";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
  }
  ArrayRef<Operation *> containingOps = state.getPayloadOps(getContainingOp());
  if (containingOps.size() != 1)
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one containing_op handle"));
  Operation *containingOp = containingOps.front();

  // Helper function to find the next producer that should be fused. Take any
  // producer that has a use inside the containing op.
  SmallVector<Operation *> remainingProducers(producerOps.begin(),
                                              producerOps.end());
  auto getNextProducer = [&]() -> FailureOr<Operation *> {
    for (const auto &it : enumerate(remainingProducers)) {
      Operation *producerOp = it.value();
      bool hasUseInContainingOp =
          any_of(producerOp->getUsers(), [&](Operation *op) {
            return containingOp->isProperAncestor(op);
          });
      // TODO: When resolving the TODO below (no duplicate ops), take an op that
      // has no use among the remaining producers. This is a topological
      // sorting.
      if (hasUseInContainingOp) {
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
      Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Note);
      diag << "could not fuse ops into container";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }

    Operation *producerOp = *nextProducer;
    // TODO: If there are multiple uses of the producer in the containing op, we
    // currently tile/clone the op multiple times (once per use). In some cases,
    // we can tile/clone once and reuse the value for each use. Futhermore,
    // producers should then be traversed according to a topological sorting.
    auto tiled = tileAndFuse(producerOp, containingOp, rewriter);
    if (succeeded(tiled))
      fusedOps.append(*tiled);

    auto cloned = cloneAndFuse(producerOp, containingOp, rewriter);
    if (succeeded(cloned))
      fusedOps.append(*cloned);

    if (failed(tiled) && failed(cloned)) {
      Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Note);
      diag << "could not fuse into containing op";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
    }
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
  if (getOps().hasValue())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  if (payloadOps.size() != 1)
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one target handle"));

  SmallVector<Operation *> res;
  auto matchFun = [&](Operation *op) {
    if (getOps().hasValue() && !strs.contains(op->getName().getStringRef()))
      return WalkResult::advance();

    // Interfaces cannot be matched by name, just by ID.
    // So we specifically encode the interfaces we care about for this op.
    if (getInterface().hasValue()) {
      auto iface = getInterface().getValue();
      if (iface == transform::MatchInterfaceEnum::LinalgOp &&
          !isa<linalg::LinalgOp>(op))
        return WalkResult::advance();
      if (iface == transform::MatchInterfaceEnum::TilingInterface &&
          isa<TilingInterface>(op))
        return WalkResult::advance();
    }

    if (getAttribute().hasValue() && !op->hasAttr(getAttribute().getValue()))
      return WalkResult::advance();

    // All constraints are satisfied.
    res.push_back(op);
    return WalkResult::advance();
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
    Attribute attr = std::get<0>(it);
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
    return emitOpError()
           << "expects padding_dimensions to contain positive integers, found "
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
  // Tiling with "scalarize_dyn_dims" actually sets the same lambda as the tile
  // sizes and asserts that it is not already set.
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
    std::tie(first.emplace_back(), second.emplace_back()) =
        linalg::splitOp(rewriter, linalgOp, getDimension(), std::get<1>(pair));
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
  if (!dynamicPointParseResult.hasValue()) {
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
    return emitOpError()
           << "expects either a dynamic or a static split point to be provided";
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
      parseOperandsOrIntegersSizesList(parser, dynamicSizes, staticSizes) ||
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
  printOperandsOrIntegersSizesList(p, getOperation(), getDynamicSizes(),
                                   getStaticSizes());
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

DiagnosedSilenceableFailure transform::TileToForeachThreadOp::applyToOne(
    TilingInterface target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  auto maybeThreadDimMappingAttr = getThreadDimMapping();
  auto dimMapping =
      llvm::to_vector(maybeThreadDimMappingAttr
                          ? extractFromI64ArrayAttr(*maybeThreadDimMappingAttr)
                          : ArrayRef<int64_t>{});

  FailureOr<ForeachThreadTilingResult> tilingResult = failure();
  if (Optional<ArrayAttr> numThreads = getNumThreads())
    tilingResult = linalg::tileToForeachThreadOp(
        rewriter, target, getAsOpFoldResult(*numThreads), dimMapping);

  if (Optional<ArrayAttr> tileSizes = getTileSizes())
    tilingResult = linalg::tileToForeachThreadOpUsingTileSizes(
        rewriter, target, getAsOpFoldResult(*tileSizes), dimMapping);

  if (failed(tilingResult))
    return emitDefaultSilenceableFailure(target);
  rewriter.replaceOp(target, tilingResult->tileOp->getResults());
  results.assign({tilingResult->tileOp, tilingResult->tiledOp});
  return DiagnosedSilenceableFailure(success());
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
/// Registers new ops and declares PDL as dependent dialect since the additional
/// ops are using PDL types for operands and results.
class LinalgTransformDialectExtension
    : public transform::TransformDialectExtension<
          LinalgTransformDialectExtension> {
public:
  LinalgTransformDialectExtension() {
    declareDependentDialect<AffineDialect>();
    declareDependentDialect<arith::ArithmeticDialect>();
    declareDependentDialect<pdl::PDLDialect>();
    declareDependentDialect<scf::SCFDialect>();
    declareDependentDialect<vector::VectorDialect>();
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
