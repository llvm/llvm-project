//===- LinalgTransformOps.cpp - Implementation of Linalg transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::transform;

/// Extracts a vector of int64_t from an array attribute. Asserts if the
/// attribute contains values other than integers.
static SmallVector<int64_t> extractI64Array(ArrayAttr attr) {
  SmallVector<int64_t> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getSExtValue());
  return result;
}

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

FailureOr<LinalgOp> transform::DecomposeOp::applyToOne(LinalgOp target,
                                                       TransformState &state) {
  FailureOr<LinalgOp> windowed =
      tryApply<DownscaleSizeOneWindowed2DConvolution>(target);
  if (succeeded(windowed))
    return windowed;

  FailureOr<LinalgOp> depthwise =
      tryApply<DownscaleDepthwiseConv2DNhwcHwcOp>(target);
  if (succeeded(depthwise))
    return depthwise;

  return reportUnknownTransformError(target);
}

//===----------------------------------------------------------------------===//
// FuseOp
//===----------------------------------------------------------------------===//

/// Apply a tiling transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
static LogicalResult
applyTilingToAll(Operation *transformOp, Value target,
                 ArrayRef<int64_t> tileSizes,
                 transform::TransformResults &transformResults,
                 transform::TransformState &state,
                 function_ref<FailureOr<TiledLinalgOp>(LinalgOp)> applyFn) {
  // Number of loops: Number of tiles sizes that are not zero.
  size_t numLoops = tileSizes.size() - llvm::count(tileSizes, 0);
  // All payload ops. These should all be LinalgOps for now.
  ArrayRef<Operation *> payloadOps = state.getPayloadOps(target);

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
      sizesArrayAttr.size() - llvm::count(extractI64Array(sizesArrayAttr), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOpType));
  if (parser.resolveOperand(targetOperand, pdlOpType, result.operands))
    return failure();
  return success();
}

DiagnosedSilenceableFailure
transform::FuseOp::apply(mlir::transform::TransformResults &transformResults,
                         mlir::transform::TransformState &state) {
  LinalgTilingAndFusionOptions fusionOptions;
  fusionOptions.tileSizes = extractI64Array(getTileSizes());
  fusionOptions.tileInterchange = extractI64Array(getTileInterchange());

  LogicalResult result = applyTilingToAll(
      getOperation(), getTarget(), fusionOptions.tileSizes, transformResults,
      state, [&](LinalgOp linalgOp) -> FailureOr<TiledLinalgOp> {
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
  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
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
  SmallVector<int64_t> permutation = extractI64Array(getTileInterchange());
  auto sequence = llvm::to_vector(llvm::seq<int64_t>(0, permutation.size()));
  if (!std::is_permutation(sequence.begin(), sequence.end(),
                           permutation.begin(), permutation.end())) {
    return emitOpError() << "expects interchange to be a permutation, found "
                         << getTileInterchange();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// GeneralizeOp
//===----------------------------------------------------------------------===//

FailureOr<LinalgOp> transform::GeneralizeOp::applyToOne(LinalgOp target,
                                                        TransformState &state) {
  // Exit early if no transformation is needed.
  if (isa<GenericOp>(target))
    return target;

  FailureOr<LinalgOp> generic = tryApply<LinalgGeneralizationPattern>(target);
  if (succeeded(generic))
    return generic;

  return reportUnknownTransformError(target);
}

//===----------------------------------------------------------------------===//
// InterchangeOp
//===----------------------------------------------------------------------===//

FailureOr<LinalgOp>
transform::InterchangeOp::applyToOne(LinalgOp target, TransformState &state) {
  SmallVector<unsigned> interchangeVector =
      extractUIntArray(getIteratorInterchange());
  // Exit early if no transformation is needed.
  if (interchangeVector.empty())
    return target;

  auto genericTarget = dyn_cast<GenericOp>(target.getOperation());
  if (!genericTarget) {
    InFlightDiagnostic diag = emitOpError()
                              << "applies to " << GenericOp::getOperationName()
                              << " ops";
    diag.attachNote(target.getLoc()) << "attempted to apply to this op";
    return diag;
  }

  return tryApply<GenericOpInterchangePattern>(target, interchangeVector);
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
// PadOp
//===---------------------------------------------------------------------===//

FailureOr<LinalgOp> transform::PadOp::applyToOne(LinalgOp target,
                                                 TransformState &state) {
  // Convert the integer packing flags to booleans.
  SmallVector<bool> packPaddings;
  for (int64_t packPadding : extractI64Array(getPackPaddings()))
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
        InFlightDiagnostic diag = emitOpError()
                                  << "expects a padding value that parses to "
                                  << elementType << ", got " << std::get<0>(it);
        diag.attachNote(target.getLoc()) << "when applied to this op";
        return diag;
      }
      continue;
    }
    // Otherwise, add the attribute directly.
    if (attr.getType() != elementType) {
      InFlightDiagnostic diag = emitOpError()
                                << "expects a padding value of type "
                                << elementType << ", got " << attr;
      diag.attachNote(target.getLoc()) << "when applied to this op";
      return diag;
    }
    paddingValues.push_back(attr);
  }

  // Extract the transpose vectors.
  SmallVector<SmallVector<int64_t>> transposePaddings;
  for (Attribute transposeVector : getTransposePaddings().cast<ArrayAttr>())
    transposePaddings.push_back(
        extractI64Array(transposeVector.cast<ArrayAttr>()));

  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValues);
  paddingOptions.setPaddingDimensions(extractI64Array(getPaddingDimensions()));
  paddingOptions.setPackPaddings(packPaddings);
  paddingOptions.setHoistPaddings(extractI64Array(getHoistPaddings()));
  paddingOptions.setTransposePaddings(transposePaddings);

  FailureOr<LinalgOp> result =
      tryApply<LinalgPaddingPattern>(target, paddingOptions);
  if (succeeded(result))
    return result;

  InFlightDiagnostic diag = emitError()
                            << "failed to apply pattern to target op";
  diag.attachNote(target.getLoc()) << "target op";
  return diag;
}

LogicalResult transform::PadOp::verify() {
  SmallVector<int64_t> packPaddings = extractI64Array(getPackPaddings());
  if (any_of(packPaddings, [](int64_t packPadding) {
        return packPadding != 0 && packPadding != 1;
      })) {
    return emitOpError()
           << "expects pack_paddings to contain booleans (0/1), found "
           << getPackPaddings();
  }

  SmallVector<int64_t> paddingDimensions =
      extractI64Array(getPaddingDimensions());
  if (any_of(paddingDimensions,
             [](int64_t paddingDimension) { return paddingDimension < 0; })) {
    return emitOpError()
           << "expects padding_dimensions to contain positive integers, found "
           << getPaddingDimensions();
  }

  SmallVector<int64_t> hoistPaddings = extractI64Array(getHoistPaddings());
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
// ScalarizeOp
//===----------------------------------------------------------------------===//

FailureOr<LinalgOp> transform::ScalarizeOp::applyToOne(LinalgOp target,
                                                       TransformState &state) {
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
    return failure();

  return result->op;
}

//===----------------------------------------------------------------------===//
// SplitReductionOp
//===----------------------------------------------------------------------===//

FailureOr<SmallVector<Operation *>>
transform::SplitReductionOp::applyToOne(LinalgOp target,
                                        TransformState &state) {
  ControlSplitReductionFn splitFn = [&](LinalgOp) {
    return std::pair<int64_t, unsigned>(getSplitFactor(),
                                        getInsertSplitDimension());
  };
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<SplitReductionResult> splitResult =
      splitReduction(rewriter, target, splitFn);
  if (failed(splitResult))
    return getOperation()->emitError("failed to apply");
  return SmallVector<Operation *>{splitResult->fillOp,
                                  splitResult->splitLinalgOp,
                                  splitResult->resultCombiningLinalgOp};
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::TileOp::apply(TransformResults &transformResults,
                         TransformState &state) {
  LinalgTilingOptions tilingOptions;
  SmallVector<int64_t> tileSizes = extractI64Array(getSizes());

  if (!tileSizes.empty())
    tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setInterchange(extractUIntArray(getInterchange()));
  LinalgTilingPattern pattern(getContext(), tilingOptions);

  LogicalResult result = applyTilingToAll(
      getOperation(), getTarget(), tileSizes, transformResults, state,
      [&](LinalgOp linalgOp) {
        SimpleRewriter rewriter(linalgOp.getContext());
        return pattern.returningMatchAndRewrite(linalgOp, rewriter);
      });
  return DiagnosedSilenceableFailure(result);
}

ParseResult transform::TileOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  return parseTileLikeOp(parser, result,
                         TileOp::getSizesAttrName(result.name).getValue());
}

void TileOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

FailureOr<Operation *> VectorizeOp::applyToOne(Operation *target,
                                               TransformState &state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    InFlightDiagnostic diag = emitOpError()
                              << "applies only to isolated-from-above targets";
    diag.attachNote(target->getLoc()) << "non-isolated target";
    return diag;
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
    return reportUnknownTransformError(target);
  return target;
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

#define GET_OP_CLASSES
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.cpp.inc"

void mlir::linalg::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<LinalgTransformDialectExtension>();
}
