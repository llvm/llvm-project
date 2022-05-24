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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FormatVariadic.h"

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

//===----------------------------------------------------------------------===//
// InterchangeOp
//===----------------------------------------------------------------------===//

FailureOr<LinalgOp> transform::InterchangeOp::applyToOne(LinalgOp target) {
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

  GenericOpInterchangePattern pattern(getContext(), interchangeVector);
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<GenericOp> result =
      pattern.returningMatchAndRewrite(genericTarget, rewriter);
  if (failed(result))
    return failure();

  return cast<LinalgOp>(result->getOperation());
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

FailureOr<LinalgOp> transform::PadOp::applyToOne(LinalgOp target) {
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

  LinalgPaddingPattern pattern(getContext(), paddingOptions);
  SimpleRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> patternResult =
      pattern.returningMatchAndRewrite(target, rewriter);
  if (failed(patternResult)) {
    InFlightDiagnostic diag = emitError()
                              << "failed to apply pattern to target op";
    diag.attachNote(target.getLoc()) << "target op";
    return diag;
  }
  return patternResult;
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

FailureOr<LinalgOp> transform::ScalarizeOp::applyToOne(LinalgOp target) {
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
// TileOp
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

LogicalResult transform::TileOp::apply(TransformResults &transformResults,
                                       TransformState &state) {
  LinalgTilingOptions tilingOptions;
  SmallVector<int64_t> tileSizes = extractI64Array(getSizes());

  if (!tileSizes.empty())
    tilingOptions.setTileSizes(tileSizes);
  tilingOptions.setInterchange(extractUIntArray(getInterchange()));
  LinalgTilingPattern pattern(getContext(), tilingOptions);

  return applyTilingToAll(getOperation(), getTarget(), tileSizes,
                          transformResults, state, [&](LinalgOp linalgOp) {
                            SimpleRewriter rewriter(linalgOp.getContext());
                            return pattern.returningMatchAndRewrite(linalgOp,
                                                                    rewriter);
                          });
}

ParseResult transform::TileOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  StringRef sizesAttrName = TileOp::getSizesAttrName(result.name).getValue();
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

void TileOp::print(OpAsmPrinter &p) {
  p << ' ';
  p << getTarget();
  p.printOptionalAttrDict((*this)->getAttrs());
}

void TileOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // `target` arg is consumed and can no longer be used.
  effects.emplace_back(MemoryEffects::Read::get(), getTarget(),
                       TransformMappingResource::get());
  effects.emplace_back(MemoryEffects::Free::get(), getTarget(),
                       TransformMappingResource::get());

  for (Value r : getResults()) {
    effects.emplace_back(MemoryEffects::Write::get(), r,
                         TransformMappingResource::get());
    effects.emplace_back(MemoryEffects::Allocate::get(), r,
                         TransformMappingResource::get());
  }

  effects.emplace_back(MemoryEffects::Read::get(), PayloadIRResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), PayloadIRResource::get());
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
