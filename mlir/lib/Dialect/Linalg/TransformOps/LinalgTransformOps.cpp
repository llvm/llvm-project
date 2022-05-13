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
