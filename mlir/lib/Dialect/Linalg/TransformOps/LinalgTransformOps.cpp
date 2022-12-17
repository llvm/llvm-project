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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::transform;

#define DEBUG_TYPE "linalg-transforms"

/// Extracts a vector of unsigned from an array attribute. Asserts if the
/// attribute contains values other than intergers. May truncate.
static SmallVector<unsigned> extractUIntArray(ArrayAttr attr) {
  SmallVector<unsigned> result;
  result.reserve(attr.size());
  for (APInt value : attr.getAsValueRange<IntegerAttr>())
    result.push_back(value.getZExtValue());
  return result;
}

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
  TrivialPatternRewriter rewriter(operation->getContext());
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
static LogicalResult applyTilingToAll(
    Operation *transformOp, ArrayRef<Operation *> payloadOps, unsigned numLoops,
    transform::TransformResults &transformResults,
    function_ref<FailureOr<scf::SCFTileAndFuseResult>(TilingInterface)>
        applyFn) {
  SmallVector<Operation *> tiledLinalgOps;
  SmallVector<SmallVector<Operation *>> loopOps(numLoops);
  for (unsigned int i = 0; i < numLoops; ++i)
    loopOps[i].reserve(payloadOps.size());

  for (Operation *target : payloadOps) {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
    if (!tilingInterfaceOp)
      return transformOp->emitError("only TilingInterface ops are supported");

    TrivialPatternRewriter rewriter(target->getContext());
    rewriter.setInsertionPoint(target);
    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        applyFn(tilingInterfaceOp);
    if (failed(tiledResults))
      return failure();

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{target};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      SmallVector<Value> replacements;
      replacements.reserve(toReplace->getNumResults());
      for (OpResult res : toReplace->getResults()) {
        auto it = tiledResults->replacements.find(res);
        if (it == tiledResults->replacements.end())
          replacements.push_back(res);
        else
          replacements.push_back(it->getSecond());
      }
      rewriter.replaceOp(toReplace, replacements);
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
  SmallVector<int64_t> tileSizes = extractFromI64ArrayAttr(getTileSizes());
  SmallVector<int64_t> tileInterchange =
      extractFromI64ArrayAttr(getTileInterchange());

  scf::SCFTilingOptions tilingOptions;
  tilingOptions.interchangeVector = tileInterchange;
  tilingOptions = tilingOptions.setTileSizes(tileSizes);
  scf::SCFTileAndFuseOptions tileAndFuseOptions;
  tileAndFuseOptions.tilingOptions = tilingOptions;
  LogicalResult result = applyTilingToAll(
      getOperation(), state.getPayloadOps(getTarget()),
      tileSizes.size() - llvm::count(tileSizes, 0), transformResults,
      [&](TilingInterface tilingInterfaceOp)
          -> FailureOr<scf::SCFTileAndFuseResult> {
        TrivialPatternRewriter rewriter(getContext());
        return tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
            rewriter, tilingInterfaceOp, tileAndFuseOptions);
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

void transform::FuseIntoContainingOp::build(OpBuilder &builder,
                                            OperationState &result,
                                            Value producerOp,
                                            Value containingOp) {
  result.addOperands({producerOp, containingOp});
  result.addTypes(pdl::OperationType::get(builder.getContext()));
}

/// Find the first "extract" user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation *tileAndFuseFirstExtractUse(RewriterBase &rewriter,
                                             Diagnostic &diag,
                                             Operation *producerOp,
                                             Operation *containingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return nullptr;
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
    return nullptr;
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  int64_t resultNumber =
      sliceOpToTile.getSource().cast<OpResult>().getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  FailureOr<Value> tiledProducer = tileableProducer.generateResultTileValue(
      rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
      sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "tiledProducer: " << *tiledProducer << "\n");

  // Replace the extract op.
  Operation *fusedOp = tiledProducer->getDefiningOp();
  rewriter.replaceOp(sliceOpToTile, fusedOp->getResult(resultNumber));
  return fusedOp;
}

/// First, find the first "scf::ForeachThreadOp" user of `producerOp` and ensure
/// it is exactly the `containingOp`, otherwise bail.
/// Then, find the first "extract" user of the tied block argument and tile it
/// right before its "extract" use. The tiled op is fused under the
/// `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation *tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
    RewriterBase &rewriter, Diagnostic &diag, Operation *producerOp,
    Operation *containingOp) {
  LLVM_DEBUG(
      llvm::dbgs() << "Try to fuse an extract use through block argument\n");

  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return nullptr;
  }

  // Search the first use by a "scf::ForeachThreadOp" user.
  scf::ForeachThreadOp foreachThreadOp;
  auto itProducerUses =
      llvm::find_if(tileableProducer->getUses(), [&](OpOperand &use) {
        foreachThreadOp = dyn_cast<scf::ForeachThreadOp>(use.getOwner());
        return foreachThreadOp;
      });
  // If it's not from the containing op, return.
  if (!foreachThreadOp || foreachThreadOp != containingOp) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find a use by the containing op: " << *tileableProducer;
    return nullptr;
  }

  // Search the producer slices accessed within the containing
  // operation.
  // TODO: Generalize to more extract/insert/parallel_insert triples.
  //   Maybe evolve into an interface.
  OpOperand *pUse = &(*itProducerUses);
  BlockArgument bbArg = foreachThreadOp.getTiedBlockArgument(pUse);

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
    return nullptr;
  }
  auto sliceOpToTile = cast<tensor::ExtractSliceOp>(*itBBArgUsers);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Replace the use in the tileableProducer before tiling: clone, replace and
  // then tile.
  int64_t resultNumber = pUse->get().cast<OpResult>().getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  // Gather destination tensors.
  SmallVector<Value> destinationTensors;
  if (failed(tensor::getOrCreateDestinations(
          rewriter, tileableProducer->getLoc(), tileableProducer,
          destinationTensors))) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to get destination tensors for: " << *tileableProducer;
    return nullptr;
  }

  BlockAndValueMapping bvm;
  bvm.map(destinationTensors[resultNumber], bbArg);
  auto tileableProducerClone =
      cast<TilingInterface>(rewriter.clone(*tileableProducer, bvm));
  auto scopeGuard =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(tileableProducerClone); });

  // Tile the producer.
  FailureOr<Value> tiledProducer =
      tileableProducerClone.generateResultTileValue(
          rewriter, resultNumber, sliceOpToTile.getMixedOffsets(),
          sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "tiledProducer: " << *tiledProducer << "\n");

  // Replace the extract op.
  Operation *fusedOp = tiledProducer->getDefiningOp();
  rewriter.replaceOp(sliceOpToTile, fusedOp->getResult(resultNumber));

  // Replace the use in containingOp.
  rewriter.updateRootInPlace(containingOp, [&]() {
    containingOp->setOperand(pUse->getOperandNumber(),
                             destinationTensors.front());
  });

  return fusedOp;
}

static Operation *cloneAndFuseFirstUse(RewriterBase &rewriter, Diagnostic &diag,
                                       Operation *producerOp,
                                       Operation *containingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Try to fuse an use by cloning\n");

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
  unsigned resultNumber = use->get().cast<OpResult>().getResultNumber();
  LLVM_DEBUG(llvm::dbgs() << "resultNumber: " << resultNumber << "\n");

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use->getOwner());
  fusedOp = rewriter.clone(*producerOp);
  rewriter.updateRootInPlace(
      use->getOwner(), [&] { use->set(fusedOp->getOpResult(resultNumber)); });

  return fusedOp;
}

DiagnosedSilenceableFailure
transform::FuseIntoContainingOp::apply(transform::TransformResults &results,
                                       transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  ArrayRef<Operation *> producerOps = state.getPayloadOps(getProducerOp());
  // If nothing to fuse, propagate success.
  if (producerOps.empty()) {
    results.set(getFusedOp().cast<OpResult>(),
                SmallVector<mlir::Operation *>{});
    return DiagnosedSilenceableFailure::success();
  }
  ArrayRef<Operation *> containingOps = state.getPayloadOps(getContainingOp());
  if (containingOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << containingOps.size() << ")";
  }
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
      results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation *>());
      Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
      diag << "could not find next producer to fuse into container";
      return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
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
    Operation *tiled =
        tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp);
    if (tiled) {
      LLVM_DEBUG(llvm::dbgs() << "\nFused a direct extract use\n"
                              << *containingOp);
      fusedOps.push_back(tiled);
      continue;
    }

    Operation *tiledContainingOpOperand =
        tileAndFuseFirstExtractUseThroughContainingOpBlockArgument(
            rewriter, diag, producerOp, containingOp);
    if (tiledContainingOpOperand) {
      LLVM_DEBUG(llvm::dbgs()
                 << "\nFused an extract use through block argument\n"
                 << *containingOp);
      fusedOps.push_back(tiledContainingOpOperand);
      continue;
    }

    Operation *cloned =
        cloneAndFuseFirstUse(rewriter, diag, producerOp, containingOp);
    if (cloned) {
      LLVM_DEBUG(llvm::dbgs() << "\nFused an use by cloning\n"
                              << *containingOp);
      fusedOps.push_back(cloned);
      continue;
    }
    results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation *>());
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
  TrivialPatternRewriter rewriter(target->getContext());
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

void transform::MatchOp::build(OpBuilder &builder, OperationState &result,
                               Value target, ArrayRef<StringRef> opNames) {
  result.addOperands(target);
  result.addAttribute(MatchOp::getOpsAttrName(result.name),
                      builder.getStrArrayAttr(opNames));
  result.addTypes(pdl::OperationType::get(builder.getContext()));
}

DiagnosedSilenceableFailure
transform::MatchOp::apply(transform::TransformResults &results,
                          transform::TransformState &state) {
  llvm::StringSet<> strs;
  if (getOps().has_value())
    strs.insert(getOps()->getAsValueRange<StringAttr>().begin(),
                getOps()->getAsValueRange<StringAttr>().end());

  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  if (payloadOps.size() != 1) {
    results.set(getResult().cast<OpResult>(), {});
    return DiagnosedSilenceableFailure(
        this->emitOpError("requires exactly one target handle"));
  }

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

  TrivialPatternRewriter rewriter(target->getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<LinalgOp> res = promoteSubViews(rewriter, target, promotionOptions);
  if (failed(res))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(target);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// ReplaceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReplaceOp::apply(TransformResults &transformResults,
                            TransformState &state) {
  ArrayRef<Operation *> payload = state.getPayloadOps(getTarget());

  // Check for invalid targets.
  for (Operation *target : payload) {
    if (target->getNumOperands() > 0)
      return emitDefiniteFailure() << "expected target without operands";
    if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
        target->getNumRegions() > 0)
      return emitDefiniteFailure()
             << "expected target that is isloated from above";
  }

  // Clone and replace.
  IRRewriter rewriter(getContext());
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
  transformResults.set(getReplacement().cast<OpResult>(), replacements);
  return DiagnosedSilenceableFailure(success());
}

void transform::ReplaceOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  producesHandle(getReplacement(), effects);
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
transform::ScalarizeOp::applyToOne(linalg::LinalgOp target,
                                   SmallVectorImpl<Operation *> &results,
                                   transform::TransformState &state) {
  scf::SCFTilingOptions tilingOptions;
  tilingOptions.setTileSizeComputationFunction([&](OpBuilder &b, Operation *) {
    SmallVector<Value, 4> tileSizes;
    Location loc = target.getLoc();
    SmallVector<OpFoldResult> allShapeSizes =
        target.createFlatListOfOperandDims(b, loc);
    AffineMap map = target.getShapesToLoopsMap();
    if (!map)
      return tileSizes;
    IRRewriter rewriter(b);
    SmallVector<OpFoldResult> shapeSizes =
        makeComposedFoldedMultiResultAffineApply(rewriter, loc, map,
                                                 allShapeSizes);
    // If the shape size is dynamic, tile by 1.
    // Otherwise, do not tile (i.e. tile size 0).
    for (OpFoldResult shapeSize : shapeSizes) {
      tileSizes.push_back(getConstantIntValue(shapeSize)
                              ? b.create<arith::ConstantIndexOp>(loc, 0)
                              : b.create<arith::ConstantIndexOp>(loc, 1));
    }
    return tileSizes;
  });
  SmallVector<int64_t> emptyTileSizes;
  TrivialPatternRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  FailureOr<scf::SCFTilingResult> maybeTilingResult = tileUsingSCFForOp(
      rewriter, cast<TilingInterface>(target.getOperation()), tilingOptions);
  if (failed(maybeTilingResult))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));

  results.append(maybeTilingResult->tiledOps);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// SplitOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure SplitOp::apply(TransformResults &results,
                                           TransformState &state) {
  // Collect the dynamic split points if provided.
  ArrayRef<Operation *> payload = state.getPayloadOps(getTarget());
  TrivialPatternRewriter rewriter(getContext());
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
    if (diag.isSilenceableFailure()) {
      results.set(getFirst().cast<OpResult>(), {});
      results.set(getSecond().cast<OpResult>(), {});
      return diag;
    }

    if (splitPoints.size() != payload.size()) {
      return emitDefiniteFailure()
             << "expected the dynamic split point handle to point to as "
                "many operations ("
             << splitPoints.size() << ") as the target handle ("
             << payload.size() << ")";
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
      results.set(getFirst().cast<OpResult>(), {});
      results.set(getSecond().cast<OpResult>(), {});
      return diag;
    }

    if (getDimension() >= linalgOp.getNumLoops()) {
      auto diag = emitSilenceableError() << "dimension " << getDimension()
                                         << " does not exist in target op";
      diag.attachNote(target->getLoc()) << "target op";
      results.set(getFirst().cast<OpResult>(), {});
      results.set(getSecond().cast<OpResult>(), {});
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
        parser.getBuilder().getI64IntegerAttr(ShapedType::kDynamic);
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
  if (staticSplitSize != ShapedType::kDynamic)
    printer << staticSplitSize;
  else
    printer << getDynamicSplitPoint();
  printer << " ";
  printer.printOptionalAttrDict(getOperation()->getAttrs(),
                                {getStaticSplitPointAttrName()});
}

LogicalResult SplitOp::verify() {
  if ((static_cast<int64_t>(getStaticSplitPoint()) != ShapedType::kDynamic) ^
      (getDynamicSplitPoint() == nullptr)) {
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
  auto resultType = pdl::OperationType::get(ctx);
  result.addTypes({resultType, resultType, resultType, resultType});
}

DiagnosedSilenceableFailure
transform::SplitReductionOp::applyToOne(linalg::LinalgOp target,
                                        SmallVectorImpl<Operation *> &results,
                                        transform::TransformState &state) {
  ControlSplitReductionFn splitFn = [&](LinalgOp) {
    return linalg::SplitReductionOptions{int64_t(getSplitFactor()),
                                         unsigned(getInsertSplitDimension()),
                                         bool(getInnerParallel())};
  };
  TrivialPatternRewriter rewriter(getContext());
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
// SplitReductionOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::TileReductionUsingScfOp::applyToOne(
    linalg::LinalgOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  TrivialPatternRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  SmallVector<int64_t> tileSizes = extractFromI64ArrayAttr(getTileSizes());
  SmallVector<OpFoldResult> sizes;
  for (int64_t size : tileSizes) {
    sizes.push_back(rewriter.getIndexAttr(size));
  }

  FailureOr<scf::SCFReductionTilingResult> result = scf::tileReductionUsingScf(
      rewriter, cast<PartialReductionOpInterface>(target.getOperation()),
      sizes);

  if (failed(result))
    return DiagnosedSilenceableFailure(reportUnknownTransformError(target));
  results.push_back(result->initialOp);
  results.push_back(result->parallelTiledOp);
  results.push_back(result->mergeOp);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// TileReductionUsingForeachThreadOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::TileReductionUsingForeachThreadOp::applyToOne(
    linalg::LinalgOp target, SmallVectorImpl<Operation *> &results,
    transform::TransformState &state) {
  TrivialPatternRewriter rewriter(getContext());
  rewriter.setInsertionPoint(target);
  SmallVector<int64_t> numThreads = extractFromI64ArrayAttr(getNumThreads());
  SmallVector<OpFoldResult> numThreadResults;
  for (int64_t num : numThreads) {
    numThreadResults.push_back(rewriter.getIndexAttr(num));
  }

  FailureOr<linalg::ForeachThreadReductionTilingResult> result =
      linalg::tileReductionUsingForeachThread(
          rewriter, cast<PartialReductionOpInterface>(target.getOperation()),
          numThreadResults, /*mapping=*/std::nullopt);

  if (failed(result)) {
    results.assign(3, nullptr);
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Remark);
    diag << "could not tile reduction in target.";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }
  results.push_back(result->initialOp);
  results.push_back(result->parallelTiledOp);
  results.push_back(result->mergeOp);
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::TileOp::apply(TransformResults &transformResults,
                         TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();

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

    scf::SCFTilingOptions tilingOptions;
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

    tilingOptions.setInterchange(getInterchange());
    TrivialPatternRewriter rewriter(linalgOp.getContext());
    FailureOr<scf::SCFTilingResult> maybeTilingResult = tileUsingSCFForOp(
        rewriter, cast<TilingInterface>(linalgOp.getOperation()),
        tilingOptions);
    if (failed(maybeTilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    if (linalgOp.hasBufferSemantics())
      rewriter.eraseOp(linalgOp);
    else
      rewriter.replaceOp(linalgOp,
                         maybeTilingResult->loops.front()->getResults());

    tiled.append(maybeTilingResult->tiledOps);
    for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(getLoops()[en.index()].cast<OpResult>(), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::TileOp::getMixedSizes() {
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

// We want to parse `DenseI64ArrayAttr` using the short form without the
// `array` prefix to be consistent in the IR with `parseDynamicIndexList`.
ParseResult parseOptionalInterchange(OpAsmParser &parser,
                                     OperationState &result) {
  if (succeeded(parser.parseOptionalLBrace())) {
    if (failed(parser.parseKeyword("interchange")))
      return parser.emitError(parser.getNameLoc()) << "expect `interchange`";
    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getNameLoc()) << "expect `=`";
    result.addAttribute("interchange",
                        DenseI64ArrayAttr::parse(parser, Type{}));
    if (failed(parser.parseRBrace()))
      return parser.emitError(parser.getNameLoc()) << "expect `}`";
  }
  return success();
}

void printOptionalInterchange(OpAsmPrinter &p,
                              ArrayRef<int64_t> interchangeVals) {
  if (!interchangeVals.empty()) {
    p << " {interchange = [";
    llvm::interleaveComma(interchangeVals, p,
                          [&](int64_t integer) { p << integer; });
    p << "]}";
  }
}

ParseResult transform::TileOp::parse(OpAsmParser &parser,
                                     OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  DenseI64ArrayAttr staticSizes;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands))
    return ParseResult::failure();

  // Parse optional interchange.
  if (failed(parseOptionalInterchange(parser, result)))
    return ParseResult::failure();
  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  size_t numExpectedLoops =
      staticSizes.size() - llvm::count(staticSizes.asArrayRef(), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOperationType));
  return success();
}

void TileOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());
  printOptionalInterchange(p, getInterchange());
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

void transform::TileToForeachThreadOp::build(OpBuilder &builder,
                                             OperationState &result,
                                             Value target,
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

void transform::TileToForeachThreadOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedTileSizes, transform::TileSizesSpec,
    ArrayAttr mapping) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes,
                             ShapedType::kDynamic);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*num_threads=*/ValueRange{},
        /*tile_sizes=*/dynamicTileSizes,
        /*static_num_threads=*/builder.getDenseI64ArrayAttr({}),
        /*static_tile_sizes=*/staticTileSizesAttr,
        /*mapping=*/mapping);
}

void transform::TileToForeachThreadOp::build(OpBuilder &builder,
                                             OperationState &result,
                                             Value target,
                                             ArrayRef<int64_t> staticNumThreads,
                                             transform::NumThreadsSpec,
                                             ArrayAttr mapping) {
  return build(builder, result, target,
               getAsOpFoldResult(builder.getI64ArrayAttr(staticNumThreads)),
               NumThreadsSpec(), mapping);
}

void transform::TileToForeachThreadOp::build(
    OpBuilder &builder, OperationState &result, Value target,
    ArrayRef<OpFoldResult> mixedNumThreads, transform::NumThreadsSpec,
    ArrayAttr mapping) {
  SmallVector<int64_t> staticNumThreads;
  SmallVector<Value> dynamicNumThreads;
  dispatchIndexOpFoldResults(mixedNumThreads, dynamicNumThreads,
                             staticNumThreads, ShapedType::kDynamic);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = pdl::OperationType::get(ctx);
  auto staticNumThreadsAttr = builder.getDenseI64ArrayAttr(staticNumThreads);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*num_threads=*/dynamicNumThreads,
        /*tile_sizes=*/ValueRange{},
        /*static_num_threads=*/staticNumThreadsAttr,
        /*static_tile_sizes=*/builder.getDenseI64ArrayAttr({}),
        /*mapping=*/mapping);
}

// Given a list of OpFoldResults that are either index attrs or op
// handles, return a list of OpFoldResults where all op handles are
// replaced with the first (and only) OpResult of that payload op. (There
// must be exactly one mapped payload op and it must have exactly one
// index result.)
static DiagnosedSilenceableFailure unpackPDLOperations(
    transform::TransformState &state, TransformOpInterface transformOp,
    SmallVector<OpFoldResult> &result, ArrayRef<OpFoldResult> ofrs) {
  for (OpFoldResult ofr : ofrs) {
    // Don't try to unpack non-PDL operation.
    if (ofr.is<Attribute>() ||
        !ofr.get<Value>().getType().isa<pdl::OperationType>()) {
      result.push_back(ofr);
      continue;
    }
    ArrayRef<Operation *> payloadOps = state.getPayloadOps(ofr.get<Value>());
    for (Operation *op : payloadOps) {
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
  }

  return DiagnosedSilenceableFailure(success());
}

DiagnosedSilenceableFailure transform::tileToForeachThreadOpImpl(
    RewriterBase &rewriter, transform::TransformState &state,
    TransformOpInterface transformOp, ArrayRef<Operation *> targets,
    ArrayRef<OpFoldResult> mixedNumThreads,
    ArrayRef<OpFoldResult> mixedTileSizes, Optional<ArrayAttr> mapping,
    SmallVector<Operation *> &tileOps, SmallVector<Operation *> &tiledOps) {
  if (targets.empty())
    return DiagnosedSilenceableFailure(success());

  // getMixedNumThreads are OpFoldResults[index attributes or PDL operation].
  // Convert to OpFoldResults[index attributes or payload op].
  SmallVector<OpFoldResult> numThreads;
  DiagnosedSilenceableFailure status =
      unpackPDLOperations(state, transformOp, numThreads, mixedNumThreads);
  if (!status.succeeded())
    return status;

  // getMixedTileSizes are OpFoldResults[index attributes or PDL operation].
  // Convert to OpFoldResults[index attributes or payload op].
  SmallVector<OpFoldResult> tileSizes;
  status = unpackPDLOperations(state, transformOp, tileSizes, mixedTileSizes);
  if (!status.succeeded())
    return status;

  // Transform all targets one by one.
  for (Operation *target : targets) {
    auto tilableOp = dyn_cast<TilingInterface>(target);
    if (!tilableOp) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "only TilingInterface ops are supported";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    rewriter.setInsertionPoint(tilableOp);
    FailureOr<linalg::ForeachThreadTilingResult> tilingResult = failure();
    if (!mixedNumThreads.empty()) {
      tilingResult = linalg::tileToForeachThreadOp(rewriter, tilableOp,
                                                   numThreads, mapping);
    } else {
      tilingResult = linalg::tileToForeachThreadOpUsingTileSizes(
          rewriter, tilableOp, tileSizes, mapping);
    }

    if (failed(tilingResult))
      return transformOp.emitDefaultSilenceableFailure(tilableOp);
    rewriter.replaceOp(tilableOp, tilingResult->tileOp->getResults());

    tileOps.push_back(tilingResult->tileOp);
    tiledOps.push_back(tilingResult->tiledOp);
  }
  return DiagnosedSilenceableFailure(success());
}

DiagnosedSilenceableFailure transform::TileToForeachThreadOp::apply(
    transform::TransformResults &transformResults,
    transform::TransformState &state) {
  IRRewriter rewriter(getContext());
  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  DiagnosedSilenceableFailure diag = tileToForeachThreadOpImpl(
      rewriter, state, cast<TransformOpInterface>(getOperation()), targets,
      getMixedNumThreads(), getMixedTileSizes(), getMapping(), tileOps,
      tiledOps);

  if (!diag.succeeded()) {
    transformResults.set(getForeachThreadOp().cast<OpResult>(), {});
    transformResults.set(getTiledOp().cast<OpResult>(), {});
    return diag;
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
  Builder b(getContext());
  return getMixedValues(getStaticNumThreads(), getNumThreads(), b);
}

SmallVector<OpFoldResult> TileToForeachThreadOp::getMixedTileSizes() {
  Builder b(getContext());
  return getMixedValues(getStaticTileSizes(), getTileSizes(), b);
}

LogicalResult TileToForeachThreadOp::verify() {
  if (getMixedNumThreads().empty() == getMixedTileSizes().empty())
    return emitOpError("either num_threads or tile_sizes must be specified");
  return success();
}

//===----------------------------------------------------------------------===//
// TileToScfForOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::TileToScfForOp::apply(TransformResults &transformResults,
                                 TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();

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
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(en.value());
    if (!tilingInterfaceOp) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "only TilingInterface ops are supported";
      diag.attachNote(en.value()->getLoc()) << "target op";
      return diag;
    }

    scf::SCFTilingOptions tilingOptions;
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

    tilingOptions.setInterchange(getInterchange());
    TrivialPatternRewriter rewriter(tilingInterfaceOp.getContext());
    FailureOr<scf::SCFTilingResult> tilingResult =
        tileUsingSCFForOp(rewriter, tilingInterfaceOp, tilingOptions);
    if (failed(tilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    rewriter.replaceOp(tilingInterfaceOp, tilingResult->replacements);

    tiled.append(tilingResult->tiledOps);
    for (const auto &en2 : llvm::enumerate(tilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(getLoops()[en.index()].cast<OpResult>(), en.value());

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::TileToScfForOp::getMixedSizes() {
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

ParseResult transform::TileToScfForOp::parse(OpAsmParser &parser,
                                             OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  DenseI64ArrayAttr staticSizes;
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands))
    return ParseResult::failure();

  // Parse optional interchange.
  if (failed(parseOptionalInterchange(parser, result)))
    return ParseResult::failure();
  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  size_t numExpectedLoops =
      staticSizes.size() - llvm::count(staticSizes.asArrayRef(), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOperationType));
  return success();
}

void TileToScfForOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());
  printOptionalInterchange(p, getInterchange());
}

void transform::TileToScfForOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  onlyReadsHandle(getDynamicSizes(), effects);
  producesHandle(getTiledLinalgOp(), effects);
  producesHandle(getLoops(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// VectorizeOp
//===----------------------------------------------------------------------===//

void transform::VectorizeOp::build(OpBuilder &builder, OperationState &result,
                                   Value target, bool vectorizePadding) {
  result.addOperands(target);
  if (vectorizePadding) {
    result.addAttribute(VectorizeOp::getVectorizePaddingAttrName(result.name),
                        builder.getUnitAttr());
  }
  result.addTypes(pdl::OperationType::get(builder.getContext()));
}

namespace {
/// This is an helper only to call vectorize via a pattern inside of
/// VectorizeOp::applyToOne.
struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
    if (!linalgOp)
      return rewriter.notifyMatchFailure(op, "expected Linalg Op");
    return vectorize(rewriter, linalgOp);
  }
};
} // namespace

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
  patterns.add<VectorizationPattern>(ctx);

  if (!getDisableTransferPermutationMapLoweringPatterns())
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);

  if (!getDisableMultiReductionToContractPatterns())
    vector::populateVectorReductionToContractPatterns(patterns);

  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                       /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);

  patterns.add<CopyVectorizationPattern>(ctx);

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
    declareDependentDialect<LinalgDialect>();
    declareGeneratedDialect<AffineDialect>();
    declareGeneratedDialect<arith::ArithDialect>();
    declareGeneratedDialect<scf::SCFDialect>();
    declareGeneratedDialect<vector::VectorDialect>();
    declareGeneratedDialect<gpu::GPUDialect>();

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
