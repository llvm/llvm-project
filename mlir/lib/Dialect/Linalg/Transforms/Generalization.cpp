//===- Generalization.cpp - linalg named ops to generic ops  --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Linalg generalization pass. It converts named
// Linalg ops to linalg.generic ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGGENERALIZENAMEDOPSPASS
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "linalg-generalization"

using namespace mlir;
using namespace mlir::linalg;

static LogicalResult generalizeNamedOpPrecondition(LinalgOp linalgOp) {
  // Bailout if `linalgOp` is already a generic.
  if (isa<GenericOp>(linalgOp))
    return failure();
  // Check if the operation has exactly one region.
  if (linalgOp->getNumRegions() != 1) {
    assert(linalgOp->getNumRegions() == 0 && "op with multiple regions");
    // TOD: Otherwise it needs to be built explicitly from the region builder.
    return failure();
  }
  return success();
}

// Converts a named matmul-like op (`matmul`, `batch_matmul`, or
// `batch_reduce_matmul`) into a `linalg.contract` category op, preserving the
// operand indexing maps and cast semantics. Returns failure for other ops.
static FailureOr<LinalgOp> generalizeToContractOp(RewriterBase &rewriter,
                                                  LinalgOp namedOp) {
  if (!isa<MatmulOp, BatchMatmulOp, BatchReduceMatmulOp>(
          namedOp.getOperation()))
    return failure();

  SmallVector<NamedAttribute> attributes;

  // Preserve operand indexing semantics (transposition, batch/reduction dims)
  // via the named op's indexing maps.
  SmallVector<Attribute> indexingMaps = llvm::map_to_vector(
      namedOp.getIndexingMapsArray(),
      [](AffineMap map) -> Attribute { return AffineMapAttr::get(map); });
  attributes.push_back(rewriter.getNamedAttr(
      "indexing_maps", rewriter.getArrayAttr(indexingMaps)));

  // Only the unsigned cast needs to be explicit; signed is the default.
  if (auto castAttr = namedOp->getAttrOfType<TypeFnAttr>("cast");
      castAttr && castAttr.getValue() == TypeFn::cast_unsigned)
    attributes.push_back(rewriter.getNamedAttr("cast", castAttr));

  LinalgOp contractOp = rewriter.replaceOpWithNewOp<ContractOp>(
      namedOp, ValueRange{namedOp.getDpsInputs()[0], namedOp.getDpsInputs()[1]},
      ValueRange{namedOp.getDpsInits()[0]}, attributes);
  return contractOp;
}

FailureOr<LinalgOp> mlir::linalg::generalizeNamedOp(RewriterBase &rewriter,
                                                    LinalgOp linalgOp,
                                                    bool emitCategoryOps) {
  if (failed(generalizeNamedOpPrecondition(linalgOp)))
    return rewriter.notifyMatchFailure(linalgOp, "preconditions not met");

  // Emit the `linalg.contract` category op for matmul-like named ops.
  if (emitCategoryOps) {
    FailureOr<LinalgOp> contractOp = generalizeToContractOp(rewriter, linalgOp);
    if (succeeded(contractOp))
      return contractOp;
    return rewriter.notifyMatchFailure(linalgOp,
                                       "failed to categorize to named op");
  }

  SmallVector<Value> inputs = linalgOp.getDpsInputs();
  ValueRange outputs = linalgOp.getDpsInits();
  SmallVector<AffineMap> indexingMaps = linalgOp.getIndexingMapsArray();
  SmallVector<utils::IteratorType> iterators = linalgOp.getIteratorTypesArray();
  SmallVector<Type> resultTypes = linalgOp.hasPureTensorSemantics()
                                      ? TypeRange(ValueRange(outputs))
                                      : TypeRange{};

  // All named ops have a region attached that can be inlined.
  assert(linalgOp->getNumRegions() == 1 &&
         "expect named op to have one region attached");
  GenericOp genericOp =
      GenericOp::create(rewriter, linalgOp.getLoc(), resultTypes, inputs,
                        outputs, indexingMaps, iterators);
  rewriter.inlineRegionBefore(linalgOp->getRegion(0), genericOp.getRegion(),
                              genericOp.getRegion().begin());

  // Discardable attributes carry user-defined metadata (e.g., annotations for
  // downstream passes). Generalization is a semantics-preserving
  // transformation, so dropping this metadata would be unexpected. This is safe
  // because discardable attributes are by definition independent of op
  // semantics.
  genericOp->setDiscardableAttrs(linalgOp->getDiscardableAttrDictionary());

  rewriter.replaceOp(linalgOp, genericOp->getResults());
  return cast<LinalgOp>(genericOp.getOperation());
}

namespace {

struct LinalgGeneralizeNamedOpsPass
    : public impl::LinalgGeneralizeNamedOpsPassBase<
          LinalgGeneralizeNamedOpsPass> {
  using impl::LinalgGeneralizeNamedOpsPassBase<
      LinalgGeneralizeNamedOpsPass>::LinalgGeneralizeNamedOpsPassBase;
  void runOnOperation() override;
};

} // namespace

void LinalgGeneralizeNamedOpsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgNamedOpsGeneralizationPatterns(patterns);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}

void mlir::linalg::populateLinalgNamedOpsGeneralizationPatterns(
    RewritePatternSet &patterns, bool emitCategoryOps) {
  patterns.add<LinalgGeneralizationPattern>(patterns.getContext(),
                                            emitCategoryOps);
}
