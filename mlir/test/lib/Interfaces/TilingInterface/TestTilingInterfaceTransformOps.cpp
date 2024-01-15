//===- TestTilingInterfaceTransformOps.cpp - Test `TilingInterface` ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines transform dialect operations used for testing
// TilingInterface
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformAttrs.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/TilingInterface.h"

#define GET_OP_CLASSES
#include "TestTilingInterfaceTransformOps.h.inc"

using namespace mlir;
using namespace mlir::transform;

//===----------------------------------------------------------------------===//
// TestFuseAndYieldOp
//===----------------------------------------------------------------------===//

static llvm::SmallDenseSet<Operation *> collectTiledAndFusedOps(Operation *op) {
  SmallVector<Operation *> worklist;
  llvm::SmallDenseSet<Operation *> producers;
  worklist.push_back(op);
  producers.insert(op);
  while (!worklist.empty()) {
    Operation *current = worklist.pop_back_val();
    for (OpOperand &operand : current->getOpOperands()) {
      Operation *producer = operand.get().getDefiningOp();
      if (!producer || !isa<TilingInterface>(producer) ||
          producers.contains(producer))
        continue;
      worklist.push_back(producer);
      producers.insert(producer);
    }
  }
  return producers;
}

/// Apply a tile and fuse transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
template <typename Range>
static LogicalResult
applyTileAndFuseToAll(RewriterBase &rewriter, Operation *transformOp,
                      Range &&payloadOps, unsigned numLoops,
                      ArrayRef<OpFoldResult> tileSizes,
                      ArrayRef<int64_t> interchange,
                      transform::TransformResults &transformResults) {
  SmallVector<Operation *> tiledOps;
  SmallVector<SmallVector<Operation *>> loopOps(numLoops);

  for (Operation *target : payloadOps) {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
    if (!tilingInterfaceOp)
      return transformOp->emitError("only TilingInterface ops are supported");
    DominanceInfo dominanceInfo(tilingInterfaceOp);

    llvm::SmallDenseSet<Operation *> tiledAndFusedOps =
        collectTiledAndFusedOps(tilingInterfaceOp);
    llvm::DenseSet<Operation *> yieldReplacementsFor;
    for (auto op : tiledAndFusedOps) {
      if (llvm::any_of(op->getUsers(), [&](Operation *user) {
            return dominanceInfo.properlyDominates(tilingInterfaceOp, user);
          })) {
        yieldReplacementsFor.insert(op);
      }
    }

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes).setInterchange(interchange);

    scf::SCFTileAndFuseOptions tileAndFuseOptions;
    tileAndFuseOptions.setTilingOptions(tilingOptions);

    scf::SCFTileAndFuseOptions::ControlFnTy controlFn =
        [&](tensor::ExtractSliceOp candidateSliceOp, OpResult originalProducer,
            bool isDestinationOperand) {
          Operation *owner = originalProducer.getOwner();
          bool yieldProducerReplacement = yieldReplacementsFor.contains(owner);
          return std::make_tuple(true, yieldProducerReplacement);
        };
    tileAndFuseOptions.setFusionControlFn(controlFn);

    rewriter.setInsertionPoint(target);
    FailureOr<scf::SCFTileAndFuseResult> tiledResults =
        scf::tileConsumerAndFuseProducerGreedilyUsingSCFForOp(
            rewriter, tilingInterfaceOp, tileAndFuseOptions);
    if (failed(tiledResults))
      return failure();

    // Perform the replacement of tiled and fused values.
    SmallVector<Operation *> opsToReplace{target};
    llvm::append_range(opsToReplace, tiledResults->fusedProducers);
    for (Operation *toReplace : opsToReplace) {
      for (OpResult res : toReplace->getResults())
        if (auto replacement = tiledResults->replacements.lookup(res)) {
          Operation *replacementOp = replacement.getDefiningOp();
          rewriter.replaceUsesWithIf(
              res, replacement, [&](mlir::OpOperand &use) {
                Operation *user = use.getOwner();
                return dominanceInfo.properlyDominates(replacementOp, user) &&
                       user->getParentOp() == replacementOp->getParentOp();
              });
        }

      if (toReplace->use_empty()) {
        rewriter.eraseOp(toReplace);
      }
    }

    // Report back the relevant handles to the transform op.
    tiledOps.push_back(tiledResults->tiledAndFusedOps.front());
    assert(tiledResults->loops.size() == numLoops &&
           "Mismatched number of loops, tile and fuse transform should have "
           "failed");
    for (unsigned int i = 0; i < numLoops; ++i)
      loopOps[i].push_back(tiledResults->loops[i]);
  }

  transformResults.set(transformOp->getOpResult(0), tiledOps);
  for (unsigned int i = 0; i < numLoops; ++i)
    transformResults.set(transformOp->getOpResult(i + 1), loopOps[i]);

  return success();
}

DiagnosedSilenceableFailure transform::TestFuseAndYieldOp::apply(
    transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &transformResults,
    mlir::transform::TransformState &state) {
  SmallVector<int64_t> tileSizes =
      extractFromIntegerArrayAttr<int64_t>(getTileSizes());
  SmallVector<int64_t> tileInterchange =
      extractFromIntegerArrayAttr<int64_t>(getTileInterchange());

  SmallVector<OpFoldResult> tileSizesOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);

  LogicalResult result = applyTileAndFuseToAll(
      rewriter, getOperation(), state.getPayloadOps(getTarget()),
      tileSizes.size() - llvm::count(tileSizes, 0), tileSizesOfr,
      tileInterchange, transformResults);
  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TestTileUsingForallOp
//===----------------------------------------------------------------------===//

/// Apply a tiling transformation to all payload ops and store both the
/// tiled operation as well as the created tile loops.
template <typename Range>
static LogicalResult
applyTileToAll(RewriterBase &rewriter, Operation *transformOp,
               Range &&payloadOps, ArrayRef<OpFoldResult> tileSizes,
               ArrayRef<int64_t> interchange, std::optional<ArrayAttr> mapping,
               transform::TransformResults &transformResults) {
  SmallVector<Operation *> tiledOps;
  SmallVector<Operation *> loopOps;

  for (Operation *target : payloadOps) {
    auto tilingInterfaceOp = dyn_cast<TilingInterface>(target);
    if (!tilingInterfaceOp)
      return transformOp->emitError("only TilingInterface ops are supported");
    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes).setInterchange(interchange);
    if (mapping) {
      auto mappingAttrs =
          llvm::map_to_vector(mapping.value(), [](Attribute attr) {
            return cast<DeviceMappingAttrInterface>(attr);
          });
      tilingOptions.setMapping(mappingAttrs);
    }

    rewriter.setInsertionPoint(target);
    FailureOr<scf::SCFTilingResult> tiledResults =
        scf::tileUsingSCFForallOp(rewriter, tilingInterfaceOp, tilingOptions);
    if (failed(tiledResults))
      return failure();

    // Perform the replacement of tiled and fused values.
    rewriter.replaceOp(tilingInterfaceOp, tiledResults->replacements);

    // Report back the relevant handles to the transform op.
    tiledOps.push_back(tiledResults->tiledOps.front());
    for (Operation *loop : tiledResults->loops)
      loopOps.push_back(loop);
  }

  transformResults.set(transformOp->getOpResult(0), tiledOps);
  for (auto [index, loop] : llvm::enumerate(loopOps))
    transformResults.set(transformOp->getOpResult(index + 1), {loop});

  return success();
}

DiagnosedSilenceableFailure transform::TestTileUsingForallOp::apply(
    transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &transformResults,
    mlir::transform::TransformState &state) {
  SmallVector<int64_t> tileSizes =
      extractFromIntegerArrayAttr<int64_t>(getTileSizes());
  SmallVector<int64_t> interchange =
      extractFromIntegerArrayAttr<int64_t>(getInterchange());
  SmallVector<OpFoldResult> tileSizesOfr =
      getAsIndexOpFoldResult(rewriter.getContext(), tileSizes);

  LogicalResult result =
      applyTileToAll(rewriter, getOperation(), state.getPayloadOps(getTarget()),
                     tileSizesOfr, interchange, getMapping(), transformResults);
  return failed(result) ? DiagnosedSilenceableFailure::definiteFailure()
                        : DiagnosedSilenceableFailure::success();
}

void transform::TestTileUsingForallOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  producesHandle(getTiledOp(), effects);
  producesHandle(getLoops(), effects);
  modifiesPayload(effects);
}

#define GET_OP_CLASSES
#include "TestTilingInterfaceTransformOps.cpp.inc"

namespace {
class TestTilingInterfaceDialectExtension
    : public transform::TransformDialectExtension<
          TestTilingInterfaceDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<affine::AffineDialect>();
    declareDependentDialect<index::IndexDialect>();
    declareDependentDialect<scf::SCFDialect>();
    declareDependentDialect<tensor::TensorDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "TestTilingInterfaceTransformOps.cpp.inc"
        >();
  }
};
} // namespace

namespace test {
void registerTestTilingInterfaceTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<TestTilingInterfaceDialectExtension>();
}
} // namespace test
