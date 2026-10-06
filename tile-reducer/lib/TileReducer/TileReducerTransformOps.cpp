//===- TileReducerTransformOps.cpp - Milestone 13 ---------------*- C++ -*-===//
//
// Custom Transform extension: transform.tr.map_row_reduction.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/TileReducerTransformOps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

#define GET_OP_CLASSES
#include "TileReducer/TileReducerTransformOps.cpp.inc"

namespace {

class TileReducerTransformDialectExtension
    : public transform::TransformDialectExtension<
          TileReducerTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TileReducerTransformDialectExtension)

  void init() {
    declareGeneratedDialect<linalg::LinalgDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "TileReducer/TileReducerTransformOps.cpp.inc"
        >();
  }
};

} // namespace

void mlir::tr::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<TileReducerTransformDialectExtension>();
}

LogicalResult transform::MapRowReductionOp::verify() {
  if (getWarpsPerBlock() <= 0 || getWarpSize() <= 0 ||
      getElementsPerLane() <= 0)
    return emitOpError("mapping parameters must be positive");
  return success();
}

void transform::MapRowReductionOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::onlyReadsHandle(getTargetMutable(), effects);
  transform::producesHandle(getOperation()->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

DiagnosedSilenceableFailure transform::MapRowReductionOp::apply(
    transform::TransformRewriter &rewriter, transform::TransformResults &results,
    transform::TransformState &state) {
  SmallVector<Operation *> payload =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  if (payload.empty())
    return emitSilenceableError() << "expected at least one payload op";

  int64_t warps = getWarpsPerBlock();
  int64_t warpSize = getWarpSize();
  int64_t elems = getElementsPerLane();

  for (Operation *op : payload) {
    auto generic = dyn_cast<linalg::GenericOp>(op);
    if (!generic)
      return emitSilenceableError() << "expected linalg.generic, got "
                                    << op->getName();

    SmallVector<utils::IteratorType> iters = generic.getIteratorTypesArray();
    if (iters.size() != 2 || iters[0] != utils::IteratorType::parallel ||
        iters[1] != utils::IteratorType::reduction)
      return emitSilenceableError()
             << "expected row reduction (iterator_types = "
                "[\"parallel\", \"reduction\"])";

    if (!generic.getInputs().empty()) {
      if (auto shaped =
              dyn_cast<ShapedType>(generic.getInputs().front().getType())) {
        if (shaped.hasRank() && shaped.getRank() >= 1 &&
            shaped.hasStaticShape() &&
            shaped.getDimSize(0) % warps != 0)
          return emitSilenceableError()
                 << "parallel extent " << shaped.getDimSize(0)
                 << " is not divisible by warps_per_block " << warps;
      }
    }

    rewriter.modifyOpInPlace(generic, [&] {
      generic->setAttr("tr.warps_per_block", rewriter.getI64IntegerAttr(warps));
      generic->setAttr("tr.warp_size", rewriter.getI64IntegerAttr(warpSize));
      generic->setAttr("tr.elements_per_lane", rewriter.getI64IntegerAttr(elems));
    });
  }

  results.set(llvm::cast<OpResult>(getResult()), payload);
  return DiagnosedSilenceableFailure::success();
}
