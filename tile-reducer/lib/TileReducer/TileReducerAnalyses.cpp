//===- TileReducerAnalyses.cpp ----------------------------------*- C++ -*-===//

#include "TileReducer/TileReducerAnalyses.h"
#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerTypes.h"

using namespace mlir;
using namespace mlir::tr;

BoundsAnalysis::BoundsAnalysis(Operation *op) { (void)op; }

int64_t BoundsAnalysis::getTileExtent(Value v, int64_t dim) const {
  auto tile = dyn_cast<TileType>(v.getType());
  if (!tile || dim < 0 || dim >= tile.getRank())
    return -1;
  return tile.getDimSize(dim);
}

LayoutAnalysis::LayoutAnalysis(Operation *op) { (void)op; }

ReductionAnalysis::ReductionAnalysis(Operation *op) {
  op->walk([&](ReduceSumOp red) {
    ReductionInfo info;
    info.axis = red.getAxis();
    auto inTy = dyn_cast<TileType>(red.getInput().getType());
    if (inTy && inTy.getRank() == 2) {
      if (info.axis == 1)
        info.kind = ReductionKind::Row;
      else if (info.axis == 0)
        info.kind = ReductionKind::Column;
    } else if (inTy && inTy.getRank() == 1) {
      info.kind = ReductionKind::Full;
    }
    info.loadReduceCandidate = isa_and_nonnull<LoadOp>(
                                   red.getInput().getDefiningOp()) &&
                               red.getInput().hasOneUse();
    infos[red] = info;
  });
}

const ReductionInfo *ReductionAnalysis::get(Operation *op) const {
  auto it = infos.find(op);
  return it == infos.end() ? nullptr : &it->second;
}
