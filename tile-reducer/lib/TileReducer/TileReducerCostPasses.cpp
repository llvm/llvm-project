//===- TileReducerCostPasses.cpp - Milestones 23, 25, 27 --------*- C++ -*-===//
//
// --tr-estimate-reduction-cost  annotate the baseline schedule
// --tr-autotune-reduction       prune, rank, cache by shape bucket
// --tr-bench-report             latency / GB/s / occupancy / kernel count
//
//===----------------------------------------------------------------------===//

#include "TileReducer/GPUTargetInfo.h"
#include "TileReducer/ReductionSchedule.h"
#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerOps.h"
#include "TileReducer/TileReducerPasses.h"
#include "TileReducer/TileReducerTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include <string>

using namespace mlir;
using namespace mlir::tr;

namespace mlir::tr {
#define GEN_PASS_DEF_ESTIMATETRREDUCTIONCOST
#define GEN_PASS_DEF_AUTOTUNETRREDUCTION
#define GEN_PASS_DEF_TRBENCHREPORT
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

static ReductionKindName kindFromFunc(func::FuncOp func) {
  ReductionKindName kind = ReductionKindName::Row;
  func.walk([&](ReduceSumOp reduce) {
    auto outTy = dyn_cast<TileType>(reduce.getType());
    if (outTy && outTy.getRank() == 0)
      kind = ReductionKindName::Full;
    else if (reduce.getAxis() == 0)
      kind = ReductionKindName::Column;
    else if (reduce.getAxis() == 1 && kind != ReductionKindName::Full)
      kind = ReductionKindName::Row;
  });
  return kind;
}

static ReductionProblem problemFrom(func::FuncOp func, int64_t M, int64_t K) {
  ReductionProblem p;
  p.kind = kindFromFunc(func);
  p.axis = p.kind == ReductionKindName::Column ? 0 : 1;
  p.M = M > 0 ? M : 1024;
  p.K = K > 0 ? K : 1024;
  func.walk([&](LoadOp load) {
    if (auto tile = dyn_cast<TileType>(load.getType())) {
      if (tile.getRank() == 2) {
        p.tileRows = tile.getDimSize(0);
        p.tileCols = tile.getDimSize(1);
        if (auto fty = dyn_cast<FloatType>(tile.getElementType())) {
          p.elemBits = fty.getWidth();
          p.dtype = "f" + std::to_string(p.elemBits);
        }
      }
    }
  });
  return p;
}

static ReductionSchedule baselineFor(ReductionKindName kind) {
  switch (kind) {
  case ReductionKindName::Column:
    return ReductionSchedule::baselineColumn();
  case ReductionKindName::Full:
    return ReductionSchedule::baselineFull();
  case ReductionKindName::Row:
    return ReductionSchedule::baselineRow();
  }
  return ReductionSchedule::baselineRow();
}

struct EstimateTRReductionCost
    : impl::EstimateTRReductionCostBase<EstimateTRReductionCost> {
  using impl::EstimateTRReductionCostBase<
      EstimateTRReductionCost>::EstimateTRReductionCostBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    GPUTargetInfo target = GPUTargetInfo::fromOp(module);
    module.walk([&](func::FuncOp func) {
      if (!func.getName().contains("sum"))
        return;
      ReductionProblem prob = problemFrom(func, problemM, problemK);
      ReductionSchedule sched = baselineFor(prob.kind);
      CostEstimate cost = estimateCost(prob, sched, target);
      applyCostAttrs(func, sched, cost);
    });
  }
};

struct AutotuneTRReduction
    : impl::AutotuneTRReductionBase<AutotuneTRReduction> {
  using impl::AutotuneTRReductionBase<AutotuneTRReduction>::AutotuneTRReductionBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    GPUTargetInfo target = GPUTargetInfo::fromOp(module);
    TuneResult last;
    bool any = false;
    module.walk([&](func::FuncOp func) {
      if (!func.getName().contains("sum"))
        return;
      ReductionProblem prob = problemFrom(func, problemM, problemK);
      TuneResult r = autotune(prob, target);
      applyTuneAttrs(func, r);
      last = r;
      any = true;
    });
    if (any)
      applyTuneAttrs(module, last);
  }
};

struct TRBenchReport : impl::TRBenchReportBase<TRBenchReport> {
  using impl::TRBenchReportBase<TRBenchReport>::TRBenchReportBase;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    GPUTargetInfo target = GPUTargetInfo::fromOp(module);
    module.walk([&](func::FuncOp func) {
      if (!func.getName().contains("sum"))
        return;
      ReductionProblem prob = problemFrom(func, problemM, problemK);
      ReductionSchedule sched = baselineFor(prob.kind);
      if (auto splits = module->getAttrOfType<IntegerAttr>("tr.tune.k_splits"))
        sched.kSplits = static_cast<int>(splits.getInt());
      CostEstimate cost = estimateCost(prob, sched, target);
      applyBenchAttrs(func, sched, cost, prob);
    });
  }
};

} // namespace
} // namespace mlir::tr
