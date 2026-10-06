//===- ReductionCostModel.cpp - Milestones 23-27 ----------------*- C++ -*-===//
//
// Roofline cost model, bounded autotune, and attribute writers.
// T ~= max(T_compute, T_memory) + T_sync + T_launch + T_tail  (microseconds).
// Not cycle-exact: occupancy, coalescing, register/smem pressure, and grid
// saturation are first-order terms only.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/ReductionSchedule.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>

using namespace mlir;

namespace mlir::tr {

namespace {

constexpr double kLaunchUs = 5.0;
constexpr double kBarrierUs = 0.25;

static int elemBytes(const ReductionProblem &prob) {
  return std::max(1, prob.elemBits / 8);
}

static int64_t nRowTiles(const ReductionProblem &prob) {
  return std::max<int64_t>(1, (prob.M + prob.tileRows - 1) / prob.tileRows);
}

static int64_t nColTiles(const ReductionProblem &prob) {
  return std::max<int64_t>(1, (prob.K + prob.tileCols - 1) / prob.tileCols);
}

static int estimateRegs(const ReductionSchedule &sched) {
  int regs = 16 + sched.elementsPerLane * 2;
  if (sched.useSharedMemory)
    regs += 8;
  regs += sched.asyncDepth * 12;
  return std::min(regs, 255);
}

static int estimateSmem(const ReductionProblem &prob,
                        const ReductionSchedule &sched) {
  if (!sched.useSharedMemory)
    return 0;
  int bytes = 0;
  if (prob.kind == ReductionKindName::Column)
    bytes = static_cast<int>(prob.tileRows * prob.tileCols * elemBytes(prob));
  else
    bytes = sched.warpsPerBlock * elemBytes(prob);
  if (sched.asyncDepth >= 2)
    bytes *= 2;
  return bytes;
}

static int kernelCount(const ReductionProblem &prob,
                       const ReductionSchedule &sched) {
  if (prob.kind == ReductionKindName::Full)
    return 2;
  if (sched.kSplits > 1)
    return 2;
  return 1;
}

static int nBlocks(const ReductionProblem &prob,
                   const ReductionSchedule &sched) {
  switch (prob.kind) {
  case ReductionKindName::Row:
    return static_cast<int>(nRowTiles(prob) * std::max(1, sched.kSplits));
  case ReductionKindName::Column:
    return static_cast<int>(nColTiles(prob));
  case ReductionKindName::Full:
    return static_cast<int>(nRowTiles(prob));
  }
  return 1;
}

static double coalescingFactor(const ReductionProblem &prob,
                               const ReductionSchedule &sched) {
  if (prob.kind == ReductionKindName::Column && !sched.useSharedMemory)
    return 1.0 / 32.0;
  return 1.0;
}

} // namespace

ReductionSchedule ReductionSchedule::baselineRow() {
  ReductionSchedule s;
  s.useSharedMemory = false;
  s.asyncDepth = 0;
  s.kSplits = 1;
  return s;
}

ReductionSchedule ReductionSchedule::baselineColumn() {
  ReductionSchedule s;
  s.useSharedMemory = true;
  s.asyncDepth = 0;
  s.kSplits = 1;
  return s;
}

ReductionSchedule ReductionSchedule::baselineFull() {
  ReductionSchedule s;
  s.useSharedMemory = true;
  s.asyncDepth = 0;
  s.kSplits = 1;
  return s;
}

CostEstimate estimateCost(const ReductionProblem &prob,
                          const ReductionSchedule &sched,
                          const GPUTargetInfo &target) {
  CostEstimate c;
  c.registersPerThread = estimateRegs(sched);
  c.sharedMemoryBytes = estimateSmem(prob, sched);
  c.kernelCount = kernelCount(prob, sched);
  c.nBlocks = nBlocks(prob, sched);
  c.coalescing = coalescingFactor(prob, sched);

  if (sched.threadsPerBlock <= 0 ||
      sched.threadsPerBlock % target.warpSize != 0 ||
      sched.threadsPerBlock > target.maxThreadsPerBlock) {
    c.legal = false;
    c.rejectReason = "illegal thread count";
    return c;
  }
  if (prob.tileRows % std::max(1, sched.warpsPerBlock) != 0 ||
      prob.tileCols % target.warpSize != 0) {
    c.legal = false;
    c.rejectReason = "tile not divisible by the GPU map";
    return c;
  }
  if (sched.asyncDepth > 0 && !sched.useSharedMemory) {
    c.legal = false;
    c.rejectReason = "async pipeline requires shared memory";
    return c;
  }
  if (sched.asyncDepth < 0 || sched.asyncDepth == 1 || sched.asyncDepth > 2) {
    c.legal = false;
    c.rejectReason = "asyncDepth must be 0 or 2";
    return c;
  }
  if (sched.kSplits < 1) {
    c.legal = false;
    c.rejectReason = "kSplits must be >= 1";
    return c;
  }
  if (c.sharedMemoryBytes > target.sharedMemoryPerBlock) {
    c.legal = false;
    c.rejectReason = "shared memory exceeds per-block limit";
    c.limiter = "shared_memory";
    return c;
  }
  if (c.registersPerThread > target.maxRegistersPerThread) {
    c.legal = false;
    c.rejectReason = "register pressure";
    c.limiter = "registers";
    return c;
  }

  int regsPerBlock = c.registersPerThread * sched.threadsPerBlock;
  int byRegs = regsPerBlock ? target.registersPerSM / regsPerBlock
                            : target.maxBlocksPerSM;
  int bySmem = c.sharedMemoryBytes
                   ? target.sharedMemoryPerSM / c.sharedMemoryBytes
                   : target.maxBlocksPerSM;
  int blocksPerSM =
      std::max(0, std::min({target.maxBlocksPerSM, byRegs, bySmem}));
  int warpsPerSM = blocksPerSM * sched.warpsPerBlock;
  c.occupancy = target.maxWarpsPerSM
                    ? static_cast<double>(warpsPerSM) / target.maxWarpsPerSM
                    : 0.0;
  if (blocksPerSM == 0) {
    c.legal = false;
    c.rejectReason = "zero occupancy";
    c.limiter = byRegs < bySmem ? "registers" : "shared_memory";
    return c;
  }
  if (byRegs <= bySmem && byRegs <= target.maxBlocksPerSM)
    c.limiter = "registers";
  else if (bySmem <= target.maxBlocksPerSM)
    c.limiter = "shared_memory";
  else
    c.limiter = "occupancy";

  int activeSMs = std::min(target.numSMs, c.nBlocks);
  c.gridSaturation =
      target.numSMs ? static_cast<double>(activeSMs) / target.numSMs : 1.0;

  double bytes = static_cast<double>(prob.M) * static_cast<double>(prob.K) *
                 elemBytes(prob);
  if (prob.kind == ReductionKindName::Row)
    bytes += static_cast<double>(prob.M) * elemBytes(prob);
  else if (prob.kind == ReductionKindName::Column)
    bytes += static_cast<double>(prob.K) * elemBytes(prob);
  else
    bytes += elemBytes(prob);
  if (sched.kSplits > 1)
    bytes += static_cast<double>(prob.M) * sched.kSplits * elemBytes(prob);

  double flops = static_cast<double>(prob.M) * static_cast<double>(prob.K);
  double sat = std::max(c.gridSaturation, 1.0 / std::max(1, target.numSMs));
  double occ = std::max(c.occupancy, 0.05);
  double bw = target.memoryBandwidthGBs * sat * c.coalescing; // GB/s
  double peak = target.fp32PeakTFLOPs * sat * occ;            // TFLOP/s
  c.tMemory = bw > 0 ? (bytes / 1e9) / bw * 1e6 : 1e9;
  c.tCompute = peak > 0 ? (flops / 1e12) / peak * 1e6 : 1e9;

  int nBarriers = 0;
  if (sched.useSharedMemory)
    nBarriers = (prob.kind == ReductionKindName::Column)
                    ? static_cast<int>(nRowTiles(prob)) * 2
                    : 1;
  if (sched.asyncDepth >= 2)
    nBarriers += 2;
  c.tSync = nBarriers * kBarrierUs;
  c.tLaunch = c.kernelCount * kLaunchUs;

  int64_t rem = prob.K % prob.tileCols;
  if (rem != 0) {
    double frac =
        static_cast<double>(prob.tileCols - rem) / prob.tileCols;
    c.tTail = frac * c.tMemory / std::max<int64_t>(1, nColTiles(prob));
  }

  // Async hides memory only when there is enough compute to overlap.
  // Row-sum intensity is ~0.25 flop/byte; do not credit a pipeline there.
  double intensity = c.tMemory > 0 ? c.tCompute / c.tMemory : 0.0;
  if (sched.asyncDepth >= 2 && sched.useSharedMemory && intensity > 0.25) {
    double hide = std::min(c.tMemory, c.tCompute) * 0.3;
    c.tMemory = std::max(0.0, c.tMemory - hide);
  }

  c.tTotal = std::max(c.tCompute, c.tMemory) + c.tSync + c.tLaunch + c.tTail;
  return c;
}

std::vector<ReductionSchedule> enumerateSchedules(const ReductionProblem &prob,
                                                  const GPUTargetInfo &target) {
  std::vector<ReductionSchedule> out;
  const int tileRows = static_cast<int>(prob.tileRows);
  const int tileCols = static_cast<int>(prob.tileCols);
  for (int threads : {128, 256, 512}) {
    if (threads > target.maxThreadsPerBlock)
      continue;
    int warps = threads / target.warpSize;
    if (warps == 0 || tileRows % warps != 0)
      continue;
    ReductionSchedule s;
    s.threadsPerBlock = threads;
    s.warpsPerBlock = warps;
    s.rowsPerWarp = tileRows / warps;
    s.elementsPerLane = tileCols / target.warpSize;
    for (bool smem : {false, true}) {
      if (prob.kind == ReductionKindName::Column && !smem)
        continue; // baseline column uses smem; strided is enumerated below
      if (prob.kind == ReductionKindName::Full && !smem)
        continue;
      s.useSharedMemory = smem;
      for (int async : {0, 2}) {
        if (async && !smem)
          continue;
        s.asyncDepth = async;
        for (int splits : {1, 4, 16, 64}) {
          if (prob.kind == ReductionKindName::Column && splits != 1)
            continue;
          // Large-K split is only for under-occupied grids. Enough row
          // tiles already saturate the SMs; do not refine further.
          if (splits > 1 && nRowTiles(prob) >= target.numSMs)
            continue;
          s.kSplits = splits;
          out.push_back(s);
        }
      }
    }
    if (prob.kind == ReductionKindName::Column) {
      s.useSharedMemory = false;
      s.asyncDepth = 0;
      s.kSplits = 1;
      out.push_back(s);
    }
  }
  return out;
}

std::string shapeBucket(const ReductionProblem &prob,
                        const GPUTargetInfo &target) {
  StringRef mB = prob.M < target.numSMs ? "M_few" : "M_many";
  StringRef kB = prob.K < 128      ? "K_tiny"
                 : prob.K < 4096   ? "K_small"
                 : prob.K < 1000000 ? "K_medium"
                                    : "K_large";
  return (mB + "_" + kB).str();
}

std::string tuneCacheKey(const ReductionProblem &prob,
                         const GPUTargetInfo &target) {
  const char *kind = prob.kind == ReductionKindName::Row      ? "row"
                     : prob.kind == ReductionKindName::Column ? "column"
                                                              : "full";
  std::ostringstream os;
  os << kind << '|' << prob.axis << '|' << prob.dtype << '|' << prob.tileRows
     << 'x' << prob.tileCols << '|' << shapeBucket(prob, target) << '|'
     << prob.arch << '|' << prob.compiler;
  return os.str();
}

TuneResult autotune(const ReductionProblem &prob, const GPUTargetInfo &target) {
  static std::map<std::string, TuneResult> cache;
  TuneResult r;
  r.cacheKey = tuneCacheKey(prob, target);
  r.shapeBucket = shapeBucket(prob, target);
  if (auto it = cache.find(r.cacheKey); it != cache.end())
    return it->second;

  auto space = enumerateSchedules(prob, target);
  r.candidates = static_cast<int>(space.size());
  CostEstimate best;
  best.tTotal = 1e300;
  best.legal = false;
  ReductionSchedule bestS = ReductionSchedule::baselineRow();
  for (const auto &s : space) {
    CostEstimate c = estimateCost(prob, s, target);
    if (!c.legal) {
      ++r.pruned;
      continue;
    }
    if (c.tTotal < best.tTotal) {
      best = c;
      bestS = s;
    }
  }
  r.winner = bestS;
  r.cost = best;

  // Measure async on the row-sum baseline. Do not assume it helps: row-sum
  // intensity is too low and extra smem/regs hurt occupancy.
  if (prob.kind == ReductionKindName::Row) {
    ReductionSchedule base = ReductionSchedule::baselineRow();
    ReductionSchedule async = base;
    async.useSharedMemory = true;
    async.asyncDepth = 2;
    CostEstimate b = estimateCost(prob, base, target);
    CostEstimate a = estimateCost(prob, async, target);
    r.baselineRowUs = b.tTotal;
    r.asyncRowUs = a.legal ? a.tTotal : 1e300;
    r.asyncHelpsRowSum =
        a.legal && b.tTotal > 0 && a.tTotal < 0.9 * b.tTotal;
  }

  cache[r.cacheKey] = r;
  return r;
}

static IntegerAttr i64(MLIRContext *ctx, int64_t v) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), v);
}

static FloatAttr f64(MLIRContext *ctx, double v) {
  return FloatAttr::get(Float64Type::get(ctx), v);
}

void applyCostAttrs(Operation *op, const ReductionSchedule &sched,
                    const CostEstimate &cost) {
  MLIRContext *ctx = op->getContext();
  op->setAttr("tr.schedule.threads_per_block", i64(ctx, sched.threadsPerBlock));
  op->setAttr("tr.schedule.warps_per_block", i64(ctx, sched.warpsPerBlock));
  op->setAttr("tr.schedule.rows_per_warp", i64(ctx, sched.rowsPerWarp));
  op->setAttr("tr.schedule.elements_per_lane", i64(ctx, sched.elementsPerLane));
  op->setAttr("tr.schedule.use_shared_memory",
              BoolAttr::get(ctx, sched.useSharedMemory));
  op->setAttr("tr.schedule.async_depth", i64(ctx, sched.asyncDepth));
  op->setAttr("tr.schedule.k_splits", i64(ctx, sched.kSplits));
  op->setAttr("tr.cost.t_compute", f64(ctx, cost.tCompute));
  op->setAttr("tr.cost.t_memory", f64(ctx, cost.tMemory));
  op->setAttr("tr.cost.t_sync", f64(ctx, cost.tSync));
  op->setAttr("tr.cost.t_launch", f64(ctx, cost.tLaunch));
  op->setAttr("tr.cost.t_tail", f64(ctx, cost.tTail));
  op->setAttr("tr.cost.t_total", f64(ctx, cost.tTotal));
  op->setAttr("tr.cost.occupancy", f64(ctx, cost.occupancy));
  op->setAttr("tr.cost.coalescing", f64(ctx, cost.coalescing));
  op->setAttr("tr.cost.grid_saturation", f64(ctx, cost.gridSaturation));
  op->setAttr("tr.cost.limiter", StringAttr::get(ctx, cost.limiter));
  op->setAttr("tr.cost.legal", BoolAttr::get(ctx, cost.legal));
}

void applyTuneAttrs(Operation *op, const TuneResult &result) {
  applyCostAttrs(op, result.winner, result.cost);
  MLIRContext *ctx = op->getContext();
  op->setAttr("tr.tune.cache_key", StringAttr::get(ctx, result.cacheKey));
  op->setAttr("tr.tune.shape_bucket", StringAttr::get(ctx, result.shapeBucket));
  op->setAttr("tr.tune.candidates", i64(ctx, result.candidates));
  op->setAttr("tr.tune.pruned", i64(ctx, result.pruned));
  op->setAttr("tr.tune.k_splits", i64(ctx, result.winner.kSplits));
  op->setAttr("tr.tune.async_depth", i64(ctx, result.winner.asyncDepth));
  op->setAttr("tr.tune.async_helps_row_sum",
              BoolAttr::get(ctx, result.asyncHelpsRowSum));
  op->setAttr("tr.cost.async_row_sum_us", f64(ctx, result.asyncRowUs));
  op->setAttr("tr.cost.baseline_row_sum_us", f64(ctx, result.baselineRowUs));
}

void applyBenchAttrs(Operation *op, const ReductionSchedule &sched,
                     const CostEstimate &cost, const ReductionProblem &prob) {
  applyCostAttrs(op, sched, cost);
  MLIRContext *ctx = op->getContext();
  double bytes = static_cast<double>(prob.M) * static_cast<double>(prob.K) *
                 elemBytes(prob);
  double gbs = cost.tTotal > 0 ? (bytes / 1e9) / (cost.tTotal / 1e6) : 0.0;
  op->setAttr("tr.bench.latency_us", f64(ctx, cost.tTotal));
  op->setAttr("tr.bench.effective_gbs", f64(ctx, gbs));
  op->setAttr("tr.bench.threads_per_block", i64(ctx, sched.threadsPerBlock));
  op->setAttr("tr.bench.registers_per_thread",
              i64(ctx, cost.registersPerThread));
  op->setAttr("tr.bench.shared_memory_bytes", i64(ctx, cost.sharedMemoryBytes));
  op->setAttr("tr.bench.occupancy", f64(ctx, cost.occupancy));
  op->setAttr("tr.bench.kernel_count", i64(ctx, cost.kernelCount));
}

} // namespace mlir::tr
