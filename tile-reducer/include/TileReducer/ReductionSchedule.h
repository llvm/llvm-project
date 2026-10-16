//===- ReductionSchedule.h - cost model / autotune --------------*- C++ -*-===//
//
// Milestone 23: candidate schedules and a roofline-style cost model.
// Not cycle-exact. T ~= max(T_compute, T_memory) + T_sync + T_launch + T_tail.
//
//===----------------------------------------------------------------------===//

#ifndef TILE_REDUCER_REDUCTIONSCHEDULE_H
#define TILE_REDUCER_REDUCTIONSCHEDULE_H

#include "TileReducer/GPUTargetInfo.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
class Operation;

namespace tr {

enum class ReductionKindName { Row, Column, Full };

struct ReductionSchedule {
  int threadsPerBlock = 256;
  int warpsPerBlock = 8;
  int rowsPerWarp = 16;
  int elementsPerLane = 4;
  bool useSharedMemory = false;
  int asyncDepth = 0;
  /// Physical blocks that refine one logical program along K. 1 means
  /// `tr.program_id` maps to one GPU block.
  int kSplits = 1;

  static ReductionSchedule baselineRow();
  static ReductionSchedule baselineColumn();
  static ReductionSchedule baselineFull();
};

struct ReductionProblem {
  ReductionKindName kind = ReductionKindName::Row;
  int axis = 1;
  int64_t tileRows = 128;
  int64_t tileCols = 128;
  int elemBits = 32;
  int64_t M = 1024;
  int64_t K = 1024;
  std::string dtype = "f32";
  std::string arch = "a100-like";
  std::string compiler = "tile-reducer-23";
};

struct CostEstimate {
  double tCompute = 0;
  double tMemory = 0;
  double tSync = 0;
  double tLaunch = 0;
  double tTail = 0;
  double tTotal = 0;
  double occupancy = 0;
  double coalescing = 1;
  double gridSaturation = 1;
  int registersPerThread = 0;
  int sharedMemoryBytes = 0;
  int kernelCount = 1;
  int nBlocks = 1;
  bool legal = true;
  std::string limiter = "none";
  std::string rejectReason;
};

struct TuneResult {
  ReductionSchedule winner;
  CostEstimate cost;
  std::string cacheKey;
  std::string shapeBucket;
  int candidates = 0;
  int pruned = 0;
  bool asyncHelpsRowSum = false;
  double baselineRowUs = 0;
  double asyncRowUs = 0;
};

/// Roofline-style estimate. Units are microseconds.
CostEstimate estimateCost(const ReductionProblem &prob,
                          const ReductionSchedule &sched,
                          const GPUTargetInfo &target);

/// Bounded legal space for autotune. Does not enumerate every M×K.
std::vector<ReductionSchedule> enumerateSchedules(const ReductionProblem &prob,
                                                  const GPUTargetInfo &target);

/// Analytically prune, rank by tTotal, cache the winner by shape bucket.
TuneResult autotune(const ReductionProblem &prob, const GPUTargetInfo &target);

std::string shapeBucket(const ReductionProblem &prob,
                        const GPUTargetInfo &target);
std::string tuneCacheKey(const ReductionProblem &prob,
                         const GPUTargetInfo &target);

/// Write schedule + cost fields as `tr.schedule.*` / `tr.cost.*` / `tr.tune.*`
/// / `tr.bench.*` attributes on `op`.
void applyCostAttrs(Operation *op, const ReductionSchedule &sched,
                    const CostEstimate &cost);
void applyTuneAttrs(Operation *op, const TuneResult &result);
void applyBenchAttrs(Operation *op, const ReductionSchedule &sched,
                     const CostEstimate &cost, const ReductionProblem &prob);

} // namespace tr
} // namespace mlir

#endif // TILE_REDUCER_REDUCTIONSCHEDULE_H
