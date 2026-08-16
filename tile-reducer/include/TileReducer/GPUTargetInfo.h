//===- GPUTargetInfo.h - GPU machine model ----------------------*- C++ -*-===//
//
// Milestone 16: target properties used by later scheduling and cost
// modeling. These are not TileReducer source semantics.
//
// Baseline launch geometry (schedule, not ISA):
//   warp size        = 32
//   threads / block  = 256
//   warps / block    = 8
//
//===----------------------------------------------------------------------===//

#ifndef TILE_REDUCER_GPUTARGETINFO_H
#define TILE_REDUCER_GPUTARGETINFO_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace tr {

struct GPUTargetInfo {
  int warpSize = 32;
  int numSMs = 108;
  int maxThreadsPerBlock = 1024;
  int maxWarpsPerSM = 64;
  int maxBlocksPerSM = 32;
  int registersPerSM = 65536;
  int maxRegistersPerThread = 255;
  int sharedMemoryPerSM = 164 * 1024;
  int sharedMemoryPerBlock = 163 * 1024;
  double memoryBandwidthGBs = 1555.0;
  double fp32PeakTFLOPs = 19.5;

  /// Baseline occupancy choice for the row-sum kernel. Not a source fact.
  static constexpr int kBaselineThreadsPerBlock = 256;

  int threadsPerBlock() const { return kBaselineThreadsPerBlock; }
  int warpsPerBlock() const { return kBaselineThreadsPerBlock / warpSize; }

  /// Rows one warp walks sequentially inside a `tileRows`-high logical tile.
  int rowsPerWarp(int tileRows) const { return tileRows / warpsPerBlock(); }
  /// Elements one lane owns along a `tileCols`-wide logical tile.
  int elementsPerLane(int tileCols) const { return tileCols / warpSize; }

  static GPUTargetInfo baseline();

  /// Read `tr.target.*` attributes from `op` or its parents; missing fields
  /// keep the baseline defaults.
  static GPUTargetInfo fromOp(Operation *op);

  /// Write every field as `tr.target.*` discardable attributes on `op`.
  void applyTo(Operation *op) const;
};

} // namespace tr
} // namespace mlir

#endif // TILE_REDUCER_GPUTARGETINFO_H
