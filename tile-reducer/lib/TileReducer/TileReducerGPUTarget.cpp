//===- TileReducerGPUTarget.cpp - Milestone 16 ------------------*- C++ -*-===//
//
// GPUTargetInfo factory, attribute I/O, and --tr-annotate-gpu-target.
//
//===----------------------------------------------------------------------===//

#include "TileReducer/GPUTargetInfo.h"
#include "TileReducer/TileReducerPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::tr;

namespace {

constexpr StringRef kWarpSize = "tr.target.warp_size";
constexpr StringRef kNumSMs = "tr.target.num_sms";
constexpr StringRef kMaxThreadsPerBlock = "tr.target.max_threads_per_block";
constexpr StringRef kMaxWarpsPerSM = "tr.target.max_warps_per_sm";
constexpr StringRef kMaxBlocksPerSM = "tr.target.max_blocks_per_sm";
constexpr StringRef kRegistersPerSM = "tr.target.registers_per_sm";
constexpr StringRef kMaxRegistersPerThread = "tr.target.max_registers_per_thread";
constexpr StringRef kSharedMemoryPerSM = "tr.target.shared_memory_per_sm";
constexpr StringRef kSharedMemoryPerBlock = "tr.target.shared_memory_per_block";
constexpr StringRef kMemoryBandwidthGBs = "tr.target.memory_bandwidth_gbs";
constexpr StringRef kFp32PeakTFLOPs = "tr.target.fp32_peak_tflops";
constexpr StringRef kThreadsPerBlock = "tr.target.threads_per_block";
constexpr StringRef kWarpsPerBlock = "tr.target.warps_per_block";

static IntegerAttr i64Attr(MLIRContext *ctx, int64_t v) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), v);
}

static std::optional<int64_t> getI64(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return std::nullopt;
}

static std::optional<double> getF64(Operation *op, StringRef name) {
  if (auto attr = op->getAttrOfType<FloatAttr>(name))
    return attr.getValueAsDouble();
  return std::nullopt;
}

} // namespace

GPUTargetInfo GPUTargetInfo::baseline() { return GPUTargetInfo(); }

GPUTargetInfo GPUTargetInfo::fromOp(Operation *op) {
  GPUTargetInfo info = baseline();
  for (Operation *cur = op; cur; cur = cur->getParentOp()) {
    if (auto v = getI64(cur, kWarpSize))
      info.warpSize = static_cast<int>(*v);
    if (auto v = getI64(cur, kNumSMs))
      info.numSMs = static_cast<int>(*v);
    if (auto v = getI64(cur, kMaxThreadsPerBlock))
      info.maxThreadsPerBlock = static_cast<int>(*v);
    if (auto v = getI64(cur, kMaxWarpsPerSM))
      info.maxWarpsPerSM = static_cast<int>(*v);
    if (auto v = getI64(cur, kMaxBlocksPerSM))
      info.maxBlocksPerSM = static_cast<int>(*v);
    if (auto v = getI64(cur, kRegistersPerSM))
      info.registersPerSM = static_cast<int>(*v);
    if (auto v = getI64(cur, kMaxRegistersPerThread))
      info.maxRegistersPerThread = static_cast<int>(*v);
    if (auto v = getI64(cur, kSharedMemoryPerSM))
      info.sharedMemoryPerSM = static_cast<int>(*v);
    if (auto v = getI64(cur, kSharedMemoryPerBlock))
      info.sharedMemoryPerBlock = static_cast<int>(*v);
    if (auto v = getF64(cur, kMemoryBandwidthGBs))
      info.memoryBandwidthGBs = *v;
    if (auto v = getF64(cur, kFp32PeakTFLOPs))
      info.fp32PeakTFLOPs = *v;
  }
  return info;
}

void GPUTargetInfo::applyTo(Operation *op) const {
  MLIRContext *ctx = op->getContext();
  auto f64 = Float64Type::get(ctx);
  op->setAttr(kWarpSize, i64Attr(ctx, warpSize));
  op->setAttr(kNumSMs, i64Attr(ctx, numSMs));
  op->setAttr(kMaxThreadsPerBlock, i64Attr(ctx, maxThreadsPerBlock));
  op->setAttr(kMaxWarpsPerSM, i64Attr(ctx, maxWarpsPerSM));
  op->setAttr(kMaxBlocksPerSM, i64Attr(ctx, maxBlocksPerSM));
  op->setAttr(kRegistersPerSM, i64Attr(ctx, registersPerSM));
  op->setAttr(kMaxRegistersPerThread, i64Attr(ctx, maxRegistersPerThread));
  op->setAttr(kSharedMemoryPerSM, i64Attr(ctx, sharedMemoryPerSM));
  op->setAttr(kSharedMemoryPerBlock, i64Attr(ctx, sharedMemoryPerBlock));
  op->setAttr(kMemoryBandwidthGBs, FloatAttr::get(f64, memoryBandwidthGBs));
  op->setAttr(kFp32PeakTFLOPs, FloatAttr::get(f64, fp32PeakTFLOPs));
  op->setAttr(kThreadsPerBlock, i64Attr(ctx, threadsPerBlock()));
  op->setAttr(kWarpsPerBlock, i64Attr(ctx, warpsPerBlock()));
}

namespace mlir::tr {
#define GEN_PASS_DEF_ANNOTATEGPUTARGET
#include "TileReducer/TileReducerPasses.h.inc"

namespace {

struct AnnotateGPUTarget : impl::AnnotateGPUTargetBase<AnnotateGPUTarget> {
  void runOnOperation() override { GPUTargetInfo::baseline().applyTo(getOperation()); }
};

} // namespace
} // namespace mlir::tr
