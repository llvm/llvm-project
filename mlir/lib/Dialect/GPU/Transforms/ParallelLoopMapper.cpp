//===- ParallelLoopMapper.cpp - Utilities for mapping parallel loops to GPU =//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements utilities to generate mappings for parallel loops to
// GPU devices.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
#define GEN_PASS_DEF_GPUMAPPARALLELLOOPSPASS
#include "mlir/Dialect/GPU/Transforms/Passes.h.inc"
} // namespace mlir

namespace mlir {

using scf::ParallelOp;

StringRef gpu::getMappingAttrName() { return "mapping"; }

LogicalResult
gpu::setMappingAttr(ParallelOp ploopOp,
                    ArrayRef<ParallelLoopDimMappingAttr> mapping) {
  // Verify that each processor is mapped to only once.
  llvm::DenseSet<gpu::Processor> specifiedMappings;
  for (auto dimAttr : mapping) {
    gpu::Processor processor = dimAttr.getProcessor();
    if (processor != gpu::Processor::Sequential &&
        specifiedMappings.count(processor))
      return ploopOp.emitError(
          "invalid mapping multiple loops to same processor");
    specifiedMappings.insert(processor);
  }
  ArrayRef<Attribute> mappingAsAttrs(mapping.data(), mapping.size());
  ploopOp->setAttr(getMappingAttrName(),
                   ArrayAttr::get(ploopOp.getContext(), mappingAsAttrs));
  return success();
}

namespace gpu {
namespace {
enum MappingLevel { MapGrid = 0, MapBlock = 1, Sequential = 2 };
enum class MappingPolicy { OutermostFirst, InnermostFirst };
} // namespace

static constexpr int kNumHardwareIds = 3;

/// Bounded increment on MappingLevel. Increments to the next
/// level unless Sequential was already reached.
static MappingLevel &operator++(MappingLevel &mappingLevel) {
  if (mappingLevel < Sequential) {
    mappingLevel = static_cast<MappingLevel>(mappingLevel + 1);
  }
  return mappingLevel;
}

// Map the policy string to a typed mapping policy.
// TODO: Revisit this and possibly use a loop interchange pass instead.
static FailureOr<MappingPolicy> getMappingPolicyFromStr(StringRef policy) {
  std::string policyCanonical = policy.trim().lower();

  std::optional<MappingPolicy> option =
      llvm::StringSwitch<std::optional<MappingPolicy>>(policyCanonical)
          .Case("innermost-first", MappingPolicy::InnermostFirst)
          .Case("outermost-first", MappingPolicy::OutermostFirst)
          .Default(std::nullopt);

  if (!option)
    return failure();
  return *option;
}

/// Computed the hardware id to use for a given mapping level. Will
/// assign x,y and z hardware ids for the first 3 dimensions and use
/// sequential after.
static Processor getHardwareIdForMapping(MappingLevel level, int dimension) {

  if (dimension >= kNumHardwareIds || level == Sequential)
    return Processor::Sequential;

  switch (level) {
  case MapGrid:
    switch (dimension) {
    case 0:
      return Processor::BlockX;
    case 1:
      return Processor::BlockY;
    case 2:
      return Processor::BlockZ;
    default:
      return Processor::Sequential;
    }
    break;
  case MapBlock:
    switch (dimension) {
    case 0:
      return Processor::ThreadX;
    case 1:
      return Processor::ThreadY;
    case 2:
      return Processor::ThreadZ;
    default:
      return Processor::Sequential;
    }
  default:;
  }
  return Processor::Sequential;
}

/// Add mapping information to the given parallel loop. Do not add
/// mapping information if the loop already has it. Also, don't
/// start a mapping at a nested loop.
static void
mapParallelOp(ParallelOp parallelOp, MappingLevel mappingLevel = MapGrid,
              MappingPolicy mappingPolicy = MappingPolicy::OutermostFirst) {
  // Do not try to add a mapping to already mapped loops or nested loops.
  if (parallelOp->getAttr(getMappingAttrName()) ||
      ((mappingLevel == MapGrid) && parallelOp->getParentOfType<ParallelOp>()))
    return;

  const int numLoops = static_cast<int>(parallelOp.getNumLoops());
  const int loopsToMap = std::min(numLoops, kNumHardwareIds);

  MLIRContext *ctx = parallelOp.getContext();
  Builder b(ctx);
  SmallVector<ParallelLoopDimMappingAttr, 4> attrs;
  attrs.reserve(numLoops);

  for (int i = 0; i < numLoops; ++i) {

    // Determine the mapping to use for this loop.
    // If the are more loops to map than HW IDs map to sequential.
    int hwMapping = kNumHardwareIds;
    if (i < loopsToMap) {
      hwMapping = (mappingPolicy == MappingPolicy::OutermostFirst)
                      ? i
                      : (loopsToMap - 1 - i);
    }

    attrs.push_back(b.getAttr<ParallelLoopDimMappingAttr>(
        getHardwareIdForMapping(mappingLevel, hwMapping), b.getDimIdentityMap(),
        b.getDimIdentityMap()));
  }
  (void)setMappingAttr(parallelOp, attrs);
  ++mappingLevel;
  // Parallel loop operations are immediately nested, so do not use
  // walk but just iterate over the operations.
  for (Operation &op : *parallelOp.getBody()) {
    if (ParallelOp nested = dyn_cast<ParallelOp>(op))
      mapParallelOp(nested, mappingLevel, mappingPolicy);
  }
}

namespace {
struct GpuMapParallelLoopsPass
    : public impl::GpuMapParallelLoopsPassBase<GpuMapParallelLoopsPass> {
  using Base::Base;

  void runOnOperation() override {
    // Parse the mapping policy.
    FailureOr<MappingPolicy> policyOrFailure =
        getMappingPolicyFromStr(mappingPolicyStr);
    if (failed(policyOrFailure)) {
      getOperation()->emitError() << "Invalid mapping policy specified.";
      return signalPassFailure();
    }

    MappingPolicy policy = *policyOrFailure;
    MappingLevel topLevel = MappingLevel::MapGrid;

    for (Region &region : getOperation()->getRegions()) {
      region.walk([&](ParallelOp parallelOp) {
        mapParallelOp(parallelOp, topLevel, policy);
      });
    }
  }
};

} // namespace
} // namespace gpu
} // namespace mlir
