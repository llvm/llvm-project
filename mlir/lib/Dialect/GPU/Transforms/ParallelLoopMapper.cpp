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

#include "mlir/Dialect/GPU/ParallelLoopMapper.h"

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::loop;

namespace {

enum MappingLevel { MapGrid = 0, MapBlock = 1, Sequential = 2 };

static constexpr int kNumHardwareIds = 3;

} // namespace

/// Bounded increment on MappingLevel. Increments to the next
/// level unless Sequential was already reached.
MappingLevel &operator++(MappingLevel &mappingLevel) {
  if (mappingLevel < Sequential) {
    mappingLevel = static_cast<MappingLevel>(mappingLevel + 1);
  }
  return mappingLevel;
}

/// Computed the hardware id to use for a given mapping level. Will
/// assign x,y and z hardware ids for the first 3 dimensions and use
/// sequential after.
static int64_t getHardwareIdForMapping(MappingLevel level, int dimension) {
  if (dimension >= kNumHardwareIds || level == Sequential)
    return Sequential * kNumHardwareIds;
  return (level * kNumHardwareIds) + dimension;
}

/// Add mapping information to the given parallel loop. Do not add
/// mapping information if the loop already has it. Also, don't
/// start a mapping at a nested loop.
static void mapParallelOp(ParallelOp parallelOp,
                          MappingLevel mappingLevel = MapGrid) {
  // Do not try to add a mapping to already mapped loops or nested loops.
  if (parallelOp.getAttr(gpu::kMappingAttributeName) ||
      ((mappingLevel == MapGrid) && parallelOp.getParentOfType<ParallelOp>()))
    return;

  MLIRContext *ctx = parallelOp.getContext();
  Builder b(ctx);
  SmallVector<Attribute, 4> attrs;
  attrs.reserve(parallelOp.getNumInductionVars());
  for (int i = 0, e = parallelOp.getNumInductionVars(); i < e; ++i) {
    SmallVector<NamedAttribute, 3> entries;
    entries.emplace_back(b.getNamedAttr(
        kProcessorEntryName,
        b.getI64IntegerAttr(getHardwareIdForMapping(mappingLevel, i))));
    entries.emplace_back(b.getNamedAttr(
        kIndexMapEntryName, AffineMapAttr::get(b.getDimIdentityMap())));
    entries.emplace_back(b.getNamedAttr(
        kBoundMapEntryName, AffineMapAttr::get(b.getDimIdentityMap())));
    attrs.push_back(DictionaryAttr::get(entries, ctx));
  }
  parallelOp.setAttr(kMappingAttributeName, ArrayAttr::get(attrs, ctx));
  ++mappingLevel;
  // Parallel loop operations are immediately nested, so do not use
  // walk but just iterate over the operations.
  for (Operation &op : *parallelOp.getBody()) {
    if (ParallelOp nested = dyn_cast<ParallelOp>(op))
      mapParallelOp(nested, mappingLevel);
  }
}

void mlir::greedilyMapParallelLoopsToGPU(Region &region) {
  region.walk([](ParallelOp parallelOp) { mapParallelOp(parallelOp); });
}
