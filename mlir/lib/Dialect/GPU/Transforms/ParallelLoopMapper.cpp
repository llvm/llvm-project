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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::gpu;
using namespace mlir::scf;

#include "mlir/Dialect/GPU/ParallelLoopMapperAttr.cpp.inc"
#include "mlir/Dialect/GPU/ParallelLoopMapperEnums.cpp.inc"
namespace mlir {
namespace gpu {

StringRef getMappingAttrName() { return "mapping"; }

ParallelLoopDimMapping getParallelLoopDimMappingAttr(Processor processor,
                                                     AffineMap map,
                                                     AffineMap bound) {
  MLIRContext *context = map.getContext();
  OpBuilder builder(context);
  return ParallelLoopDimMapping::get(
      ProcessorAttr::get(builder.getContext(), processor),
      AffineMapAttr::get(map), AffineMapAttr::get(bound), context);
}

LogicalResult setMappingAttr(scf::ParallelOp ploopOp,
                             ArrayRef<ParallelLoopDimMapping> mapping) {
  // Verify that each processor is mapped to only once.
  llvm::DenseSet<gpu::Processor> specifiedMappings;
  for (auto dimAttr : mapping) {
    gpu::Processor processor = getProcessor(dimAttr);
    if (processor != gpu::Processor::Sequential &&
        specifiedMappings.count(processor))
      return ploopOp.emitError(
          "invalid mapping multiple loops to same processor");
  }
  ArrayRef<Attribute> mappingAsAttrs(mapping.data(), mapping.size());
  ploopOp->setAttr(getMappingAttrName(),
                   ArrayAttr::get(ploopOp.getContext(), mappingAsAttrs));
  return success();
}
} // namespace gpu
} // namespace mlir

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
/// TODO: Make this use x for the inner-most loop that is
/// distributed to map to x, the next innermost to y and the next innermost to
/// z.
static gpu::Processor getHardwareIdForMapping(MappingLevel level,
                                              int dimension) {

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
static void mapParallelOp(ParallelOp parallelOp,
                          MappingLevel mappingLevel = MapGrid) {
  // Do not try to add a mapping to already mapped loops or nested loops.
  if (parallelOp->getAttr(getMappingAttrName()) ||
      ((mappingLevel == MapGrid) && parallelOp->getParentOfType<ParallelOp>()))
    return;

  MLIRContext *ctx = parallelOp.getContext();
  Builder b(ctx);
  SmallVector<ParallelLoopDimMapping, 4> attrs;
  attrs.reserve(parallelOp.getNumLoops());
  for (int i = 0, e = parallelOp.getNumLoops(); i < e; ++i) {
    attrs.push_back(getParallelLoopDimMappingAttr(
        getHardwareIdForMapping(mappingLevel, i), b.getDimIdentityMap(),
        b.getDimIdentityMap()));
  }
  (void)setMappingAttr(parallelOp, attrs);
  ++mappingLevel;
  // Parallel loop operations are immediately nested, so do not use
  // walk but just iterate over the operations.
  for (Operation &op : *parallelOp.getBody()) {
    if (ParallelOp nested = dyn_cast<ParallelOp>(op))
      mapParallelOp(nested, mappingLevel);
  }
}

void mlir::greedilyMapParallelSCFToGPU(Region &region) {
  region.walk([](ParallelOp parallelOp) { mapParallelOp(parallelOp); });
}
