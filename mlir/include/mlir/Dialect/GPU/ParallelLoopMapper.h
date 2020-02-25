//===- ParallelLoopMapper.h - Utilities for mapping parallel loops to GPU ====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares the utilities to generate mappings for parallel
// loops to GPU devices.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_PARALLELLOOPMAPPER_H
#define MLIR_DIALECT_GPU_PARALLELLOOPMAPPER_H

namespace mlir {

struct Region;

namespace gpu {

/// Name of the mapping attribute produced by loop mappers.
static constexpr const char *kMappingAttributeName = "mapping";
/// Name of the processor sub-attribute that identifies the hardware id
/// to map a loop to.
static constexpr const char *kProcessorEntryName = "processor";
/// Name of the map sub-attribute that identifies the affine map to apply
/// to the hardware id to compute the iteration number of the loop. This
/// map is expected to be extended by step and lower bound computations:
///   index = map(hardware_id) * step + lowerbound
static constexpr const char *kIndexMapEntryName = "map";
/// Name of the bound sub-attribute that itendities the affine map to
/// compute an upper bound of iterations for the hardware id. This is
/// applied to an upper bound on the number of iterations:
///   launchBound = bound(upperbound-lowerbound ceildiv step)
static constexpr const char *kBoundMapEntryName = "bound";

} // end namespace gpu

/// Maps the parallel loops found in the given function to workgroups. The first
/// loop encountered will be mapped to the global workgroup and the second loop
/// encountered to the local workgroup. Within each mapping, the first three
/// dimensions are mapped to x/y/z hardware ids and all following dimensions are
/// mapped to sequential loops.
void greedilyMapParallelLoopsToGPU(Region &region);

} // end namespace mlir

#endif // MLIR_DIALECT_GPU_PARALLELLOOPMAPPER_H
