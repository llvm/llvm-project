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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"

#include "mlir/Dialect/GPU/ParallelLoopMapperEnums.h.inc"

namespace mlir {

class AffineMap;
struct LogicalResult;
class Operation;
class Region;

} // namespace mlir

#include "mlir/Dialect/GPU/ParallelLoopMapperAttr.h.inc"

namespace mlir {
namespace scf {
class ParallelOp;
}

namespace gpu {

/// Name of the mapping attribute produced by loop mappers.
StringRef getMappingAttrName();

/// Get the value of the processor in the ParallelLoopDimMapping attribute.
inline Processor getProcessor(ParallelLoopDimMapping attr) {
  return static_cast<Processor>(attr.processor().getInt());
}

/// Helper function to create a ParallelDimMapperAttr.
/// TODO: Replace its uses with an auto-gened method.
ParallelLoopDimMapping getParallelLoopDimMappingAttr(Processor processor,
                                                     AffineMap map,
                                                     AffineMap bound);

/// Sets the mapping attribute of a scf.parallel operation. Verifies that the
/// mapping passed is valid.
/// - the number of DimMapperAttr provided is same as the number of loops of
///   the `ploopOp`.
/// - the mapping does not map multiple loops to the same processor.
LogicalResult setMappingAttr(scf::ParallelOp ploopOp,
                             ArrayRef<ParallelLoopDimMapping> mapping);
} // namespace gpu

/// Maps the parallel loops found in the given function to workgroups. The first
/// loop encountered will be mapped to the global workgroup and the second loop
/// encountered to the local workgroup. Within each mapping, the first three
/// dimensions are mapped to x/y/z hardware ids and all following dimensions are
/// mapped to sequential loops.
void greedilyMapParallelSCFToGPU(Region &region);

} // namespace mlir
#endif // MLIR_DIALECT_GPU_PARALLELLOOPMAPPER_H
