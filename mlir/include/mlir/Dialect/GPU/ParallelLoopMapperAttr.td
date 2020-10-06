//===-- ParallelLoopMapperAttr.td - Attribute definition ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the attribute used for driving conversion from scf.parallel to
// gpu.launch operations
//
//===----------------------------------------------------------------------===//

#ifndef PARALLEL_LOOP_MAPPER_ATTR
#define PARALLEL_LOOP_MAPPER_ATTR

include "mlir/Dialect/GPU/GPUBase.td"

def BlockX : I64EnumAttrCase<"BlockX", 0>;
def BlockY : I64EnumAttrCase<"BlockY", 1>;
def BlockZ : I64EnumAttrCase<"BlockZ", 2>;
def ThreadX : I64EnumAttrCase<"ThreadX", 3>;
def ThreadY : I64EnumAttrCase<"ThreadY", 4>;
def ThreadZ : I64EnumAttrCase<"ThreadZ", 5>;
def Sequential : I64EnumAttrCase<"Sequential", 6>;

def ProcessorAttr : I64EnumAttr<"Processor", "processor for loop mapping", [
    BlockX, BlockY, BlockZ, ThreadX, ThreadY, ThreadZ, Sequential]> {
  let cppNamespace = "::mlir::gpu";
}

// Attribute that drives conversion of a scf.parallel to gpu.launch
// operation.
// processor: the hardware id to map to.
// map : An affine map that is used to pre-process hardware ids before
//       substitution.
// bound : An affine map that is used to compute the bound of the hardware
//         id based on an upper bound of the number of iterations.
def ParallelLoopDimMappingAttr :
    StructAttr<"ParallelLoopDimMapping", GPU_Dialect,
               [StructFieldAttr<"processor", ProcessorAttr>,
                StructFieldAttr<"map", AffineMapAttr>,
                StructFieldAttr<"bound", AffineMapAttr>]>;


def ParallelLoopMappingAttr :
    TypedArrayAttrBase<ParallelLoopDimMappingAttr,
                       "parallel loop to processor mapping attribute">;

#endif // PARALLEL_LOOP_MAPPER_ATTR
