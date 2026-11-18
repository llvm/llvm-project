//===-- Passes.td - Vector pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
#define MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def EliminateVectorMasks : Pass<"eliminate-vector-masks", "func::FuncOp"> {
  let summary = "Replace masks that are provably all-true with constant masks";
  let description = [{
    Rewrites `vector.create_mask` operations that can be proven all-true into
    `vector.constant_mask`. Canonicalization then folds those away, turning a
    masked transfer into an unmasked one.

    A mask dimension is all-true when its size is at least the size of the
    corresponding vector dimension. Sizes are bounded with
    `ValueBoundsOpInterface`, so sizes derived from a loop, such as
    `%dim - %iv`, can be proven too.

    Proving a scalable dimension needs the range of `vscale`: pass `vscale-min`
    and `vscale-max` when it is known for the target (1 and 16 for SVE, say).
    Without a range, only fixed-size dimensions can be proven.
  }];
  let options = [
    Option<"vscaleMin", "vscale-min", "unsigned", /*default=*/"0",
           "Minimum possible value of vscale, or 0 when unknown.">,
    Option<"vscaleMax", "vscale-max", "unsigned", /*default=*/"0",
           "Maximum possible value of vscale, or 0 when unknown.">,
  ];
}

def LowerVectorMaskPass : Pass<"lower-vector-mask", "func::FuncOp"> {
  let summary = "Lower 'vector.mask' operations";
  let constructor = "mlir::vector::createLowerVectorMaskPass()";
}

def LowerVectorMultiReduction : Pass<"lower-vector-multi-reduction", "func::FuncOp"> {
  let summary = "Lower 'vector.multi_reduction' operations";
  let constructor = "mlir::vector::createLowerVectorMultiReductionPass()";
  let options = [
    Option<"loweringStrategy", "lowering-strategy", "mlir::vector::VectorMultiReductionLowering",
           /*default=*/"mlir::vector::VectorMultiReductionLowering::InnerParallel",
           "Select the strategy to control how multi_reduction is lowered.",
           [{::llvm::cl::values(
            clEnumValN(mlir::vector::VectorMultiReductionLowering::InnerParallel,
                       "inner-parallel",
                       "Lower multi_reduction into outer-reduction and inner-parallel ops."),
            clEnumValN(mlir::vector::VectorMultiReductionLowering::InnerReduction,
                       "inner-reduction",
                       "Lower multi_reduction into outer-parallel and inner-reduction ops.")
        )}]>
  ];
}

def LowerVectorToFromElementsToShuffleTree
  : Pass<"lower-vector-to-from-elements-to-shuffle-tree", "func::FuncOp"> {
  let summary = "Lower `vector.to_elements` and `vector.from_elements` to a tree of `vector.shuffle` operations";
}

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
