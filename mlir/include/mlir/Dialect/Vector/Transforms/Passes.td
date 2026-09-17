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

def VectorInferInBounds : Pass<"vector-infer-in-bounds", "func::FuncOp"> {
  let summary = "Infer the `in_bounds` attribute of vector transfer operations";
  let description = [{
    Marks dimensions of `vector.transfer_read` and `vector.transfer_write`
    operations as in-bounds when value-bounds analysis can prove that the
    transfer, including its starting point, stays within the source.

    The canonicalizer already handles transfers whose indices are constants.
    This pass additionally handles indices that are loop induction variables,
    or affine expressions of them, which is the common shape after tiling and
    vectorization. That analysis is too expensive to run from a folder, so it
    is opt-in here.

    The pass only ever adds information: a dimension already marked in-bounds
    is left untouched, and no dimension is ever marked out-of-bounds.
  }];
}

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
