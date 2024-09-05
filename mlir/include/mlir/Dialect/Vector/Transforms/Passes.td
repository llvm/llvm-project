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

#endif // MLIR_DIALECT_VECTOR_TRANSFORMS_PASSES
