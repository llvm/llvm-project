//===- VectorTransformOps.td - Vector transform ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef VECTOR_TRANSFORM_OPS
#define VECTOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformEffects.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Dialect/Vector/Transforms/VectorTransformsBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def LowerVectorsOp : Op<Transform_Dialect, "vector.lower_vectors",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let description = [{
    Indicates that the vector operations nested under the isolated from above op
    `target` should be lowered to finer-grained vector primitives.

    At this time, the transform is all or nothing.

    This is usally a late step that is run after bufferization as part of the
    process of lowering to e.g. LLVM or NVVM.
  }];

  // TODO: evolve this to proper enums.
  let arguments = (ins PDL_Operation:$target,
     DefaultValuedAttr<VectorContractLoweringAttr,
       "vector::VectorContractLowering::OuterProduct">:$contraction_lowering,
     DefaultValuedAttr<VectorMultiReductionLoweringAttr,
       "vector::VectorMultiReductionLowering::InnerParallel">:
         $multireduction_lowering,
     DefaultValuedAttr<VectorTransferSplitAttr,
       "vector::VectorTransferSplit::LinalgCopy">:$split_transfers,
     DefaultValuedAttr<VectorTransposeLoweringAttr,
       "vector::VectorTransposeLowering::EltWise">:$transpose_lowering,
     DefaultValuedAttr<BoolAttr, "false">:$transpose_avx2_lowering,
     DefaultValuedAttr<BoolAttr, "true">:$unroll_vector_transfers
  );
  let results = (outs PDL_Operation:$results);

  let assemblyFormat = [{
    $target
    oilist (
      `contraction_lowering` `=` $contraction_lowering
      | `multireduction_lowering` `=` $multireduction_lowering
      | `split_transfers` `=` $split_transfers
      | `transpose_lowering` `=` $transpose_lowering
    )
    attr-dict
  }];
}

#endif // VECTOR_TRANSFORM_OPS
