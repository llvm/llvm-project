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
     DefaultValuedAttr<StrAttr, "\"outerproduct\"">:$contraction_lowering,
     DefaultValuedAttr<StrAttr, "\"innerparallel\"">:$multireduction_lowering,
     DefaultValuedAttr<StrAttr, "\"linalg-copy\"">:$split_transfers,
     DefaultValuedAttr<StrAttr, "\"eltwise\"">:$transpose_lowering,
     DefaultValuedAttr<BoolAttr, "false">:$transpose_avx2_lowering,
     DefaultValuedAttr<BoolAttr, "true">:$unroll_vector_transfers
  );
  let results = (outs PDL_Operation:$results);

  let assemblyFormat = "$target attr-dict";
}

#endif // VECTOR_TRANSFORM_OPS
