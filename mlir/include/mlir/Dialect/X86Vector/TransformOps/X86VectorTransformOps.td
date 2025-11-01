//===- X86VectorTransformOps.td - X86Vector transform ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef X86VECTOR_TRANSFORM_OPS
#define X86VECTOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/RegionKindInterface.td"

def ApplyVectorContractNanokernelLoweringPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.vector_contract_nanokernel_lowering",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contract operation can be lowered to target
    specific nanokernels.
  }];

  let arguments = (ins DefaultValuedAttr<I64Attr, "8">:$vector_size);

  let assemblyFormat = [{
    (`vector_size` `=` $vector_size^)? attr-dict
  }];

}


#endif // X86VECTOR_TRANSFORM_OPS
