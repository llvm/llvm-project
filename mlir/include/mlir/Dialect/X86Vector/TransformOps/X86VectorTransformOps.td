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

def ApplyVectorContractToFMAPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.vector_contract_to_fma",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contract operation can be lowered to a FMA.
  }];
  
  let assemblyFormat = "attr-dict";
}

def ApplyVectorContractToPackedTypeDotProductPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.vector_contract_to_packed_type_dot_product",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contract operation can be lowered to a BF16/Int8 dot-product.
  }];

  let assemblyFormat = "attr-dict";
}


#endif // X86VECTOR_TRANSFORM_OPS

