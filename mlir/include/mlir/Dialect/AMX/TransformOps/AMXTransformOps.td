//===- AMXTransformOps.td - AMX transform ops --*- tablegen -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AMX_TRANSFORM_OPS
#define AMX_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/RegionKindInterface.td"

def ApplyVectorContractToPackedTypeTiledDotProductPatternsOp : Op<Transform_Dialect,
    "apply_patterns.amx.vector_contract_to_packed_type_tiled_dot_product",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect patterns to lower a BF16/Int8 type vector.contract operation 
	to a BF16/Int8 tiled dot-product.
  }];

  let assemblyFormat = "attr-dict";
}


#endif // AMX_TRANSFORM_OPS
