//===- X86VectorTransformOps.td - X86Vector transform ops --*- tablegen -*-===//
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
    Collect patterns to lower a F32 type vector.contract operation to a FMA.
  }];
  
  let assemblyFormat = "attr-dict";
}

def ApplyVectorContractToPackedTypeDotProductPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.vector_contract_to_packed_type_dot_product",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect patterns to lower a BF16/Int8 type vector.contract operation 
	to a BF16/Int8 dot-product.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyVectorContractBF16ToFMAPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.vector_contract_bf16_to_fma",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect patterns to lower a BF16 type vector.contract operation
        to a FMA via emulation lowering using BF16 packed operations.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplySinkVectorProducerOpsPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.sink_vector_producer_ops",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect patterns to sink vector producer operations forward in a block to 
         place them immediately before their first use.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyShuffleVectorFMAOpsPatternsOp : Op<Transform_Dialect,
    "apply_patterns.x86vector.shuffle_vector_fma_ops",
    [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Collect patterns to shuffle FMAs with x86vector operations as operands 
    such that FMAs are grouped with respect to odd/even packed index.
  }];

  let assemblyFormat = "attr-dict";
}


#endif // X86VECTOR_TRANSFORM_OPS

