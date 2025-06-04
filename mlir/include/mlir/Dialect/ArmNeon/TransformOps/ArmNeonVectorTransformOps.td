//===- ArmNeonVectorTransformOps.td - Arm Neon TD ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ARM_NEON_VECTOR_TRANSFORM_OPS
#define ARM_NEON_VECTOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

def ApplyArmNeonContractionToI8MMPatternsOp
    : Op<Transform_Dialect,
         "apply_patterns.arm_neon.vector_contract_to_i8mm",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector.contract operations should be lowered to
    finer-grained vector primitives from the ArmNeon dialect.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // ARM_NEON_VECTOR_TRANSFORM_OPS
