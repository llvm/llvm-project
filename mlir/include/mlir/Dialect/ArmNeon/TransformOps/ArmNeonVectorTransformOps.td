//===- ArmNeonTransformOps.td - Arm Neon transform ops------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ARMNEON_TRANSFORM_OPS
#define ARMNEON_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

def ApplyArmNeonLowerContractionPatternsOp
    : Op<Transform_Dialect, "apply_patterns.vector.arm_neon.lower_contraction",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contraction-like operations should be lowered to
    finer-grained vector primitives using the ArmNeon dialect.
  }];

  let assemblyFormat = "attr-dict";
}

#endif // ARMNEON_TRANSFORM_OPS
