//===- ArmSVEVectorTransformOps.td - Arm SVE transform ops--*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef ARMSVE_VECTOR_TRANSFORM_OPS
#define ARMSVE_VECTOR_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformAttrs.td"
include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"

def ApplyArmSVELowerContractionToI8MMPatternsOp
    : Op<Transform_Dialect, "apply_patterns.arm_sve.vector_contract_to_i8mm",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contract operations should be lowered to
    to ArmSVE dialect operations mapping to instructions from FEAT_I8MM.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyArmSVELowerContractionToBFMMLAPatternsOp
    : Op<Transform_Dialect, "apply_patterns.arm_sve.vector_contract_to_bfmmla",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Indicates that vector contract operations should be lowered to
    ArmSVE dialect operations mapping to instructions from FEAT_BF16.
  }];

  let assemblyFormat = "attr-dict";
}

def ApplyArmSVELowerContractionToIndexedFMLAPatternsOp
    : Op<Transform_Dialect, "apply_patterns.arm_sve.vector_contract_to_indexed_fmla",
         [DeclareOpInterfaceMethods<PatternDescriptorOpInterface>]> {
  let description = [{
    Apply patterns to lower `vector.contract` to SVE's indexed FMLAs, where
    `vector.contract` is a matrix multiplication with types matching the
    following: 

    ```
      vector.contract #matmul_trait %lhs, %rhs, %acc
        vector<1x4xf32>, vector<4x[8]xf32> into vector<1x[8]xf32>
    ```

    (`#matmul_trait` contains indexing maps and iterator specification specific
    to matrix multiplication).
  }];

  let assemblyFormat = "attr-dict";
}

#endif // ARMSVE_VECTOR_TRANSFORM_OPS
