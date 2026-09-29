//===-- SPIRVPrimitiveOps.td - MLIR SPIR-V Primitive Ops ------*- tablegen -*------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------===//
//
// This file contains primitive ops for the SPIR-V dialect. It corresponds
// to "3.52.19. Primitive Instructions" of the SPIR-V specification.
//
//===-----------------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_PRIMITIVE_OPS
#define MLIR_DIALECT_SPIRV_PRIMITIVE_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

// -----

def SPIRV_EmitVertexOp : SPIRV_Op<"EmitVertex", []> {
  let summary = [{
    Emits the current values of all output variables to the current output
    primitive. After execution, the values of all output variables are
    undefined.
  }];

  let description = [{
    This instruction must only be used when only one stream is present.

    #### Example:

    ```mlir
    spirv.EmitVertex
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Geometry]>
  ];

  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 0;
  let assemblyFormat = "attr-dict";
}

// -----

def SPIRV_EndPrimitiveOp : SPIRV_Op<"EndPrimitive", []> {
  let summary = [{
    Finish the current primitive and start a new one. No vertex is emitted.
  }];

  let description = [{
    This instruction must only be used when only one stream is present.

    #### Example:

    ```mlir
    spirv.EndPrimitive
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Geometry]>
  ];

  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 0;
  let assemblyFormat = "attr-dict";
}

#endif // MLIR_DIALECT_SPIRV_PRIMITIVE_OPS
