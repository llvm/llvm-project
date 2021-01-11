//===-- SPIRVBarrierOps.td - MLIR SPIR-V Barrier Ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous ops for the SPIR-V dialect. It corresponds
// to "3.32.1. Miscellaneous Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_MISC_OPS
#define MLIR_DIALECT_SPIRV_IR_MISC_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

// -----

def SPV_UndefOp : SPV_Op<"undef", []> {
  let summary = "Make an intermediate object whose value is undefined.";

  let description = [{
    Result Type is the type of object to make.

    Each consumption of Result <id> yields an arbitrary, possibly different
    bit pattern or abstract value resulting in possibly different concrete,
    abstract, or opaque values.

    <!-- End of AutoGen section -->

    ```
    undef-op ::= `spv.undef` `:` spirv-type
    ```

    #### Example:

    ```mlir
    %0 = spv.undef : f32
    %1 = spv.undef : !spv.struct<!spv.array<4 x vector<4xi32>>>
    ```
  }];

  let arguments = (ins);

  let results = (outs
    SPV_Type:$result
  );

  let verifier = [{ return success(); }];

  let hasOpcode = 0;
  let autogenSerialization = 0;

  let assemblyFormat = "attr-dict `:` type($result)";
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_MISC_OPS
