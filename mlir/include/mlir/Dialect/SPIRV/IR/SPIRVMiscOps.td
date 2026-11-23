//===-- SPIRVMiscOps.td - MLIR SPIR-V Misc Ops -------------*- tablegen -*-===//
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

def SPIRV_KHRAssumeTrueOp : SPIRV_KhrVendorOp<"AssumeTrue", []> {
  let summary = "TBD";

  let description = [{


    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    spirv.KHR.AssumeTrue %arg
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_expect_assume]>,
    Capability<[SPIRV_C_ExpectAssumeKHR]>
  ];

  let arguments = (ins
    SPIRV_Bool:$condition
  );

  let results = (outs);

  let hasVerifier = 0;
  let assemblyFormat = "$condition attr-dict";
}

// -----

def SPIRV_KHRExpectOp : SPIRV_KhrVendorOp<"Expect",
    [Pure, AllTypesMatch<["value", "expectedValue", "result"]>]> {
  let summary = "Provides a hint to the optimizer that the value is likely "
                "to equal the expected value.";

  let description = [{
    Result Type must be a scalar or vector of integer type or Boolean type.

    Value and Expected Value must be of Result Type. Result is the same as
    Value.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %2 = spirv.KHR.Expect %0, %1 : i32
    %5 = spirv.KHR.Expect %3, %4 : vector<4xi1>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_expect_assume]>,
    Capability<[SPIRV_C_ExpectAssumeKHR]>
  ];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<AnyTypeOf<[SPIRV_Integer, SPIRV_Bool]>>:$value,
    SPIRV_ScalarOrVectorOf<AnyTypeOf<[SPIRV_Integer, SPIRV_Bool]>>:$expectedValue
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<AnyTypeOf<[SPIRV_Integer, SPIRV_Bool]>>:$result
  );

  let hasVerifier = 0;
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

// -----

def SPIRV_UndefOp : SPIRV_Op<"Undef", [Pure]> {
  let summary = "Make an intermediate object whose value is undefined.";

  let description = [{
    Result Type is the type of object to make.

    Each consumption of Result `<id>` yields an arbitrary, possibly different
    bit pattern or abstract value resulting in possibly different concrete,
    abstract, or opaque values.

    #### Example:

    ```mlir
    %0 = spirv.Undef : f32
    %1 = spirv.Undef : !spirv.struct<!spirv.array<4 x vector<4xi32>>>
    ```
  }];

  let arguments = (ins);

  let results = (outs
    SPIRV_Type:$result
  );

  let hasVerifier = 0;

  let hasOpcode = 0;
  let autogenSerialization = 0;

  let assemblyFormat = "attr-dict `:` type($result)";
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_MISC_OPS
