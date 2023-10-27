//===- SPIRVIntelExtOps.td - Intel SPIR-V extensions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the op definition spec of Intel-specific SPIR-V extensions
// These extensions are not part of Khronos specification but publicly available
// at (https://github.com/intel/llvm)
// Supported extensions
// * SPV_INTEL_bfloat16_conversion
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS
#define MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS

// -----

def SPIRV_INTELConvertFToBF16Op : SPIRV_IntelVendorOp<"ConvertFToBF16", []> {
  let summary = "See extension SPV_INTEL_bfloat16_conversion";

  let description = [{
    Convert value numerically from 32-bit floating point to bfloat16,
    which is represented as a 16-bit unsigned integer.

    Result Type must be a scalar or vector of integer type.
    The component width must be 16 bits. Bit pattern in the Result represents a bfloat16 value.

    Float Value must be a scalar or vector of floating-point type.
    It must have the same number of components as Result Type. The component width must be 32 bits.

    Results are computed per component.

    ```
    convert-f-to-bf16-op ::= ssa-id `=` `spirv.INTEL.ConvertFToBF16` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertFToBF16 %0 : f32 to i16
    %3 = spirv.ConvertFToBF16 %2 : vector<3xf32> to vector<3xi16>
    ```

  }];


  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_bfloat16_conversion]>,
    Capability<[SPIRV_C_Bfloat16ConversionINTEL]>
  ];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Float32>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Int16>:$result
  );
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasVerifier = 1;
}

// -----

def SPIRV_INTELConvertBF16ToFOp : SPIRV_IntelVendorOp<"ConvertBF16ToF", []> {
  let summary = "See extension SPV_INTEL_bfloat16_conversion";

  let description = [{
    Interpret a 16-bit integer as bfloat16 and convert the value numerically to 32-bit floating point type.

    Result Type must be a scalar or vector of floating-point. The component width must be 32 bits.

    Bfloat16 Value must be a scalar or vector of integer type, which is interpreted as a bfloat16 type.
    The type must have the same number of components as the Result Type. The component width must be 16 bits.

    Results are computed per component.

    ```
    convert-bf16-to-f-op ::= ssa-id `=` `spirv.INTEL.ConvertBF16ToF` ssa-use
                          `:` operand-type `to` result-type
    ```

    #### Example:

    ```mlir
    %1 = spirv.ConvertBF16ToF %0 : i16 to f32
    %3 = spirv.ConvertBF16ToF %2 : vector<3xi16> to vector<3xf32>
    ```

  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_bfloat16_conversion]>,
    Capability<[SPIRV_C_Bfloat16ConversionINTEL]>
  ];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Int16>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float32>:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];
  let hasVerifier = 1;
}


// -----

#endif // MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS