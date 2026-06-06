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
// * SPV_INTEL_tensor_float32_conversion
//===----------------------------------------------------------------------===//


#ifndef MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS
#define MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS

// -----

def SPIRV_INTELConvertFToBF16Op : SPIRV_IntelVendorOp<"ConvertFToBF16", [SameOperandsAndResultShape]> {
  let summary = "See extension SPV_INTEL_bfloat16_conversion";

  let description = [{
    Convert value numerically from 32-bit floating point to bfloat16,
    which is represented as a 16-bit unsigned integer.

    Result Type must be a scalar or vector of integer type.
    The component width must be 16 bits. Bit pattern in the Result represents a bfloat16 value.

    Float Value must be a scalar or vector of floating-point type.
    It must have the same number of components as Result Type. The component width must be 32 bits.

    Results are computed per component.

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

  let hasVerifier = 0;
}

// -----

def SPIRV_INTELConvertBF16ToFOp : SPIRV_IntelVendorOp<"ConvertBF16ToF", [SameOperandsAndResultShape]> {
  let summary = "See extension SPV_INTEL_bfloat16_conversion";

  let description = [{
    Interpret a 16-bit integer as bfloat16 and convert the value numerically to 32-bit floating point type.

    Result Type must be a scalar or vector of floating-point. The component width must be 32 bits.

    Bfloat16 Value must be a scalar or vector of integer type, which is interpreted as a bfloat16 type.
    The type must have the same number of components as the Result Type. The component width must be 16 bits.

    Results are computed per component.

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

  let hasVerifier = 0;
}

// -----

def SPIRV_INTELRoundFToTF32Op : SPIRV_IntelVendorOp<"RoundFToTF32", [SameOperandsAndResultShape]> {
  let summary = "See extension SPV_INTEL_tensor_float32_conversion";

  let description = [{
    Convert value numerically from a 32-bit floating point type to tensor float32,
    with rounding to the nearest even.

    Result Type must be a scalar or vector of 32-bit floating-point type.
    The component width must be 32 bits. Bit pattern in the Result represents a tensor float32 value.

    Float Value must be a scalar or vector of floating-point type.
    It must have the same number of components as Result Type. The component width must be 32 bits.

    Results are computed per component.

    #### Example:

    ```mlir
    %1 = spirv.RoundFToTF32 %0 : f32 to f32
    %3 = spirv.RoundFToTF32 %2 : vector<3xf32> to vector<3xf32>
    ```

  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_tensor_float32_conversion]>,
    Capability<[SPIRV_C_TensorFloat32RoundingINTEL]>
  ];

  let arguments = (ins
    SPIRV_ScalarOrVectorOf<SPIRV_Float32>:$operand
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float32>:$result
  );

  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `to` type($result)
  }];

  let hasVerifier = 0;
}

// -----

class SPIRV_IntelSplitBarrierOp<string mnemonic>
    : SPIRV_IntelVendorOp<mnemonic, []> {
  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_split_barrier]>,
    Capability<[SPIRV_C_SplitBarrierINTEL]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$memory_semantics
  );

  let results = (outs);

  let assemblyFormat = [{
    $execution_scope $memory_scope $memory_semantics attr-dict
  }];

  let hasVerifier = 0;
}

def SPIRV_INTELControlBarrierArriveOp
    : SPIRV_IntelSplitBarrierOp<"ControlBarrierArrive"> {
  let summary = "See extension SPV_INTEL_split_barrier";

  let description = [{
    Indicates that an invocation has arrived at a split control barrier. This
    may allow other invocations waiting on the split control barrier to continue
    executing.

    When `Execution` is `Workgroup` or larger, behavior is undefined unless all
    invocations within `Execution` execute the same dynamic instance of this
    instruction. When `Execution` is `Subgroup` or `Invocation`, the behavior of
    this instruction in non-uniform control flow is defined by the client API.

    If `Semantics` is not `None`, this instruction also serves as the start of a
    memory barrier similar to an `OpMemoryBarrier` instruction with the same
    `Memory` and `Semantics` operands. This allows atomically specifying both a
    control barrier and a memory barrier (that is, without needing two
    instructions). If `Semantics` is `None`, `Memory` is ignored.

    #### Example:

    ```mlir
    spirv.ControlBarrierArrive <Workgroup> <Device> <Acquire|UniformMemory>
    ```
  }];
}


// -----

def SPIRV_INTELControlBarrierWaitOp
    : SPIRV_IntelSplitBarrierOp<"ControlBarrierWait"> {
  let summary = "See extension SPV_INTEL_split_barrier";

  let description = [{
    Waits for other invocations of this module to arrive at a split control
    barrier.

    When `Execution` is `Workgroup` or larger, behavior is undefined unless all
    invocations within `Execution` execute the same dynamic instance of this
    instruction. When `Execution` is `Subgroup` or `Invocation`, the behavior of
    this instruction in non-uniform control flow is defined by the client API.

    If `Semantics` is not `None`, this instruction also serves as the end of a
    memory barrier similar to an `OpMemoryBarrier` instruction with the same
    `Memory` and `Semantics` operands. This ensures that memory accesses issued
    before arriving at the split barrier are observed before memory accesses
    issued after this instruction. This control is ensured only for memory
    accesses issued by this invocation and observed by another invocation
    executing within `Memory` scope. This allows atomically specifying both a
    control barrier and a memory barrier (that is, without needing two
    instructions). If `Semantics` is `None`, `Memory` is ignored.

    #### Example:

    ```mlir
    spirv.ControlBarrierWait <Workgroup> <Device> <Acquire|UniformMemory>
    ```
  }];
}


// -----

def SPIRV_INTELMaskedGatherOp : SPIRV_IntelVendorOp<"MaskedGather",
    [AllTypesMatch<["fill_empty", "result"]>,
     TypesMatchWith<"pointee type of ptr_vector must match result element type",
                    "ptr_vector", "result",
                    "VectorType::get("
                      "::llvm::cast<VectorType>($_self).getShape(), "
                      "::llvm::cast<spirv::PointerType>("
                        "::llvm::cast<VectorType>($_self).getElementType())"
                      ".getPointeeType())">,
     TypesMatchWith<"mask must be a vector of i1 matching result shape",
                    "result", "mask",
                    "getMatchingBoolType($_self)">]> {
  let summary = "Gather values from memory using a vector of pointers and a mask";

  let description = [{
    Reads values from a vector of pointers gathering them into a result
    vector. Lanes where the mask is false receive the corresponding
    FillEmpty value.

    Result Type must be a vector of numerical type.

    PtrVector must be a vector of pointers to the scalar element type of
    Result Type. It must have the same number of components as Result Type.

    Alignment is the known minimum alignment in bytes of each pointer in
    PtrVector.

    Mask must be a vector of boolean type with the same number of components
    as Result Type.

    FillEmpty must have the same type as Result Type.

    #### Example:

    ```mlir
    %result = spirv.INTEL.MaskedGather %ptrs, %alignment, %mask, %fill
              : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
                vector<4xi1>, vector<4xf32> -> vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_masked_gather_scatter]>,
    Capability<[SPIRV_C_MaskedGatherScatterINTEL]>
  ];

  let arguments = (ins
    SPIRV_VectorOf<SPIRV_AnyPtr>:$ptr_vector,
    SPIRV_Int32:$alignment,
    SPIRV_VectorOf<SPIRV_Bool>:$mask,
    SPIRV_VectorOf<SPIRV_Numerical>:$fill_empty
  );

  let results = (outs
    SPIRV_VectorOf<SPIRV_Numerical>:$result
  );

  let assemblyFormat = [{
    $ptr_vector `,` $alignment `,` $mask `,` $fill_empty attr-dict `:`
      type($ptr_vector) `,` type($alignment) `,`
      type($mask) `,` type($fill_empty) `->` type($result)
  }];

  let hasVerifier = 0;
}

// -----

def SPIRV_INTELMaskedScatterOp : SPIRV_IntelVendorOp<"MaskedScatter",
    [TypesMatchWith<"pointee type of ptr_vector must match input element type",
                    "ptr_vector", "input_vector",
                    "VectorType::get("
                      "::llvm::cast<VectorType>($_self).getShape(), "
                      "::llvm::cast<spirv::PointerType>("
                        "::llvm::cast<VectorType>($_self).getElementType())"
                      ".getPointeeType())">,
     TypesMatchWith<"mask must be a vector of i1 matching input shape",
                    "input_vector", "mask",
                    "getMatchingBoolType($_self)">]> {
  let summary = "Scatter values to memory using a vector of pointers and a mask";

  let description = [{
    Writes values from a vector into memory locations pointed to by a
    vector of pointers. Only lanes where the mask is true are written.

    PtrVector must be a vector of pointers to the scalar element type of
    InputVector. It must have the same number of components as InputVector.

    Alignment is the known minimum alignment in bytes of each pointer in
    PtrVector.

    Mask must be a vector of boolean type with the same number of components
    as InputVector.

    InputVector is the vector of values to scatter into memory.

    #### Example:

    ```mlir
    spirv.INTEL.MaskedScatter %ptrs, %alignment, %mask, %values
              : vector<4x!spirv.ptr<f32, CrossWorkgroup>>, i32,
                vector<4xi1>, vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_masked_gather_scatter]>,
    Capability<[SPIRV_C_MaskedGatherScatterINTEL]>
  ];

  let arguments = (ins
    SPIRV_VectorOf<SPIRV_AnyPtr>:$ptr_vector,
    SPIRV_Int32:$alignment,
    SPIRV_VectorOf<SPIRV_Bool>:$mask,
    SPIRV_VectorOf<SPIRV_Numerical>:$input_vector
  );

  let results = (outs);

  let assemblyFormat = [{
    $ptr_vector `,` $alignment `,` $mask `,` $input_vector attr-dict `:`
      type($ptr_vector) `,` type($alignment) `,`
      type($mask) `,` type($input_vector)
  }];

  let hasVerifier = 0;
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_INTEL_EXT_OPS
