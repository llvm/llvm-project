//===- QuantOps.td - Quantization operation definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the operation definition file for Quantization.
//
//===----------------------------------------------------------------------===//

#ifndef QUANT_OPS
#define QUANT_OPS

include "mlir/Dialect/Quant/IR/QuantBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Base classes
//===----------------------------------------------------------------------===//

class quant_Op<string mnemonic, list<Trait> traits> :
    Op<Quant_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Quantization casts
//===----------------------------------------------------------------------===//

def quant_DequantizeCastOp : quant_Op<"dcast", [
    Pure,
    quant_SameScalarOrTensorShape]> {
  let summary = "Dequantize cast operation";
  let description = [{
    Convert an input quantized value into its expressed floating-point value.
    The dequantization process consists of the following steps:

    ```
    def dequantize(quantizedValue: quantizedType) -> expressedType:
        storedValue = reinterpretCast(quantizedValue, storageType)
        storedValueFloat = convertIntToFloat(storedValue, expressedType)
        zeroPointFloat = convertIntToFloat(zeroPoint, expressedType)
        expressedValue = (storedValueFloat - zeroPointFloat) * scale
        return expressedValue
    ```

    Here, `storageType`, `expressedType`, `scale`, and `zeroPoint` are obtained
    from the corresponding parameters encoded in `quantizedType`. For
    per-channel quantization, the appropriate `scale` and `zeroPoint` values
    are used for each tensor element computation according to the channel the
    element belongs to.
    
    The numerical results produced by the algorithm above may vary depending on
    the rounding methods used by `convertIntToFloat()`, subtraction (`-`), and
    multiplication (`*`). This operation does not define specific rounding
    methods; instead, it is the responsibility of a transform pipeline to
    determine which rounding method to apply when this operation is broken down
    into lower-level dialects.

    The operation must satisfy the following syntactic constraints:

    - Operand `input` must be a scalar or tensor of type `!quant.uniform`.

    - The result type must be a floating-point scalar or tensor.

    - The `expressedType` parameter of the `!quant.uniform` type of the input
      must match the floating-point type of the result.

    - The operand and result types must be both scalars or both tensors. If
      tensors, they must be both ranked or both unranked. If ranked, both must
      have the same shape, including matching static and dynamic dimensions.

    - If the operand uses per-channel quantization, its `!quant.uniform` type
      must adhere to the [Per-axis quantization
      integrity](#per-axis-quantization-integrity) guidelines.

    Examples:

    ```
    // Dequantize a scalar quantized value
    %result = quant.dcast %input : !quant.uniform<i8:f32, 2.0> to f32

    // Dequantize a dynamically shaped tensor of quantized values
    %result = quant.dcast %input : tensor<?x!quant.uniform<i8:f32, 2.0>> to tensor<?xf32>

    // Dequantize an unranked tensor using per-axis quantization information
    %result = quant.dcast %input : tensor<*x!quant.uniform<i8:f32:1, {2.0, 3.0}>> to tensor<*xf32>
    ```
  }];
  let arguments = (ins quant_QuantizedScalarOrTensor:$input);
  let results = (outs quant_FloatScalarOrTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let extraClassDeclaration = [{
    /// Return the float type of the scalar or tensor result.
    FloatType getFloatType();
    
    /// Return the quantized type of the scalar or tensor input.
    quant::QuantizedType getQuantizedType();
  }];
}

def quant_QuantizeCastOp : quant_Op<"qcast", [
    Pure,
    quant_SameScalarOrTensorShape]> {
  let summary = "Quantize cast operation";
  let description = [{
    Convert a floating-point value to a quantized type. The quantization
    process consists of the following steps:

    ```
    def quantize(expressedValue: expressedType) -> quantizedType:
        zeroPointFloat = convertIntToFloat(zeroPoint, expressedType)
        scaledValue = expressedValue / scale
        storedValueFloat = scaledValue + zeroPointFloat
        storedValue = convertFloatToInt(storedValueFloat, storageType)
        storedValueClamped = clamp(storedValue, storageMin, storageMax)
        quantizedValue = reinterpretCast(storedValueClamped, quantizedType)
        return quantizedValue
    ```

    Here, `storageType`, `storageMin`, `storageMax`, `expressedType`, `scale`,
    and `zeroPoint` are obtained from the corresponding parameters encoded in
    `quantizedType`. For per-channel quantization, the appropriate `scale` and
    `zeroPoint` values are used for each tensor element computation according
    to the channel the element belongs to.

    The numerical results produced by the algorithm above may vary depending on
    the rounding methods used by `convertIntToFloat()`, `convertFloatToInt()`,
    `clamp()`, division (`/`), and addition (`+`). This operation does not
    define specific rounding methods; instead, it is the responsibility of a
    transform pipeline to determine which rounding method to apply when this
    operation is broken down into lower-level dialects.

    The operation must satisfy the following syntactic constraints:

    - Operand `input` must be a floating-point scalar or tensor.

    - The result type must be a scalar or tensor of type `!quant.uniform`.

    - The `expressedType` parameter in the `!quant.uniform` type of the result
      must match the floating-point type of the input.

    - The operand and result types must be both scalars or both tensors. If
      tensors, they must be both ranked or both unranked. If ranked, both must
      have the same shape, including matching static and dynamic dimensions.

    - If the result uses per-channel quantization, its `!quant.uniform` type
      must adhere to the [Per-axis quantization
      integrity](#per-axis-quantization-integrity) guidelines.

    Examples:

    ```
    // Quantize a scalar floating-point value
    %result = quant.qcast %input : f32 to !quant.uniform<i8:f32, 2.0>

    // Quantize a dynamically shaped tensor of quantized values
    %result = quant.qcast %input : tensor<?xf32> to tensor<?x!quant.uniform<i8:f32, 2.0>>

    // Quantize an unranked tensor using per-axis quantization information
    %result = quant.qcast %input : tensor<*xf32> to tensor<*x!quant.uniform<i8:f32:1, {2.0, 3.0}>>
    ```
  }];
  let arguments = (ins quant_FloatScalarOrTensor:$input);
  let results = (outs quant_QuantizedScalarOrTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let extraClassDeclaration = [{
    /// Return the float type of the scalar or tensor input.
    FloatType getFloatType();
    
    /// Return the quantized type of the scalar or tensor result.
    quant::QuantizedType getQuantizedType();
  }];
}

def quant_StorageCastOp : quant_Op<"scast", [
    Pure,
    quant_SameScalarOrTensorShape,
    quant_IntegerAndQuantizedCombination]> {
  let summary = "Storage cast operation";
  let description = [{
    Convert a value from a quantized type to the corresponding signless integer
    storage type, or vice versa. This conversion simply involves a
    reinterpretation of the input bits and does not involve any data
    manipulation.

    The following syntactic restrictions must be met:

    - Operand `input` must be a scalar or tensor of a signless integer or
      `!quant.uniform` type.

    - The result must be a scalar or tensor of a signless integer or
      `!quant.uniform` type.

    - If the operand is a scalar or tensor of type integer, the result must be
      a scalar or tensor of type `!quant.uniform`, and vice versa.

    - The operand and result must be both scalars or both tensors. If tensors,
      they must be both ranked or both unranked. If ranked, both must have the
      same shape, including matching static and dynamic dimensions.

    - The width of the `storageType` parameter of the quantized type of the
      operand or result must match the width of the signless integer type of
      the operand or result.

    - If the operand or result uses per-channel quantization, its
      `!quant.uniform` type must adhere to the [Per-axis quantization
      integrity](#per-axis-quantization-integrity) guidelines.

    Examples:

    ```
    // Cast a scalar quantized value into its storage type
    %result = quant.scast %input : !quant.uniform<i8:f32, 2.0> to i8

    // Cast a dynamically shaped tensor of quantized values into their storage type
    %result = quant.scast %input : tensor<?x!quant.uniform<i8:f32, 2.0>> to tensor<?xi8>

    // Cast an unranked tensor of signless integers into a quantized type using
    // per-channel quantization
    %result = quant.scast %input : tensor<*xi8> to tensor<*x!quant.uniform<i8:f32:1, {2.0, 3.0}>>
    ```
  }];
  let arguments = (ins quant_IntegerOrQuantizedScalarOrTensor:$input);
  let results = (outs quant_IntegerOrQuantizedScalarOrTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let extraClassDeclaration = [{
    /// Return the integer type used either in the input or the result.
    IntegerType getIntegerType();
    
    /// Return the quantized type used either in the input or the result.
    quant::QuantizedType getQuantizedType();
  }];
}

#endif // QUANT_OPS
