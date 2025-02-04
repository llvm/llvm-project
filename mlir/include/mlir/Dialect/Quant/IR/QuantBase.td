//===- QuantBase.td - Quantization dialect base ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Quantization dialect, types, and traits.
//
//===----------------------------------------------------------------------===//

#ifndef QUANT_BASE
#define QUANT_BASE

include "mlir/IR/OpBase.td"

def Quant_Dialect : Dialect {
  let name = "quant";
  let description = [{
    The `quant` dialect offers a framework for defining and manipulating
    quantized values. Central to this framework is the `!quant.uniform` data
    type, used to represent quantized values. This dialect also provides a
    suite of operations to handle and convert quantized values between their
    original floating-point representations and the optimized, lower bit-width
    integer representations. The `quant` dialect is instrumented with
    transformation passes to lower these operations into other core MLIR
    dialects, while also flattening all occurrences of quantized types into
    their integer counterparts.


    ## The `!quant.uniform` type

    The quantization process establishes a relationship between two types of
    values: an *expressed value* and a *stored value*. The former refers to the
    floating-point representation used in an original machine learning model,
    capturing the precise numerical characteristics needed for accurate
    calculations. The latter is the simplified integer representation that
    resides in memory after quantization. The `!quant.uniform` data type
    encodes the necessary information for (lossy) round-trip conversion between
    an expressed and a stored value.

    The `quant.uniform` type has two variants: per-layer quantization and
    per-channel (or per-axis) quantization. In per-layer quantization, the
    quantization information affects an entire tensor uniformly. Conversely, in
    per-channel quantization, the data type encodes the specific tensor axis
    that serves as the channel and includes quantization information for each
    individual channel within the tensor. Below are the specific syntactic and
    semantic considerations for each modality.


    ### Per-layer quantization

    This is the general syntax of the `!quant.uniform` type representing
    per-layer quantization:

    ```
    `!quant.uniform` `<`
      storedType (`<` storageMin `:` storageMax `>`)? `:`
      expressedType `,`
      scale (`:` zeroPoint)?
    `>`
    ```

    The type contains the following parameters:

    - `storedType`: Integer type of the value stored in memory. This type
      conveys the bit width and signedness of the quantized stored value.
      Signed integer types are represented as `'i' bitWidth` (e.g., `i8`),
      while unsigned integer types are represented as `'u' bitWidth` (e.g.,
      `u8`).

    - `storageMin`, `storageMax`: Optional bounds for the stored value. If
      given, they must be within the range of `storedType`. If omitted, the
      entire range of `storedType` is allowed (e.g., `-128...127` for `i8` or
      `0...255` for `u8`).

    - `expressedType`: Floating-point type of the value expressed by this
      quantized type (e.g., `f32`, `f80`, `bf16`, or `tf32`).

    - `scale`: Floating-point value of type `expressedType` used in the
      conversion between stored and expressed values.

    - `zeroPoint`: Optional integer value of type `storageType` used in the
      conversion between stored and expressed values. If omitted, the default
      is 0.

    Type conversions, rounding methods, and clamping actions aside, the
    relationship between the expressed and stored values as encoded in a
    quantized type is denoted by the following formula:

    $$
    expressedValue = (storedValue ~-~ zeroPoint) ~\times~ scale
    $$

    Operations `quant.qcast` (quantize cast) and `quant.dcast` (dequantize
    cast) can be used to quantize a floating-point value and dequantize a
    stored value, respectively. See the documentation for these operations for
    details on how the quantization and dequantization processes are influenced
    by the `!quant.uniform` type parameters.

    Here are some examples of the use of `!quant.uniform` with per-layer
    quantization:

    ```
    // An 8-bit signed integer type is used to represent a 32-bit float. No
    // clamping information is provided, so the full [-128, 127] range is
    // available. The scale is set to 3.0, and the zero point takes its default
    // 0 value.
    !quant.uniform<i8:f32, 3.0>

    // A 16-bit unsigned integer type is used to represent a 32-bit float. Out
    // of the 16 bits, only 10 are used, acoording to the 0..1023 clamping
    // range. The type sets the scale to 1.23 and the zero point to 512.
    !quant.uniform<u16<0:1023>:f32, 1.23:512>
    ```

    ### Per-channel quantization

    The general syntax of the `!quant.uniform` type representing per-channel
    quantization is as follows:

    ```
    `!quant.uniform` `<`
      storedType (`<` storageMin `:` storageMax `>`)? `:`
      expressedType `:`
      channelAxis `,`
      `{`
        scale0 (`:` zeroPoint0)? `,`
        scale1 (`:` zeroPoint1)? ...
      '}'
    `>`
    ```

    In this data type, there are multiple pairs of `scale` and `zeroPoint`
    values. The `channelAxis` field represents the dimension of the containing
    tensor acting as the channel. The size of the tensor along this dimension
    is expected to match the number of provided `scale`-`zeroPoint` pairs, and
    a given pair *i* applies to all elements in the tensor whose index along
    dimension `channelAxis` is *i*. A quantized data type using per-channel
    quantization is always expected to be contained within a tensor type.

    Here are some examples:

    ```
    // A 2x3x4 tensor contains 8-bit signed integers representing 32-bit
    // floats. Dimension 1 of the tensor acts as the channel dimension. Its
    // size 3 matches the number of provided scale values. Tensor elemenets at
    // positions [*][0][*], [*][1][*], and [*][2][*] use scales 3.0, 4.0, and
    // 5.0, respectively.
    tensor<2x3x4x!quant.uniform<i8:f32:1, {3.0, 4.0, 5.0}>>

    // A 2D dynamically sized tensor contains 16-bit unsigned integers
    // representing 32-bit floats. Dimension 0 of the tensor acts as the
    // channel dimension. Since 2 scale and zero-point values are provided, the
    // size of dimension 0 is expected to be 2 at runtime. Tensor elements
    // [0][*] use scale 2.0 and zero point 10, while elements [1][*] use scale
    // 3.0 and zero point 20.
    tensor<?x?x!quant.uniform<u16:f32:0, {2.0:10, 3.0:20}>>
    ```


    ## Per-axis quantization integrity

    When type `!quant.uniform` contains per-axis quantization information, the
    rules below are enforced. These rules guarantee that the quantization
    information encoded in the data type is applicable to the context in which
    the quantized type is used. For efficiency, these rules are actively
    enforced by the verifiers of `quant` dialect ops, but they must be
    respected in any context in which the `!quant.uniform` data type is used,
    such as the header of a `func.func` op, or the input of an arithmetic
    operation.
 
    - A quantized type with per-channel quantization information must be the
      element type of a tensor container type, and may not occur directly as
      the data type of a scalar value.

    ```
    // Incorrect. Type !quant.uniform specifies per-channel quantization for a
    // scalar type.
    %result = quant.qcast %input : f32 to !quant.uniform<i8:f32:0, {1.0, 2.0}>

    // Correct. Type `!quant.uniform` with per-channel quantization is wrapped
    // in a `tensor` type.
    %result = quant.qcast %input : tensor<2xf32> to tensor<2x!quant.uniform<i8:f32:0, {1.0, 2.0}>>
    ```

    - If the tensor containing the `!quant.uniform` type is ranked, its rank
      must be greater than the channel axis specified in the quantized type.

    ```
    // Incorrect. The tensor rank (2) is not greater than the channel axis in
    // the quantized type (3).
    %result = quant.qcast %input : tensor<1x2xf32> to tensor<1x2x!quant.uniform<i8:f32:3, {1.0, 2.0}>>

    // Correct. The tensor rank (2) is now greater than the channel axis (1):
    %result = quant.qcast %input : tensor<1x2xf32> to tensor<1x2x!quant.uniform<i8:f32:1, {1.0, 2.0}>>
    ```

    - If the axis dimension in the containing tensor is static, its size must
      be equal to the number of scales present in the quantized type.

    ```
    // Incorrect. The channel axis is 1, and the size of dimension 1 in the
    // containing tensor is 3. However, there are 4 scale values present in the
    // quantized type.
    %result = quant.qcast %input : tensor<?x3xf32> to tensor<?x3x!quant.uniform<i8:f32:1, {1.0, 2.0, 3.0, 4.0}>>

    // Correct. The quantized type now includes 3 scale values, matching the
    // size of dimension 1 of the result tensor.
    %result = quant.qcast %input : tensor<?x3xf32> to tensor<?x3x!quant.uniform<i8:f32:1, {2.0, 3.0, 4.0}>>
    ```
  }];
  let cppNamespace = "::mlir::quant";
  let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

class quant_ScalarOrTensorOf<Type etype> :
    Type<Or<[etype.predicate, TensorOf<[etype]>.predicate]>,
         "scalar or tensor of " # etype.summary>;

def quant_QuantizedType :
    Type<CPred<"::llvm::isa<mlir::quant::QuantizedType>($_self)">, "quantized type">;

def quant_ScalarType :
    Type<Or<[
      AnySignlessInteger.predicate,
      AnyFloat.predicate,
      quant_QuantizedType.predicate
    ]>,
    "signless integer, float, or quantized scalar">;

def quant_IntegerOrQuantizedType :
    Type<Or<[
      AnySignlessInteger.predicate,
      quant_QuantizedType.predicate
    ]>,
    "signless integer or quantized type">;

def quant_FloatScalarOrTensor :
    quant_ScalarOrTensorOf<AnyFloat>;

def quant_IntegerScalarOrTensor :
    quant_ScalarOrTensorOf<AnySignlessInteger>;

def quant_QuantizedScalarOrTensor :
    quant_ScalarOrTensorOf<quant_QuantizedType>;

def quant_IntegerOrQuantizedScalarOrTensor :
    quant_ScalarOrTensorOf<quant_IntegerOrQuantizedType>;


//===----------------------------------------------------------------------===//
// Traits
//===----------------------------------------------------------------------===//

def quant_SameScalarOrTensorShape :
    PredOpTrait<
      "input and result are both scalars or both tensors with matching shape",
      Or<[
        And<[
          TypeIsPred<"input", quant_ScalarType>,
          TypeIsPred<"result", quant_ScalarType>
        ]>,
        And<[
          TypeIsPred<"input", AnyUnrankedTensor>,
          TypeIsPred<"result", AnyUnrankedTensor>
        ]>,
        And<[
          TypeIsPred<"input", AnyRankedTensor>,
          TypeIsPred<"result", AnyRankedTensor>,
          AllShapesMatch<["input", "result"]>.predicate
        ]>
      ]>
    >;

def quant_IntegerAndQuantizedCombination :
    PredOpTrait<
      "input must be integer and result must be quantized, or vice versa",
      Or<[
        And<[
          TypeIsPred<"input", quant_QuantizedScalarOrTensor>,
          TypeIsPred<"result", quant_IntegerScalarOrTensor>
        ]>,
        And<[
          TypeIsPred<"input", quant_IntegerScalarOrTensor>,
          TypeIsPred<"result", quant_QuantizedScalarOrTensor>
        ]>
      ]>
    >;

#endif // QUANT_BASE
