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

    The `quant.uniform` type has three variants: per-layer quantization,
    per-channel (or per-axis) quantization, and sub-channel (or blockwize)
    quantization.  In per-layer quantization, the quantization information
    affects an entire tensor uniformly. Conversely, in per-channel
    quantization, the data type encodes the specific tensor axis that serves
    as the channel and includes quantization information for each individual
    channel within the tensor. Sub-channel quantization is a generalization
    of per-tensor and per-channel quantization, where the quantization
    parameters are defined for blocks of elements along one or more
    dimensions of the tensor. Below are the specific syntactic and semantic
    considerations for each modality.


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
    // size 3 matches the number of provided scale values. Tensor elements at
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

    ### Sub-channel quantization

    Sub-channel quantization, also known as blockwise quantization, provides
    finer-grained control than per-tensor or per-channel quantization. It
    divides a tensor into blocks of elements, each with its own quantization
    parameters (scale and zero point). This is particularly useful when
    different regions of a tensor exhibit distinct value ranges.

    The `!quant.uniform` type represents sub-channel quantization with the
    following syntax:

    ```
    `!quant.uniform` `<`
      storedType (`<` storageMin `:` storageMax `>`)? `:`
      expressedType `:` blockSizeInfo
      scaleZeroTensor `>`

    blockSizeInfo ::= `{` `}` | `{` axisBlock (`,` axisBlock)*)? `}`
    axisBlock ::= axis `:` blockSize
    scaleZeroTensor ::= scaleZeroDenseExp | scaleZeroList
    scaleZeroDenseExp ::= `{` scaleZeroTensor (`,` scaleZeroTensor)* `}`
    scaleZeroList  ::= scaleZero (`,` scaleZero)*
    scaleZero ::= scale (`:` zeroPoint)?
    
    scaleZeroTensor ::= scale-zero-dense-exp | scale-zero-list
    scale-zero-dense-exp ::= `{` scale-zero-tensor (`,` scale-zero-tensor)* `}`
    scale-zero-list ::= scale (`:` zeroPoint)? (`,` scale (`:` zeroPoint)?)*
    ```

    The `blockSize` field specifies the size of the blocks along dimension
    `axis` of the tensor. The `scale` and `zeroPoint` fields specify the
    quantization parameters for a particular block. Specifically, the tensor
    element at position [i0...iN] uses
    `scaleZeroTensor[i/blockSize0...i/blockSizeN].scale` and
    `scaleZeroTensor[i/blockSize0...i/blockSizeN].zeroPoint` as scale
    and zeroPoint respectively.

    Here are some examples:

    ```
    // A 3x4 tensor of i8 values representing f32 values, quantized 
    // along axis-0 and axis-1 with block sizes 1 and 2,
    // respectively. As a result, the shape of the scales (or zero-points) will
    // be `[3,4]/[1,2] = [3,2]`, which essentially represents the number of
    // blocks along each axis. Tensor elements at positions 
    // [0][0] and [0][1] use scale `s00` and zero point `z00`,
    // [0][2] and [0][3] use scale `s01` and zero point `z01`,
    // [1][0] and [1][1] use scale `s10` and zero point `z10`,
    // [1][2] and [1][3] use scale `s11` and zero point `z11`,
    // [2][0] and [2][1] use scale `s20` and zero point `z20`,
    // [2][2] and [2][3] use scale `s21` and zero point `z21`,
    tensor<3x4x!quant.uniform<i8:f32:{0:1, 1:2},
      {{s00:z00, s01:z01}, {s10:z10,s11:z11}, {s20:z20,s21:z21}}>>

    // A 2D dynamically sized tensor contains u16 values
    // representing f32 values. Since the shape of the quantization
    // parameters (i.e. scales and zero-points) is given as [2,2] and
    // the blocks-sizes are given as [1,2], the shape of the tensor is expected
    // to be [2,4] (= [2,2] * [1,2]) at runtime. Tensor elements at positions
    // [0][0] and [0][1] use scale `s00` and zero point `z00`,
    // [0][2] and [0][3] use scale `s01` and zero point `z01`,
    // [1][0] and [1][1] use scale `s10` and zero point `z10`,
    // [1][2] and [1][3] use scale `s11` and zero point `z11`,
    tensor<?x?x!quant.uniform<u16:f32:{0:1, 1:2},
      {{s00:z00, s01:z01}, {s10:z10,s11:z11}}>>
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

    ## Sub-channel quantization integrity

    When type `!quant.uniform` contains sub-channel quantization information,
    the following rules are enforced.  For efficiency, these rules are actively
    enforced by the verifiers of `quant` dialect ops, but they must be
    respected in any context in which the `!quant.uniform` data type is used,
    such as the header of a `func.func` op, or the input of an arithmetic
    operation.

    - A quantized type with sub-channel quantization information must be the
      element type of a tensor container type, and may not occur directly as
      the data type of a scalar value.

    ```
    // Incorrect. Type !quant.uniform specifies sub-channel quantization for a
    // scalar type.
    %result = quant.qcast %input : f32 to !quant.uniform<i8:f32:{0:1, 1:2}, {{1.0}, {2.0}}>

    // Correct. Type `!quant.uniform` with sub-channel quantization is wrapped
    // in a `tensor` type.
    %result = quant.qcast %input : tensor<2x2xf32> to
                tensor<2x2x!quant.uniform<i8:f32:{0:1, 1:2}, {{1.0}, {2.0}}>>
    ```

    - The tensor containing the sub-channel quantized type must be ranked.

    ```
    // Incorrect. Type !quant.uniform specifies sub-channel quantization for a
    // unranked tensor type.
    %result = quant.qcast %input : tensor<*xf32> to
                tensor<*x!quant.uniform<i8:f32:{0:1, 1:2}, {{1.0}, {2.0}}>>
    ```

    - The axis for which a block size is specified should be valid for a tensor
    of a given rank. Block sizes can be specified for a subset of axes. 
    Any unspecified block size for an axis i defaults to the tensor dimension
    size of that axis (shape(tensor)[i]).

    ```
    // Incorrect. The block-size is specified for axis 2 which is greater than
    // the rank of the tensor.
    %result = quant.qcast %input : tensor<2x2xf32> to
                tensor<2x2x!quant.uniform<i8:f32:{2:1, 1:2}, {{1.0}, {2.0}}>>

    // Incorrect. The block-size is specified for a negative axis.
    %result = quant.qcast %input : tensor<2x2xf32> to
                tensor<2x2x!quant.uniform<i8:f32:{-1:1, 1:2}, {{1.0}, {2.0}}>>

    // Correct. The block size for axis 1 is skipped which should be assumed as
    // 2, the dim-size of tensor at axis 1.
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:3}, {{1.0}, {3.0}}>>

    // Correct. The block size for all the axes are skipped making the
    // sub-channel type essentially a per-tensor type.
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{}, {{1.0}}>>
    ```

    - Block size for a particular axis should be a positive integer and should
      be less than the dimension size of the tensor along that axis.

    ```
    // Incorrect. The block size for axis 0 is -1.
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:-1}, {{1.0, 2.0}}>>

    // Incorrect. The block size for axis 0 is 8 which is greater than the
    // dimension size of tensor at axis 0 (which is 6).
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:8}, {{1.0, 2.0}}>>

    // Correct. The block size for axis 0 is now 3.
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:3}, {{1.0}, {2.0}}>>
    ```

    - shape(tensor) % blockSizes = 0 where blockSizes = [block sizes for
      axis i in [0, 1, ..., rank(tensor)-1]].

    ```
    // Incorrect. The block size for axis 0 is 4 and the corresponding
    // dimension size is 6 and 6 % 4 != 0.
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:4}, {{1.0, 2.0}}>>

    // Correct. The block size for axis 0 is now 3 making 6 % 3 = 0.
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:3}, {{1.0}, {2.0}}>>
    ```

    - shape(scales) = shape(zeroPoints) = shape(tensor) / blockSizes.

    ```
    // Incorrect. shape(tensor) = [6,2], blockSizes = [3,2], but
    // shape(scales) is [1,2] which is not equal to [6,2]/[3,2].
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:3}, {{1.0, 2.0}}>>

    // Correct. shape(tensor) = [6,2], blockSizes = [3,2], and
    // shape(scales) equals [6,2]/[3,2].
    %result = quant.qcast %input : tensor<6x2xf32> to
                tensor<6x2x!quant.uniform<i8:f32:{0:3}, {{1.0}, {2.0}}>>
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
