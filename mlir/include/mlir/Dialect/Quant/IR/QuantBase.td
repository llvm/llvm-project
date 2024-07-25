//===- QuantBase.td - Quantization dialect base ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Predicates for types in the Quantization dialect.
//
//===----------------------------------------------------------------------===//

#ifndef QUANT_BASE
#define QUANT_BASE

include "mlir/IR/OpBase.td"

def Quant_Dialect : Dialect {
  let name = "quant";
  let description = [{
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

    // Correct. Type `!quant.uniform` with per-channel quantization is wrapped in
    // a `tensor` type.
    %result = quant.qcast %input : tensor<2xf32> to tensor<2x!quant.uniform<i8:f32:0, {1.0, 2.0}>>
    ```

    - If the tensor containing the `!quant.uniform` type is ranked, its rank
      must be greater than the channel axis specified in the quantized type.

    ```
    // Incorrect. The tensor rank (2) is not greater than the channel axis in the
    // quantized type (3).
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

    // Correct. The quantized type now includes 3 scale values, matching the size
    // of dimension 1 of the result tensor.
    %result = quant.qcast %input : tensor<?x3xf32> to tensor<?x3x!quant.uniform<i8:f32:1, {2.0, 3.0, 4.0}>>
    ```
  }];
  let cppNamespace = "::mlir::quant";
  let useDefaultTypePrinterParser = 1;
}


//===----------------------------------------------------------------------===//
// Type definitions
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
