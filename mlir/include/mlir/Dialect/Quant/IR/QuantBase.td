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
      AnyInteger.predicate,
      AnyFloat.predicate,
      quant_QuantizedType.predicate
    ]>, "integer, float, or quantized scalar">;

def quant_IntegerOrQuantizedType :
    Type<Or<[AnyInteger.predicate, quant_QuantizedType.predicate]>>;

def quant_FloatScalarOrTensor :
    quant_ScalarOrTensorOf<AnyFloat>;

def quant_IntegerScalarOrTensor :
    quant_ScalarOrTensorOf<AnyInteger>;

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

#endif // QUANT_BASE
