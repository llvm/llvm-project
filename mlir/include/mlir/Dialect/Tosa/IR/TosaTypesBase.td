//===-- TosaTypesBase.td - TOSA type definitions *- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type definitions for the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifdef TOSA_TYPES_BASE
#else
#define TOSA_TYPES_BASE

include "mlir/IR/OpBase.td"

///////////////////////////
// Tosa Type Definitions
///////////////////////////

// The base class of a quantized type.
// Param tuple is: [bitwidth, zeropt, smantissa, sexp, low_end, high_end]
// Where low and high ends are 0,255 when unsigned, -128,127 when signe, for the 8-bit case
class Tosa_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"$_self.isa<mlir::quant::QuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>]>,
    "Q" # !if (signed, "int", "uint") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr =
    StrJoinInt<params>.result # !if(signed, ", true", ", false");
}

/* Non-Quantized Signed Integer Types
   Used to express accumulator results or compare results */

// Booleans are currently assumed to be expressed using int8 or
// built in U1 / U1Tensor type

def Tosa_Int32 : SI<32>;
def Tosa_Int48 : SI<48>;
def Tosa_Int64 : SI<64>;

// Any signed integer type
def Tosa_SignedInt : AnyTypeOf<[Tosa_Int32,
                                Tosa_Int48,
                                Tosa_Int64]>;

def Tosa_Int : AnyTypeOf<[Tosa_SignedInt]>;

def Tosa_Int32Or64 : AnyTypeOf<[Tosa_Int32,
                   	        Tosa_Int64]>;

// Any integer tensor types
def Tosa_IntTensor : TensorOf<[Tosa_SignedInt]>;

// Any integer tensor types
def Tosa_Int32Or64Tensor : TensorOf<[Tosa_Int32Or64]>;

/* Quantized Integer Types
   Datatype for network feature map or weight content */

def Tosa_Quint8 : Tosa_QuantizedType<"Uniform", [8], 0>;
def Tosa_Qint8  : Tosa_QuantizedType<"Uniform", [8], 1>;
def Tosa_Qint16 : Tosa_QuantizedType<"Uniform", [16], 1>;

// Any quantized type
// aint8 : asymmetric per tensor, signed
// uint8: asymmetric per tensor , unsigned
// int4: symmetric per channel, signed
// int8 : symmetric per tensor/per channel, signed
// int16: symmetric per tensor, signed
def Tosa_QuantizedInt	: AnyTypeOf<[Tosa_QuantizedType<"aint8", [8], 1>,
                                     Tosa_QuantizedType<"uint8", [8], 0>,
                                     Tosa_QuantizedType<"int4", [4, 0], 1>,
                                     Tosa_QuantizedType<"int8", [8, 0], 1>,
                                     Tosa_QuantizedType<"int16", [16, 0], 1>]>;

/* Floating-point types */

def Tosa_Float : AnyTypeOf<[F32,
			    F16,
			    BF16]>;

def Tosa_FpTensor : TensorOf<[Tosa_Float]>;

// Multi-category type constraints

def Tosa_AnyNumber : AnyTypeOf<[Tosa_Int, Tosa_QuantizedInt, Tosa_Float],
                               "number">;

// Tensors exclusively of numerical types
def Tosa_Tensor : TensorOf<[Tosa_AnyNumber]>;

// Tensors exclusively of quantized types
def Tosa_QTypeTensor : TensorOf<[Tosa_QuantizedInt]>;

class Tosa_TensorOrNone<list<Type> possibleTypes, string description = ""> :
    AnyTypeOf<[TensorOf<possibleTypes>, NoneType], description>;

// Any tensor element type allowed in Tosa ops
def Tosa_ElementType : Type<Or<[Tosa_Int.predicate, Tosa_QuantizedInt.predicate, Tosa_Float.predicate]>,
                          "tosa.dtype">;

// Tensor or None type.
class Tosa_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

// Any Tosa tensor type including string and bool
def Tosa_AnyTensor : TensorOf<[Tosa_ElementType]>;

// String attribute constraints

def Tosa_ResizeTypeAttr : StringBasedAttr<
    CPred<"$_self.cast<StringAttr>().getValue() == \"TRANSPOSE\" || " #
          "$_self.cast<StringAttr>().getValue() == \"BILINEAR\"  || " #
          "$_self.cast<StringAttr>().getValue() == \"NEAREST_NEIGHBOR\"">,
    "Supported resize/upsampling strategies">;

def Tosa_TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

// Tensor to buffer
def Tosa_Buffer : MemRefOf<[Tosa_AnyNumber]>;
def Tosa_TupleBuffer : NestedTupleOf<[Tosa_Buffer]>;
def Tosa_BufOrTuple : AnyTypeOf<[Tosa_Buffer, Tosa_TupleBuffer]>;

#endif // TOSA_TYPES_BASE
