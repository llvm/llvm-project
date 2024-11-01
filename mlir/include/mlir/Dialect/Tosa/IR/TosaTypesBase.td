//===-- TosaTypesBase.td - TOSA type definitions -----------*- tablegen -*-===//
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

#ifndef TOSA_TYPES_BASE
#define TOSA_TYPES_BASE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Tosa Type Definitions.
//===----------------------------------------------------------------------===//

// The base class of a quantized type.
// Param tuple is: [bitwidth, zeropt, smantissa, sexp, low_end, high_end].
// Where low and high ends are 0,255 when unsigned, -128,127 when signed, for
// the 8-bit case.
class Tosa_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"$_self.isa<mlir::quant::QuantizedType>()">,
              CPred<"$_self.cast<mlir::quant::QuantizedType>()" #
                    ".getStorageTypeIntegralWidth() == " # !head(params)>]>,
    "Q" # !if (signed, "int", "uint") # !head(params) # " type"> {
  string name = n;
  string asTraitArgsStr = !interleave(params, ", ") #
                          !if(signed, ", true", ", false");
}

//===----------------------------------------------------------------------===//
// Non-Quantized Signed Integer Types.
// Used to express accumulator results or compare results.
//===----------------------------------------------------------------------===//

def Tosa_UInt8 : UI<8>;
def Tosa_UInt16 : UI<16>;

def Tosa_Int8 : I<8>;
def Tosa_Int16 : I<16>;
def Tosa_Int32 : I<32>;
def Tosa_Int48 : I<48>;
def Tosa_Int64 : I<64>;

def Tosa_SignedInt : AnyTypeOf<[Tosa_Int8,
                                Tosa_Int16,
                                Tosa_Int32,
                                Tosa_Int48,
                                Tosa_Int64]>;

def Tosa_Bool : I<1>;

// No unsigned unquantized int types.
def Tosa_Int : AnyTypeOf<[Tosa_Bool,
                          Tosa_UInt8,
                          Tosa_UInt16,
                          Tosa_SignedInt]>;

def Tosa_Int32Or64 : AnyTypeOf<[Tosa_Int32,
                   	        Tosa_Int64]>;

//===----------------------------------------------------------------------===//
// Quantized Integer Types.
// Datatype for network feature map or weight content.
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//
// Name    Symmetry   Grouping                Sign
//===----------------------------------------------------------------------===//
// uint8 : asymmetric per tensor ,            unsigned
// int4  : symmetric  per channel,            signed
// int8  : symmetric  per tensor/per channel, signed
// int16 : symmetric  per tensor,             signed
//===----------------------------------------------------------------------===//
def Tosa_QuantizedInt	: AnyTypeOf<[ Tosa_QuantizedType<"uint8", [8], 0>,
                                     Tosa_QuantizedType<"int4", [4, 0], 1>,
                                     Tosa_QuantizedType<"int8", [8, 0], 1>,
                                     Tosa_QuantizedType<"int16", [16, 0], 1>,
                                     Tosa_QuantizedType<"int32", [32, 0], 1>]>;

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//
def Tosa_Float : AnyTypeOf<[
                            F32,
			    F16,
			    BF16]>;

//===----------------------------------------------------------------------===//
// Multi-category types.
//===----------------------------------------------------------------------===//
def Tosa_AnyNumber : AnyTypeOf<[Tosa_Int, Tosa_QuantizedInt, Tosa_Float],
                               "number">;
// Add F64 type support for just for tosa::CastOp
def Tosa_AnyNumber_Cast : AnyTypeOf<[Tosa_Int, Tosa_QuantizedInt, Tosa_Float, F64],
                               "number_cast">;

//===----------------------------------------------------------------------===//
// Tensor types
//===----------------------------------------------------------------------===//

def Tosa_Int32Tensor : TensorOf<[Tosa_Int32]>;
def Tosa_Int32Or64Tensor : TensorOf<[Tosa_Int32Or64]>;

// Either ranked or unranked tensor of TOSA supported element types.
def Tosa_Tensor : TensorOf<[Tosa_AnyNumber]>;
def Tosa_Tensor_Cast : TensorOf<[Tosa_AnyNumber_Cast]>;
// Must be ranked but no further constraints
def Tosa_RankedTensor : RankedTensorOf<[Tosa_AnyNumber]>;

// Any tensor element type allowed in Tosa ops.
def Tosa_ElementType : Type<Or<[Tosa_Int.predicate, Tosa_QuantizedInt.predicate,
                                Tosa_Float.predicate]>, "tosa.dtype">;

class Tosa_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TensorOf<allowedTypes>, NoneType], description>;

//===----------------------------------------------------------------------===//
// Tensor types with constrained ranks.
//===----------------------------------------------------------------------===//

// Rank-0 (scalar) tensor
def Tosa_ScalarTensor : TensorRankOf<[Tosa_AnyNumber], [0]>;

// We include unranked tensors as a supported type for all possible tosa
// Tensors as unranked does not guarantee invalid. If unranked tensors exist
// they should be shape propagate used Tosa's shape inference pass and verified
// to not include any remaining unranked tensors.
def Tosa_UnrankedTensor : UnrankedTensorOf<[Tosa_AnyNumber]>;

def Tosa_Tensor1D : AnyTypeOf<[Tosa_UnrankedTensor, 1DTensorOf<[Tosa_AnyNumber]>]>;
def Tosa_Tensor2D : AnyTypeOf<[Tosa_UnrankedTensor, 2DTensorOf<[Tosa_AnyNumber]>]>;
def Tosa_Tensor3D : AnyTypeOf<[Tosa_UnrankedTensor, 3DTensorOf<[Tosa_AnyNumber]>]>;
def Tosa_Tensor4D : AnyTypeOf<[Tosa_UnrankedTensor, 4DTensorOf<[Tosa_AnyNumber]>]>;
def Tosa_Tensor5D : AnyTypeOf<[Tosa_UnrankedTensor, TensorRankOf<[Tosa_AnyNumber], [5]>]>;

// Ranked tensors up to given rank.
def Tosa_Tensor1Dto4D : AnyTypeOf<[
  Tosa_UnrankedTensor, TensorRankOf<[Tosa_AnyNumber], [1,2,3,4]>]>;
def Tosa_Tensor1Dto6D : AnyTypeOf<[
  Tosa_UnrankedTensor, TensorRankOf<[Tosa_AnyNumber], [1,2,3,4,5,6]>]>;

def Tosa_TensorUpto4D : AnyTypeOf<[
  Tosa_UnrankedTensor, TensorRankOf<[Tosa_AnyNumber], [0,1,2,3,4]>]>;

def Tosa_Int32TensorUpto4D : AnyTypeOf<[
  Tosa_UnrankedTensor, TensorRankOf<[Tosa_Int32], [0,1,2,3,4]>]>;

//===----------------------------------------------------------------------===//
// Generic scalar, vector, or tensor of a particular type.
//===----------------------------------------------------------------------===//

class Tosa_TypeLike<list<Type> types, string description = ""> : TypeConstraint<Or<[
     AnyTypeOf<types>.predicate,
     VectorOf<types>.predicate,
     TensorOf<types>.predicate]>,
     description>;

def Tosa_IntLike : Tosa_TypeLike<[Tosa_Int], "signless-integer-like">;
def Tosa_Int8Like : Tosa_TypeLike<[Tosa_Int8], "signless-integer-8-bit-like">;
def Tosa_Int16Like : Tosa_TypeLike<[Tosa_Int16], "signless-integer-16-bit-like">;
def Tosa_Int32Like : Tosa_TypeLike<[Tosa_Int32], "signless-integer-32-bit-like">;
def Tosa_Int64Like : Tosa_TypeLike<[Tosa_Int64], "signless-integer-64-bit-like">;

//===----------------------------------------------------------------------===//
// Attribute predicates and classes.
//===----------------------------------------------------------------------===//
class DenseArrayMaxCt<int n> : AttrConstraint<
    CPred<"$_self.cast<::mlir::DenseArrayAttr>().size() <= " # n>,
    "with at least " # n # " elements">;

def Tosa_Fp32ArrayAttr2 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<2>]>;
def Tosa_Fp32ArrayAttr3 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<3>]>;
def Tosa_Fp32ArrayAttr4 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<4>]>;
def Tosa_Fp32ArrayAttr5 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<5>]>;
def Tosa_Fp32ArrayAttr6 : ConfinedAttr<DenseF32ArrayAttr, [DenseArrayCount<6>]>;

def Tosa_IntArrayAttr2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<2>]>;
def Tosa_IntArrayAttr3 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<3>]>;
def Tosa_IntArrayAttr4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>;
def Tosa_IntArrayAttr5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<5>]>;
def Tosa_IntArrayAttr6 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<6>]>;

def Tosa_IntArrayAttrUpto2 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<2>]>;
def Tosa_IntArrayAttrUpto4 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<4>]>;
def Tosa_IntArrayAttrUpto5 : ConfinedAttr<DenseI64ArrayAttr, [DenseArrayMaxCt<5>]>;

//===----------------------------------------------------------------------===//
// Iterable attributes.
//===----------------------------------------------------------------------===//
// Supported regimes for tosa.resize.
def Tosa_ResizeTypeAttr : StringBasedAttr<
    CPred<"$_self.cast<StringAttr>().getValue() == \"BILINEAR\"  || " #
          "$_self.cast<StringAttr>().getValue() == \"NEAREST_NEIGHBOR\"">,
    "Supported resize/upsampling strategies">;

def Tosa_TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

// Tensor to buffer types.
def Tosa_Buffer : MemRefOf<[Tosa_AnyNumber]>;
def Tosa_TupleBuffer : NestedTupleOf<[Tosa_Buffer]>;
def Tosa_BufOrTuple : AnyTypeOf<[Tosa_Buffer, Tosa_TupleBuffer]>;

#endif // TOSA_TYPES_BASE
