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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

//===----------------------------------------------------------------------===//
// Tosa Type Definitions.
//===----------------------------------------------------------------------===//

// The base class of a quantized type.
// Param tuple is: [bitwidth, zeropt, smantissa, sexp, low_end, high_end].
// Where low and high ends are 0,255 when unsigned, -128,127 when signed, for
// the 8-bit case.
class Tosa_QuantizedType<string n, list<int> params, bit signed>
  : Type<And<[CPred<"::llvm::isa<mlir::quant::QuantizedType>($_self)">,
              CPred<"::llvm::cast<mlir::quant::QuantizedType>($_self)" #
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

def Tosa_Int4 : I<4>;
def Tosa_Int8 : I<8>;
def Tosa_Int16 : I<16>;
def Tosa_Int32 : I<32>;
def Tosa_Int64 : I<64>;

// The TOSA dialect allows more types than the TOSA standard to allow for
// experimentation. For historical reasons, signless is used in the place of
// signed.
// The TosaValidation pass can be used to check for standard conformance.
def Tosa_Int : AnyTypeOf<[AnyUnsignedInteger,
                          AnySignlessInteger]>;

def Tosa_Int32Or64 : AnyTypeOf<[Tosa_Int32,
                                Tosa_Int64]>;

def Tosa_Int16Or32 : AnyTypeOf<[Tosa_Int16,
                                Tosa_Int32]>;

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
def Tosa_QuantizedInt : AnyTypeOf<[Tosa_QuantizedType<"uint8", [8], 0>,
                                   Tosa_QuantizedType<"int4", [4, 0], 1>,
                                   Tosa_QuantizedType<"int8", [8, 0], 1>,
                                   Tosa_QuantizedType<"int16", [16, 0], 1>,
                                   Tosa_QuantizedType<"int32", [32, 0], 1>]>;

//===----------------------------------------------------------------------===//
// Multi-category types.
//===----------------------------------------------------------------------===//
def Tosa_AnyNumber : AnyTypeOf<[Tosa_Int, Tosa_QuantizedInt, AnyFloat],
                                "number">;

//===----------------------------------------------------------------------===//
// TOSA Tensor Conformance
//===----------------------------------------------------------------------===//

def HasNo0Dimensions : And<[
    IsRankedTensorTypePred,
    CPred<"::llvm::all_of(::llvm::cast<::mlir::RankedTensorType>($_self).getShape(), [](auto v) { return v != 0; })">]>;

def AllDimensionsAreSizeOne : And<[
    IsRankedTensorTypePred,
    CPred<"::llvm::all_of(::llvm::cast<::mlir::RankedTensorType>($_self).getShape(), [](auto v) { return v == 1; })">]>;

def AtLeastRankOne : And<[
  IsRankedTensorTypePred,
  CPred<"::llvm::cast<::mlir::RankedTensorType>($_self).getRank() >= 1">]>;

class TosaTensorOf<
    list<Type> allowedTypes, string summary = "tosa-conformant tensor">
    : TensorOf<allowedTypes, [Or<[HasNo0Dimensions, IsUnrankedTensorTypePred]>], summary>;

class TosaRankedTensorOf<
    list<Type> allowedTypes, list<Pred> preds = [], string summary = "tosa-conformant ranked tensor">
    : RankedTensorOf<allowedTypes, !listconcat([HasNo0Dimensions], preds), summary>;

class TosaUnrankedTensorOf<list<Type> allowedTypes, list<Pred> preds = [], string summary = "tosa-conformant unranked tensor">
    : UnrankedTensorOf<allowedTypes, preds, summary>;

class TosaTensorRankOf<list<Type> allowedTypes, list<int> ranks>
    : TosaRankedTensorOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tensor">;

class TosaScalarTensorOf<list<Type> allowedTypes, list<int> ranks>
    : TosaRankedTensorOf<allowedTypes,
      [HasAnyRankOfPred<ranks>, AllDimensionsAreSizeOne],
      "tosa-conformant scalar tensor">;

//===----------------------------------------------------------------------===//
// Tensor types
//===----------------------------------------------------------------------===//

def Tosa_I1Tensor : TosaTensorOf<[I1]>;
def Tosa_Int32Tensor : TosaTensorOf<[Tosa_Int32]>;
def Tosa_Int32Or64Tensor : TosaTensorOf<[Tosa_Int32Or64]>;

def Tosa_FloatTensor : TosaTensorOf<[AnyFloat]>;

// Either ranked or unranked tensor of TOSA supported element types.
def Tosa_Tensor : TosaTensorOf<[Tosa_AnyNumber]>;

// Must be ranked but no further constraints
def Tosa_RankedTensor : TosaRankedTensorOf<[Tosa_AnyNumber]>;

// Any tensor element type allowed in Tosa ops.
def Tosa_ElementType : Type<Or<[Tosa_Int.predicate, Tosa_QuantizedInt.predicate,
                                AnyFloat.predicate]>, "tosa.dtype">;

class Tosa_TensorOfOrNone<list<Type> allowedTypes, string description = ""> :
  AnyTypeOf<[TosaTensorOf<allowedTypes>, NoneType], description>;

//===----------------------------------------------------------------------===//
// Tensor types with constrained ranks.
//===----------------------------------------------------------------------===//

def Tosa_Rank0Tensor : TosaTensorRankOf<[Tosa_AnyNumber], [0]>;

def Tosa_ScalarTensor : TosaScalarTensorOf<[Tosa_AnyNumber], [1]>;
def Tosa_ScalarInt8Tensor : TosaScalarTensorOf<[Tosa_Int8], [1]>;
def Tosa_ScalarIntOrFloatTensor : TosaScalarTensorOf<[Tosa_Int, AnyFloat], [1]>;

// We include unranked tensors as a supported type for all possible tosa
// Tensors as unranked does not guarantee invalid. If unranked tensors exist
// they should be shape propagate used Tosa's shape inference pass and verified
// to not include any remaining unranked tensors.
def Tosa_UnrankedTensor : TosaUnrankedTensorOf<[Tosa_AnyNumber]>;

def Tosa_Tensor1D : AnyTypeOf<[Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [1]>], "1-d tosa-conformant tensor", "::mlir::TensorType">;
def Tosa_Tensor2D : AnyTypeOf<[Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [2]>], "2-d tosa-conformant tensor", "::mlir::TensorType">;
def Tosa_Tensor3D : AnyTypeOf<[Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [3]>], "3-d tosa-conformant tensor", "::mlir::TensorType">;
def Tosa_Tensor4D : AnyTypeOf<[Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [4]>], "4-d tosa-conformant tensor", "::mlir::TensorType">;
def Tosa_Tensor5D : AnyTypeOf<[Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [5]>], "5-d tosa-conformant tensor", "::mlir::TensorType">;

// 1D tensor of specific types
def Tosa_1DInt8Tensor : 1DTensorOf<[Tosa_Int8]>;
def Tosa_1DInt16Or32Tensor : 1DTensorOf<[Tosa_Int16Or32]>;

// Ranked tensors up to given rank.
def Tosa_Tensor1Dto4D : AnyTypeOf<[
  Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [1,2,3,4]>]>;
def Tosa_Tensor1Dto6D : AnyTypeOf<[
  Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [1,2,3,4,5,6]>]>;

def Tosa_TensorUpto4D : AnyTypeOf<[
  Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_AnyNumber], [0,1,2,3,4]>]>;

def Tosa_Int32TensorUpto4D : AnyTypeOf<[
  Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_Int32], [0,1,2,3,4]>]>;
def Tosa_Int32Tensor2D : AnyTypeOf<[
  Tosa_UnrankedTensor, TosaTensorRankOf<[Tosa_Int32], [2]>]>;

def Tosa_TensorAtLeast1D : AnyTypeOf<[
  Tosa_UnrankedTensor, TosaRankedTensorOf<[Tosa_AnyNumber], [AtLeastRankOne]>], "tosa-conformant tensor of at least rank 1", "::mlir::TensorType">;

//===----------------------------------------------------------------------===//
// Generic scalar, vector, or tensor of a particular type.
//===----------------------------------------------------------------------===//

class Tosa_TypeLike<list<Type> types, string description = ""> : TypeConstraint<Or<[
     AnyTypeOf<types>.predicate,
     VectorOfNonZeroRankOf<types>.predicate,
     TosaTensorOf<types>.predicate]>,
     description>;

def Tosa_IntLike : Tosa_TypeLike<[Tosa_Int], "signless-integer-like">;
def Tosa_Int8Like : Tosa_TypeLike<[Tosa_Int8], "signless-integer-8-bit-like">;

//===----------------------------------------------------------------------===//
// Attribute predicates and classes.
//===----------------------------------------------------------------------===//

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

def Tosa_FloatAttr : Attr<CPred<"::llvm::isa<::mlir::FloatAttr>($_self)">,
                          "arbitrary float attribute"> {
  let storageType = [{ ::mlir::FloatAttr }];
  let returnType = [{ ::mlir::APFloat }];
}

def Tosa_IntegerAttr : Attr<CPred<"::llvm::isa<::mlir::IntegerAttr>($_self)">,
                          "arbitrary integer attribute"> {
  let storageType = [{ ::mlir::IntegerAttr }];
  let returnType = [{ ::llvm::APInt }];
}

def Tosa_IntOrFloatAttr : AnyAttrOf<[Tosa_IntegerAttr, Tosa_FloatAttr]>;

//===----------------------------------------------------------------------===//
// Iterable attributes.
//===----------------------------------------------------------------------===//
// Defined in `section 3. Enumerations` of the TOSA specification.

// Supported regimes for tosa.resize.
def Tosa_ResizeTypeAttr : StringBasedAttr<
    CPred<"::llvm::cast<StringAttr>($_self).getValue() == \"BILINEAR\"  || " #
          "::llvm::cast<StringAttr>($_self).getValue() == \"NEAREST_NEIGHBOR\"">,
    "Supported resize/upsampling strategies">;

// Supported NaN propagation strategies.
def Tosa_NanPropagationAttr : StringBasedAttr<
    CPred<"::llvm::cast<StringAttr>($_self).getValue() == \"PROPAGATE\"  || " #
          "::llvm::cast<StringAttr>($_self).getValue() == \"IGNORE\"">,
    "Supported NaN propagation strategies">;

// Rounding mode for tosa.rescale
def Tosa_RoundingTypeAttr : StringBasedAttr<
    CPred<"::llvm::cast<StringAttr>($_self).getValue() == \"SINGLE_ROUND\"  || " #
          "::llvm::cast<StringAttr>($_self).getValue() == \"INEXACT_ROUND\" || " #
          "::llvm::cast<StringAttr>($_self).getValue() == \"DOUBLE_ROUND\"">,
    "Supported rounding modes">;

def Tosa_TensorTypeAttr : TypeAttrBase<"TensorType", "Tensor type attribute">;

// Tensor to buffer types.
def Tosa_Buffer : MemRefOf<[Tosa_AnyNumber]>;
def Tosa_TupleBuffer : NestedTupleOf<[Tosa_Buffer]>;
def Tosa_BufOrTuple : AnyTypeOf<[Tosa_Buffer, Tosa_TupleBuffer]>;

//===----------------------------------------------------------------------===//
// Tosa Type Definitions.
//===----------------------------------------------------------------------===//

// The base class for Tosa dialect types.
class Tosa_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tosa_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//
def Tosa_Shape : Tosa_Type<"shape", "shape"> {
  let summary = "Shape with static rank and Index element type";
  let description = [{
    Syntax:

    ``` shape - type :: = `shape` `<` rank `>`
    ``` Values with shape type represents a shape with a fixed rank and a list
                               of dimensions
                                   .Rank must be zero or a positive integer
                                   .Each dimension is represented by the builtin
                                       Index type.

    Examples:

    ```mlir
     // Shape with rank of four, for example, [1, 1, 8, 16]:
     !tosa
         .shape<4>

     // Shape with rank of one, for example, [16]:
     !tosa
         .shape<1>

     // Shape with rank zero, for example, [] (i.e., shape of scalar values):
     !tosa.shape<0>
    ```
  }];
  let parameters = (ins "int" : $rank);
  let builders = [TypeBuilder<(ins "int" : $rank)>];
  let assemblyFormat = "`<` $rank `>`";

  let genVerifyDecl = 1;
}

def IsTosaShapeType : CPred<"mlir::tosa::isa_tosa_shape_type($_self)">;

// Whether a Tosa Shape type has a rank equal to the specified rank.
class IsTosaShapeOfRankPred<int rank> : And<[
  IsTosaShapeType,
  CPred<[{::llvm::cast<::mlir::tosa::shapeType>($_self).getRank() == }] # rank>
]>;

class TosaShapeOfRank<int rank>
    : Type<IsTosaShapeOfRankPred<rank>, "Tosa shape type of rank " #rank>;

def Rank1TosaShape : TosaShapeOfRank<1>;
def Rank2TosaShape : TosaShapeOfRank<2>;
def Rank4TosaShape : TosaShapeOfRank<4>;

#endif // TOSA_TYPES_BASE
