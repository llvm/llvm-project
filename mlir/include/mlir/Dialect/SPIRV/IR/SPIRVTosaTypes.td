//===- SPIRVTosaTypes.td - Tosa Types insts spec file --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This specifies Tosa types used by the Graph Extension and Tosa Ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_TOSA_TYPES
#define MLIR_DIALECT_SPIRV_IR_TOSA_TYPES

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

def SPIRV_I8OrI16 : AnyIntOfWidths<[8, 16]>;
def SPIRV_I8OrI16OrI32 : AnyIntOfWidths<[8, 16, 32]>;
def SPIRV_I8OrI16OrI32OrI64 : AnyIntOfWidths<[8, 16, 32, 64]>;
def SPIRV_I16OrI32 : AnyIntOfWidths<[16, 32]>;
def SPIRV_I32OrI64 : AnyIntOfWidths<[32, 64]>;
def SPIRV_F16OrF32OrBF16 : AnyTypeOf<[SPIRV_Float16, SPIRV_Float32, SPIRV_BFloat16KHR]>;
def SPIRV_F16OrF32OrBF16OrFP8 : AnyTypeOf<[SPIRV_Float16, SPIRV_Float32, SPIRV_BFloat16KHR, SPIRV_Float8E4M3EXT, SPIRV_Float8E5M2EXT]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16 : AnyTypeOf<[SPIRV_I8OrI16, SPIRV_F16OrF32OrBF16]>;
def SPIRV_I8OrF16OrF32OrBF16OrFP8 : AnyTypeOf<[SPIRV_Int8, SPIRV_F16OrF32OrBF16OrFP8]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8 : AnyTypeOf<[SPIRV_I8OrI16, SPIRV_F16OrF32OrBF16OrFP8]>;
def SPIRV_I32OrF16OrF32OrBF16 : AnyTypeOf<[SPIRV_Int32, SPIRV_F16OrF32OrBF16]>;
def SPIRV_I8OrI16OrI32OrF16OrF32OrBF16 : AnyTypeOf<[SPIRV_I8OrI16OrI32, SPIRV_F16OrF32OrBF16]>;
def SPIRV_I8OrI16OrI32OrF16OrF32OrBF16OrFP8 : AnyTypeOf<[SPIRV_I8OrI16OrI32, SPIRV_F16OrF32OrBF16OrFP8]>;
def SPIRV_I32OrI64OrF16OrF32OrBF16 : AnyTypeOf<[SPIRV_I32OrI64, SPIRV_F16OrF32OrBF16]>;
def SPIRV_I32OrI64OrF16OrF32 : AnyTypeOf<[SPIRV_I32OrI64, SPIRV_Float16, SPIRV_Float32]>;
def SPIRV_I32OrI8OrI64OrI16OrF16OrF32OrBF16 : AnyTypeOf<[SPIRV_I8OrI16OrI32OrI64, SPIRV_F16OrF32OrBF16]>;
def SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16 : AnyTypeOf<[SPIRV_Bool, SPIRV_I8OrI16OrI32, SPIRV_F16OrF32OrBF16]>;
def SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16OrFP8 : AnyTypeOf<[SPIRV_Bool, SPIRV_I8OrI16OrI32, SPIRV_F16OrF32OrBF16OrFP8]>;
def SPIRV_I8OrI16OrI32OrBoolOrF16OrF32OrBF16OrFP8 : AnyTypeOf<[SPIRV_I8OrI16OrI32, SPIRV_Bool, SPIRV_F16OrF32OrBF16OrFP8]>;
def SPIRV_I8OrI32 : AnyTypeOf<[SPIRV_Int8, SPIRV_Int32]>;

def SPIRV_TensorArmAxisAttr : ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<5>]>;
def SPIRV_BoolConstAttr : ConfinedAttr<BoolAttr, []>;
def SPIRV_I8OrI16OrF16OrF32OrBF16ConstAttr : AnyAttrOf<[I8Attr, I16Attr, F16Attr, F32Attr, BF16Attr]>;

// TensorARM Types

class RankedTensorArmOf<list<Type> allowedTypes, list<Pred> preds = [],
                     string summary = "ranked tensorArm">
  : ShapedContainerType<
      allowedTypes, And<!listconcat([SPIRV_IsTensorArmType], preds)>,
      summary, "::mlir::spirv::TensorArmType">;

class TensorArmRankOf<list<Type> allowedTypes, list<int> ranks>
  : RankedTensorArmOf<allowedTypes,
      [HasAnyRankOfPred<ranks>],
      !interleave(!foreach(rank, ranks, rank # "D"), "/") # " tensorArm">;

def SPIRV_I8_TensorArm1D : TensorArmRankOf<[SPIRV_Int8], [1]>;
def SPIRV_I16OrI32_TensorArm1D : TensorArmRankOf<[SPIRV_I16OrI32], [1]>;
def SPIRV_I32_TensorArm2D : TensorArmRankOf<[SPIRV_Int32], [2]>;
def SPIRV_F32_TensorArm3D: TensorArmRankOf<[SPIRV_Float32], [3]>;
def SPIRV_I32OrI64OrF16OrF32OrBF16_TensorArm1D : TensorArmRankOf<[SPIRV_I32OrI64OrF16OrF32OrBF16], [1]>;
def SPIRV_I8OrI16_TensorArm1D : TensorArmRankOf<[SPIRV_I8OrI16], [1]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8_TensorArm3D : TensorArmRankOf<[SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8], [3]>;
def SPIRV_I8OrI16OrI32OrF16OrF32OrBF16OrFP8_TensorArm3D : TensorArmRankOf<[SPIRV_I8OrI16OrI32OrF16OrF32OrBF16OrFP8], [3]>;
def SPIRV_I32OrI64OrF16OrF32_TensorArm3D : TensorArmRankOf<[SPIRV_I32OrI64OrF16OrF32], [3]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16_TensorArm4D : TensorArmRankOf<[SPIRV_I8OrI16OrF16OrF32OrBF16], [4]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8_TensorArm4D : TensorArmRankOf<[SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8], [4]>;
def SPIRV_I8OrF16OrF32OrBF16OrFP8_TensorArm4D : TensorArmRankOf<[SPIRV_I8OrF16OrF32OrBF16OrFP8], [4]>;
def SPIRV_I32OrI64OrF16OrF32OrBF16_TensorArm4D : TensorArmRankOf<[SPIRV_I32OrI64OrF16OrF32OrBF16], [4]>;
def SPIRV_I32OrI8OrI64OrI16OrF16OrF32OrBF16_TensorArm4D : TensorArmRankOf<[SPIRV_I32OrI8OrI64OrI16OrF16OrF32OrBF16], [4]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8_TensorArm5D : TensorArmRankOf<[SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8], [5]>;
def SPIRV_I8OrF16OrF32OrBF16OrFP8_TensorArm5D : TensorArmRankOf<[SPIRV_I8OrF16OrF32OrBF16OrFP8], [5]>;
def SPIRV_I32OrI64OrF16OrF32OrBF16_TensorArm5D : TensorArmRankOf<[SPIRV_I32OrI64OrF16OrF32OrBF16], [5]>;

def SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16_TensorArm : TensorArmRankOf<[SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16], [1, 2, 3, 4, 5, 6]>;
def SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16OrFP8_TensorArm : TensorArmRankOf<[SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16OrFP8], [1, 2, 3, 4, 5, 6]>;
def SPIRV_F16OrF32OrBF16_TensorArm : TensorArmRankOf<[SPIRV_F16OrF32OrBF16], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16OrF16OrF32OrBF16], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I32OrF16OrF32OrBF16_TensorArm : TensorArmRankOf<[SPIRV_I32OrF16OrF32OrBF16], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16OrI32OrBoolOrF16OrF32OrBF16OrFP8_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16OrI32OrBoolOrF16OrF32OrBF16OrFP8], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16OrI32OrF16OrF32OrBF16_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16OrI32OrF16OrF32OrBF16], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16OrI32OrI64_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16OrI32OrI64], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16OrI32_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16OrI32], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI16_TensorArm : TensorArmRankOf<[SPIRV_I8OrI16], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I8OrI32_TensorArm : TensorArmRankOf<[SPIRV_I8OrI32], [1, 2, 3, 4, 5, 6]>;
def SPIRV_Bool_TensorArm : TensorArmRankOf<[SPIRV_Bool], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I32_TensorArm : TensorArmRankOf<[SPIRV_Int32], [1, 2, 3, 4, 5, 6]>;
def SPIRV_I32_TensorArmUpTo5D : TensorArmRankOf<[SPIRV_Int32], [1, 2, 3, 4, 5]>;

class Is1DTensorArmOfLength<list<int> allowedLengths> :
  And<[HasAnyRankOfPred<[1]>,
       Or<!foreach(allowedlength, allowedLengths,
                   CPred<[{::llvm::cast<::mlir::spirv::TensorArmType>($_self).getShape()[0] == }]
                         # allowedlength>)>]>;

class SPIRV_1DTensorArmOfLengthAndType<list<int> allowedLengths, list<Type> allowedTypes> :
  ContainerType<AnyTypeOf<allowedTypes>, Is1DTensorArmOfLength<allowedLengths>,
    "::llvm::cast<::mlir::spirv::TensorArmType>($_self).getElementType()",
    "rank 1 tensorArm of length " # !interleave(allowedLengths, "/"),
    "::mlir::spirv::TensorArmType">;

def SPIRV_I32_1DTensorArmOfLength2 : SPIRV_1DTensorArmOfLengthAndType<[2], [SPIRV_Int32]>;
def SPIRV_I32_1DTensorArmOfLength4 : SPIRV_1DTensorArmOfLengthAndType<[4], [SPIRV_Int32]>;

def SPIRV_I32_1DTensorArmOfLength1To6 : SPIRV_1DTensorArmOfLengthAndType<[1, 2, 3, 4, 5, 6], [SPIRV_Int32]>;
def SPIRV_I32_1DTensorArmOfEvenLength2To12 : SPIRV_1DTensorArmOfLengthAndType<[2, 4, 6, 8, 10, 12], [SPIRV_Int32]>;

def SPIRV_DenseElementAttrsWithTensorArmType : AttrConstraint<
  CPred<"::llvm::isa<::mlir::spirv::TensorArmType>(::llvm::cast<::mlir::DenseElementsAttr>($_self).getType())">,
  "Attr with type = spirv::TensorArmType">;

def SPIRV_I32_1DTensorArmOfLength2Attr : ConfinedAttr<RankedI32ElementsAttr<[2]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_I32_1DTensorArmOfLength3Attr : ConfinedAttr<RankedI32ElementsAttr<[3]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_I32_1DTensorArmOfLength4Attr : ConfinedAttr<RankedI32ElementsAttr<[4]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_I32_1DTensorArmOfLength5Attr : ConfinedAttr<RankedI32ElementsAttr<[5]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_I32_1DTensorArmOfLength6Attr : ConfinedAttr<RankedI32ElementsAttr<[6]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
class IntElementsAttrAllValuesAtLeast<int minValue> : AttrConstraint<
  CPred<"::llvm::all_of(::llvm::cast<::mlir::DenseElementsAttr>($_self).getValues<::llvm::APInt>(), "
        "[](const ::llvm::APInt &value) { return value.getSExtValue() >= " #
        minValue # "; })">,
  "all values must be >= " # minValue>;

def SPIRV_PositiveI32_1DTensorArmOfLength2Attr : ConfinedAttr<RankedI32ElementsAttr<[2]>, [SPIRV_DenseElementAttrsWithTensorArmType, IntElementsAttrAllValuesAtLeast<1>]>;
def SPIRV_PositiveI32_1DTensorArmOfLength3Attr : ConfinedAttr<RankedI32ElementsAttr<[3]>, [SPIRV_DenseElementAttrsWithTensorArmType, IntElementsAttrAllValuesAtLeast<1>]>;
def SPIRV_NonNegativeI32_1DTensorArmOfLength4Attr : ConfinedAttr<RankedI32ElementsAttr<[4]>, [SPIRV_DenseElementAttrsWithTensorArmType, IntElementsAttrAllValuesAtLeast<0>]>;
def SPIRV_NonNegativeI32_1DTensorArmOfLength6Attr : ConfinedAttr<RankedI32ElementsAttr<[6]>, [SPIRV_DenseElementAttrsWithTensorArmType, IntElementsAttrAllValuesAtLeast<0>]>;

class Is1DTensorArmAttrOfLength<list<int> allowedLengths> :
  AttrConstraint<And<[CPred<[{::llvm::cast<::mlir::spirv::TensorArmType>(::llvm::cast<::mlir::DenseElementsAttr>($_self).getType()).getShape().size() == 1 }]>,
    Or<!foreach(allowedlength, allowedLengths,
      CPred<[{::llvm::cast<::mlir::spirv::TensorArmType>(::llvm::cast<::mlir::DenseElementsAttr>($_self).getType()).getShape()[0] == }]
              # allowedlength>)>]>>;

def SPIRV_I32_1DTensorArmOfLength1To6Attr : ConfinedAttr<
  I32ElementsAttr, [SPIRV_DenseElementAttrsWithTensorArmType, Is1DTensorArmAttrOfLength<[1, 2, 3, 4, 5, 6]>]>;

def SPIRV_I8_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_Int8]>;
def SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_I8OrI16OrF16OrF32OrBF16OrFP8]>;
def SPIRV_I8OrI16OrI32OrF16OrF32OrBF16_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_I8OrI16OrI32OrF16OrF32OrBF16]>;
def SPIRV_I8OrI16OrI32OrI64_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_I8OrI16OrI32OrI64]>;
def SPIRV_I8OrI16OrI32_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_I8OrI16OrI32]>;
def SPIRV_I8OrF16OrF32OrBF16OrFP8_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_I8OrF16OrF32OrBF16OrFP8]>;
def SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16OrFP8_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_BoolOrI8OrI16OrI32OrF16OrF32OrBF16OrFP8]>;

// Struct type

class IsStructOfNumElementsPred<int numElements> :
  And<[SPIRV_IsStructType,
      CPred<[{::llvm::cast<::mlir::spirv::StructType>($_self).getNumElements()
              == }]
            # numElements>]>;

class IsStructOfNumElementsAndType<int numElements, list<Type> allowedTypes>
    : MixedContainerType<AnyTypeOf<allowedTypes>, IsStructOfNumElementsPred<numElements>,
                         "::llvm::cast<::mlir::spirv::StructType>($_self).getElementTypes()",
                         "Struct">;

def SPIRV_Struct_2_F32_TensorArm3D : IsStructOfNumElementsAndType<2, [SPIRV_F32_TensorArm3D]>;

// Op Trait constraints:

class Implies<Pred left, list<Pred> right>: Or<[Neg<left>, Or<right>]>;

class TypeConstraintImplicationOn<string name, Type type, string other, list<Type> allowedTypes>:
  PredOpTrait<"if " # name # " has type " # type.summary # " then " #
              other # " must have a type in [" #
              !interleave(!foreach(type, allowedTypes, type.summary), ",") # "]",
    Implies<ElementTypeIsPred<name, type>,
    !foreach(allowedType, allowedTypes, ElementTypeIsPred<other, allowedType>)>>;

class BoolAttrTypeConstraintImplicationOn<string boolAttr, string other, list<Type> allowedTypes>:
  PredOpTrait<"if " # boolAttr # " is true then " #
              other # " must have a type in [" #
              !interleave(!foreach(type, allowedTypes, type.summary), ",") # "]",
    Implies<CPred<"get" # snakeCaseToCamelCase<boolAttr>.ret # "()" >,
    !foreach(allowedType, allowedTypes, ElementTypeIsPred<other, allowedType>)>>;

class AxisValueLessThanRankOf<string input>:
PredOpTrait<"axis attribute value should be lower than rank(" # input # ")",
  Implies<CPred<HasRank<input>.result>, [CPred<"getAxis() < " # Rank<input>.result>]>>;

class OutputRankIsInputRankMinusOne<string input, string output>:
PredOpTrait<output # " rank must be equal to max(1, rank(" # input # "))",
  Implies<
      And<[CPred<HasRank<input>.result>, CPred<HasRank<input>.result>]>,
      [CPred<"std::max(int64_t(1), " # Rank<input>.result # " - int64_t(1)) == " # Rank<output>.result>]>>;

class DimOfType<string type, int dim> :
  StrFunc<type # ".getDimSize(" # dim # ")">;

class ShapedTypeOf<string input> :
  StrFunc<"::llvm::cast<::mlir::ShapedType>($" # input # ".getType())">;

class DimOf<string input, int dim> :
  DimOfType<ShapedTypeOf<input>.result, dim>;

class DimIsDynamic<string input, int dim> :
  CPred<"::mlir::ShapedType::isDynamic(" # DimOf<input, dim>.result # ")">;

class DimOfTypeIsDynamic<string type, int dim> :
  CPred<"::mlir::ShapedType::isDynamic(" #
        DimOfType<type, dim>.result # ")">;

class DimIsOne<string input, int dim> :
  CPred<DimOf<input, dim>.result # " == 1">;

class DimsMatch<string lhs, int lhsDim, string rhs, int rhsDim> :
  CPred<DimOf<lhs, lhsDim>.result # " == " # DimOf<rhs, rhsDim>.result>;

class DimsOfTypeMatch<string lhsType, int lhsDim, string rhsType,
                      int rhsDim> :
  CPred<DimOfType<lhsType, lhsDim>.result # " == " #
        DimOfType<rhsType, rhsDim>.result>;

class ProductDimsMatch<string lhs, int lhsDim, string rhs, int rhsDim,
                       string output, int outputDim> :
  CPred<DimOf<lhs, lhsDim>.result # " * " # DimOf<rhs, rhsDim>.result #
        " == " # DimOf<output, outputDim>.result>;

class SameDimsOrDynamicPred<string lhs, int lhsDim, string rhs, int rhsDim> :
  Or<[
    DimIsDynamic<lhs, lhsDim>,
    DimIsDynamic<rhs, rhsDim>,
    DimsMatch<lhs, lhsDim, rhs, rhsDim>
  ]>;

class SameDimsOrOneOrDynamicPred<string lhs, int lhsDim, string rhs, int rhsDim> :
  Or<[
    SameDimsOrDynamicPred<lhs, lhsDim, rhs, rhsDim>,
    DimIsOne<lhs, lhsDim>
  ]>;

class ProductDimOrOneOrDynamicPred<string lhs, int lhsDim, string rhs,
                                   int rhsDim, string output, int outputDim> :
  Or<[
    DimIsDynamic<lhs, lhsDim>,
    DimIsDynamic<rhs, rhsDim>,
    DimIsDynamic<output, outputDim>,
    ProductDimsMatch<lhs, lhsDim, rhs, rhsDim, output, outputDim>,
    DimIsOne<output, outputDim>
  ]>;

class AttrIsFloat<string attr> :
  CPred<"::llvm::isa<::mlir::FloatAttr>(get" #
        snakeCaseToCamelCase<attr>.ret # "())">;

class AttrIsInteger<string attr> :
  CPred<"::llvm::isa<::mlir::IntegerAttr>(get" #
        snakeCaseToCamelCase<attr>.ret # "())">;

class IntegerAttrValue<string attr> :
  StrFunc<"::llvm::cast<::mlir::IntegerAttr>(get" #
          snakeCaseToCamelCase<attr>.ret # "()).getValue()">;

class FloatAttrValue<string attr> :
  StrFunc<"::llvm::cast<::mlir::FloatAttr>(get" #
          snakeCaseToCamelCase<attr>.ret # "()).getValue()">;

class FloatAttrIsNaN<string attr> :
  And<[AttrIsFloat<attr>,
       CPred<FloatAttrValue<attr>.result # ".isNaN()">]>;

class ReductionDimMatchPred<string input, string output, int dim> :
  Or<[
    DimIsDynamic<input, dim>,
    DimIsDynamic<output, dim>,
    And<[CPred<"getAxis() == " # dim>, DimIsOne<output, dim>]>,
    And<[CPred<"getAxis() != " # dim>, DimsMatch<input, dim, output, dim>]>
  ]>;

class ReductionOutputShapeMatchesInputAndAxis<string input, string output>:
PredOpTrait<"shape of " # output # " must match the shape of " # input #
                " with the dimension at axis replaced by 1",
  And<!foreach(dim, [0, 1, 2, 3, 4, 5],
               Implies<
                 And<[CPred<HasRank<input>.result>,
                      CPred<HasRank<output>.result>,
                      CPred<Rank<input>.result # " == " # Rank<output>.result>,
                      CPred<Rank<input>.result # " > " # dim>]>,
                 [ReductionDimMatchPred<input, output, dim>]>)>>;

class AccTypeIn<list<string> allowedValues> :
  CPred<"llvm::is_contained({" # !interleave(!foreach(value, allowedValues, "::mlir::spirv::TosaExtAccType::" # value), ",") # "}, getAccType())">;

class TypeImpliesAccType<string input, Type type, list<string> allowedAccTypes>:
  PredOpTrait<"acc_type must be one in [" # !interleave(allowedAccTypes, ",") # "] when type has value " # type.summary,
  Implies<ElementTypeIsPred<input, type>, [AccTypeIn<allowedAccTypes>]>>;

class MatchBroadcastableShapes<string input1, string input2, string output>:
  PredOpTrait<"the shape of " # input1 # " and " # input2 # " are compatible for broadcasting and the broadcast shape is equal to the " # output # " shape",
  Implies<And<[CPred<HasRank<input1>.result>, CPred<HasRank<input2>.result>, CPred<HasRank<output>.result>,
               CPred<Rank<input1>.result # " == " # Rank<input2>.result # " && " # Rank<input1>.result # " == " # Rank<output>.result>]>,
    [CPred<"llvm::all_of_zip(" # Shape<input1>.result # ", " # Shape<input2>.result # ", " # Shape<output>.result # ", " #
    "[](int64_t input1Dim, int64_t input2Dim, int64_t outputDim) { " #
    "   bool dynamic = ShapedType::isDynamic(input1Dim) || ShapedType::isDynamic(input2Dim) || ShapedType::isDynamic(outputDim);"
    "   bool broadcastableInputs = input1Dim == input2Dim || input1Dim == 1 || input2Dim == 1;" #
    "   bool broacastDimMatchesOutputDim = std::max(input1Dim, input2Dim) == outputDim;"
    "   return dynamic || (broadcastableInputs && broacastDimMatchesOutputDim);" #
    "})">]>
  >;

class ValuesIndicesShapesMatch<string values, string indices, string tensor>:
  PredOpTrait<"shapes of " # values # ", " # indices # ", and " # tensor #
                  " must satisfy [N,K,C], [N,W], [N,W,C]",
    And<[
      SameDimsOrDynamicPred<values, 0, indices, 0>,
      SameDimsOrDynamicPred<values, 0, tensor, 0>,
      SameDimsOrDynamicPred<indices, 0, tensor, 0>,
      SameDimsOrDynamicPred<indices, 1, tensor, 1>,
      SameDimsOrDynamicPred<values, 2, tensor, 2>
    ]>>;

// The tensor shapes are [N,H,W,C] where N,H,W,C are the dimension values.
class NHWCInputOutputShapeMatch<string input, string output>:
  PredOpTrait<"shapes of " # input # " and " # output #
                  " must satisfy [N,*,*,C] and [N,*,*,C]",
    And<[
      SameDimsOrDynamicPred<input, 0, output, 0>,
      SameDimsOrDynamicPred<input, 3, output, 3>
    ]>>;


class Conv2DShapeMatch<string input, string weight, string bias, string output>:
  PredOpTrait<"shapes of " # input # ", " # weight # ", " # bias # ", and " #
                  output # " must satisfy [N,IH,IW,IC], [OC,KH,KW,IC], "
                  "[OC/1], [N,OH,OW,OC]",
    And<[
      SameDimsOrDynamicPred<input, 0, output, 0>,
      SameDimsOrDynamicPred<input, 3, weight, 3>,
      SameDimsOrDynamicPred<weight, 0, output, 3>,
      SameDimsOrOneOrDynamicPred<bias, 0, output, 3>
    ]>>;

class Conv3DShapeMatch<string input, string weight, string bias, string output>:
  PredOpTrait<"shapes of " # input # ", " # weight # ", " # bias # ", and " #
                  output # " must satisfy [N,ID,IH,IW,IC], [OC,KD,KH,KW,IC], "
                  "[OC/1], [N,OD,OH,OW,OC]",
    And<[
      SameDimsOrDynamicPred<input, 0, output, 0>,
      SameDimsOrDynamicPred<input, 4, weight, 4>,
      SameDimsOrDynamicPred<weight, 0, output, 4>,
      SameDimsOrOneOrDynamicPred<bias, 0, output, 4>
    ]>>;

class DepthwiseConv2DShapeMatch<string input, string weight, string bias,
                                string output>:
  PredOpTrait<"shapes of " # input # ", " # weight # ", " # bias # ", and " #
                  output # " must satisfy [N,IH,IW,IC], [KH,KW,IC,M], "
                  "[IC*M/1], [N,OH,OW,IC*M]",
    And<[
      SameDimsOrDynamicPred<input, 0, output, 0>,
      SameDimsOrDynamicPred<input, 3, weight, 2>,
      ProductDimOrOneOrDynamicPred<input, 3, weight, 3, output, 3>,
      SameDimsOrOneOrDynamicPred<bias, 0, output, 3>
    ]>>;

class FetchNthIntElementsAttr<string attrName, int idx> :
  StrFunc<"get" # snakeCaseToCamelCase<attrName>.ret # "().getValues<APInt>()[" # idx # "].getSExtValue()">;

class ElementsAttrValueLessThan<string leftAttrName, int leftIdx,string rightAttrName, int rightIdx> :
   CPred<FetchNthIntElementsAttr<leftAttrName, leftIdx>.result # " < " # FetchNthIntElementsAttr<rightAttrName, rightIdx>.result>;

class Pool2DPadValuesLessThanKernel<string padAttr, string kernelAttr> :
    PredOpTrait<"op pad values must satisfy pad_top/pad_bottom < kernel_y and pad_left/pad_right < kernel_x",
    And<[ElementsAttrValueLessThan<padAttr, 0, kernelAttr, 0>,
         ElementsAttrValueLessThan<padAttr, 1, kernelAttr, 0>,
         ElementsAttrValueLessThan<padAttr, 2, kernelAttr, 1>,
         ElementsAttrValueLessThan<padAttr, 3, kernelAttr, 1>]
    >>;

class MatMulShapesMatch<string lhs, string rhs, string output> : PredOpTrait<
  "shapes of " # lhs # ", " # rhs # ", and " # output #
      " must satisfy [N,H,C], [N,C,W], [N,H,W]",
  And<[
    SameDimsOrDynamicPred<lhs, 0, rhs, 0>,
    SameDimsOrDynamicPred<lhs, 0, output, 0>,
    SameDimsOrDynamicPred<lhs, 1, output, 1>,
    SameDimsOrDynamicPred<lhs, 2, rhs, 1>,
    SameDimsOrDynamicPred<rhs, 2, output, 2>
  ]>>;

class DimsOfTypeAllMatchOrDynamic<string lhsType, string rhsType,
                                  string otherType, int dim> :
  Or<[
    DimOfTypeIsDynamic<lhsType, dim>,
    DimOfTypeIsDynamic<rhsType, dim>,
    DimOfTypeIsDynamic<otherType, dim>,
    And<[DimsOfTypeMatch<lhsType, dim, rhsType, dim>,
         DimsOfTypeMatch<lhsType, dim, otherType, dim>]>
  ]>;

class DimOfTypeMatchesHalfPlusOne<string inputType, int inputDim,
                                  string outputType, int outputDim> :
  Or<[
    DimOfTypeIsDynamic<inputType, inputDim>,
    DimOfTypeIsDynamic<outputType, outputDim>,
    CPred<DimOfType<outputType, outputDim>.result # " == " #
          DimOfType<inputType, inputDim>.result # " / 2 + 1">
  ]>;

class Struct2TensorArmShapesMatchInput<string input> : PredOpTrait<
  "shapes of output_real and output_imag must match the shape of " # input,
  And<!foreach(dim, [0, 1, 2],
               DimsOfTypeAllMatchOrDynamic<
                 ShapedTypeOf<input>.result,
                 "getResultRealType()", "getResultImagType()", dim>)>>;

class Struct2TensorArmRFFT2DShapesMatchInput<string input> : PredOpTrait<
  "shapes of output_real and output_imag must satisfy [N,H,W/2 + 1] when " #
      input # " has shape [N,H,W]",
  And<[
    DimsOfTypeAllMatchOrDynamic<ShapedTypeOf<input>.result,
      "getResultRealType()", "getResultImagType()", 0>,
    DimsOfTypeAllMatchOrDynamic<ShapedTypeOf<input>.result,
      "getResultRealType()", "getResultImagType()", 1>,
    DimOfTypeMatchesHalfPlusOne<ShapedTypeOf<input>.result,
      2, "getResultRealType()", 2>,
    DimOfTypeMatchesHalfPlusOne<ShapedTypeOf<input>.result,
      2, "getResultImagType()", 2>
  ]>>;

class DimIsPowerOfTwo<string input, int dim> :
  Or<[DimIsDynamic<input, dim>,
      CPred<"llvm::isPowerOf2_64(static_cast<uint64_t>(" # DimOf<input, dim>.result # "))">]>;

class TensorDimsArePowerOfTwo<string input, list<int> dims> : PredOpTrait<
  "dimensions " # !interleave(dims, ", ") # " of " # input # " must be powers of two",
  And<!foreach(dim, dims, DimIsPowerOfTwo<input, dim>)>>;

class ClampBoundsAreNotNaN<string minVal, string maxVal>:
  PredOpTrait<minVal # " and " # maxVal # " must not be NaN",
    Neg<Or<[FloatAttrIsNaN<minVal>, FloatAttrIsNaN<maxVal>]>>>;

class IntegerAttrLessThanOrEqual<string minVal, string maxVal> :
  Implies<AttrIsInteger<minVal>,
          [CPred<IntegerAttrValue<minVal>.result # ".sle(" #
                 IntegerAttrValue<maxVal>.result # ")">]>;

class FloatAttrLessThanOrEqual<string minVal, string maxVal> :
  Implies<AttrIsFloat<minVal>,
          [CPred<FloatAttrValue<minVal>.result # ".compare(" #
                 FloatAttrValue<maxVal>.result #
                 ") != ::llvm::APFloat::cmpGreaterThan">]>;

class ClampMinValLessThanOrEqualToMaxVal<string minVal, string maxVal>:
  PredOpTrait<minVal # " must be <= " # maxVal,
    And<[IntegerAttrLessThanOrEqual<minVal, maxVal>,
         FloatAttrLessThanOrEqual<minVal, maxVal>]>>;

class TableSizeConstraint<string input, Type type, int size>:
  PredOpTrait<"table must have size " # size # " if " # input # " has element type " # type.summary,
      Implies<ElementTypeIsPred<input, type>, [CPred<"::llvm::cast<::mlir::ShapedType>(getTable().getType()).getShape()[0] == " # size>]>
    >;

class ShapeConstraintFromInputRank<string input, string other, int mul=1>:
  PredOpTrait< "the number of elements of " # other # " must be rank(" # input # ")" # !if(!eq(mul, 1), "", " * " # mul),
      Implies<CPred<HasRank<input>.result>,
        [CPred<ElementCount<other>.result # " == " # mul # " * " # Rank<input>.result>]>
    >;

class VariadicInputWithMinSize<string input, int min_size>:
    PredOpTrait<"variadic " # input # " must has at least " # min_size # " elements",
      CPred<"static_cast<int64_t>($" # input # ".getTypes().size()) >= " # min_size>>;

class VariadicInputAllSameElementType<string reference, string input>:
    PredOpTrait<"all elements of variadic " # input # " must have same element type",
      CPred<"::llvm::all_of($" # input # ".getTypes(), "
                     "[&](::mlir::Type t) { return ::llvm::cast<::mlir::ShapedType>(t).getElementType() == "
                     # ElementType<reference>.result # "; })">>;

class VariadicInputAllSameRank<string reference, string input>:
    PredOpTrait<"all elements of variadic " # input # " must have same element type",
      CPred<"::llvm::all_of($" # input # ".getTypes(), "
                     "[&](::mlir::Type t) { return ::llvm::cast<::mlir::ShapedType>(t).hasRank() && "
                     # HasRank<reference>.result #
                     " && ::llvm::cast<::mlir::ShapedType>(t).getRank() == "
                     # Rank<reference>.result # "; })">>;

class ElementTypeMatchesScale32<string tensor> :
  PredOpTrait<tensor # " must have element type i32 when scale32 is true, otherwise i16",
    CPred<"::llvm::cast<::mlir::ShapedType>($" # tensor # ".getType()).getElementType()."
          "isInteger(getScale32() ? 32 : 16)">>;

class TensorLengthMatchesPerChannel<string tensor> :
  PredOpTrait<tensor # " must have length input_shape[rank(input) - 1] when per_channel is true, otherwise length 1",
    CPred<"::llvm::cast<::mlir::ShapedType>($" # tensor # ".getType()).getShape()[0] == "
          "(getPerChannel() ? "
          "::llvm::cast<::mlir::ShapedType>($input.getType()).getShape().back() : 1)">>;


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_TYPES
