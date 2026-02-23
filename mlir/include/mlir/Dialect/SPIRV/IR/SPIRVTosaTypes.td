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

def SPIRV_TosaInteger : AnyIntOfWidths<[8, 16, 32, 64]>;
def SPIRV_TosaFloat : AnyTypeOf<[SPIRV_Float16, SPIRV_Float32, SPIRV_BFloat16KHR]>;
def SPIRV_TosaNumerical : AnyTypeOf<[SPIRV_TosaInteger, SPIRV_TosaFloat]>;
def SPIRV_TosaAny : AnyTypeOf<[SPIRV_TosaNumerical, SPIRV_Bool]>;

def SPIRV_TensorArmAxisAttr : ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<5>]>;
def SPIRV_BoolConstAttr : ConfinedAttr<BoolAttr, []>;
def SPIRV_TosaNumericalAttr: AnyAttrOf<[I8Attr, I16Attr, I32Attr, I64Attr, F16Attr, F32Attr, BF16Attr]>;

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

def SPIRV_Float32_TensorArm3D: TensorArmRankOf<[SPIRV_Float32], [3]>;
def SPIRV_TosaNumerical_TensorArm1D : TensorArmRankOf<[SPIRV_TosaNumerical], [1]>;
def SPIRV_TosaNumerical_TensorArm3D : TensorArmRankOf<[SPIRV_TosaNumerical], [3]>;
def SPIRV_TosaNumerical_TensorArm4D : TensorArmRankOf<[SPIRV_TosaNumerical], [4]>;
def SPIRV_TosaNumerical_TensorArm5D : TensorArmRankOf<[SPIRV_TosaNumerical], [5]>;

def SPIRV_TosaNumerical_TensorArm : TensorArmRankOf<[SPIRV_TosaNumerical], [1, 2, 3, 4, 5, 6]>;
def SPIRV_TosaFloat_TensorArm : TensorArmRankOf<[SPIRV_TosaFloat], [1, 2, 3, 4, 5, 6]>;
def SPIRV_Int32_TensorArmUpTo5D : TensorArmRankOf<[SPIRV_Int32], [1, 2, 3, 4, 5]>;

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

def SPIRV_DenseElementAttrsWithTensorArmType : AttrConstraint<
  CPred<"::llvm::isa<::mlir::spirv::TensorArmType>(::llvm::cast<::mlir::DenseElementsAttr>($_self).getType())">,
  "Attr with type = spirv::TensorArmType">;

def SPIRV_Int32_1DTensorArmOfLength2Attr : ConfinedAttr<RankedI32ElementsAttr<[2]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_Int32_1DTensorArmOfLength3Attr : ConfinedAttr<RankedI32ElementsAttr<[3]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_Int32_1DTensorArmOfLength4Attr : ConfinedAttr<RankedI32ElementsAttr<[4]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;
def SPIRV_Int32_1DTensorArmOfLength6Attr : ConfinedAttr<RankedI32ElementsAttr<[6]>, [SPIRV_DenseElementAttrsWithTensorArmType]>;

def SPIRV_TosaNumerical_1DTensorArmOfLength1 : SPIRV_1DTensorArmOfLengthAndType<[1], [SPIRV_TosaNumerical]>;

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

def SPIRV_Struct_2_Float32_TensorArm3D : IsStructOfNumElementsAndType<2, [SPIRV_Float32_TensorArm3D]>;

// Op Trait constraints:

class Implies<Pred left, list<Pred> right>: Or<[Neg<left>, Or<right>]>;

class TypeConstraintImplicationOn<string name, Type type, string other, list<Type> allowedTypes>:
  PredOpTrait<"if " # name # " has type " # type.summary # " then " #
              other # " must have a type in [" #
              !interleave(!foreach(type, allowedTypes, type.summary), ",") # "]",
    Implies<ElementTypeIsPred<name, type>,
    !foreach(allowedType, allowedTypes, ElementTypeIsPred<other, allowedType>)>>;

class AxisValueLessThanRankOf<string input>:
PredOpTrait<"axis attribute value should be lower than rank(" # input # ")",
  Implies<CPred<HasRank<input>.result>, [CPred<"getAxis() < " # Rank<input>.result>]>>;

class OutputRankIsInputRankMinusOne<string input, string output>:
PredOpTrait<output # " rank must be equal to max(1, rank(" # input # "))",
  Implies<
      And<[CPred<HasRank<input>.result>, CPred<HasRank<input>.result>]>,
      [CPred<"std::max(int64_t(1), " # Rank<input>.result # " - int64_t(1)) == " # Rank<output>.result>]>>;

class AccTypeIn<list<string> allowedValues> :
  CPred<"llvm::is_contained({" # !interleave(!foreach(value, allowedValues, "::mlir::spirv::TosaExtAccType::" # value), ",") # "}, getAccType())">;

class TypeImpliesAccType<string input, Type type, list<string> allowedAccTypes>:
  PredOpTrait<"acc_type must be one in [" # !interleave(allowedAccTypes, ",") # "] when type has value " # type.summary,
  Implies<ElementTypeIsPred<input, type>, [AccTypeIn<allowedAccTypes>]>>;


#endif // MLIR_DIALECT_SPIRV_IR_TOSA_TYPES
