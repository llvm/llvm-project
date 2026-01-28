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

def SPIRV_TosaNumerical_TensorArm1D : TensorArmRankOf<[SPIRV_TosaNumerical], [1]>;
def SPIRV_TosaNumerical_TensorArm4D : TensorArmRankOf<[SPIRV_TosaNumerical], [4]>;
def SPIRV_TosaNumerical_TensorArm5D : TensorArmRankOf<[SPIRV_TosaNumerical], [5]>;

def SPIRV_TosaNumerical_TensorArm : TensorArmRankOf<[SPIRV_TosaNumerical], [1, 2, 3, 4, 5, 6]>;
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

#endif // MLIR_DIALECT_SPIRV_IR_TOSA_TYPES
