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

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

def SPIRV_TosaInteger : AnyIntOfWidths<[8, 16, 32, 64]>;
def SPIRV_TosaFloat : AnyTypeOf<[SPIRV_Float16, SPIRV_Float32, SPIRV_BFloat16KHR]>;
def SPIRV_TosaNumerical : AnyTypeOf<[SPIRV_TosaInteger, SPIRV_TosaFloat]>;
def SPIRV_TosaAny : AnyTypeOf<[SPIRV_TosaNumerical, SPIRV_Bool]>;

def SPIRV_TensorArmAxisAttr : ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<5>]>;

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

def SPIRV_TosaNumerical_TensorArm : TensorArmRankOf<[SPIRV_TosaNumerical], [1, 2, 3, 4, 5, 6]>;
def SPIRV_Int32_TensorArmUpTo5D : TensorArmRankOf<[SPIRV_Int32], [1, 2, 3, 4, 5]>;

#endif // MLIR_DIALECT_SPIRV_IR_TOSA_TYPES
