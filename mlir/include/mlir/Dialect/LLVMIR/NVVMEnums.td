//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the NVVM IR enum attributes.
///
//===----------------------------------------------------------------------===//

#ifndef NVVMIR_ENUMS
#define NVVMIR_ENUMS

include "mlir/Dialect/LLVMIR/NVVMDialect.td"
include "mlir/IR/EnumAttr.td"

// Attributes for the floating point rounding modes supported by PTX
def FPRoundingModeNone : I32EnumAttrCase<"NONE", 0, "none">;
def FPRoundingModeRN   : I32EnumAttrCase<"RN",   1, "rn">;
def FPRoundingModeRM   : I32EnumAttrCase<"RM",   2, "rm">;
def FPRoundingModeRP   : I32EnumAttrCase<"RP",   3, "rp">;
def FPRoundingModeRZ   : I32EnumAttrCase<"RZ",   4, "rz">;
def FPRoundingModeRNA  : I32EnumAttrCase<"RNA",  5, "rna">;
def FPRoundingModeRS   : I32EnumAttrCase<"RS",   6, "rs">;

def FPRoundingMode : I32EnumAttr<"FPRoundingMode", "NVVM FPRoundingMode kind",
  [FPRoundingModeNone, FPRoundingModeRN, FPRoundingModeRM,
    FPRoundingModeRP, FPRoundingModeRZ, FPRoundingModeRNA, FPRoundingModeRS]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def FPRoundingModeAttr : EnumAttr<NVVM_Dialect, FPRoundingMode, "fp_rnd_mode"> {
  let assemblyFormat = "`<` $value `>`";
}

def SaturationModeNone   : I32EnumAttrCase<"NONE", 0, "none">;
def SaturationModeFinite : I32EnumAttrCase<"SATFINITE", 1, "satfinite">;
def SaturationModeSat    : I32EnumAttrCase<"SAT", 2, "sat">;

def SaturationMode : I32EnumAttr<"SaturationMode", "NVVM SaturationMode kind",
  [SaturationModeNone, SaturationModeFinite, SaturationModeSat]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::NVVM";
}
def SaturationModeAttr : EnumAttr<NVVM_Dialect, SaturationMode, "sat_mode"> {
  let summary = "Describes the saturation mode";
  let description = [{
    A `nvvm.sat_mode` attribute specifies the saturation mode for instructions
    involving floating points or integers. It can be one of the following
    values:
    - `none`: No saturation is applied.
    - `satfinite`: If the absolute value of input (ignoring sign) is greater
      than the `MAX_NORM` of the specified destination format, then the result
      is the sign-preserved `MAX_NORM` of the destination format and a positive
      `MAX_NORM` in unsigned datatypes for which the destination sign is not
      supported. If the input is `NaN`, then the result can be `NaN` or the
      `MAX_NORM` of the destination format, depending on the format.
    - `sat`: For integer destination types, this limits the value to `MININT..
      MAXINT` and applies to both signed and unsigned integer datatypes. For
      floating point destination types (applies to only `F16`, `F32`, and `F64`
      types), this limits the value to the range `[0.0, 1.0]` and flushes NaN
      results to positive zero.

    [For more information, see PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cvt)
}];
 let assemblyFormat = "`<` $value `>`";
}

#endif // NVVMIR_ENUMS
