//===- VCIXAttrs.td - VCIX dialect attributes ----*- tablegen -----------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_VCIX_VCIXATTRS
#define MLIR_DIALECT_VCIX_VCIXATTRS

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// VCIX helper type definitions
//===----------------------------------------------------------------------===//
def VectorOfRank1 : AnyTypeOf<[ScalableVectorOfRank<[1]>, VectorOfRank<[1]>]>;
def VectorOfRank1OrScalar
    : AnyTypeOf<[VectorOfRank1, I<64>, I<32>, F<16>, F<32>, F<64>, I<5>]>;
def OpcodeI1Attr : AnyIntegerAttrBase<AnyI<1>, "1-bit integer attribute">;
def OpcodeI2Attr : AnyIntegerAttrBase<AnyI<2>, "2-bit integer attribute">;
def OpcodeIntAttr : AnyAttrOf<[OpcodeI1Attr, OpcodeI2Attr]>;
def RAttr : AnyAttrOf<[AnyIntegerAttrBase<AnyI<5>, "5-bit integer attribute">]>;
def RVLType : AnyTypeOf<[UI<64>, UI<32>]>;

// Special version for intrinsic version where int attr is zext to i32 or i64
// depending on xlen of the target
def VectorOfRank1OrScalarIntrin
    : AnyTypeOf<[VectorOfRank1, I<64>, I<32>, F<16>, F<32>, F<64>]>;
def OpcodeIntrinIntAttr : AnyAttrOf<[I64Attr, I32Attr]>;
def RIntrinAttr : AnyAttrOf<[I64Attr, I32Attr]>;
def RVLIntrinType : AnyTypeOf<[I<64>, I<32>]>;

def VCIX_e8mf8 : I32EnumAttrCase<"e8mf8", 0, "e8mf8">;
def VCIX_e8mf4 : I32EnumAttrCase<"e8mf4", 1, "e8mf4">;
def VCIX_e8mf2 : I32EnumAttrCase<"e8mf2", 2, "e8mf2">;
def VCIX_e8m1  : I32EnumAttrCase<"e8m1",  3, "e8m1">;
def VCIX_e8m2  : I32EnumAttrCase<"e8m2",  4, "e8m2">;
def VCIX_e8m4  : I32EnumAttrCase<"e8m4",  5, "e8m4">;
def VCIX_e8m8  : I32EnumAttrCase<"e8m8",  6, "e8m8">;

def VCIX_e16mf4 : I32EnumAttrCase<"e16mf4", 7,  "e16mf4">;
def VCIX_e16mf2 : I32EnumAttrCase<"e16mf2", 8,  "e16mf2">;
def VCIX_e16m1  : I32EnumAttrCase<"e16m1",  9,  "e16m1">;
def VCIX_e16m2  : I32EnumAttrCase<"e16m2",  10, "e16m2">;
def VCIX_e16m4  : I32EnumAttrCase<"e16m4",  11,  "e16m4">;
def VCIX_e16m8  : I32EnumAttrCase<"e16m8",  12,  "e16m8">;

def VCIX_e32mf2 : I32EnumAttrCase<"e32mf2", 13, "e32mf2">;
def VCIX_e32m1  : I32EnumAttrCase<"e32m1",  14, "e32m1">;
def VCIX_e32m2  : I32EnumAttrCase<"e32m2",  15, "e32m2">;
def VCIX_e32m4  : I32EnumAttrCase<"e32m4",  16, "e32m4">;
def VCIX_e32m8  : I32EnumAttrCase<"e32m8",  17, "e32m8">;

def VCIX_e64m1  : I32EnumAttrCase<"e64m1",  18, "e64m1">;
def VCIX_e64m2  : I32EnumAttrCase<"e64m2",  19, "e64m2">;
def VCIX_e64m4  : I32EnumAttrCase<"e64m4",  20, "e64m4">;
def VCIX_e64m8  : I32EnumAttrCase<"e64m8",  21, "e64m8">;

def VCIX_SewLmulAttr : I32EnumAttr<"SewLmul",
    "A list of all possible SEW and LMUL",
    [
      VCIX_e8mf8,
      VCIX_e8mf4,
      VCIX_e8mf2,
      VCIX_e8m1,
      VCIX_e8m2,
      VCIX_e8m4,
      VCIX_e8m8,

      VCIX_e16mf4,
      VCIX_e16mf2,
      VCIX_e16m1,
      VCIX_e16m2,
      VCIX_e16m4,
      VCIX_e16m8,

      VCIX_e32mf2,
      VCIX_e32m1,
      VCIX_e32m2,
      VCIX_e32m4,
      VCIX_e32m8,

      VCIX_e64m1,
      VCIX_e64m2,
      VCIX_e64m4,
      VCIX_e64m8,
    ]> {
  let cppNamespace = "::mlir::vcix";
}

#endif // MLIR_DIALECT_VCIX_VCIXATTRS
