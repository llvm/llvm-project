//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ValueTypes.td - list of ValueType instances supported by the the
// CodeGen infrastructure.
//
//===----------------------------------------------------------------------===//

class ValueType<int size, int value> {
  string Namespace = "MVT";
  string LLVMName = NAME;
  int Size = size;
  int Value = value;
  int nElem = 1;
  ValueType ElementType = ?;
  bit isOverloaded = false;
  bit isInteger = false;
  bit isFP = false;
  bit isVector = false;
  bit isScalable = false;
  // Indicates this VT should be included in the
  // [FIRST_VALUETYPE,LAST_VALUETYPE] range.
  bit isNormalValueType = true;
}

class VTAny<int value> : ValueType<0, value> {
  let isOverloaded = true;
}

class VTInt<int size, int value>
    : ValueType<size, value> {
  let isInteger = true;
}

class VTFP<int size, int value>
    : ValueType<size, value> {
  let isFP = true;
}

class VTVec<int nelem, ValueType elt, int value>
    : ValueType<!mul(nelem, elt.Size), value> {
  let nElem = nelem;
  let ElementType = elt;
  let isInteger = elt.isInteger;
  let isFP = elt.isFP;
  let isVector = true;
}

class VTScalableVec<int nelem, ValueType elt, int value>
    : VTVec<nelem, elt, value> {
  let isScalable = true;
}

defset list<ValueType> ValueTypes = {

def OtherVT : ValueType<0,   1> {  // "Other" value
  let LLVMName = "Other";
}

def i1      : VTInt<1,   2>;  // One bit boolean value
def i2      : VTInt<2,   3>;  // 2-bit integer value
def i4      : VTInt<4,   4>;  // 4-bit integer value
def i8      : VTInt<8,   5>;  // 8-bit integer value
def i16     : VTInt<16,  6>;  // 16-bit integer value
def i32     : VTInt<32,  7>;  // 32-bit integer value
def i64     : VTInt<64,  8>;  // 64-bit integer value
def i128    : VTInt<128, 9>;  // 128-bit integer value

def bf16    : VTFP<16,  10>;  // 16-bit brain floating point value
def f16     : VTFP<16,  11>;  // 16-bit floating point value
def f32     : VTFP<32,  12>;  // 32-bit floating point value
def f64     : VTFP<64,  13>;  // 64-bit floating point value
def f80     : VTFP<80,  14>;  // 80-bit floating point value
def f128    : VTFP<128, 15>;  // 128-bit floating point value
def ppcf128 : VTFP<128, 16>;  // PPC 128-bit floating point value

def v1i1    : VTVec<1,    i1, 17>;  //    1 x i1 vector value
def v2i1    : VTVec<2,    i1, 18>;  //    2 x i1 vector value
def v3i1    : VTVec<3,    i1, 19>;  //    3 x i1 vector value
def v4i1    : VTVec<4,    i1, 20>;  //    4 x i1 vector value
def v8i1    : VTVec<8,    i1, 21>;  //    8 x i1 vector value
def v16i1   : VTVec<16,   i1, 22>;  //   16 x i1 vector value
def v32i1   : VTVec<32,   i1, 23>;  //   32 x i1 vector value
def v64i1   : VTVec<64,   i1, 24>;  //   64 x i1 vector value
def v128i1  : VTVec<128,  i1, 25>;  //  128 x i1 vector value
def v256i1  : VTVec<256,  i1, 26>;  //  256 x i1 vector value
def v512i1  : VTVec<512,  i1, 27>;  //  512 x i1 vector value
def v1024i1 : VTVec<1024, i1, 28>;  // 1024 x i1 vector value
def v2048i1 : VTVec<2048, i1, 29>;  // 2048 x i1 vector value

def v128i2  : VTVec<128,  i2, 30>;   //  128 x i2 vector value
def v256i2  : VTVec<256,  i2, 31>;   //  256 x i2 vector value

def v64i4   : VTVec<64,   i4, 32>;   //   64 x i4 vector value
def v128i4  : VTVec<128,  i4, 33>;   //  128 x i4 vector value

def v1i8    : VTVec<1,    i8, 34>;  //    1 x i8 vector value
def v2i8    : VTVec<2,    i8, 35>;  //    2 x i8 vector value
def v3i8    : VTVec<3,    i8, 36>;  //    3 x i8 vector value
def v4i8    : VTVec<4,    i8, 37>;  //    4 x i8 vector value
def v8i8    : VTVec<8,    i8, 38>;  //    8 x i8 vector value
def v16i8   : VTVec<16,   i8, 39>;  //   16 x i8 vector value
def v32i8   : VTVec<32,   i8, 40>;  //   32 x i8 vector value
def v64i8   : VTVec<64,   i8, 41>;  //   64 x i8 vector value
def v128i8  : VTVec<128,  i8, 42>;  //  128 x i8 vector value
def v256i8  : VTVec<256,  i8, 43>;  //  256 x i8 vector value
def v512i8  : VTVec<512,  i8, 44>;  //  512 x i8 vector value
def v1024i8 : VTVec<1024, i8, 45>;  // 1024 x i8 vector value

def v1i16   : VTVec<1,   i16, 46>;  //   1 x i16 vector value
def v2i16   : VTVec<2,   i16, 47>;  //   2 x i16 vector value
def v3i16   : VTVec<3,   i16, 48>;  //   3 x i16 vector value
def v4i16   : VTVec<4,   i16, 49>;  //   4 x i16 vector value
def v6i16   : VTVec<6,   i16, 50>;  //   6 x i16 vector value
def v8i16   : VTVec<8,   i16, 51>;  //   8 x i16 vector value
def v10i16  : VTVec<10,  i16, 52>;  //  10 x i16 vector value
def v16i16  : VTVec<16,  i16, 53>;  //  16 x i16 vector value
def v18i16  : VTVec<18,  i16, 54>;  //  18 x i16 vector value
def v32i16  : VTVec<32,  i16, 55>;  //  32 x i16 vector value
def v36i16  : VTVec<36,  i16, 56>;  //  36 x i16 vector value
def v64i16  : VTVec<64,  i16, 57>;  //  64 x i16 vector value
def v128i16 : VTVec<128, i16, 58>;  // 128 x i16 vector value
def v256i16 : VTVec<256, i16, 59>;  // 256 x i16 vector value
def v512i16 : VTVec<512, i16, 60>;  // 512 x i16 vector value

def v1i32    : VTVec<1,    i32, 61>;  //    1 x i32 vector value
def v2i32    : VTVec<2,    i32, 62>;  //    2 x i32 vector value
def v3i32    : VTVec<3,    i32, 63>;  //    3 x i32 vector value
def v4i32    : VTVec<4,    i32, 64>;  //    4 x i32 vector value
def v5i32    : VTVec<5,    i32, 65>;  //    5 x i32 vector value
def v6i32    : VTVec<6,    i32, 66>;  //    6 x f32 vector value
def v7i32    : VTVec<7,    i32, 67>;  //    7 x f32 vector value
def v8i32    : VTVec<8,    i32, 68>;  //    8 x i32 vector value
def v9i32    : VTVec<9,    i32, 69>;  //    9 x i32 vector value
def v10i32   : VTVec<10,   i32, 70>;  //   10 x i32 vector value
def v11i32   : VTVec<11,   i32, 71>;  //   11 x i32 vector value
def v12i32   : VTVec<12,   i32, 72>;  //   12 x i32 vector value
def v16i32   : VTVec<16,   i32, 73>;  //   16 x i32 vector value
def v18i32   : VTVec<18,   i32, 74>;  //   18 x i32 vector value
def v32i32   : VTVec<32,   i32, 75>;  //   32 x i32 vector value
def v64i32   : VTVec<64,   i32, 76>;  //   64 x i32 vector value
def v128i32  : VTVec<128,  i32, 77>;  //  128 x i32 vector value
def v256i32  : VTVec<256,  i32, 78>;  //  256 x i32 vector value
def v512i32  : VTVec<512,  i32, 79>;  //  512 x i32 vector value
def v1024i32 : VTVec<1024, i32, 80>;  // 1024 x i32 vector value
def v2048i32 : VTVec<2048, i32, 81>;  // 2048 x i32 vector value

def v1i64   : VTVec<1,   i64, 82>;  //   1 x i64 vector value
def v2i64   : VTVec<2,   i64, 83>;  //   2 x i64 vector value
def v3i64   : VTVec<3,   i64, 84>;  //   3 x i64 vector value
def v4i64   : VTVec<4,   i64, 85>;  //   4 x i64 vector value
def v8i64   : VTVec<8,   i64, 86>;  //   8 x i64 vector value
def v16i64  : VTVec<16,  i64, 87>;  //  16 x i64 vector value
def v32i64  : VTVec<32,  i64, 88>;  //  32 x i64 vector value
def v64i64  : VTVec<64,  i64, 89>;  //  64 x i64 vector value
def v128i64 : VTVec<128, i64, 90>;  // 128 x i64 vector value
def v256i64 : VTVec<256, i64, 91>;  // 256 x i64 vector value

def v1i128  : VTVec<1,  i128, 92>;  //  1 x i128 vector value

def v1f16    : VTVec<1,    f16,  93>;  //    1 x f16 vector value
def v2f16    : VTVec<2,    f16,  94>;  //    2 x f16 vector value
def v3f16    : VTVec<3,    f16,  95>;  //    3 x f16 vector value
def v4f16    : VTVec<4,    f16,  96>;  //    4 x f16 vector value
def v6f16    : VTVec<6,    f16,  97>;  //    6 x f16 vector value
def v8f16    : VTVec<8,    f16,  98>;  //    8 x f16 vector value
def v10f16   : VTVec<10,   f16,  99>;  //   10 x f16 vector value
def v16f16   : VTVec<16,   f16,  100>; //   16 x f16 vector value
def v18f16   : VTVec<18,   f16,  101>; //   18 x f16 vector value
def v32f16   : VTVec<32,   f16,  102>; //   32 x f16 vector value
def v36f16   : VTVec<36,   f16,  103>; //   36 x f16 vector value
def v64f16   : VTVec<64,   f16,  104>; //   64 x f16 vector value
def v128f16  : VTVec<128,  f16,  105>; //  128 x f16 vector value
def v256f16  : VTVec<256,  f16,  106>; //  256 x f16 vector value
def v512f16  : VTVec<512,  f16,  107>; //  512 x f16 vector value

def v2bf16   : VTVec<2,   bf16, 108>;  //    2 x bf16 vector value
def v3bf16   : VTVec<3,   bf16, 109>;  //    3 x bf16 vector value
def v4bf16   : VTVec<4,   bf16, 110>;  //    4 x bf16 vector value
def v6bf16   : VTVec<6,   bf16, 111>;  //    6 x bf16 vector value
def v8bf16   : VTVec<8,   bf16, 112>;  //    8 x bf16 vector value
def v10bf16  : VTVec<10,  bf16, 113>;  //   10 x bf16 vector value
def v16bf16  : VTVec<16,  bf16, 114>;  //   16 x bf16 vector value
def v18bf16  : VTVec<18,  bf16, 115>;  //   18 x bf16 vector value
def v32bf16  : VTVec<32,  bf16, 116>;  //   32 x bf16 vector value
def v36bf16  : VTVec<36,  bf16, 117>;  //   36 x bf16 vector value
def v64bf16  : VTVec<64,  bf16, 118>;  //   64 x bf16 vector value
def v128bf16 : VTVec<128, bf16, 119>;  //  128 x bf16 vector value

def v1f32    : VTVec<1,    f32, 120>;  //    1 x f32 vector value
def v2f32    : VTVec<2,    f32, 121>;  //    2 x f32 vector value
def v3f32    : VTVec<3,    f32, 122>;  //    3 x f32 vector value
def v4f32    : VTVec<4,    f32, 123>;  //    4 x f32 vector value
def v5f32    : VTVec<5,    f32, 124>;  //    5 x f32 vector value
def v6f32    : VTVec<6,    f32, 125>;  //    6 x f32 vector value
def v7f32    : VTVec<7,    f32, 126>;  //    7 x f32 vector value
def v8f32    : VTVec<8,    f32, 127>;  //    8 x f32 vector value
def v9f32    : VTVec<9,    f32, 128>;  //    9 x f32 vector value
def v10f32   : VTVec<10,   f32, 129>;  //   10 x f32 vector value
def v11f32   : VTVec<11,   f32, 130>;  //   11 x f32 vector value
def v12f32   : VTVec<12,   f32, 131>;  //   12 x f32 vector value
def v16f32   : VTVec<16,   f32, 132>;  //   16 x f32 vector value
def v18f32   : VTVec<18,   f32, 133>;  //   18 x f32 vector value
def v32f32   : VTVec<32,   f32, 134>;  //   32 x f32 vector value
def v64f32   : VTVec<64,   f32, 135>;  //   64 x f32 vector value
def v128f32  : VTVec<128,  f32, 136>;  //  128 x f32 vector value
def v256f32  : VTVec<256,  f32, 137>;  //  256 x f32 vector value
def v512f32  : VTVec<512,  f32, 138>;  //  512 x f32 vector value
def v1024f32 : VTVec<1024, f32, 139>;  // 1024 x f32 vector value
def v2048f32 : VTVec<2048, f32, 140>;  // 2048 x f32 vector value

def v1f64    : VTVec<1,    f64, 141>;  //    1 x f64 vector value
def v2f64    : VTVec<2,    f64, 142>;  //    2 x f64 vector value
def v3f64    : VTVec<3,    f64, 143>;  //    3 x f64 vector value
def v4f64    : VTVec<4,    f64, 144>;  //    4 x f64 vector value
def v8f64    : VTVec<8,    f64, 145>;  //    8 x f64 vector value
def v16f64   : VTVec<16,   f64, 146>;  //   16 x f64 vector value
def v32f64   : VTVec<32,   f64, 147>;  //   32 x f64 vector value
def v64f64   : VTVec<64,   f64, 148>;  //   64 x f64 vector value
def v128f64  : VTVec<128,  f64, 149>;  //  128 x f64 vector value
def v256f64  : VTVec<256,  f64, 150>;  //  256 x f64 vector value

def nxv1i1  : VTScalableVec<1,  i1, 151>;  // n x  1 x i1  vector value
def nxv2i1  : VTScalableVec<2,  i1, 152>;  // n x  2 x i1  vector value
def nxv4i1  : VTScalableVec<4,  i1, 153>;  // n x  4 x i1  vector value
def nxv8i1  : VTScalableVec<8,  i1, 154>;  // n x  8 x i1  vector value
def nxv16i1 : VTScalableVec<16, i1, 155>;  // n x 16 x i1  vector value
def nxv32i1 : VTScalableVec<32, i1, 156>;  // n x 32 x i1  vector value
def nxv64i1 : VTScalableVec<64, i1, 157>;  // n x 64 x i1  vector value

def nxv1i8  : VTScalableVec<1,  i8, 158>;  // n x  1 x i8  vector value
def nxv2i8  : VTScalableVec<2,  i8, 159>;  // n x  2 x i8  vector value
def nxv4i8  : VTScalableVec<4,  i8, 160>;  // n x  4 x i8  vector value
def nxv8i8  : VTScalableVec<8,  i8, 161>;  // n x  8 x i8  vector value
def nxv16i8 : VTScalableVec<16, i8, 162>;  // n x 16 x i8  vector value
def nxv32i8 : VTScalableVec<32, i8, 163>;  // n x 32 x i8  vector value
def nxv64i8 : VTScalableVec<64, i8, 164>;  // n x 64 x i8  vector value

def nxv1i16  : VTScalableVec<1,  i16, 165>;  // n x  1 x i16 vector value
def nxv2i16  : VTScalableVec<2,  i16, 166>;  // n x  2 x i16 vector value
def nxv4i16  : VTScalableVec<4,  i16, 167>;  // n x  4 x i16 vector value
def nxv8i16  : VTScalableVec<8,  i16, 168>;  // n x  8 x i16 vector value
def nxv16i16 : VTScalableVec<16, i16, 169>;  // n x 16 x i16 vector value
def nxv32i16 : VTScalableVec<32, i16, 170>;  // n x 32 x i16 vector value

def nxv1i32  : VTScalableVec<1,  i32, 171>;  // n x  1 x i32 vector value
def nxv2i32  : VTScalableVec<2,  i32, 172>;  // n x  2 x i32 vector value
def nxv4i32  : VTScalableVec<4,  i32, 173>;  // n x  4 x i32 vector value
def nxv8i32  : VTScalableVec<8,  i32, 174>;  // n x  8 x i32 vector value
def nxv16i32 : VTScalableVec<16, i32, 175>;  // n x 16 x i32 vector value
def nxv32i32 : VTScalableVec<32, i32, 176>;  // n x 32 x i32 vector value

def nxv1i64  : VTScalableVec<1,  i64, 177>;  // n x  1 x i64 vector value
def nxv2i64  : VTScalableVec<2,  i64, 178>;  // n x  2 x i64 vector value
def nxv4i64  : VTScalableVec<4,  i64, 179>;  // n x  4 x i64 vector value
def nxv8i64  : VTScalableVec<8,  i64, 180>;  // n x  8 x i64 vector value
def nxv16i64 : VTScalableVec<16, i64, 181>;  // n x 16 x i64 vector value
def nxv32i64 : VTScalableVec<32, i64, 182>;  // n x 32 x i64 vector value

def nxv1f16  : VTScalableVec<1,  f16, 183>;  // n x  1 x  f16 vector value
def nxv2f16  : VTScalableVec<2,  f16, 184>;  // n x  2 x  f16 vector value
def nxv4f16  : VTScalableVec<4,  f16, 185>;  // n x  4 x  f16 vector value
def nxv8f16  : VTScalableVec<8,  f16, 186>;  // n x  8 x  f16 vector value
def nxv16f16 : VTScalableVec<16, f16, 187>;  // n x 16 x  f16 vector value
def nxv32f16 : VTScalableVec<32, f16, 188>;  // n x 32 x  f16 vector value

def nxv1bf16  : VTScalableVec<1,  bf16, 189>;  // n x  1 x bf16 vector value
def nxv2bf16  : VTScalableVec<2,  bf16, 190>;  // n x  2 x bf16 vector value
def nxv4bf16  : VTScalableVec<4,  bf16, 191>;  // n x  4 x bf16 vector value
def nxv8bf16  : VTScalableVec<8,  bf16, 192>;  // n x  8 x bf16 vector value
def nxv16bf16 : VTScalableVec<16, bf16, 193>;  // n x 16 x bf16 vector value
def nxv32bf16 : VTScalableVec<32, bf16, 194>;  // n x 32 x bf16 vector value

def nxv1f32  : VTScalableVec<1,  f32, 195>;  // n x  1 x  f32 vector value
def nxv2f32  : VTScalableVec<2,  f32, 196>;  // n x  2 x  f32 vector value
def nxv4f32  : VTScalableVec<4,  f32, 197>;  // n x  4 x  f32 vector value
def nxv8f32  : VTScalableVec<8,  f32, 198>;  // n x  8 x  f32 vector value
def nxv16f32 : VTScalableVec<16, f32, 199>;  // n x 16 x  f32 vector value

def nxv1f64  : VTScalableVec<1,  f64, 200>;  // n x  1 x  f64 vector value
def nxv2f64  : VTScalableVec<2,  f64, 201>;  // n x  2 x  f64 vector value
def nxv4f64  : VTScalableVec<4,  f64, 202>;  // n x  4 x  f64 vector value
def nxv8f64  : VTScalableVec<8,  f64, 203>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   204>;  // X86 MMX value
def FlagVT    : ValueType<0,    205> { // Pre-RA sched glue
  let LLVMName = "Glue";
}
def isVoid    : ValueType<0,    206>;  // Produces no value
def untyped   : ValueType<8,    207> { // Produces an untyped value
  let LLVMName = "Untyped";
}
def funcref   : ValueType<0,    208>;  // WebAssembly's funcref type
def externref : ValueType<0,    209>;  // WebAssembly's externref type
def exnref    : ValueType<0,    210>;  // WebAssembly's exnref type
def x86amx    : ValueType<8192, 211>;  // X86 AMX value
def i64x8     : ValueType<512,  212>;  // 8 Consecutive GPRs (AArch64)
def aarch64svcount
              : ValueType<16,  213>;  // AArch64 predicate-as-counter
def spirvbuiltin : ValueType<0, 214>; // SPIR-V's builtin type

let isNormalValueType = false in {
def token      : ValueType<0, 504>;  // TokenTy
def MetadataVT : ValueType<0, 505> { // Metadata
  let LLVMName = "Metadata";
}

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny    : VTAny<506>;

// Pseudo valuetype to represent "vector of any size"
// Should only be used in TableGen.
def vAny       : VTAny<507>;

// Pseudo valuetype to represent "float of any format"
// Should only be used in TableGen.
def fAny       : VTAny<508>;

// Pseudo valuetype to represent "integer of any bit width"
// Should only be used in TableGen.
def iAny       : VTAny<509>;

// Pseudo valuetype mapped to the current pointer size.
// Should only be used in TableGen.
def iPTR       : ValueType<0, 510>;

// Pseudo valuetype to represent "any type of any size".
// Should only be used in TableGen.
def Any        : VTAny<511>;

} // isNormalValueType = false

} // end defset ValueTypes

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the pointer.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
