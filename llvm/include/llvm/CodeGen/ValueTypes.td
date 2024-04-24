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
  int isOverloaded = false;
  int isInteger = false;
  int isFP = false;
  int isVector = false;
  int isScalable = false;
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

def f8e4m3fn: VTFP<8,   10>;  // 8-bit floating point value
def f8e5m2  : VTFP<8,   11>;  // 8-bit floating point value
def bf16    : VTFP<16,  12>;  // 16-bit brain floating point value
def f16     : VTFP<16,  13>;  // 16-bit floating point value
def f32     : VTFP<32,  14>;  // 32-bit floating point value
def f64     : VTFP<64,  15>;  // 64-bit floating point value
def f80     : VTFP<80,  16>;  // 80-bit floating point value
def f128    : VTFP<128, 17>;  // 128-bit floating point value
def ppcf128 : VTFP<128, 18>;  // PPC 128-bit floating point value

def v1i1    : VTVec<1,    i1, 19>;  //    1 x i1 vector value
def v2i1    : VTVec<2,    i1, 20>;  //    2 x i1 vector value
def v3i1    : VTVec<3,    i1, 21>;  //    3 x i1 vector value
def v4i1    : VTVec<4,    i1, 22>;  //    4 x i1 vector value
def v8i1    : VTVec<8,    i1, 23>;  //    8 x i1 vector value
def v16i1   : VTVec<16,   i1, 24>;  //   16 x i1 vector value
def v32i1   : VTVec<32,   i1, 25>;  //   32 x i1 vector value
def v64i1   : VTVec<64,   i1, 26>;  //   64 x i1 vector value
def v128i1  : VTVec<128,  i1, 27>;  //  128 x i1 vector value
def v256i1  : VTVec<256,  i1, 28>;  //  256 x i1 vector value
def v512i1  : VTVec<512,  i1, 29>;  //  512 x i1 vector value
def v1024i1 : VTVec<1024, i1, 30>;  // 1024 x i1 vector value
def v2048i1 : VTVec<2048, i1, 31>;  // 2048 x i1 vector value

def v128i2  : VTVec<128,  i2, 32>;   //  128 x i2 vector value
def v256i2  : VTVec<256,  i2, 33>;   //  256 x i2 vector value

def v64i4   : VTVec<64,   i4, 34>;   //   64 x i4 vector value
def v128i4  : VTVec<128,  i4, 35>;   //  128 x i4 vector value

def v1i8    : VTVec<1,    i8, 36>;  //    1 x i8 vector value
def v2i8    : VTVec<2,    i8, 37>;  //    2 x i8 vector value
def v3i8    : VTVec<3,    i8, 38>;  //    3 x i8 vector value
def v4i8    : VTVec<4,    i8, 39>;  //    4 x i8 vector value
def v8i8    : VTVec<8,    i8, 40>;  //    8 x i8 vector value
def v16i8   : VTVec<16,   i8, 41>;  //   16 x i8 vector value
def v32i8   : VTVec<32,   i8, 42>;  //   32 x i8 vector value
def v64i8   : VTVec<64,   i8, 43>;  //   64 x i8 vector value
def v128i8  : VTVec<128,  i8, 44>;  //  128 x i8 vector value
def v256i8  : VTVec<256,  i8, 45>;  //  256 x i8 vector value
def v512i8  : VTVec<512,  i8, 46>;  //  512 x i8 vector value
def v1024i8 : VTVec<1024, i8, 47>;  // 1024 x i8 vector value

def v1i16   : VTVec<1,   i16, 48>;  //   1 x i16 vector value
def v2i16   : VTVec<2,   i16, 49>;  //   2 x i16 vector value
def v3i16   : VTVec<3,   i16, 50>;  //   3 x i16 vector value
def v4i16   : VTVec<4,   i16, 51>;  //   4 x i16 vector value
def v8i16   : VTVec<8,   i16, 52>;  //   8 x i16 vector value
def v16i16  : VTVec<16,  i16, 53>;  //  16 x i16 vector value
def v32i16  : VTVec<32,  i16, 54>;  //  32 x i16 vector value
def v64i16  : VTVec<64,  i16, 55>;  //  64 x i16 vector value
def v128i16 : VTVec<128, i16, 56>;  // 128 x i16 vector value
def v256i16 : VTVec<256, i16, 57>;  // 256 x i16 vector value
def v512i16 : VTVec<512, i16, 58>;  // 512 x i16 vector value

def v1i32    : VTVec<1,    i32, 59>;  //    1 x i32 vector value
def v2i32    : VTVec<2,    i32, 60>;  //    2 x i32 vector value
def v3i32    : VTVec<3,    i32, 61>;  //    3 x i32 vector value
def v4i32    : VTVec<4,    i32, 62>;  //    4 x i32 vector value
def v5i32    : VTVec<5,    i32, 63>;  //    5 x i32 vector value
def v6i32    : VTVec<6,    i32, 64>;  //    6 x f32 vector value
def v7i32    : VTVec<7,    i32, 65>;  //    7 x f32 vector value
def v8i32    : VTVec<8,    i32, 66>;  //    8 x i32 vector value
def v9i32    : VTVec<9,    i32, 67>;  //    9 x i32 vector value
def v10i32   : VTVec<10,   i32, 68>;  //   10 x i32 vector value
def v11i32   : VTVec<11,   i32, 69>;  //   11 x i32 vector value
def v12i32   : VTVec<12,   i32, 70>;  //   12 x i32 vector value
def v16i32   : VTVec<16,   i32, 71>;  //   16 x i32 vector value
def v32i32   : VTVec<32,   i32, 72>;  //   32 x i32 vector value
def v64i32   : VTVec<64,   i32, 73>;  //   64 x i32 vector value
def v128i32  : VTVec<128,  i32, 74>;  //  128 x i32 vector value
def v256i32  : VTVec<256,  i32, 75>;  //  256 x i32 vector value
def v512i32  : VTVec<512,  i32, 76>;  //  512 x i32 vector value
def v1024i32 : VTVec<1024, i32, 77>;  // 1024 x i32 vector value
def v2048i32 : VTVec<2048, i32, 78>;  // 2048 x i32 vector value

def v1i64   : VTVec<1,   i64, 79>;  //   1 x i64 vector value
def v2i64   : VTVec<2,   i64, 80>;  //   2 x i64 vector value
def v3i64   : VTVec<3,   i64, 81>;  //   3 x i64 vector value
def v4i64   : VTVec<4,   i64, 82>;  //   4 x i64 vector value
def v8i64   : VTVec<8,   i64, 83>;  //   8 x i64 vector value
def v16i64  : VTVec<16,  i64, 84>;  //  16 x i64 vector value
def v32i64  : VTVec<32,  i64, 85>;  //  32 x i64 vector value
def v64i64  : VTVec<64,  i64, 86>;  //  64 x i64 vector value
def v128i64 : VTVec<128, i64, 87>;  // 128 x i64 vector value
def v256i64 : VTVec<256, i64, 88>;  // 256 x i64 vector value

def v1i128  : VTVec<1,  i128, 89>;  //  1 x i128 vector value

def v1f8e4m3fn    : VTVec<1,    f8e4m3fn,  90>;  //    1 x f8e4m3fn vector value
def v2f8e4m3fn    : VTVec<2,    f8e4m3fn,  91>;  //    2 x f8e4m3fn vector value
def v3f8e4m3fn    : VTVec<3,    f8e4m3fn,  92>;  //    3 x f8e4m3fn vector value
def v4f8e4m3fn    : VTVec<4,    f8e4m3fn,  93>;  //    4 x f8e4m3fn vector value
def v8f8e4m3fn    : VTVec<8,    f8e4m3fn,  94>;  //    8 x f8e4m3fn vector value
def v16f8e4m3fn   : VTVec<16,   f8e4m3fn,  95>;  //   16 x f8e4m3fn vector value
def v32f8e4m3fn   : VTVec<32,   f8e4m3fn,  96>;  //   32 x f8e4m3fn vector value
def v64f8e4m3fn   : VTVec<64,   f8e4m3fn,  97>;  //   64 x f8e4m3fn vector value
def v128f8e4m3fn  : VTVec<128,  f8e4m3fn,  98>;  //  128 x f8e4m3fn vector value
def v256f8e4m3fn  : VTVec<256,  f8e4m3fn,  99>;  //  256 x f8e4m3fn vector value
def v512f8e4m3fn  : VTVec<512,  f8e4m3fn,  100>;  //  512 x f8e4m3fn vector value

def v1f8e5m2    : VTVec<1,    f8e5m2,  101>;  //    1 x f8e5m2 vector value
def v2f8e5m2    : VTVec<2,    f8e5m2,  102>;  //    2 x f8e5m2 vector value
def v3f8e5m2    : VTVec<3,    f8e5m2,  103>;  //    3 x f8e5m2 vector value
def v4f8e5m2    : VTVec<4,    f8e5m2,  104>;  //    4 x f8e5m2 vector value
def v8f8e5m2    : VTVec<8,    f8e5m2,  105>;  //    8 x f8e5m2 vector value
def v16f8e5m2   : VTVec<16,   f8e5m2,  106>;  //   16 x f8e5m2 vector value
def v32f8e5m2   : VTVec<32,   f8e5m2,  107>;  //   32 x f8e5m2 vector value
def v64f8e5m2   : VTVec<64,   f8e5m2,  108>;  //   64 x f8e5m2 vector value
def v128f8e5m2  : VTVec<128,  f8e5m2,  109>;  //  128 x f8e5m2 vector value
def v256f8e5m2  : VTVec<256,  f8e5m2,  110>;  //  256 x f8e5m2 vector value
def v512f8e5m2  : VTVec<512,  f8e5m2,  111>;  //  512 x f8e5m2 vector value

def v1f16    : VTVec<1,    f16,  112>;  //    1 x f16 vector value
def v2f16    : VTVec<2,    f16,  113>;  //    2 x f16 vector value
def v3f16    : VTVec<3,    f16,  114>;  //    3 x f16 vector value
def v4f16    : VTVec<4,    f16,  115>;  //    4 x f16 vector value
def v8f16    : VTVec<8,    f16,  116>;  //    8 x f16 vector value
def v16f16   : VTVec<16,   f16,  117>;  //   16 x f16 vector value
def v32f16   : VTVec<32,   f16,  118>;  //   32 x f16 vector value
def v64f16   : VTVec<64,   f16,  119>;  //   64 x f16 vector value
def v128f16  : VTVec<128,  f16,  120>;  //  128 x f16 vector value
def v256f16  : VTVec<256,  f16,  121>;  //  256 x f16 vector value
def v512f16  : VTVec<512,  f16,  122>;  //  512 x f16 vector value

def v2bf16   : VTVec<2,   bf16,  123>;  //    2 x bf16 vector value
def v3bf16   : VTVec<3,   bf16, 124>;  //    3 x bf16 vector value
def v4bf16   : VTVec<4,   bf16, 125>;  //    4 x bf16 vector value
def v8bf16   : VTVec<8,   bf16, 126>;  //    8 x bf16 vector value
def v16bf16  : VTVec<16,  bf16, 127>;  //   16 x bf16 vector value
def v32bf16  : VTVec<32,  bf16, 128>;  //   32 x bf16 vector value
def v64bf16  : VTVec<64,  bf16, 129>;  //   64 x bf16 vector value
def v128bf16 : VTVec<128, bf16, 130>;  //  128 x bf16 vector value

def v1f32    : VTVec<1,    f32, 131>;  //    1 x f32 vector value
def v2f32    : VTVec<2,    f32, 132>;  //    2 x f32 vector value
def v3f32    : VTVec<3,    f32, 133>;  //    3 x f32 vector value
def v4f32    : VTVec<4,    f32, 134>;  //    4 x f32 vector value
def v5f32    : VTVec<5,    f32, 135>;  //    5 x f32 vector value
def v6f32    : VTVec<6,    f32, 136>;  //    6 x f32 vector value
def v7f32    : VTVec<7,    f32, 137>;  //    7 x f32 vector value
def v8f32    : VTVec<8,    f32, 138>;  //    8 x f32 vector value
def v9f32    : VTVec<9,    f32, 139>;  //    9 x f32 vector value
def v10f32   : VTVec<10,   f32, 140>;  //   10 x f32 vector value
def v11f32   : VTVec<11,   f32, 141>;  //   11 x f32 vector value
def v12f32   : VTVec<12,   f32, 142>;  //   12 x f32 vector value
def v16f32   : VTVec<16,   f32, 143>;  //   16 x f32 vector value
def v32f32   : VTVec<32,   f32, 144>;  //   32 x f32 vector value
def v64f32   : VTVec<64,   f32, 145>;  //   64 x f32 vector value
def v128f32  : VTVec<128,  f32, 146>;  //  128 x f32 vector value
def v256f32  : VTVec<256,  f32, 147>;  //  256 x f32 vector value
def v512f32  : VTVec<512,  f32, 148>;  //  512 x f32 vector value
def v1024f32 : VTVec<1024, f32, 149>;  // 1024 x f32 vector value
def v2048f32 : VTVec<2048, f32, 150>;  // 2048 x f32 vector value

def v1f64    : VTVec<1,    f64, 151>;  //    1 x f64 vector value
def v2f64    : VTVec<2,    f64, 152>;  //    2 x f64 vector value
def v3f64    : VTVec<3,    f64, 153>;  //    3 x f64 vector value
def v4f64    : VTVec<4,    f64, 154>;  //    4 x f64 vector value
def v8f64    : VTVec<8,    f64, 155>;  //    8 x f64 vector value
def v16f64   : VTVec<16,   f64, 156>;  //   16 x f64 vector value
def v32f64   : VTVec<32,   f64, 157>;  //   32 x f64 vector value
def v64f64   : VTVec<64,   f64, 158>;  //   64 x f64 vector value
def v128f64  : VTVec<128,  f64, 159>;  //  128 x f64 vector value
def v256f64  : VTVec<256,  f64, 160>;  //  256 x f64 vector value

def nxv1i1  : VTScalableVec<1,  i1, 161>;  // n x  1 x i1  vector value
def nxv2i1  : VTScalableVec<2,  i1, 162>;  // n x  2 x i1  vector value
def nxv4i1  : VTScalableVec<4,  i1, 163>;  // n x  4 x i1  vector value
def nxv8i1  : VTScalableVec<8,  i1, 164>;  // n x  8 x i1  vector value
def nxv16i1 : VTScalableVec<16, i1, 165>;  // n x 16 x i1  vector value
def nxv32i1 : VTScalableVec<32, i1, 166>;  // n x 32 x i1  vector value
def nxv64i1 : VTScalableVec<64, i1, 167>;  // n x 64 x i1  vector value

def nxv1i8  : VTScalableVec<1,  i8, 168>;  // n x  1 x i8  vector value
def nxv2i8  : VTScalableVec<2,  i8, 169>;  // n x  2 x i8  vector value
def nxv4i8  : VTScalableVec<4,  i8, 170>;  // n x  4 x i8  vector value
def nxv8i8  : VTScalableVec<8,  i8, 171>;  // n x  8 x i8  vector value
def nxv16i8 : VTScalableVec<16, i8, 172>;  // n x 16 x i8  vector value
def nxv32i8 : VTScalableVec<32, i8, 173>;  // n x 32 x i8  vector value
def nxv64i8 : VTScalableVec<64, i8, 174>;  // n x 64 x i8  vector value

def nxv1i16  : VTScalableVec<1,  i16, 175>;  // n x  1 x i16 vector value
def nxv2i16  : VTScalableVec<2,  i16, 176>;  // n x  2 x i16 vector value
def nxv4i16  : VTScalableVec<4,  i16, 177>;  // n x  4 x i16 vector value
def nxv8i16  : VTScalableVec<8,  i16, 178>;  // n x  8 x i16 vector value
def nxv16i16 : VTScalableVec<16, i16, 179>;  // n x 16 x i16 vector value
def nxv32i16 : VTScalableVec<32, i16, 180>;  // n x 32 x i16 vector value

def nxv1i32  : VTScalableVec<1,  i32, 181>;  // n x  1 x i32 vector value
def nxv2i32  : VTScalableVec<2,  i32, 182>;  // n x  2 x i32 vector value
def nxv4i32  : VTScalableVec<4,  i32, 183>;  // n x  4 x i32 vector value
def nxv8i32  : VTScalableVec<8,  i32, 184>;  // n x  8 x i32 vector value
def nxv16i32 : VTScalableVec<16, i32, 185>;  // n x 16 x i32 vector value
def nxv32i32 : VTScalableVec<32, i32, 186>;  // n x 32 x i32 vector value

def nxv1i64  : VTScalableVec<1,  i64, 187>;  // n x  1 x i64 vector value
def nxv2i64  : VTScalableVec<2,  i64, 188>;  // n x  2 x i64 vector value
def nxv4i64  : VTScalableVec<4,  i64, 189>;  // n x  4 x i64 vector value
def nxv8i64  : VTScalableVec<8,  i64, 190>;  // n x  8 x i64 vector value
def nxv16i64 : VTScalableVec<16, i64, 191>;  // n x 16 x i64 vector value
def nxv32i64 : VTScalableVec<32, i64, 192>;  // n x 32 x i64 vector value

def nxv1f8e4m3fn  : VTScalableVec<1,  f8e4m3fn, 193>;  // n x  1 x  f8e4m3fn vector value
def nxv2f8e4m3fn  : VTScalableVec<2,  f8e4m3fn, 194>;  // n x  2 x  f8e4m3fn vector value
def nxv4f8e4m3fn  : VTScalableVec<4,  f8e4m3fn, 195>;  // n x  4 x  f8e4m3fn vector value
def nxv8f8e4m3fn  : VTScalableVec<8,  f8e4m3fn, 196>;  // n x  8 x  f8e4m3fn vector value
def nxv16f8e4m3fn : VTScalableVec<16, f8e4m3fn, 197>;  // n x 16 x  f8e4m3fn vector value
def nxv32f8e4m3fn : VTScalableVec<32, f8e4m3fn, 198>;  // n x 32 x  f8e4m3fn vector value

def nxv1f8e5m2  : VTScalableVec<1,  f8e5m2, 199>;  // n x  1 x  f8e5m2 vector value
def nxv2f8e5m2  : VTScalableVec<2,  f8e5m2, 200>;  // n x  2 x  f8e5m2 vector value
def nxv4f8e5m2  : VTScalableVec<4,  f8e5m2, 201>;  // n x  4 x  f8e5m2 vector value
def nxv8f8e5m2  : VTScalableVec<8,  f8e5m2, 202>;  // n x  8 x  f8e5m2 vector value
def nxv16f8e5m2 : VTScalableVec<16, f8e5m2, 203>;  // n x 16 x  f8e5m2 vector value
def nxv32f8e5m2 : VTScalableVec<32, f8e5m2, 204>;  // n x 32 x  f8e5m2 vector value

def nxv1f16  : VTScalableVec<1,  f16, 205>;  // n x  1 x  f16 vector value
def nxv2f16  : VTScalableVec<2,  f16, 206>;  // n x  2 x  f16 vector value
def nxv4f16  : VTScalableVec<4,  f16, 207>;  // n x  4 x  f16 vector value
def nxv8f16  : VTScalableVec<8,  f16, 208>;  // n x  8 x  f16 vector value
def nxv16f16 : VTScalableVec<16, f16, 209>;  // n x 16 x  f16 vector value
def nxv32f16 : VTScalableVec<32, f16, 210>;  // n x 32 x  f16 vector value

def nxv1bf16  : VTScalableVec<1,  bf16, 211>;  // n x  1 x bf16 vector value
def nxv2bf16  : VTScalableVec<2,  bf16, 212>;  // n x  2 x bf16 vector value
def nxv4bf16  : VTScalableVec<4,  bf16, 213>;  // n x  4 x bf16 vector value
def nxv8bf16  : VTScalableVec<8,  bf16, 214>;  // n x  8 x bf16 vector value
def nxv16bf16 : VTScalableVec<16, bf16, 215>;  // n x 16 x bf16 vector value
def nxv32bf16 : VTScalableVec<32, bf16, 216>;  // n x 32 x bf16 vector value

def nxv1f32  : VTScalableVec<1,  f32, 217>;  // n x  1 x  f32 vector value
def nxv2f32  : VTScalableVec<2,  f32, 218>;  // n x  2 x  f32 vector value
def nxv4f32  : VTScalableVec<4,  f32, 219>;  // n x  4 x  f32 vector value
def nxv8f32  : VTScalableVec<8,  f32, 220>;  // n x  8 x  f32 vector value
def nxv16f32 : VTScalableVec<16, f32, 221>;  // n x 16 x  f32 vector value

def nxv1f64  : VTScalableVec<1,  f64, 222>;  // n x  1 x  f64 vector value
def nxv2f64  : VTScalableVec<2,  f64, 223>;  // n x  2 x  f64 vector value
def nxv4f64  : VTScalableVec<4,  f64, 224>;  // n x  4 x  f64 vector value
def nxv8f64  : VTScalableVec<8,  f64, 225>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   226>;  // X86 MMX value
def FlagVT    : ValueType<0,    227> { // Pre-RA sched glue
  let LLVMName = "Glue";
}
def isVoid    : ValueType<0,    228>;  // Produces no value
def untyped   : ValueType<8,    229> { // Produces an untyped value
  let LLVMName = "Untyped";
}
def funcref   : ValueType<0,    230>;  // WebAssembly's funcref type
def externref : ValueType<0,    231>;  // WebAssembly's externref type
def x86amx    : ValueType<8192, 232>;  // X86 AMX value
def i64x8     : ValueType<512,  233>;  // 8 Consecutive GPRs (AArch64)
def aarch64svcount
              : ValueType<16,  234>;  // AArch64 predicate-as-counter
def spirvbuiltin : ValueType<0, 235>; // SPIR-V's builtin type

def token      : ValueType<0, 248>;  // TokenTy
def MetadataVT : ValueType<0, 249> { // Metadata
  let LLVMName = "Metadata";
}

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny    : VTAny<250>;

// Pseudo valuetype to represent "vector of any size"
def vAny       : VTAny<251>;

// Pseudo valuetype to represent "float of any format"
def fAny       : VTAny<252>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny       : VTAny<253>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR       : ValueType<0, 254>;

// Pseudo valuetype to represent "any type of any size".
def Any        : VTAny<255>;

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
