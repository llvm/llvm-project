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
  int NF = 0;
  bit isRISCVVecTuple = false;
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

class VTVecTup<int size, int nf, ValueType dummy_elt, int value>
    : ValueType<size, value> {
  let NF = nf;
  let ElementType = dummy_elt;
  let isRISCVVecTuple = true;
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
def v5i1    : VTVec<5,    i1, 21>;  //    5 x i1 vector value
def v6i1    : VTVec<6,    i1, 22>;  //    6 x i1 vector value
def v7i1    : VTVec<7,    i1, 23>;  //    7 x i1 vector value
def v8i1    : VTVec<8,    i1, 24>;  //    8 x i1 vector value
def v16i1   : VTVec<16,   i1, 25>;  //   16 x i1 vector value
def v32i1   : VTVec<32,   i1, 26>;  //   32 x i1 vector value
def v64i1   : VTVec<64,   i1, 27>;  //   64 x i1 vector value
def v128i1  : VTVec<128,  i1, 28>;  //  128 x i1 vector value
def v256i1  : VTVec<256,  i1, 29>;  //  256 x i1 vector value
def v512i1  : VTVec<512,  i1, 30>;  //  512 x i1 vector value
def v1024i1 : VTVec<1024, i1, 31>;  // 1024 x i1 vector value
def v2048i1 : VTVec<2048, i1, 32>;  // 2048 x i1 vector value
def v4096i1 : VTVec<4096, i1, 33>;  // 4096 x i1 vector value

def v128i2  : VTVec<128,  i2, 34>;   //  128 x i2 vector value
def v256i2  : VTVec<256,  i2, 35>;   //  256 x i2 vector value

def v64i4   : VTVec<64,   i4, 36>;   //   64 x i4 vector value
def v128i4  : VTVec<128,  i4, 37>;   //  128 x i4 vector value

def v1i8    : VTVec<1,    i8, 38>;  //    1 x i8 vector value
def v2i8    : VTVec<2,    i8, 39>;  //    2 x i8 vector value
def v3i8    : VTVec<3,    i8, 40>;  //    3 x i8 vector value
def v4i8    : VTVec<4,    i8, 41>;  //    4 x i8 vector value
def v5i8    : VTVec<5,    i8, 42>;  //    5 x i8 vector value
def v6i8    : VTVec<6,    i8, 43>;  //    6 x i8 vector value
def v7i8    : VTVec<7,    i8, 44>;  //    7 x i8 vector value
def v8i8    : VTVec<8,    i8, 45>;  //    8 x i8 vector value
def v16i8   : VTVec<16,   i8, 46>;  //   16 x i8 vector value
def v32i8   : VTVec<32,   i8, 47>;  //   32 x i8 vector value
def v64i8   : VTVec<64,   i8, 48>;  //   64 x i8 vector value
def v128i8  : VTVec<128,  i8, 49>;  //  128 x i8 vector value
def v256i8  : VTVec<256,  i8, 50>;  //  256 x i8 vector value
def v512i8  : VTVec<512,  i8, 51>;  //  512 x i8 vector value
def v1024i8 : VTVec<1024, i8, 52>;  // 1024 x i8 vector value

def v1i16    : VTVec<1,    i16, 53>;  //    1 x i16 vector value
def v2i16    : VTVec<2,    i16, 54>;  //    2 x i16 vector value
def v3i16    : VTVec<3,    i16, 55>;  //    3 x i16 vector value
def v4i16    : VTVec<4,    i16, 56>;  //    4 x i16 vector value
def v5i16    : VTVec<5,    i16, 57>;  //    5 x i16 vector value
def v6i16    : VTVec<6,    i16, 58>;  //    6 x i16 vector value
def v7i16    : VTVec<7,    i16, 59>;  //    7 x i16 vector value
def v8i16    : VTVec<8,    i16, 60>;  //    8 x i16 vector value
def v16i16   : VTVec<16,   i16, 61>;  //   16 x i16 vector value
def v32i16   : VTVec<32,   i16, 62>;  //   32 x i16 vector value
def v64i16   : VTVec<64,   i16, 63>;  //   64 x i16 vector value
def v128i16  : VTVec<128,  i16, 64>;  //  128 x i16 vector value
def v256i16  : VTVec<256,  i16, 65>;  //  256 x i16 vector value
def v512i16  : VTVec<512,  i16, 66>;  //  512 x i16 vector value
def v4096i16 : VTVec<4096, i16, 67>;  // 4096 x i16 vector value

def v1i32    : VTVec<1,    i32, 68>;  //    1 x i32 vector value
def v2i32    : VTVec<2,    i32, 69>;  //    2 x i32 vector value
def v3i32    : VTVec<3,    i32, 70>;  //    3 x i32 vector value
def v4i32    : VTVec<4,    i32, 71>;  //    4 x i32 vector value
def v5i32    : VTVec<5,    i32, 72>;  //    5 x i32 vector value
def v6i32    : VTVec<6,    i32, 73>;  //    6 x i32 vector value
def v7i32    : VTVec<7,    i32, 74>;  //    7 x i32 vector value
def v8i32    : VTVec<8,    i32, 75>;  //    8 x i32 vector value
def v9i32    : VTVec<9,    i32, 76>;  //    9 x i32 vector value
def v10i32   : VTVec<10,   i32, 77>;  //   10 x i32 vector value
def v11i32   : VTVec<11,   i32, 78>;  //   11 x i32 vector value
def v12i32   : VTVec<12,   i32, 79>;  //   12 x i32 vector value
def v16i32   : VTVec<16,   i32, 80>;  //   16 x i32 vector value
def v32i32   : VTVec<32,   i32, 81>;  //   32 x i32 vector value
def v64i32   : VTVec<64,   i32, 82>;  //   64 x i32 vector value
def v128i32  : VTVec<128,  i32, 83>;  //  128 x i32 vector value
def v256i32  : VTVec<256,  i32, 84>;  //  256 x i32 vector value
def v512i32  : VTVec<512,  i32, 85>;  //  512 x i32 vector value
def v1024i32 : VTVec<1024, i32, 86>;  // 1024 x i32 vector value
def v2048i32 : VTVec<2048, i32, 87>;  // 2048 x i32 vector value
def v4096i32 : VTVec<4096, i32, 88>;  // 4096 x i32 vector value

def v1i64   : VTVec<1,   i64, 89>;  //   1 x i64 vector value
def v2i64   : VTVec<2,   i64, 90>;  //   2 x i64 vector value
def v3i64   : VTVec<3,   i64, 91>;  //   3 x i64 vector value
def v4i64   : VTVec<4,   i64, 92>;  //   4 x i64 vector value
def v8i64   : VTVec<8,   i64, 93>;  //   8 x i64 vector value
def v16i64  : VTVec<16,  i64, 94>;  //  16 x i64 vector value
def v32i64  : VTVec<32,  i64, 95>;  //  32 x i64 vector value
def v64i64  : VTVec<64,  i64, 96>;  //  64 x i64 vector value
def v128i64 : VTVec<128, i64, 97>;  // 128 x i64 vector value
def v256i64 : VTVec<256, i64, 98>;  // 256 x i64 vector value

def v1i128  : VTVec<1,  i128, 99>;  //  1 x i128 vector value

def v1f16    : VTVec<1,    f16, 100>;  //    1 x f16 vector value
def v2f16    : VTVec<2,    f16, 101>;  //    2 x f16 vector value
def v3f16    : VTVec<3,    f16, 102>;  //    3 x f16 vector value
def v4f16    : VTVec<4,    f16, 103>;  //    4 x f16 vector value
def v5f16    : VTVec<5,    f16, 104>;  //    5 x f16 vector value
def v6f16    : VTVec<6,    f16, 105>;  //    6 x f16 vector value
def v7f16    : VTVec<7,    f16, 106>;  //    7 x f16 vector value
def v8f16    : VTVec<8,    f16, 107>;  //    8 x f16 vector value
def v16f16   : VTVec<16,   f16, 108>;  //   16 x f16 vector value
def v32f16   : VTVec<32,   f16, 109>;  //   32 x f16 vector value
def v64f16   : VTVec<64,   f16, 110>;  //   64 x f16 vector value
def v128f16  : VTVec<128,  f16, 111>;  //  128 x f16 vector value
def v256f16  : VTVec<256,  f16, 112>;  //  256 x f16 vector value
def v512f16  : VTVec<512,  f16, 113>;  //  512 x f16 vector value
def v4096f16 : VTVec<4096, f16, 114>;  // 4096 x f16 vector value

def v1bf16    : VTVec<1,    bf16, 115>;  //    1 x bf16 vector value
def v2bf16    : VTVec<2,    bf16, 116>;  //    2 x bf16 vector value
def v3bf16    : VTVec<3,    bf16, 117>;  //    3 x bf16 vector value
def v4bf16    : VTVec<4,    bf16, 118>;  //    4 x bf16 vector value
def v8bf16    : VTVec<8,    bf16, 119>;  //    8 x bf16 vector value
def v16bf16   : VTVec<16,   bf16, 120>;  //   16 x bf16 vector value
def v32bf16   : VTVec<32,   bf16, 121>;  //   32 x bf16 vector value
def v64bf16   : VTVec<64,   bf16, 122>;  //   64 x bf16 vector value
def v128bf16  : VTVec<128,  bf16, 123>;  //  128 x bf16 vector value
def v4096bf16 : VTVec<4096, bf16, 124>;  // 4096 x bf16 vector value

def v1f32    : VTVec<1,    f32, 125>;  //    1 x f32 vector value
def v2f32    : VTVec<2,    f32, 126>;  //    2 x f32 vector value
def v3f32    : VTVec<3,    f32, 127>;  //    3 x f32 vector value
def v4f32    : VTVec<4,    f32, 128>;  //    4 x f32 vector value
def v5f32    : VTVec<5,    f32, 129>;  //    5 x f32 vector value
def v6f32    : VTVec<6,    f32, 130>;  //    6 x f32 vector value
def v7f32    : VTVec<7,    f32, 131>;  //    7 x f32 vector value
def v8f32    : VTVec<8,    f32, 132>;  //    8 x f32 vector value
def v9f32    : VTVec<9,    f32, 133>;  //    9 x f32 vector value
def v10f32   : VTVec<10,   f32, 134>;  //   10 x f32 vector value
def v11f32   : VTVec<11,   f32, 135>;  //   11 x f32 vector value
def v12f32   : VTVec<12,   f32, 136>;  //   12 x f32 vector value
def v16f32   : VTVec<16,   f32, 137>;  //   16 x f32 vector value
def v32f32   : VTVec<32,   f32, 138>;  //   32 x f32 vector value
def v64f32   : VTVec<64,   f32, 139>;  //   64 x f32 vector value
def v128f32  : VTVec<128,  f32, 140>;  //  128 x f32 vector value
def v256f32  : VTVec<256,  f32, 141>;  //  256 x f32 vector value
def v512f32  : VTVec<512,  f32, 142>;  //  512 x f32 vector value
def v1024f32 : VTVec<1024, f32, 143>;  // 1024 x f32 vector value
def v2048f32 : VTVec<2048, f32, 144>;  // 2048 x f32 vector value

def v1f64    : VTVec<1,    f64, 145>;  //    1 x f64 vector value
def v2f64    : VTVec<2,    f64, 146>;  //    2 x f64 vector value
def v3f64    : VTVec<3,    f64, 147>;  //    3 x f64 vector value
def v4f64    : VTVec<4,    f64, 148>;  //    4 x f64 vector value
def v8f64    : VTVec<8,    f64, 149>;  //    8 x f64 vector value
def v16f64   : VTVec<16,   f64, 150>;  //   16 x f64 vector value
def v32f64   : VTVec<32,   f64, 151>;  //   32 x f64 vector value
def v64f64   : VTVec<64,   f64, 152>;  //   64 x f64 vector value
def v128f64  : VTVec<128,  f64, 153>;  //  128 x f64 vector value
def v256f64  : VTVec<256,  f64, 154>;  //  256 x f64 vector value

def nxv1i1  : VTScalableVec<1,  i1, 155>;  // n x  1 x i1  vector value
def nxv2i1  : VTScalableVec<2,  i1, 156>;  // n x  2 x i1  vector value
def nxv4i1  : VTScalableVec<4,  i1, 157>;  // n x  4 x i1  vector value
def nxv8i1  : VTScalableVec<8,  i1, 158>;  // n x  8 x i1  vector value
def nxv16i1 : VTScalableVec<16, i1, 159>;  // n x 16 x i1  vector value
def nxv32i1 : VTScalableVec<32, i1, 160>;  // n x 32 x i1  vector value
def nxv64i1 : VTScalableVec<64, i1, 161>;  // n x 64 x i1  vector value

def nxv1i8  : VTScalableVec<1,  i8, 162>;  // n x  1 x i8  vector value
def nxv2i8  : VTScalableVec<2,  i8, 163>;  // n x  2 x i8  vector value
def nxv4i8  : VTScalableVec<4,  i8, 164>;  // n x  4 x i8  vector value
def nxv8i8  : VTScalableVec<8,  i8, 165>;  // n x  8 x i8  vector value
def nxv16i8 : VTScalableVec<16, i8, 166>;  // n x 16 x i8  vector value
def nxv32i8 : VTScalableVec<32, i8, 167>;  // n x 32 x i8  vector value
def nxv64i8 : VTScalableVec<64, i8, 168>;  // n x 64 x i8  vector value

def nxv1i16  : VTScalableVec<1,  i16, 169>;  // n x  1 x i16 vector value
def nxv2i16  : VTScalableVec<2,  i16, 170>;  // n x  2 x i16 vector value
def nxv4i16  : VTScalableVec<4,  i16, 171>;  // n x  4 x i16 vector value
def nxv8i16  : VTScalableVec<8,  i16, 172>;  // n x  8 x i16 vector value
def nxv16i16 : VTScalableVec<16, i16, 173>;  // n x 16 x i16 vector value
def nxv32i16 : VTScalableVec<32, i16, 174>;  // n x 32 x i16 vector value

def nxv1i32  : VTScalableVec<1,  i32, 175>;  // n x  1 x i32 vector value
def nxv2i32  : VTScalableVec<2,  i32, 176>;  // n x  2 x i32 vector value
def nxv4i32  : VTScalableVec<4,  i32, 177>;  // n x  4 x i32 vector value
def nxv8i32  : VTScalableVec<8,  i32, 178>;  // n x  8 x i32 vector value
def nxv16i32 : VTScalableVec<16, i32, 179>;  // n x 16 x i32 vector value
def nxv32i32 : VTScalableVec<32, i32, 180>;  // n x 32 x i32 vector value

def nxv1i64  : VTScalableVec<1,  i64, 181>;  // n x  1 x i64 vector value
def nxv2i64  : VTScalableVec<2,  i64, 182>;  // n x  2 x i64 vector value
def nxv4i64  : VTScalableVec<4,  i64, 183>;  // n x  4 x i64 vector value
def nxv8i64  : VTScalableVec<8,  i64, 184>;  // n x  8 x i64 vector value
def nxv16i64 : VTScalableVec<16, i64, 185>;  // n x 16 x i64 vector value
def nxv32i64 : VTScalableVec<32, i64, 186>;  // n x 32 x i64 vector value

def nxv1f16  : VTScalableVec<1,  f16, 187>;  // n x  1 x  f16 vector value
def nxv2f16  : VTScalableVec<2,  f16, 188>;  // n x  2 x  f16 vector value
def nxv4f16  : VTScalableVec<4,  f16, 189>;  // n x  4 x  f16 vector value
def nxv8f16  : VTScalableVec<8,  f16, 190>;  // n x  8 x  f16 vector value
def nxv16f16 : VTScalableVec<16, f16, 191>;  // n x 16 x  f16 vector value
def nxv32f16 : VTScalableVec<32, f16, 192>;  // n x 32 x  f16 vector value

def nxv1bf16  : VTScalableVec<1,  bf16, 193>;  // n x  1 x bf16 vector value
def nxv2bf16  : VTScalableVec<2,  bf16, 194>;  // n x  2 x bf16 vector value
def nxv4bf16  : VTScalableVec<4,  bf16, 195>;  // n x  4 x bf16 vector value
def nxv8bf16  : VTScalableVec<8,  bf16, 196>;  // n x  8 x bf16 vector value
def nxv16bf16 : VTScalableVec<16, bf16, 197>;  // n x 16 x bf16 vector value
def nxv32bf16 : VTScalableVec<32, bf16, 198>;  // n x 32 x bf16 vector value

def nxv1f32  : VTScalableVec<1,  f32, 199>;  // n x  1 x  f32 vector value
def nxv2f32  : VTScalableVec<2,  f32, 200>;  // n x  2 x  f32 vector value
def nxv4f32  : VTScalableVec<4,  f32, 201>;  // n x  4 x  f32 vector value
def nxv8f32  : VTScalableVec<8,  f32, 202>;  // n x  8 x  f32 vector value
def nxv16f32 : VTScalableVec<16, f32, 203>;  // n x 16 x  f32 vector value

def nxv1f64  : VTScalableVec<1,  f64, 204>;  // n x  1 x  f64 vector value
def nxv2f64  : VTScalableVec<2,  f64, 205>;  // n x  2 x  f64 vector value
def nxv4f64  : VTScalableVec<4,  f64, 206>;  // n x  4 x  f64 vector value
def nxv8f64  : VTScalableVec<8,  f64, 207>;  // n x  8 x  f64 vector value

// Sz = NF * MinNumElts * 8(bits)
def riscv_nxv1i8x2   : VTVecTup<16,  2, i8, 208>;  // RISCV vector tuple(min_num_elts=1,  nf=2)
def riscv_nxv1i8x3   : VTVecTup<24,  3, i8, 209>;  // RISCV vector tuple(min_num_elts=1,  nf=3)
def riscv_nxv1i8x4   : VTVecTup<32,  4, i8, 210>;  // RISCV vector tuple(min_num_elts=1,  nf=4)
def riscv_nxv1i8x5   : VTVecTup<40,  5, i8, 211>;  // RISCV vector tuple(min_num_elts=1,  nf=5)
def riscv_nxv1i8x6   : VTVecTup<48,  6, i8, 212>;  // RISCV vector tuple(min_num_elts=1,  nf=6)
def riscv_nxv1i8x7   : VTVecTup<56,  7, i8, 213>;  // RISCV vector tuple(min_num_elts=1,  nf=7)
def riscv_nxv1i8x8   : VTVecTup<64,  8, i8, 214>;  // RISCV vector tuple(min_num_elts=1,  nf=8)
def riscv_nxv2i8x2   : VTVecTup<32,  2, i8, 215>;  // RISCV vector tuple(min_num_elts=2,  nf=2)
def riscv_nxv2i8x3   : VTVecTup<48,  3, i8, 216>;  // RISCV vector tuple(min_num_elts=2,  nf=3)
def riscv_nxv2i8x4   : VTVecTup<64,  4, i8, 217>;  // RISCV vector tuple(min_num_elts=2,  nf=4)
def riscv_nxv2i8x5   : VTVecTup<80,  5, i8, 218>;  // RISCV vector tuple(min_num_elts=2,  nf=5)
def riscv_nxv2i8x6   : VTVecTup<96,  6, i8, 219>;  // RISCV vector tuple(min_num_elts=2,  nf=6)
def riscv_nxv2i8x7   : VTVecTup<112, 7, i8, 220>;  // RISCV vector tuple(min_num_elts=2,  nf=7)
def riscv_nxv2i8x8   : VTVecTup<128, 8, i8, 221>;  // RISCV vector tuple(min_num_elts=2,  nf=8)
def riscv_nxv4i8x2   : VTVecTup<64,  2, i8, 222>;  // RISCV vector tuple(min_num_elts=4,  nf=2)
def riscv_nxv4i8x3   : VTVecTup<96,  3, i8, 223>;  // RISCV vector tuple(min_num_elts=4,  nf=3)
def riscv_nxv4i8x4   : VTVecTup<128, 4, i8, 224>;  // RISCV vector tuple(min_num_elts=4,  nf=4)
def riscv_nxv4i8x5   : VTVecTup<160, 5, i8, 225>;  // RISCV vector tuple(min_num_elts=4,  nf=5)
def riscv_nxv4i8x6   : VTVecTup<192, 6, i8, 226>;  // RISCV vector tuple(min_num_elts=4,  nf=6)
def riscv_nxv4i8x7   : VTVecTup<224, 7, i8, 227>;  // RISCV vector tuple(min_num_elts=4,  nf=7)
def riscv_nxv4i8x8   : VTVecTup<256, 8, i8, 228>;  // RISCV vector tuple(min_num_elts=4,  nf=8)
def riscv_nxv8i8x2   : VTVecTup<128, 2, i8, 229>;  // RISCV vector tuple(min_num_elts=8,  nf=2)
def riscv_nxv8i8x3   : VTVecTup<192, 3, i8, 230>;  // RISCV vector tuple(min_num_elts=8,  nf=3)
def riscv_nxv8i8x4   : VTVecTup<256, 4, i8, 231>;  // RISCV vector tuple(min_num_elts=8,  nf=4)
def riscv_nxv8i8x5   : VTVecTup<320, 5, i8, 232>;  // RISCV vector tuple(min_num_elts=8,  nf=5)
def riscv_nxv8i8x6   : VTVecTup<384, 6, i8, 233>;  // RISCV vector tuple(min_num_elts=8,  nf=6)
def riscv_nxv8i8x7   : VTVecTup<448, 7, i8, 234>;  // RISCV vector tuple(min_num_elts=8,  nf=7)
def riscv_nxv8i8x8   : VTVecTup<512, 8, i8, 235>;  // RISCV vector tuple(min_num_elts=8,  nf=8)
def riscv_nxv16i8x2  : VTVecTup<256, 2, i8, 236>;  // RISCV vector tuple(min_num_elts=16, nf=2)
def riscv_nxv16i8x3  : VTVecTup<384, 3, i8, 237>;  // RISCV vector tuple(min_num_elts=16, nf=3)
def riscv_nxv16i8x4  : VTVecTup<512, 4, i8, 238>;  // RISCV vector tuple(min_num_elts=16, nf=4)
def riscv_nxv32i8x2  : VTVecTup<512, 2, i8, 239>;  // RISCV vector tuple(min_num_elts=32, nf=2)

def x86mmx    : ValueType<64,   240>;  // X86 MMX value
def Glue      : ValueType<0,    241>;  // Pre-RA sched glue
def isVoid    : ValueType<0,    242>;  // Produces no value
def untyped   : ValueType<8,    243> { // Produces an untyped value
  let LLVMName = "Untyped";
}
def funcref   : ValueType<0,    244>;  // WebAssembly's funcref type
def externref : ValueType<0,    245>;  // WebAssembly's externref type
def exnref    : ValueType<0,    246>;  // WebAssembly's exnref type
def x86amx    : ValueType<8192, 247>;  // X86 AMX value
def i64x8     : ValueType<512,  248>;  // 8 Consecutive GPRs (AArch64)
def aarch64svcount
              : ValueType<16,  249>;  // AArch64 predicate-as-counter
def spirvbuiltin : ValueType<0, 250>; // SPIR-V's builtin type
// AMDGPU buffer fat pointer, buffer rsrc + offset, rewritten before MIR translation.
// FIXME: Remove this and the getPointerType() override if MVT::i160 is added.
def amdgpuBufferFatPointer : ValueType<160, 251>;
// AMDGPU buffer strided pointer, buffer rsrc + index + offset, doesn't reach MIR.
// FIXME: Remove this and the getPointerType() override if MVT::i82 is added.
def amdgpuBufferStridedPointer : ValueType<192, 252>;

def aarch64mfp8 : ValueType<8,  253>;  // 8-bit value in FPR (AArch64)

let isNormalValueType = false in {
def token      : ValueType<0, 504>;  // TokenTy
def MetadataVT : ValueType<0, 505> { // Metadata
  let LLVMName = "Metadata";
}

// Pseudo valuetype to represent "pointer to any address space"
// Should only be used in TableGen.
def pAny       : VTAny<506>;

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
