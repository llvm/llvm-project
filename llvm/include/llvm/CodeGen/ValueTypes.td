//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ValueTypes.td - list of ValueType instances supported by the
// CodeGen infrastructure.
//
//===----------------------------------------------------------------------===//

class ValueType<int size, string llvm_name = NAME> {
  string Namespace = "MVT";
  string LLVMName = llvm_name;
  int Size = size;
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
  bit isCheriCapability = false;
}

class VTAny : ValueType<0> {
  let isOverloaded = true;
}

class VTInt<int size>
    : ValueType<size> {
  let isInteger = true;
}

class VTFP<int size>
    : ValueType<size> {
  let isFP = true;
}

class VTVec<int nelem, ValueType elt, string llvm_name = NAME>
    : ValueType<!mul(nelem, elt.Size), llvm_name> {
  let nElem = nelem;
  let ElementType = elt;
  let isInteger = elt.isInteger;
  let isFP = elt.isFP;
  let isVector = true;
}

class VTScalableVec<int nelem, ValueType elt>
    : VTVec<nelem, elt> {
  let isScalable = true;
}

class VTVecTup<int size, int nf, ValueType dummy_elt>
    : ValueType<size> {
  let NF = nf;
  let ElementType = dummy_elt;
  let isRISCVVecTuple = true;
}

class VTCheriCapability<int size> : ValueType<size> {
  let isCheriCapability = true;
}

defset list<ValueType> ValueTypes = {

def OtherVT : ValueType<0, "Other">; // "Other" value

def i1      : VTInt<1>;   // One bit boolean value
def i2      : VTInt<2>;   // 2-bit integer value
def i4      : VTInt<4>;   // 4-bit integer value
def i8      : VTInt<8>;   // 8-bit integer value
def i16     : VTInt<16>;  // 16-bit integer value
def i32     : VTInt<32>;  // 32-bit integer value
def i64     : VTInt<64>;  // 64-bit integer value
def i128    : VTInt<128>; // 128-bit integer value
def i256    : VTInt<256>; // 256-bit integer value
def i512    : VTInt<512>; // 512-bit integer value

def bf16    : VTFP<16>;  // 16-bit brain floating point value
def f16     : VTFP<16>;  // 16-bit floating point value
def f32     : VTFP<32>;  // 32-bit floating point value
def f64     : VTFP<64>;  // 64-bit floating point value
def f80     : VTFP<80>;  // 80-bit floating point value
def f128    : VTFP<128>; // 128-bit floating point value
def ppcf128 : VTFP<128>; // PPC 128-bit floating point value

def v1i1    : VTVec<1,    i1>; //    1 x i1 vector value
def v2i1    : VTVec<2,    i1>; //    2 x i1 vector value
def v3i1    : VTVec<3,    i1>; //    3 x i1 vector value
def v4i1    : VTVec<4,    i1>; //    4 x i1 vector value
def v5i1    : VTVec<5,    i1>; //    5 x i1 vector value
def v6i1    : VTVec<6,    i1>; //    6 x i1 vector value
def v7i1    : VTVec<7,    i1>; //    7 x i1 vector value
def v8i1    : VTVec<8,    i1>; //    8 x i1 vector value
def v16i1   : VTVec<16,   i1>; //   16 x i1 vector value
def v32i1   : VTVec<32,   i1>; //   32 x i1 vector value
def v64i1   : VTVec<64,   i1>; //   64 x i1 vector value
def v128i1  : VTVec<128,  i1>; //  128 x i1 vector value
def v256i1  : VTVec<256,  i1>; //  256 x i1 vector value
def v512i1  : VTVec<512,  i1>; //  512 x i1 vector value
def v1024i1 : VTVec<1024, i1>; // 1024 x i1 vector value
def v2048i1 : VTVec<2048, i1>; // 2048 x i1 vector value
def v4096i1 : VTVec<4096, i1>; // 4096 x i1 vector value

def v128i2  : VTVec<128,  i2>; //  128 x i2 vector value
def v256i2  : VTVec<256,  i2>; //  256 x i2 vector value

def v64i4   : VTVec<64,   i4>; //   64 x i4 vector value
def v128i4  : VTVec<128,  i4>; //  128 x i4 vector value

def v1i8    : VTVec<1,    i8>; //    1 x i8 vector value
def v2i8    : VTVec<2,    i8>; //    2 x i8 vector value
def v3i8    : VTVec<3,    i8>; //    3 x i8 vector value
def v4i8    : VTVec<4,    i8>; //    4 x i8 vector value
def v5i8    : VTVec<5,    i8>; //    5 x i8 vector value
def v6i8    : VTVec<6,    i8>; //    6 x i8 vector value
def v7i8    : VTVec<7,    i8>; //    7 x i8 vector value
def v8i8    : VTVec<8,    i8>; //    8 x i8 vector value
def v16i8   : VTVec<16,   i8>; //   16 x i8 vector value
def v32i8   : VTVec<32,   i8>; //   32 x i8 vector value
def v64i8   : VTVec<64,   i8>; //   64 x i8 vector value
def v128i8  : VTVec<128,  i8>; //  128 x i8 vector value
def v256i8  : VTVec<256,  i8>; //  256 x i8 vector value
def v512i8  : VTVec<512,  i8>; //  512 x i8 vector value
def v1024i8 : VTVec<1024, i8>; // 1024 x i8 vector value

def v1i16    : VTVec<1,    i16>; //    1 x i16 vector value
def v2i16    : VTVec<2,    i16>; //    2 x i16 vector value
def v3i16    : VTVec<3,    i16>; //    3 x i16 vector value
def v4i16    : VTVec<4,    i16>; //    4 x i16 vector value
def v5i16    : VTVec<5,    i16>; //    5 x i16 vector value
def v6i16    : VTVec<6,    i16>; //    6 x i16 vector value
def v7i16    : VTVec<7,    i16>; //    7 x i16 vector value
def v8i16    : VTVec<8,    i16>; //    8 x i16 vector value
def v16i16   : VTVec<16,   i16>; //   16 x i16 vector value
def v32i16   : VTVec<32,   i16>; //   32 x i16 vector value
def v64i16   : VTVec<64,   i16>; //   64 x i16 vector value
def v128i16  : VTVec<128,  i16>; //  128 x i16 vector value
def v256i16  : VTVec<256,  i16>; //  256 x i16 vector value
def v512i16  : VTVec<512,  i16>; //  512 x i16 vector value
def v4096i16 : VTVec<4096, i16>; // 4096 x i16 vector value

def v1i32    : VTVec<1,    i32>; //    1 x i32 vector value
def v2i32    : VTVec<2,    i32>; //    2 x i32 vector value
def v3i32    : VTVec<3,    i32>; //    3 x i32 vector value
def v4i32    : VTVec<4,    i32>; //    4 x i32 vector value
def v5i32    : VTVec<5,    i32>; //    5 x i32 vector value
def v6i32    : VTVec<6,    i32>; //    6 x i32 vector value
def v7i32    : VTVec<7,    i32>; //    7 x i32 vector value
def v8i32    : VTVec<8,    i32>; //    8 x i32 vector value
def v9i32    : VTVec<9,    i32>; //    9 x i32 vector value
def v10i32   : VTVec<10,   i32>; //   10 x i32 vector value
def v11i32   : VTVec<11,   i32>; //   11 x i32 vector value
def v12i32   : VTVec<12,   i32>; //   12 x i32 vector value
def v16i32   : VTVec<16,   i32>; //   16 x i32 vector value
def v24i32   : VTVec<24,   i32>; //   24 x i32 vector value
def v32i32   : VTVec<32,   i32>; //   32 x i32 vector value
def v48i32   : VTVec<48,   i32>; //   48 x i32 vector value
def v64i32   : VTVec<64,   i32>; //   64 x i32 vector value
def v128i32  : VTVec<128,  i32>; //  128 x i32 vector value
def v256i32  : VTVec<256,  i32>; //  256 x i32 vector value
def v512i32  : VTVec<512,  i32>; //  512 x i32 vector value
def v1024i32 : VTVec<1024, i32>; // 1024 x i32 vector value
def v2048i32 : VTVec<2048, i32>; // 2048 x i32 vector value
def v4096i32 : VTVec<4096, i32>; // 4096 x i32 vector value

def v1i64   : VTVec<1,   i64>; //   1 x i64 vector value
def v2i64   : VTVec<2,   i64>; //   2 x i64 vector value
def v3i64   : VTVec<3,   i64>; //   3 x i64 vector value
def v4i64   : VTVec<4,   i64>; //   4 x i64 vector value
def v8i64   : VTVec<8,   i64>; //   8 x i64 vector value
def v16i64  : VTVec<16,  i64>; //  16 x i64 vector value
def v32i64  : VTVec<32,  i64>; //  32 x i64 vector value
def v64i64  : VTVec<64,  i64>; //  64 x i64 vector value
def v128i64 : VTVec<128, i64>; // 128 x i64 vector value
def v256i64 : VTVec<256, i64>; // 256 x i64 vector value

def v1i128  : VTVec<1,  i128>; //  1 x i128 vector value

def v1f16    : VTVec<1,    f16>; //    1 x f16 vector value
def v2f16    : VTVec<2,    f16>; //    2 x f16 vector value
def v3f16    : VTVec<3,    f16>; //    3 x f16 vector value
def v4f16    : VTVec<4,    f16>; //    4 x f16 vector value
def v5f16    : VTVec<5,    f16>; //    5 x f16 vector value
def v6f16    : VTVec<6,    f16>; //    6 x f16 vector value
def v7f16    : VTVec<7,    f16>; //    7 x f16 vector value
def v8f16    : VTVec<8,    f16>; //    8 x f16 vector value
def v16f16   : VTVec<16,   f16>; //   16 x f16 vector value
def v32f16   : VTVec<32,   f16>; //   32 x f16 vector value
def v64f16   : VTVec<64,   f16>; //   64 x f16 vector value
def v128f16  : VTVec<128,  f16>; //  128 x f16 vector value
def v256f16  : VTVec<256,  f16>; //  256 x f16 vector value
def v512f16  : VTVec<512,  f16>; //  512 x f16 vector value
def v4096f16 : VTVec<4096, f16>; // 4096 x f16 vector value

def v1bf16    : VTVec<1,    bf16>; //    1 x bf16 vector value
def v2bf16    : VTVec<2,    bf16>; //    2 x bf16 vector value
def v3bf16    : VTVec<3,    bf16>; //    3 x bf16 vector value
def v4bf16    : VTVec<4,    bf16>; //    4 x bf16 vector value
def v8bf16    : VTVec<8,    bf16>; //    8 x bf16 vector value
def v16bf16   : VTVec<16,   bf16>; //   16 x bf16 vector value
def v32bf16   : VTVec<32,   bf16>; //   32 x bf16 vector value
def v64bf16   : VTVec<64,   bf16>; //   64 x bf16 vector value
def v128bf16  : VTVec<128,  bf16>; //  128 x bf16 vector value
def v256bf16  : VTVec<256,  bf16>; //  256 x bf16 vector value
def v512bf16  : VTVec<512,  bf16>; //  512 x bf16 vector value
def v4096bf16 : VTVec<4096, bf16>; // 4096 x bf16 vector value

def v1f32    : VTVec<1,    f32>; //    1 x f32 vector value
def v2f32    : VTVec<2,    f32>; //    2 x f32 vector value
def v3f32    : VTVec<3,    f32>; //    3 x f32 vector value
def v4f32    : VTVec<4,    f32>; //    4 x f32 vector value
def v5f32    : VTVec<5,    f32>; //    5 x f32 vector value
def v6f32    : VTVec<6,    f32>; //    6 x f32 vector value
def v7f32    : VTVec<7,    f32>; //    7 x f32 vector value
def v8f32    : VTVec<8,    f32>; //    8 x f32 vector value
def v9f32    : VTVec<9,    f32>; //    9 x f32 vector value
def v10f32   : VTVec<10,   f32>; //   10 x f32 vector value
def v11f32   : VTVec<11,   f32>; //   11 x f32 vector value
def v12f32   : VTVec<12,   f32>; //   12 x f32 vector value
def v16f32   : VTVec<16,   f32>; //   16 x f32 vector value
def v32f32   : VTVec<32,   f32>; //   32 x f32 vector value
def v64f32   : VTVec<64,   f32>; //   64 x f32 vector value
def v128f32  : VTVec<128,  f32>; //  128 x f32 vector value
def v256f32  : VTVec<256,  f32>; //  256 x f32 vector value
def v512f32  : VTVec<512,  f32>; //  512 x f32 vector value
def v1024f32 : VTVec<1024, f32>; // 1024 x f32 vector value
def v2048f32 : VTVec<2048, f32>; // 2048 x f32 vector value

def v1f64    : VTVec<1,    f64>; //    1 x f64 vector value
def v2f64    : VTVec<2,    f64>; //    2 x f64 vector value
def v3f64    : VTVec<3,    f64>; //    3 x f64 vector value
def v4f64    : VTVec<4,    f64>; //    4 x f64 vector value
def v8f64    : VTVec<8,    f64>; //    8 x f64 vector value
def v16f64   : VTVec<16,   f64>; //   16 x f64 vector value
def v32f64   : VTVec<32,   f64>; //   32 x f64 vector value
def v64f64   : VTVec<64,   f64>; //   64 x f64 vector value
def v128f64  : VTVec<128,  f64>; //  128 x f64 vector value
def v256f64  : VTVec<256,  f64>; //  256 x f64 vector value

def nxv1i1  : VTScalableVec<1,  i1>; // n x  1 x i1  vector value
def nxv2i1  : VTScalableVec<2,  i1>; // n x  2 x i1  vector value
def nxv4i1  : VTScalableVec<4,  i1>; // n x  4 x i1  vector value
def nxv8i1  : VTScalableVec<8,  i1>; // n x  8 x i1  vector value
def nxv16i1 : VTScalableVec<16, i1>; // n x 16 x i1  vector value
def nxv32i1 : VTScalableVec<32, i1>; // n x 32 x i1  vector value
def nxv64i1 : VTScalableVec<64, i1>; // n x 64 x i1  vector value

def nxv1i8  : VTScalableVec<1,  i8>; // n x  1 x i8  vector value
def nxv2i8  : VTScalableVec<2,  i8>; // n x  2 x i8  vector value
def nxv4i8  : VTScalableVec<4,  i8>; // n x  4 x i8  vector value
def nxv8i8  : VTScalableVec<8,  i8>; // n x  8 x i8  vector value
def nxv16i8 : VTScalableVec<16, i8>; // n x 16 x i8  vector value
def nxv32i8 : VTScalableVec<32, i8>; // n x 32 x i8  vector value
def nxv64i8 : VTScalableVec<64, i8>; // n x 64 x i8  vector value

def nxv1i16  : VTScalableVec<1,  i16>; // n x  1 x i16 vector value
def nxv2i16  : VTScalableVec<2,  i16>; // n x  2 x i16 vector value
def nxv4i16  : VTScalableVec<4,  i16>; // n x  4 x i16 vector value
def nxv8i16  : VTScalableVec<8,  i16>; // n x  8 x i16 vector value
def nxv16i16 : VTScalableVec<16, i16>; // n x 16 x i16 vector value
def nxv32i16 : VTScalableVec<32, i16>; // n x 32 x i16 vector value

def nxv1i32  : VTScalableVec<1,  i32>; // n x  1 x i32 vector value
def nxv2i32  : VTScalableVec<2,  i32>; // n x  2 x i32 vector value
def nxv4i32  : VTScalableVec<4,  i32>; // n x  4 x i32 vector value
def nxv8i32  : VTScalableVec<8,  i32>; // n x  8 x i32 vector value
def nxv16i32 : VTScalableVec<16, i32>; // n x 16 x i32 vector value
def nxv32i32 : VTScalableVec<32, i32>; // n x 32 x i32 vector value

def nxv1i64  : VTScalableVec<1,  i64>; // n x  1 x i64 vector value
def nxv2i64  : VTScalableVec<2,  i64>; // n x  2 x i64 vector value
def nxv4i64  : VTScalableVec<4,  i64>; // n x  4 x i64 vector value
def nxv8i64  : VTScalableVec<8,  i64>; // n x  8 x i64 vector value
def nxv16i64 : VTScalableVec<16, i64>; // n x 16 x i64 vector value
def nxv32i64 : VTScalableVec<32, i64>; // n x 32 x i64 vector value

def nxv1f16  : VTScalableVec<1,  f16>; // n x  1 x  f16 vector value
def nxv2f16  : VTScalableVec<2,  f16>; // n x  2 x  f16 vector value
def nxv4f16  : VTScalableVec<4,  f16>; // n x  4 x  f16 vector value
def nxv8f16  : VTScalableVec<8,  f16>; // n x  8 x  f16 vector value
def nxv16f16 : VTScalableVec<16, f16>; // n x 16 x  f16 vector value
def nxv32f16 : VTScalableVec<32, f16>; // n x 32 x  f16 vector value

def nxv1bf16  : VTScalableVec<1,  bf16>; // n x  1 x bf16 vector value
def nxv2bf16  : VTScalableVec<2,  bf16>; // n x  2 x bf16 vector value
def nxv4bf16  : VTScalableVec<4,  bf16>; // n x  4 x bf16 vector value
def nxv8bf16  : VTScalableVec<8,  bf16>; // n x  8 x bf16 vector value
def nxv16bf16 : VTScalableVec<16, bf16>; // n x 16 x bf16 vector value
def nxv32bf16 : VTScalableVec<32, bf16>; // n x 32 x bf16 vector value

def nxv1f32  : VTScalableVec<1,  f32>; // n x  1 x  f32 vector value
def nxv2f32  : VTScalableVec<2,  f32>; // n x  2 x  f32 vector value
def nxv4f32  : VTScalableVec<4,  f32>; // n x  4 x  f32 vector value
def nxv8f32  : VTScalableVec<8,  f32>; // n x  8 x  f32 vector value
def nxv16f32 : VTScalableVec<16, f32>; // n x 16 x  f32 vector value

def nxv1f64  : VTScalableVec<1,  f64>; // n x  1 x  f64 vector value
def nxv2f64  : VTScalableVec<2,  f64>; // n x  2 x  f64 vector value
def nxv4f64  : VTScalableVec<4,  f64>; // n x  4 x  f64 vector value
def nxv8f64  : VTScalableVec<8,  f64>; // n x  8 x  f64 vector value

// Sz = NF * MinNumElts * 8(bits)
def riscv_nxv1i8x2   : VTVecTup<16,  2, i8>; // RISCV vector tuple(min_num_elts=1,  nf=2)
def riscv_nxv1i8x3   : VTVecTup<24,  3, i8>; // RISCV vector tuple(min_num_elts=1,  nf=3)
def riscv_nxv1i8x4   : VTVecTup<32,  4, i8>; // RISCV vector tuple(min_num_elts=1,  nf=4)
def riscv_nxv1i8x5   : VTVecTup<40,  5, i8>; // RISCV vector tuple(min_num_elts=1,  nf=5)
def riscv_nxv1i8x6   : VTVecTup<48,  6, i8>; // RISCV vector tuple(min_num_elts=1,  nf=6)
def riscv_nxv1i8x7   : VTVecTup<56,  7, i8>; // RISCV vector tuple(min_num_elts=1,  nf=7)
def riscv_nxv1i8x8   : VTVecTup<64,  8, i8>; // RISCV vector tuple(min_num_elts=1,  nf=8)
def riscv_nxv2i8x2   : VTVecTup<32,  2, i8>; // RISCV vector tuple(min_num_elts=2,  nf=2)
def riscv_nxv2i8x3   : VTVecTup<48,  3, i8>; // RISCV vector tuple(min_num_elts=2,  nf=3)
def riscv_nxv2i8x4   : VTVecTup<64,  4, i8>; // RISCV vector tuple(min_num_elts=2,  nf=4)
def riscv_nxv2i8x5   : VTVecTup<80,  5, i8>; // RISCV vector tuple(min_num_elts=2,  nf=5)
def riscv_nxv2i8x6   : VTVecTup<96,  6, i8>; // RISCV vector tuple(min_num_elts=2,  nf=6)
def riscv_nxv2i8x7   : VTVecTup<112, 7, i8>; // RISCV vector tuple(min_num_elts=2,  nf=7)
def riscv_nxv2i8x8   : VTVecTup<128, 8, i8>; // RISCV vector tuple(min_num_elts=2,  nf=8)
def riscv_nxv4i8x2   : VTVecTup<64,  2, i8>; // RISCV vector tuple(min_num_elts=4,  nf=2)
def riscv_nxv4i8x3   : VTVecTup<96,  3, i8>; // RISCV vector tuple(min_num_elts=4,  nf=3)
def riscv_nxv4i8x4   : VTVecTup<128, 4, i8>; // RISCV vector tuple(min_num_elts=4,  nf=4)
def riscv_nxv4i8x5   : VTVecTup<160, 5, i8>; // RISCV vector tuple(min_num_elts=4,  nf=5)
def riscv_nxv4i8x6   : VTVecTup<192, 6, i8>; // RISCV vector tuple(min_num_elts=4,  nf=6)
def riscv_nxv4i8x7   : VTVecTup<224, 7, i8>; // RISCV vector tuple(min_num_elts=4,  nf=7)
def riscv_nxv4i8x8   : VTVecTup<256, 8, i8>; // RISCV vector tuple(min_num_elts=4,  nf=8)
def riscv_nxv8i8x2   : VTVecTup<128, 2, i8>; // RISCV vector tuple(min_num_elts=8,  nf=2)
def riscv_nxv8i8x3   : VTVecTup<192, 3, i8>; // RISCV vector tuple(min_num_elts=8,  nf=3)
def riscv_nxv8i8x4   : VTVecTup<256, 4, i8>; // RISCV vector tuple(min_num_elts=8,  nf=4)
def riscv_nxv8i8x5   : VTVecTup<320, 5, i8>; // RISCV vector tuple(min_num_elts=8,  nf=5)
def riscv_nxv8i8x6   : VTVecTup<384, 6, i8>; // RISCV vector tuple(min_num_elts=8,  nf=6)
def riscv_nxv8i8x7   : VTVecTup<448, 7, i8>; // RISCV vector tuple(min_num_elts=8,  nf=7)
def riscv_nxv8i8x8   : VTVecTup<512, 8, i8>; // RISCV vector tuple(min_num_elts=8,  nf=8)
def riscv_nxv16i8x2  : VTVecTup<256, 2, i8>; // RISCV vector tuple(min_num_elts=16, nf=2)
def riscv_nxv16i8x3  : VTVecTup<384, 3, i8>; // RISCV vector tuple(min_num_elts=16, nf=3)
def riscv_nxv16i8x4  : VTVecTup<512, 4, i8>; // RISCV vector tuple(min_num_elts=16, nf=4)
def riscv_nxv32i8x2  : VTVecTup<512, 2, i8>; // RISCV vector tuple(min_num_elts=32, nf=2)

def x86mmx    : ValueType<64>;           // X86 MMX value
def Glue      : ValueType<0>;            // Pre-RA sched glue
def isVoid    : ValueType<0>;            // Produces no value
def untyped   : ValueType<8, "Untyped">; // Produces an untyped value
def funcref   : ValueType<0>;            // WebAssembly's funcref type
def externref : ValueType<0>;            // WebAssembly's externref type
def exnref    : ValueType<0>;            // WebAssembly's exnref type
def x86amx    : ValueType<8192>;         // X86 AMX value
def i64x8     : ValueType<512>;          // 8 Consecutive GPRs (AArch64)
def aarch64svcount
              : ValueType<16>;   // AArch64 predicate-as-counter
def spirvbuiltin : ValueType<0>; // SPIR-V's builtin type
// AMDGPU buffer fat pointer, buffer rsrc + offset, rewritten before MIR translation.
// FIXME: Remove this and the getPointerType() override if MVT::i160 is added.
def amdgpuBufferFatPointer : ValueType<160>;
// AMDGPU buffer strided pointer, buffer rsrc + index + offset, doesn't reach MIR.
// FIXME: Remove this and the getPointerType() override if MVT::i82 is added.
def amdgpuBufferStridedPointer : ValueType<192>;

def aarch64mfp8 : ValueType<8>; // 8-bit value in FPR (AArch64)

// CHERI capabilities. Pointer-like values that carry additional metadata
// for enforcing safety guarantees on CHERI-enabled targets.
def c64 : VTCheriCapability<64>;   // 64-bit CHERI capability value
def c128 : VTCheriCapability<128>; // 128-bit CHERI capability value

let isNormalValueType = false in {
// Pseudo valuetype mapped to the current CHERI capability pointer size.
// Should only be used in TableGen.
def cPTR : VTAny;

def token      : ValueType<0>;             // TokenTy
def MetadataVT : ValueType<0, "Metadata">; // Metadata

// Pseudo valuetype to represent "pointer to any address space"
// Should only be used in TableGen.
def pAny       : VTAny;

// Pseudo valuetype to represent "vector of any size"
// Should only be used in TableGen.
def vAny       : VTAny;

// Pseudo valuetype to represent "float of any format"
// Should only be used in TableGen.
def fAny       : VTAny;

// Pseudo valuetype to represent "integer of any bit width"
// Should only be used in TableGen.
def iAny       : VTAny;

// Pseudo valuetype mapped to the current pointer size.
// Should only be used in TableGen.
def iPTR       : ValueType<0>;

// Pseudo valuetype to represent "any type of any size".
// Should only be used in TableGen.
def Any        : VTAny;

} // isNormalValueType = false

} // end defset ValueTypes

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the pointer.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.LLVMName> {
  int AddrSpace = addrspace;
}
