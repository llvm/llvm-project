//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// MachineValueTypes.h file.  If you update anything here, you must update it
// there as well!
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
  bit ForLegalizer = true;
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

// Convenience classes allowing for ordered Value definitions so new
// additions to the middle do not require the user to update the Values
// of all subsequent VTs.
class VTIntOrd<int size, ValueType prev> : VTInt<size, 0> {
  let Value = !add(prev.Value, 1);
}

class VTFPOrd<int size, ValueType prev> : VTFP<size, 0> {
  let Value = !add(prev.Value, 1);
}
class VTVecOrd<int nelem, ValueType elt, ValueType prev> :
  VTVec<nelem, elt, 0> {
  let Value = !add(prev.Value, 1);
}
class VTScalVecOrd<int nelem, ValueType elt, ValueType prev> :
  VTScalableVec<nelem, elt, 0> {
  let Value = !add(prev.Value, 1);
}
class VTOrd<int size, ValueType prev> : ValueType<size, 0> {
  let Value = !add(prev.Value, 1);
}
class VTAnyOrd<ValueType prev> : VTAny<0> {
  let Value = !add(prev.Value, 1);
}

defset list<ValueType> ValueTypes = {

def OtherVT : ValueType<0,   1> {  // "Other" value
  let LLVMName = "Other";
}

def i1      : VTIntOrd<1,   OtherVT>;  // One bit boolean value
def i2      : VTIntOrd<2,   i1>;   // 2-bit integer value
def i4      : VTIntOrd<4,   i2>;   // 4-bit integer value
def i8      : VTIntOrd<8,   i4>;   // 8-bit integer value
def i16     : VTIntOrd<16,  i8>;   // 16-bit integer value
def i32     : VTIntOrd<32,  i16>;  // 32-bit integer value
def i64     : VTIntOrd<64,  i32>;  // 64-bit integer value
def i128    : VTIntOrd<128, i64>;  // 128-bit integer value

def bf16    : VTFPOrd<16,  i128>;  // 16-bit brain floating point value
def f16     : VTFPOrd<16,  bf16>;  // 16-bit floating point value
def f32     : VTFPOrd<32,  f16>;   // 32-bit floating point value
def f64     : VTFPOrd<64,  f32>;   // 64-bit floating point value
def f80     : VTFPOrd<80,  f64>;   // 80-bit floating point value
def f128    : VTFPOrd<128, f80>;   // 128-bit floating point value
def ppcf128 : VTFPOrd<128, f128>;  // PPC 128-bit floating point value

def v1i1    : VTVecOrd<1,    i1, ppcf128>; //    1 x i1 vector value
def v2i1    : VTVecOrd<2,    i1, v1i1   >; //    2 x i1 vector value
def v4i1    : VTVecOrd<4,    i1, v2i1   >; //    4 x i1 vector value
def v8i1    : VTVecOrd<8,    i1, v4i1   >; //    8 x i1 vector value
def v16i1   : VTVecOrd<16,   i1, v8i1   >; //   16 x i1 vector value
def v32i1   : VTVecOrd<32,   i1, v16i1  >; //   32 x i1 vector value
def v64i1   : VTVecOrd<64,   i1, v32i1  >; //   64 x i1 vector value
def v128i1  : VTVecOrd<128,  i1, v64i1  >; //  128 x i1 vector value
def v256i1  : VTVecOrd<256,  i1, v128i1 >; //  256 x i1 vector value
def v512i1  : VTVecOrd<512,  i1, v256i1 >; //  512 x i1 vector value
def v1024i1 : VTVecOrd<1024, i1, v512i1 >; // 1024 x i1 vector value
def v2048i1 : VTVecOrd<2048, i1, v1024i1>; // 2048 x i1 vector value

def v128i2  : VTVecOrd<128,  i2, v2048i1>; //  128 x i2 vector value
def v256i2  : VTVecOrd<256,  i2, v128i2 >; //  256 x i2 vector value

def v64i4   : VTVecOrd<64,   i4, v256i2 >; //   64 x i4 vector value
def v128i4  : VTVecOrd<128,  i4, v64i4  >; //  128 x i4 vector value

def v1i8    : VTVecOrd<1,    i8, v128i4>;  //    1 x i8 vector value
def v2i8    : VTVecOrd<2,    i8, v1i8  >;  //    2 x i8 vector value
def v4i8    : VTVecOrd<4,    i8, v2i8  >;  //    4 x i8 vector value
def v8i8    : VTVecOrd<8,    i8, v4i8  >;  //    8 x i8 vector value
def v16i8   : VTVecOrd<16,   i8, v8i8  >;  //   16 x i8 vector value
def v32i8   : VTVecOrd<32,   i8, v16i8 >;  //   32 x i8 vector value
def v64i8   : VTVecOrd<64,   i8, v32i8 >;  //   64 x i8 vector value
def v128i8  : VTVecOrd<128,  i8, v64i8 >;  //  128 x i8 vector value
def v256i8  : VTVecOrd<256,  i8, v128i8>;  //  256 x i8 vector value
def v512i8  : VTVecOrd<512,  i8, v256i8>;  //  512 x i8 vector value
def v1024i8 : VTVecOrd<1024, i8, v512i8>;  // 1024 x i8 vector value

def v1i16   : VTVecOrd<1,   i16, v1024i8 >;  //   1 x i16 vector value
def v2i16   : VTVecOrd<2,   i16, v1i16   >;  //   2 x i16 vector value
def v3i16   : VTVecOrd<3,   i16, v2i16   >;  //   3 x i16 vector value
def v4i16   : VTVecOrd<4,   i16, v3i16   >;  //   4 x i16 vector value
def v8i16   : VTVecOrd<8,   i16, v4i16   >;  //   8 x i16 vector value
def v16i16  : VTVecOrd<16,  i16, v8i16   >;  //  16 x i16 vector value
def v32i16  : VTVecOrd<32,  i16, v16i16  >;  //  32 x i16 vector value
def v64i16  : VTVecOrd<64,  i16, v32i16  >;  //  64 x i16 vector value
def v128i16 : VTVecOrd<128, i16, v64i16  >;  // 128 x i16 vector value
def v256i16 : VTVecOrd<256, i16, v128i16 >;  // 256 x i16 vector value
def v512i16 : VTVecOrd<512, i16, v256i16 >;  // 512 x i16 vector value

def v1i32    : VTVecOrd<1,    i32, v512i16>;  //    1 x i32 vector value
def v2i32    : VTVecOrd<2,    i32, v1i32   >; //    2 x i32 vector value
def v3i32    : VTVecOrd<3,    i32, v2i32   >; //    3 x i32 vector value
def v4i32    : VTVecOrd<4,    i32, v3i32   >; //    4 x i32 vector value
def v5i32    : VTVecOrd<5,    i32, v4i32   >; //    5 x i32 vector value
def v6i32    : VTVecOrd<6,    i32, v5i32   >; //    6 x f32 vector value
def v7i32    : VTVecOrd<7,    i32, v6i32   >; //    7 x f32 vector value
def v8i32    : VTVecOrd<8,    i32, v7i32   >; //    8 x i32 vector value
def v9i32    : VTVecOrd<9,    i32, v8i32   >; //    9 x i32 vector value
def v10i32   : VTVecOrd<10,   i32, v9i32   >; //   10 x i32 vector value
def v11i32   : VTVecOrd<11,   i32, v10i32  >; //   11 x i32 vector value
def v12i32   : VTVecOrd<12,   i32, v11i32  >; //   12 x i32 vector value
def v16i32   : VTVecOrd<16,   i32, v12i32  >; //   16 x i32 vector value
def v32i32   : VTVecOrd<32,   i32, v16i32  >; //   32 x i32 vector value
def v64i32   : VTVecOrd<64,   i32, v32i32  >; //   64 x i32 vector value
def v128i32  : VTVecOrd<128,  i32, v64i32  >; //  128 x i32 vector value
def v256i32  : VTVecOrd<256,  i32, v128i32 >; //  256 x i32 vector value
def v512i32  : VTVecOrd<512,  i32, v256i32 >; //  512 x i32 vector value
def v1024i32 : VTVecOrd<1024, i32, v512i32 >; // 1024 x i32 vector value
def v2048i32 : VTVecOrd<2048, i32, v1024i32>; // 2048 x i32 vector value

def v1i64   : VTVecOrd<1,   i64, v2048i32>;   //   1 x i64 vector value
def v2i64   : VTVecOrd<2,   i64, v1i64   >;   //   2 x i64 vector value
def v3i64   : VTVecOrd<3,   i64, v2i64   >;   //   3 x i64 vector value
def v4i64   : VTVecOrd<4,   i64, v3i64   >;   //   4 x i64 vector value
def v8i64   : VTVecOrd<8,   i64, v4i64   >;   //   8 x i64 vector value
def v16i64  : VTVecOrd<16,  i64, v8i64   >;   //  16 x i64 vector value
def v32i64  : VTVecOrd<32,  i64, v16i64  >;   //  32 x i64 vector value
def v64i64  : VTVecOrd<64,  i64, v32i64  >;   //  64 x i64 vector value
def v128i64 : VTVecOrd<128, i64, v64i64  >;   // 128 x i64 vector value
def v256i64 : VTVecOrd<256, i64, v128i64 >;   // 256 x i64 vector value

def v1i128  : VTVecOrd<1,  i128, v256i64 >;   //  1 x i128 vector value

def v1f16    : VTVecOrd<1,    f16,  v1i128 >; //    1 x f16 vector value
def v2f16    : VTVecOrd<2,    f16,  v1f16  >; //    2 x f16 vector value
def v3f16    : VTVecOrd<3,    f16,  v2f16  >; //    3 x f16 vector value
def v4f16    : VTVecOrd<4,    f16,  v3f16  >; //    4 x f16 vector value
def v8f16    : VTVecOrd<8,    f16,  v4f16  >; //    8 x f16 vector value
def v16f16   : VTVecOrd<16,   f16,  v8f16  >; //   16 x f16 vector value
def v32f16   : VTVecOrd<32,   f16,  v16f16 >; //   32 x f16 vector value
def v64f16   : VTVecOrd<64,   f16,  v32f16 >; //   64 x f16 vector value
def v128f16  : VTVecOrd<128,  f16,  v64f16 >; //  128 x f16 vector value
def v256f16  : VTVecOrd<256,  f16,  v128f16>; //  256 x f16 vector value
def v512f16  : VTVecOrd<512,  f16,  v256f16>; //  512 x f16 vector value
def v2bf16   : VTVecOrd<2,   bf16,  v512f16>; //    2 x bf16 vector value
def v3bf16   : VTVecOrd<3,   bf16,  v2bf16 >; //    3 x bf16 vector value
def v4bf16   : VTVecOrd<4,   bf16,  v3bf16 >; //    4 x bf16 vector value
def v8bf16   : VTVecOrd<8,   bf16, v4bf16  >; //    8 x bf16 vector value
def v16bf16  : VTVecOrd<16,  bf16, v8bf16  >; //   16 x bf16 vector value
def v32bf16  : VTVecOrd<32,  bf16, v16bf16 >; //   32 x bf16 vector value
def v64bf16  : VTVecOrd<64,  bf16, v32bf16 >; //   64 x bf16 vector value
def v128bf16 : VTVecOrd<128, bf16, v64bf16 >; //  128 x bf16 vector value

def v1f32    : VTVecOrd<1,    f32, v128bf16>; //    1 x f32 vector value
def v2f32    : VTVecOrd<2,    f32, v1f32   >; //    2 x f32 vector value
def v3f32    : VTVecOrd<3,    f32, v2f32   >; //    3 x f32 vector value
def v4f32    : VTVecOrd<4,    f32, v3f32   >; //    4 x f32 vector value
def v5f32    : VTVecOrd<5,    f32, v4f32   >; //    5 x f32 vector value
def v6f32    : VTVecOrd<6,    f32, v5f32   >; //    6 x f32 vector value
def v7f32    : VTVecOrd<7,    f32, v6f32   >; //    7 x f32 vector value
def v8f32    : VTVecOrd<8,    f32, v7f32   >; //    8 x f32 vector value
def v9f32    : VTVecOrd<9,    f32, v8f32   >; //    9 x f32 vector value
def v10f32   : VTVecOrd<10,   f32, v9f32   >; //   10 x f32 vector value
def v11f32   : VTVecOrd<11,   f32, v10f32  >; //   11 x f32 vector value
def v12f32   : VTVecOrd<12,   f32, v11f32  >; //   12 x f32 vector value
def v16f32   : VTVecOrd<16,   f32, v12f32  >; //   16 x f32 vector value
def v32f32   : VTVecOrd<32,   f32, v16f32  >; //   32 x f32 vector value
def v64f32   : VTVecOrd<64,   f32, v32f32  >; //   64 x f32 vector value
def v128f32  : VTVecOrd<128,  f32, v64f32  >; //  128 x f32 vector value
def v256f32  : VTVecOrd<256,  f32, v128f32 >; //  256 x f32 vector value
def v512f32  : VTVecOrd<512,  f32, v256f32 >; //  512 x f32 vector value
def v1024f32 : VTVecOrd<1024, f32, v512f32 >; // 1024 x f32 vector value
def v2048f32 : VTVecOrd<2048, f32, v1024f32>; // 2048 x f32 vector value

def v1f64    : VTVecOrd<1,    f64, v2048f32>; //    1 x f64 vector value
def v2f64    : VTVecOrd<2,    f64, v1f64   >; //    2 x f64 vector value
def v3f64    : VTVecOrd<3,    f64, v2f64   >; //    3 x f64 vector value
def v4f64    : VTVecOrd<4,    f64, v3f64   >; //    4 x f64 vector value
def v8f64    : VTVecOrd<8,    f64, v4f64   >; //    8 x f64 vector value
def v16f64   : VTVecOrd<16,   f64, v8f64   >; //   16 x f64 vector value
def v32f64   : VTVecOrd<32,   f64, v16f64  >; //   32 x f64 vector value
def v64f64   : VTVecOrd<64,   f64, v32f64  >; //   64 x f64 vector value
def v128f64  : VTVecOrd<128,  f64, v64f64  >; //  128 x f64 vector value
def v256f64  : VTVecOrd<256,  f64, v128f64 >; //  256 x f64 vector value

def nxv1i1  : VTScalVecOrd<1,  i1, v256f64>;  // n x  1 x i1  vector value
def nxv2i1  : VTScalVecOrd<2,  i1, nxv1i1  >;  // n x  2 x i1  vector value
def nxv4i1  : VTScalVecOrd<4,  i1, nxv2i1  >;  // n x  4 x i1  vector value
def nxv8i1  : VTScalVecOrd<8,  i1, nxv4i1  >;  // n x  8 x i1  vector value
def nxv16i1 : VTScalVecOrd<16, i1, nxv8i1  >;  // n x 16 x i1  vector value
def nxv32i1 : VTScalVecOrd<32, i1, nxv16i1 >;  // n x 32 x i1  vector value
def nxv64i1 : VTScalVecOrd<64, i1, nxv32i1 >;  // n x 64 x i1  vector value
def nxv1i8  : VTScalVecOrd<1,  i8, nxv64i1 >;  // n x  1 x i8  vector value
def nxv2i8  : VTScalVecOrd<2,  i8, nxv1i8  >;  // n x  2 x i8  vector value
def nxv4i8  : VTScalVecOrd<4,  i8, nxv2i8  >;  // n x  4 x i8  vector value
def nxv8i8  : VTScalVecOrd<8,  i8, nxv4i8  >;  // n x  8 x i8  vector value
def nxv16i8 : VTScalVecOrd<16, i8, nxv8i8  >;  // n x 16 x i8  vector value
def nxv32i8 : VTScalVecOrd<32, i8, nxv16i8 >;  // n x 32 x i8  vector value
def nxv64i8 : VTScalVecOrd<64, i8, nxv32i8 >;  // n x 64 x i8  vector value

def nxv1i16  : VTScalVecOrd<1,  i16, nxv64i8 >; // n x  1 x i16 vector value
def nxv2i16  : VTScalVecOrd<2,  i16, nxv1i16 >; // n x  2 x i16 vector value
def nxv4i16  : VTScalVecOrd<4,  i16, nxv2i16 >; // n x  4 x i16 vector value
def nxv8i16  : VTScalVecOrd<8,  i16, nxv4i16 >; // n x  8 x i16 vector value
def nxv16i16 : VTScalVecOrd<16, i16, nxv8i16 >; // n x 16 x i16 vector value
def nxv32i16 : VTScalVecOrd<32, i16, nxv16i16>; // n x 32 x i16 vector value

def nxv1i32  : VTScalVecOrd<1,  i32, nxv32i16>; // n x  1 x i32 vector value
def nxv2i32  : VTScalVecOrd<2,  i32, nxv1i32 >; // n x  2 x i32 vector value
def nxv4i32  : VTScalVecOrd<4,  i32, nxv2i32 >; // n x  4 x i32 vector value
def nxv8i32  : VTScalVecOrd<8,  i32, nxv4i32 >; // n x  8 x i32 vector value
def nxv16i32 : VTScalVecOrd<16, i32, nxv8i32 >; // n x 16 x i32 vector value
def nxv32i32 : VTScalVecOrd<32, i32, nxv16i32>; // n x 32 x i32 vector value

def nxv1i64  : VTScalVecOrd<1,  i64, nxv32i32>; // n x  1 x i64 vector value
def nxv2i64  : VTScalVecOrd<2,  i64, nxv1i64 >; // n x  2 x i64 vector value
def nxv4i64  : VTScalVecOrd<4,  i64, nxv2i64 >; // n x  4 x i64 vector value
def nxv8i64  : VTScalVecOrd<8,  i64, nxv4i64 >; // n x  8 x i64 vector value
def nxv16i64 : VTScalVecOrd<16, i64, nxv8i64 >; // n x 16 x i64 vector value
def nxv32i64 : VTScalVecOrd<32, i64, nxv16i64>; // n x 32 x i64 vector value

def nxv1f16  : VTScalVecOrd<1,  f16, nxv32i64   >; // n x  1 x  f16 vector value
def nxv2f16  : VTScalVecOrd<2,  f16, nxv1f16    >; // n x  2 x  f16 vector value
def nxv4f16  : VTScalVecOrd<4,  f16, nxv2f16    >; // n x  4 x  f16 vector value
def nxv8f16  : VTScalVecOrd<8,  f16, nxv4f16    >; // n x  8 x  f16 vector value
def nxv16f16 : VTScalVecOrd<16, f16, nxv8f16    >; // n x 16 x  f16 vector value
def nxv32f16 : VTScalVecOrd<32, f16, nxv16f16   >; // n x 32 x  f16 vector value
def nxv1bf16  : VTScalVecOrd<1,  bf16, nxv32f16 >; // n x  1 x bf16 vector value
def nxv2bf16  : VTScalVecOrd<2,  bf16, nxv1bf16 >; // n x  2 x bf16 vector value
def nxv4bf16  : VTScalVecOrd<4,  bf16, nxv2bf16 >; // n x  4 x bf16 vector value
def nxv8bf16  : VTScalVecOrd<8,  bf16, nxv4bf16 >; // n x  8 x bf16 vector value
def nxv16bf16 : VTScalVecOrd<16, bf16, nxv8bf16 >; // n x 16 x bf16 vector value
def nxv32bf16 : VTScalVecOrd<32, bf16, nxv16bf16>; // n x 16 x bf16 vector value

def nxv1f32  : VTScalVecOrd<1,  f32, nxv32bf16>;  // n x  1 x  f32 vector value
def nxv2f32  : VTScalVecOrd<2,  f32, nxv1f32  >;  // n x  2 x  f32 vector value
def nxv4f32  : VTScalVecOrd<4,  f32, nxv2f32  >;  // n x  4 x  f32 vector value
def nxv8f32  : VTScalVecOrd<8,  f32, nxv4f32  >;  // n x  8 x  f32 vector value
def nxv16f32 : VTScalVecOrd<16, f32, nxv8f32  >;  // n x 16 x  f32 vector value

def nxv1f64  : VTScalVecOrd<1,  f64, nxv16f32>;  // n x  1 x  f64 vector value
def nxv2f64  : VTScalVecOrd<2,  f64, nxv1f64 >;  // n x  2 x  f64 vector value
def nxv4f64  : VTScalVecOrd<4,  f64, nxv2f64 >;  // n x  4 x  f64 vector value
def nxv8f64  : VTScalVecOrd<8,  f64, nxv4f64 >;  // n x  8 x  f64 vector value

def x86mmx    : VTOrd<64,   nxv8f64>;  // X86 MMX value
def FlagVT    : VTOrd<0,    x86mmx> {  // Pre-RA sched glue
  let LLVMName = "Glue";
}
def isVoid    : VTOrd<0,    FlagVT>;  // Produces no value
def untyped   : VTOrd<8,    isVoid> { // Produces an untyped value
  let LLVMName = "Untyped";
}
def funcref   : VTOrd<0,    untyped>;    // WebAssembly's funcref type
def externref : VTOrd<0,    funcref>;    // WebAssembly's externref type
def x86amx    : VTOrd<8192, externref>;  // X86 AMX value
def i64x8     : VTOrd<512,  x86amx>;     // 8 Consecutive GPRs (AArch64)
def aarch64svcount
              : VTOrd<16,   i64x8>;   // AArch64 predicate-as-counter
def spirvbuiltin : VTOrd<0,  aarch64svcount>; // SPIR-V's builtin type

// The remaining list isn't used by the Legalizer so their values are meant
// to be in the range [VALUETYPE_SIZE, MAX_ALLOWED_VALUETYPE). The token
// type is used as the anchor so if the previous list grows past its current
// value of 248, it simply needs to be updated so that the rest of the list
// fills out the last available values. Effectively, this is a decrement if
// a VT is added with a value larger than that of token and an increase by
// 64 if VALUETYPE_SIZE grows past the current value of token (248).
let ForLegalizer = false in {
  def token      : ValueType<0, 248>;  // TokenTy
  def MetadataVT : VTOrd<0, token> {   // Metadata
    let LLVMName = "Metadata";
  }

  // Pseudo valuetype mapped to the current pointer size to any address space.
  // Should only be used in TableGen.
  def iPTRAny    : VTAnyOrd<MetadataVT>;

  // Pseudo valuetype to represent "vector of any size"
  def vAny       : VTAnyOrd<iPTRAny>;

  // Pseudo valuetype to represent "float of any format"
  def fAny       : VTAnyOrd<vAny>;

  // Pseudo valuetype to represent "integer of any bit width"
  def iAny       : VTAnyOrd<fAny>;

  // Pseudo valuetype mapped to the current pointer size.
  def iPTR       : VTOrd<0, iAny>;

  // Pseudo valuetype to represent "any type of any size".
  def Any        : VTAnyOrd<iPTR>;
}

} // end defset ValueTypes

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the pointer.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
  let ForLegalizer = false;
}
