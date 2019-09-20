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
  int Size = size;
  int Value = value;
}

def OtherVT: ValueType<0  ,  1>;   // "Other" value
def i1     : ValueType<1  ,  2>;   // One bit boolean value
def i8     : ValueType<8  ,  3>;   // 8-bit integer value
def i16    : ValueType<16 ,  4>;   // 16-bit integer value
def i32    : ValueType<32 ,  5>;   // 32-bit integer value
def i64    : ValueType<64 ,  6>;   // 64-bit integer value
def i128   : ValueType<128,  7>;   // 128-bit integer value
def f16    : ValueType<16 ,  8>;   // 16-bit floating point value
def f32    : ValueType<32 ,  9>;   // 32-bit floating point value
def f64    : ValueType<64 , 10>;   // 64-bit floating point value
def f80    : ValueType<80 , 11>;   // 80-bit floating point value
def f128   : ValueType<128, 12>;   // 128-bit floating point value
def ppcf128: ValueType<128, 13>;   // PPC 128-bit floating point value

def v1i1   : ValueType<1 ,  14>;   //   1 x i1 vector value
def v2i1   : ValueType<2 ,  15>;   //   2 x i1 vector value
def v4i1   : ValueType<4 ,  16>;   //   4 x i1 vector value
def v8i1   : ValueType<8 ,  17>;   //   8 x i1 vector value
def v16i1  : ValueType<16,  18>;   //  16 x i1 vector value
def v32i1  : ValueType<32 , 19>;   //  32 x i1 vector value
def v64i1  : ValueType<64 , 20>;   //  64 x i1 vector value
def v128i1 : ValueType<128, 21>;   // 128 x i1 vector value
def v256i1 : ValueType<256, 22>;   // 256 x i1 vector value
def v512i1 : ValueType<512, 23>;   // 512 x i1 vector value
def v1024i1: ValueType<1024,24>;   //1024 x i1 vector value

def v1i8   : ValueType<8,   25>;   //  1 x i8  vector value
def v2i8   : ValueType<16 , 26>;   //  2 x i8  vector value
def v4i8   : ValueType<32 , 27>;   //  4 x i8  vector value
def v8i8   : ValueType<64 , 28>;   //  8 x i8  vector value
def v16i8  : ValueType<128, 29>;   // 16 x i8  vector value
def v32i8  : ValueType<256, 30>;   // 32 x i8  vector value
def v64i8  : ValueType<512, 31>;   // 64 x i8  vector value
def v128i8 : ValueType<1024,32>;   //128 x i8  vector value
def v256i8 : ValueType<2048,33>;   //256 x i8  vector value

def v1i16  : ValueType<16 , 34>;   //  1 x i16 vector value
def v2i16  : ValueType<32 , 35>;   //  2 x i16 vector value
def v3i16  : ValueType<48 , 36>;   //  3 x i16 vector value
def v4i16  : ValueType<64 , 37>;   //  4 x i16 vector value
def v8i16  : ValueType<128, 38>;   //  8 x i16 vector value
def v16i16 : ValueType<256, 39>;   // 16 x i16 vector value
def v32i16 : ValueType<512, 40>;   // 32 x i16 vector value
def v64i16 : ValueType<1024,41>;   // 64 x i16 vector value
def v128i16: ValueType<2048,42>;   //128 x i16 vector value

def v1i32    : ValueType<32 , 43>;   //  1 x i32 vector value
def v2i32    : ValueType<64 , 44>;   //  2 x i32 vector value
def v3i32    : ValueType<96 , 45>;   //  3 x i32 vector value
def v4i32    : ValueType<128, 46>;   //  4 x i32 vector value
def v5i32    : ValueType<160, 47>;   //  5 x i32 vector value
def v8i32    : ValueType<256, 48>;   //  8 x i32 vector value
def v16i32   : ValueType<512, 49>;   // 16 x i32 vector value
def v32i32   : ValueType<1024,50>;   // 32 x i32 vector value
def v64i32   : ValueType<2048,51>;   // 64 x i32 vector value
def v128i32  : ValueType<4096,52>;   // 128 x i32 vector value
def v256i32  : ValueType<8182,53>;   // 256 x i32 vector value
def v512i32  : ValueType<16384,54>;  // 512 x i32 vector value
def v1024i32 : ValueType<32768,55>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536,56>;  // 2048 x i32 vector value

def v1i64  : ValueType<64 , 57>;   //  1 x i64 vector value
def v2i64  : ValueType<128, 58>;   //  2 x i64 vector value
def v4i64  : ValueType<256, 59>;   //  4 x i64 vector value
def v8i64  : ValueType<512, 60>;   //  8 x i64 vector value
def v16i64 : ValueType<1024,61>;   // 16 x i64 vector value
def v32i64 : ValueType<2048,62>;   // 32 x i64 vector value

def v1i128 : ValueType<128, 63>;   //  1 x i128 vector value

def v2f16    : ValueType<32 , 64>;    //    2 x f16 vector value
def v3f16    : ValueType<48 , 65>;    //    3 x f16 vector value
def v4f16    : ValueType<64 , 66>;    //    4 x f16 vector value
def v8f16    : ValueType<128, 67>;    //    8 x f16 vector value
def v16f16   : ValueType<256, 68>;    //    8 x f16 vector value
def v32f16   : ValueType<512, 69>;    //    8 x f16 vector value
def v1f32    : ValueType<32 , 70>;    //    1 x f32 vector value
def v2f32    : ValueType<64 , 71>;    //    2 x f32 vector value
def v3f32    : ValueType<96 , 72>;    //    3 x f32 vector value
def v4f32    : ValueType<128, 73>;    //    4 x f32 vector value
def v5f32    : ValueType<160, 74>;    //    5 x f32 vector value
def v8f32    : ValueType<256, 75>;    //    8 x f32 vector value
def v16f32   : ValueType<512,  76>;   //   16 x f32 vector value
def v32f32   : ValueType<1024, 77>;   //   32 x f32 vector value
def v64f32   : ValueType<2048, 78>;   //   64 x f32 vector value
def v128f32  : ValueType<4096, 79>;   //  128 x f32 vector value
def v256f32  : ValueType<8182, 80>;   //  256 x f32 vector value
def v512f32  : ValueType<16384, 81>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 82>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 83>;  // 2048 x f32 vector value
def v1f64    : ValueType<64, 84>;     //    1 x f64 vector value
def v2f64    : ValueType<128, 85>;    //    2 x f64 vector value
def v4f64    : ValueType<256, 86>;    //    4 x f64 vector value
def v8f64    : ValueType<512, 87>;    //    8 x f64 vector value

def nxv1i1  : ValueType<1,   88>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,   89>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,   90>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,   91>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16,  92>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32,  93>;  // n x 32 x i1  vector value

def nxv1i8  : ValueType<8,   94>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  95>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  96>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  97>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 98>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 99>;  // n x 32 x i8  vector value

def nxv1i16 : ValueType<16,  100>; // n x  1 x i16 vector value
def nxv2i16 : ValueType<32,  101>; // n x  2 x i16 vector value
def nxv4i16 : ValueType<64,  102>; // n x  4 x i16 vector value
def nxv8i16 : ValueType<128, 103>; // n x  8 x i16 vector value
def nxv16i16: ValueType<256, 104>; // n x 16 x i16 vector value
def nxv32i16: ValueType<512, 105>; // n x 32 x i16 vector value

def nxv1i32 : ValueType<32,  106>; // n x  1 x i32 vector value
def nxv2i32 : ValueType<64,  107>; // n x  2 x i32 vector value
def nxv4i32 : ValueType<128, 108>; // n x  4 x i32 vector value
def nxv8i32 : ValueType<256, 109>; // n x  8 x i32 vector value
def nxv16i32: ValueType<512, 110>; // n x 16 x i32 vector value
def nxv32i32: ValueType<1024,111>; // n x 32 x i32 vector value

def nxv1i64 : ValueType<64,  112>; // n x  1 x i64 vector value
def nxv2i64 : ValueType<128, 113>; // n x  2 x i64 vector value
def nxv4i64 : ValueType<256, 114>; // n x  4 x i64 vector value
def nxv8i64 : ValueType<512, 115>; // n x  8 x i64 vector value
def nxv16i64: ValueType<1024,116>; // n x 16 x i64 vector value
def nxv32i64: ValueType<2048,117>; // n x 32 x i64 vector value

def nxv2f16  : ValueType<32 , 118>; // n x  2 x f16 vector value
def nxv4f16  : ValueType<64 , 119>; // n x  4 x f16 vector value
def nxv8f16  : ValueType<128, 120>; // n x  8 x f16 vector value
def nxv1f32  : ValueType<32 , 121>; // n x  1 x f32 vector value
def nxv2f32  : ValueType<64 , 122>; // n x  2 x f32 vector value
def nxv4f32  : ValueType<128, 123>; // n x  4 x f32 vector value
def nxv8f32  : ValueType<256, 124>; // n x  8 x f32 vector value
def nxv16f32 : ValueType<512, 125>; // n x 16 x f32 vector value
def nxv1f64  : ValueType<64,  126>; // n x  1 x f64 vector value
def nxv2f64  : ValueType<128, 127>; // n x  2 x f64 vector value
def nxv4f64  : ValueType<256, 128>; // n x  4 x f64 vector value
def nxv8f64  : ValueType<512, 129>; // n x  8 x f64 vector value

def x86mmx : ValueType<64 , 130>;   // X86 MMX value
def FlagVT : ValueType<0  , 131>;   // Pre-RA sched glue
def isVoid : ValueType<0  , 132>;   // Produces no value
def untyped: ValueType<8  , 133>;   // Produces an untyped value
def exnref: ValueType<0, 134>;      // WebAssembly's exnref type
def token  : ValueType<0  , 248>;   // TokenTy
def MetadataVT: ValueType<0, 249>;  // Metadata

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny   : ValueType<0, 250>;

// Pseudo valuetype to represent "vector of any size"
def vAny   : ValueType<0  , 251>;

// Pseudo valuetype to represent "float of any format"
def fAny   : ValueType<0  , 252>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny   : ValueType<0  , 253>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR   : ValueType<0  , 254>;

// Pseudo valuetype to represent "any type of any size".
def Any    : ValueType<0  , 255>;

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the ponter.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
