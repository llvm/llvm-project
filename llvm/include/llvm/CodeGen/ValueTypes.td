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

def bf16   : ValueType<16 ,  8>;   // 16-bit brain floating point value
def f16    : ValueType<16 ,  9>;   // 16-bit floating point value
def f32    : ValueType<32 , 10>;   // 32-bit floating point value
def f64    : ValueType<64 , 11>;   // 64-bit floating point value
def f80    : ValueType<80 , 12>;   // 80-bit floating point value
def f128   : ValueType<128, 13>;   // 128-bit floating point value
def ppcf128: ValueType<128, 14>;   // PPC 128-bit floating point value

def v1i1   : ValueType<1 ,  15>;   //   1 x i1 vector value
def v2i1   : ValueType<2 ,  16>;   //   2 x i1 vector value
def v4i1   : ValueType<4 ,  17>;   //   4 x i1 vector value
def v8i1   : ValueType<8 ,  18>;   //   8 x i1 vector value
def v16i1  : ValueType<16,  19>;   //  16 x i1 vector value
def v32i1  : ValueType<32 , 20>;   //  32 x i1 vector value
def v64i1  : ValueType<64 , 21>;   //  64 x i1 vector value
def v128i1 : ValueType<128, 22>;   // 128 x i1 vector value
def v256i1 : ValueType<256, 23>;   // 256 x i1 vector value
def v512i1 : ValueType<512, 24>;   // 512 x i1 vector value
def v1024i1: ValueType<1024,25>;   //1024 x i1 vector value

def v1i8   : ValueType<8,   26>;   //  1 x i8  vector value
def v2i8   : ValueType<16 , 27>;   //  2 x i8  vector value
def v4i8   : ValueType<32 , 28>;   //  4 x i8  vector value
def v8i8   : ValueType<64 , 29>;   //  8 x i8  vector value
def v16i8  : ValueType<128, 30>;   // 16 x i8  vector value
def v32i8  : ValueType<256, 31>;   // 32 x i8  vector value
def v64i8  : ValueType<512, 32>;   // 64 x i8  vector value
def v128i8 : ValueType<1024,33>;   //128 x i8  vector value
def v256i8 : ValueType<2048,34>;   //256 x i8  vector value

def v1i16  : ValueType<16 , 35>;   //  1 x i16 vector value
def v2i16  : ValueType<32 , 36>;   //  2 x i16 vector value
def v3i16  : ValueType<48 , 37>;   //  3 x i16 vector value
def v4i16  : ValueType<64 , 38>;   //  4 x i16 vector value
def v8i16  : ValueType<128, 39>;   //  8 x i16 vector value
def v16i16 : ValueType<256, 40>;   // 16 x i16 vector value
def v32i16 : ValueType<512, 41>;   // 32 x i16 vector value
def v64i16 : ValueType<1024,42>;   // 64 x i16 vector value
def v128i16: ValueType<2048,43>;   //128 x i16 vector value

def v1i32    : ValueType<32 , 44>;   //  1 x i32 vector value
def v2i32    : ValueType<64 , 45>;   //  2 x i32 vector value
def v3i32    : ValueType<96 , 46>;   //  3 x i32 vector value
def v4i32    : ValueType<128, 47>;   //  4 x i32 vector value
def v5i32    : ValueType<160, 48>;   //  5 x i32 vector value
def v8i32    : ValueType<256, 49>;   //  8 x i32 vector value
def v16i32   : ValueType<512, 50>;   // 16 x i32 vector value
def v32i32   : ValueType<1024,51>;   // 32 x i32 vector value
def v64i32   : ValueType<2048,52>;   // 64 x i32 vector value
def v128i32  : ValueType<4096,53>;   // 128 x i32 vector value
def v256i32  : ValueType<8182,54>;   // 256 x i32 vector value
def v512i32  : ValueType<16384,55>;  // 512 x i32 vector value
def v1024i32 : ValueType<32768,56>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536,57>;  // 2048 x i32 vector value

def v1i64  : ValueType<64 , 58>;   //  1 x i64 vector value
def v2i64  : ValueType<128, 59>;   //  2 x i64 vector value
def v4i64  : ValueType<256, 60>;   //  4 x i64 vector value
def v8i64  : ValueType<512, 61>;   //  8 x i64 vector value
def v16i64 : ValueType<1024,62>;   // 16 x i64 vector value
def v32i64 : ValueType<2048,63>;   // 32 x i64 vector value

def v1i128 : ValueType<128, 64>;   //  1 x i128 vector value

def v2f16    : ValueType<32 , 65>;    //    2 x f16 vector value
def v3f16    : ValueType<48 , 66>;    //    3 x f16 vector value
def v4f16    : ValueType<64 , 67>;    //    4 x f16 vector value
def v8f16    : ValueType<128, 68>;    //    8 x f16 vector value
def v16f16   : ValueType<256, 69>;    //   16 x f16 vector value
def v32f16   : ValueType<512, 70>;    //   32 x f16 vector value
def v64f16   : ValueType<1024, 71>;   //   64 x f16 vector value
def v128f16  : ValueType<2048, 72>;   //  128 x f16 vector value
def v2bf16   : ValueType<32 , 73>;    //    2 x bf16 vector value
def v3bf16   : ValueType<48 , 74>;    //    3 x bf16 vector value
def v4bf16   : ValueType<64 , 75>;    //    4 x bf16 vector value
def v8bf16   : ValueType<128, 76>;    //    8 x bf16 vector value
def v16bf16  : ValueType<256, 77>;    //   16 x bf16 vector value
def v32bf16  : ValueType<512, 78>;    //   32 x bf16 vector value
def v64bf16  : ValueType<1024, 79>;   //   64 x bf16 vector value
def v128bf16 : ValueType<2048, 80>;   //  128 x bf16 vector value
def v1f32    : ValueType<32 , 81>;    //    1 x f32 vector value
def v2f32    : ValueType<64 , 82>;    //    2 x f32 vector value
def v3f32    : ValueType<96 , 83>;    //    3 x f32 vector value
def v4f32    : ValueType<128, 84>;    //    4 x f32 vector value
def v5f32    : ValueType<160, 85>;    //    5 x f32 vector value
def v8f32    : ValueType<256, 86>;    //    8 x f32 vector value
def v16f32   : ValueType<512,  87>;   //   16 x f32 vector value
def v32f32   : ValueType<1024, 88>;   //   32 x f32 vector value
def v64f32   : ValueType<2048, 89>;   //   64 x f32 vector value
def v128f32  : ValueType<4096, 90>;   //  128 x f32 vector value
def v256f32  : ValueType<8182, 91>;   //  256 x f32 vector value
def v512f32  : ValueType<16384, 92>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 93>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 94>;  // 2048 x f32 vector value
def v1f64    : ValueType<64, 95>;     //    1 x f64 vector value
def v2f64    : ValueType<128, 96>;    //    2 x f64 vector value
def v4f64    : ValueType<256, 97>;    //    4 x f64 vector value
def v8f64    : ValueType<512, 98>;    //    8 x f64 vector value
def v16f64   : ValueType<1024, 99>;   //   16 x f64 vector value
def v32f64   : ValueType<2048, 100>;  //   32 x f64 vector value

def nxv1i1  : ValueType<1,  101>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,  102>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,  103>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,  104>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16, 105>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32, 106>;  // n x 32 x i1  vector value
def nxv64i1  : ValueType<64,107>;  // n x  64 x i1  vector value

def nxv1i8  : ValueType<8,   108>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  109>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  110>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  111>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 112>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 113>;  // n x 32 x i8  vector value
def nxv64i8  : ValueType<512,  114>;  // n x  64 x i8  vector value

def nxv1i16 : ValueType<16,  115>; // n x  1 x i16 vector value
def nxv2i16 : ValueType<32,  116>; // n x  2 x i16 vector value
def nxv4i16 : ValueType<64,  117>; // n x  4 x i16 vector value
def nxv8i16 : ValueType<128, 118>; // n x  8 x i16 vector value
def nxv16i16: ValueType<256, 119>; // n x 16 x i16 vector value
def nxv32i16: ValueType<512, 120>; // n x 32 x i16 vector value

def nxv1i32 : ValueType<32,  121>; // n x  1 x i32 vector value
def nxv2i32 : ValueType<64,  122>; // n x  2 x i32 vector value
def nxv4i32 : ValueType<128, 123>; // n x  4 x i32 vector value
def nxv8i32 : ValueType<256, 124>; // n x  8 x i32 vector value
def nxv16i32: ValueType<512, 125>; // n x 16 x i32 vector value
def nxv32i32: ValueType<1024,126>; // n x 32 x i32 vector value

def nxv1i64 : ValueType<64,  127>; // n x  1 x i64 vector value
def nxv2i64 : ValueType<128, 128>; // n x  2 x i64 vector value
def nxv4i64 : ValueType<256, 129>; // n x  4 x i64 vector value
def nxv8i64 : ValueType<512, 130>; // n x  8 x i64 vector value
def nxv16i64: ValueType<1024,131>; // n x 16 x i64 vector value
def nxv32i64: ValueType<2048,132>; // n x 32 x i64 vector value

def nxv1f16   : ValueType<32, 133>; // n x   1 x f16 vector value
def nxv2f16  : ValueType<32 , 134>; // n x  2 x f16 vector value
def nxv4f16  : ValueType<64 , 135>; // n x  4 x f16 vector value
def nxv8f16  : ValueType<128, 136>; // n x  8 x f16 vector value
def nxv16f16  : ValueType<256,137>; // n x  16 x f16 vector value
def nxv32f16  : ValueType<512,138>; // n x  32 x f16 vector value
def nxv2bf16 : ValueType<32 , 139>; // n x  2 x bf16 vector value
def nxv4bf16 : ValueType<64 , 140>; // n x  4 x bf16 vector value
def nxv8bf16 : ValueType<128, 141>; // n x  8 x bf16 vector value
def nxv1f32  : ValueType<32 , 142>; // n x  1 x f32 vector value
def nxv2f32  : ValueType<64 , 143>; // n x  2 x f32 vector value
def nxv4f32  : ValueType<128, 144>; // n x  4 x f32 vector value
def nxv8f32  : ValueType<256, 145>; // n x  8 x f32 vector value
def nxv16f32 : ValueType<512, 146>; // n x 16 x f32 vector value
def nxv1f64  : ValueType<64,  147>; // n x  1 x f64 vector value
def nxv2f64  : ValueType<128, 148>; // n x  2 x f64 vector value
def nxv4f64  : ValueType<256, 149>; // n x  4 x f64 vector value
def nxv8f64  : ValueType<512, 150>; // n x  8 x f64 vector value

def x86mmx : ValueType<64 , 151>;   // X86 MMX value
def FlagVT : ValueType<0  , 152>;   // Pre-RA sched glue
def isVoid : ValueType<0  , 153>;   // Produces no value
def untyped: ValueType<8  , 154>;   // Produces an untyped value
def exnref : ValueType<0  , 155>;   // WebAssembly's exnref type

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
