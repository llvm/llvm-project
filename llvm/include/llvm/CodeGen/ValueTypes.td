//===- ValueTypes.td - ValueType definitions ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Value types - These values correspond to the register types defined in the
// ValueTypes.h file.  If you update anything here, you must update it there as
// well!
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

def v2i1   : ValueType<2 ,  14>;   //   2 x i1 vector value
def v4i1   : ValueType<4 ,  15>;   //   4 x i1 vector value
def v8i1   : ValueType<8 ,  16>;   //   8 x i1 vector value
def v16i1  : ValueType<16,  17>;   //  16 x i1 vector value
def v32i1  : ValueType<32 , 18>;   //  32 x i1 vector value
def v64i1  : ValueType<64 , 19>;   //  64 x i1 vector value
def v512i1 : ValueType<512, 20>;   // 512 x i1 vector value
def v1024i1: ValueType<1024,21>;   //1024 x i1 vector value

def v1i8   : ValueType<8,   22>;   //  1 x i8  vector value
def v2i8   : ValueType<16 , 23>;   //  2 x i8  vector value
def v4i8   : ValueType<32 , 24>;   //  4 x i8  vector value
def v8i8   : ValueType<64 , 25>;   //  8 x i8  vector value
def v16i8  : ValueType<128, 26>;   // 16 x i8  vector value
def v32i8  : ValueType<256, 27>;   // 32 x i8  vector value
def v64i8  : ValueType<512, 28>;   // 64 x i8  vector value
def v128i8 : ValueType<1024,29>;   //128 x i8  vector value
def v256i8 : ValueType<2048,30>;   //256 x i8  vector value

def v1i16  : ValueType<16 , 31>;   //  1 x i16 vector value
def v2i16  : ValueType<32 , 32>;   //  2 x i16 vector value
def v4i16  : ValueType<64 , 33>;   //  4 x i16 vector value
def v8i16  : ValueType<128, 34>;   //  8 x i16 vector value
def v16i16 : ValueType<256, 35>;   // 16 x i16 vector value
def v32i16 : ValueType<512, 36>;   // 32 x i16 vector value
def v64i16 : ValueType<1024,37>;   // 64 x i16 vector value
def v128i16: ValueType<2048,38>;   //128 x i16 vector value

def v1i32  : ValueType<32 , 39>;   //  1 x i32 vector value
def v2i32  : ValueType<64 , 40>;   //  2 x i32 vector value
def v4i32  : ValueType<128, 41>;   //  4 x i32 vector value
def v8i32  : ValueType<256, 42>;   //  8 x i32 vector value
def v16i32 : ValueType<512, 43>;   // 16 x i32 vector value
def v32i32 : ValueType<1024,44>;   // 32 x i32 vector value
def v64i32 : ValueType<2048,45>;   // 32 x i32 vector value

def v1i64  : ValueType<64 , 46>;   //  1 x i64 vector value
def v2i64  : ValueType<128, 47>;   //  2 x i64 vector value
def v4i64  : ValueType<256, 48>;   //  4 x i64 vector value
def v8i64  : ValueType<512, 49>;   //  8 x i64 vector value
def v16i64 : ValueType<1024,50>;   // 16 x i64 vector value
def v32i64 : ValueType<2048,51>;   // 32 x i64 vector value

def v1i128 : ValueType<128, 52>;   //  1 x i128 vector value

def nxv2i1  : ValueType<2,   53>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,   54>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,   55>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16,  56>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32,  57>;  // n x 32 x i1  vector value

def nxv1i8  : ValueType<8,   58>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  59>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  60>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  61>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 62>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 63>;  // n x 32 x i8  vector value

def nxv1i16 : ValueType<16,  64>;  // n x  1 x i16 vector value
def nxv2i16 : ValueType<32,  65>;  // n x  2 x i16 vector value
def nxv4i16 : ValueType<64,  66>;  // n x  4 x i16 vector value
def nxv8i16 : ValueType<128, 67>;  // n x  8 x i16 vector value
def nxv16i16: ValueType<256, 68>;  // n x 16 x i16 vector value
def nxv32i16: ValueType<512, 69>;  // n x 32 x i16 vector value

def nxv1i32 : ValueType<32,  70>;  // n x  1 x i32 vector value
def nxv2i32 : ValueType<64,  71>;  // n x  2 x i32 vector value
def nxv4i32 : ValueType<128, 72>;  // n x  4 x i32 vector value
def nxv8i32 : ValueType<256, 73>;  // n x  8 x i32 vector value
def nxv16i32: ValueType<512, 74>;  // n x 16 x i32 vector value
def nxv32i32: ValueType<1024,75>;  // n x 32 x i32 vector value

def nxv1i64 : ValueType<64,  76>;  // n x  1 x i64 vector value
def nxv2i64 : ValueType<128, 77>;  // n x  2 x i64 vector value
def nxv4i64 : ValueType<256, 78>;  // n x  4 x i64 vector value
def nxv8i64 : ValueType<512, 79>;  // n x  8 x i64 vector value
def nxv16i64: ValueType<1024,80>;  // n x 16 x i64 vector value
def nxv32i64: ValueType<2048,81>;  // n x 32 x i64 vector value

def v2f16  : ValueType<32 , 82>;   //  2 x f16 vector value
def v4f16  : ValueType<64 , 83>;   //  4 x f16 vector value
def v8f16  : ValueType<128, 84>;   //  8 x f16 vector value
def v1f32  : ValueType<32 , 85>;   //  1 x f32 vector value
def v2f32  : ValueType<64 , 86>;   //  2 x f32 vector value
def v4f32  : ValueType<128, 87>;   //  4 x f32 vector value
def v8f32  : ValueType<256, 88>;   //  8 x f32 vector value
def v16f32 : ValueType<512, 89>;   // 16 x f32 vector value
def v1f64  : ValueType<64,  90>;   //  1 x f64 vector value
def v2f64  : ValueType<128, 91>;   //  2 x f64 vector value
def v4f64  : ValueType<256, 92>;   //  4 x f64 vector value
def v8f64  : ValueType<512, 93>;   //  8 x f64 vector value

def nxv2f16  : ValueType<32 ,  94>; // n x  2 x f16 vector value
def nxv4f16  : ValueType<64 ,  95>; // n x  4 x f16 vector value
def nxv8f16  : ValueType<128,  96>; // n x  8 x f16 vector value
def nxv1f32  : ValueType<32 ,  97>; // n x  1 x f32 vector value
def nxv2f32  : ValueType<64 ,  98>; // n x  2 x f32 vector value
def nxv4f32  : ValueType<128,  99>; // n x  4 x f32 vector value
def nxv8f32  : ValueType<256, 100>; // n x  8 x f32 vector value
def nxv16f32 : ValueType<512, 101>; // n x 16 x f32 vector value
def nxv1f64  : ValueType<64,  102>; // n x  1 x f64 vector value
def nxv2f64  : ValueType<128, 103>; // n x  2 x f64 vector value
def nxv4f64  : ValueType<256, 104>; // n x  4 x f64 vector value
def nxv8f64  : ValueType<512, 105>; // n x  8 x f64 vector value

def x86mmx : ValueType<64 , 106>;   // X86 MMX value
def FlagVT : ValueType<0  , 107>;   // Pre-RA sched glue
def isVoid : ValueType<0  , 108>;   // Produces no value
def untyped: ValueType<8  , 109>;   // Produces an untyped value
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
