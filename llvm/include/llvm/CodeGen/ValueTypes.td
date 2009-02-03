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

def OtherVT: ValueType<0  ,  0>;   // "Other" value
def i1     : ValueType<1  ,  1>;   // One bit boolean value
def i8     : ValueType<8  ,  2>;   // 8-bit integer value
def i16    : ValueType<16 ,  3>;   // 16-bit integer value
def i32    : ValueType<32 ,  4>;   // 32-bit integer value
def i64    : ValueType<64 ,  5>;   // 64-bit integer value
def i128   : ValueType<128,  6>;   // 128-bit integer value
def f32    : ValueType<32 ,  7>;   // 32-bit floating point value
def f64    : ValueType<64 ,  8>;   // 64-bit floating point value
def f80    : ValueType<80 ,  9>;   // 80-bit floating point value
def f128   : ValueType<128, 10>;   // 128-bit floating point value
def ppcf128: ValueType<128, 11>;   // PPC 128-bit floating point value
def FlagVT : ValueType<0  , 12>;   // Condition code or machine flag
def isVoid : ValueType<0  , 13>;   // Produces no value
def v2i8   : ValueType<16 , 14>;   //  2 x i8  vector value
def v4i8   : ValueType<32 , 15>;   //  4 x i8  vector value
def v2i16  : ValueType<32 , 16>;   //  2 x i16 vector value
def v8i8   : ValueType<64 , 17>;   //  8 x i8  vector value
def v4i16  : ValueType<64 , 18>;   //  4 x i16 vector value
def v2i32  : ValueType<64 , 19>;   //  2 x i32 vector value
def v1i64  : ValueType<64 , 20>;   //  1 x i64 vector value

def v16i8  : ValueType<128, 21>;   // 16 x i8  vector value
def v8i16  : ValueType<128, 22>;   //  8 x i16 vector value
def v3i32  : ValueType<96 , 23>;   //  3 x i32 vector value
def v4i32  : ValueType<128, 24>;   //  4 x i32 vector value
def v2i64  : ValueType<128, 25>;   //  2 x i64 vector value

def v2f32  : ValueType<64,  26>;   //  2 x f32 vector value
def v3f32  : ValueType<96 , 27>;   //  3 x f32 vector value
def v4f32  : ValueType<128, 28>;   //  4 x f32 vector value
def v2f64  : ValueType<128, 29>;   //  2 x f64 vector value

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny   : ValueType<0, 252>;

// Pseudo valuetype to represent "float of any format"
def fAny   : ValueType<0  , 253>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny   : ValueType<0  , 254>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR   : ValueType<0  , 255>;
