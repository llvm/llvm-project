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

def OtherVT : ValueType<0,   1>;  // "Other" value
def i1      : ValueType<1,   2>;  // One bit boolean value
def i8      : ValueType<8,   3>;  // 8-bit integer value
def i16     : ValueType<16,  4>;  // 16-bit integer value
def i32     : ValueType<32,  5>;  // 32-bit integer value
def i64     : ValueType<64,  6>;  // 64-bit integer value
def i128    : ValueType<128, 7>;  // 128-bit integer value

def bf16    : ValueType<16,   8>;  // 16-bit brain floating point value
def f16     : ValueType<16,   9>;  // 16-bit floating point value
def f32     : ValueType<32,  10>;  // 32-bit floating point value
def f64     : ValueType<64,  11>;  // 64-bit floating point value
def f80     : ValueType<80,  12>;  // 80-bit floating point value
def f128    : ValueType<128, 13>;  // 128-bit floating point value
def ppcf128 : ValueType<128, 14>;  // PPC 128-bit floating point value

def v1i1    : ValueType<1,    15>;  //    1 x i1 vector value
def v2i1    : ValueType<2,    16>;  //    2 x i1 vector value
def v4i1    : ValueType<4,    17>;  //    4 x i1 vector value
def v8i1    : ValueType<8,    18>;  //    8 x i1 vector value
def v16i1   : ValueType<16,   19>;  //   16 x i1 vector value
def v32i1   : ValueType<32,   20>;  //   32 x i1 vector value
def v64i1   : ValueType<64,   21>;  //   64 x i1 vector value
def v128i1  : ValueType<128,  22>;  //  128 x i1 vector value
def v256i1  : ValueType<256,  23>;  //  256 x i1 vector value
def v512i1  : ValueType<512,  24>;  //  512 x i1 vector value
def v1024i1 : ValueType<1024, 25>;  // 1024 x i1 vector value

def v1i8    : ValueType<8,    26>;  //    1 x i8 vector value
def v2i8    : ValueType<16,   27>;  //    2 x i8 vector value
def v4i8    : ValueType<32,   28>;  //    4 x i8 vector value
def v8i8    : ValueType<64,   29>;  //    8 x i8 vector value
def v16i8   : ValueType<128,  30>;  //   16 x i8 vector value
def v32i8   : ValueType<256,  31>;  //   32 x i8 vector value
def v64i8   : ValueType<512,  32>;  //   64 x i8 vector value
def v128i8  : ValueType<1024, 33>;  //  128 x i8 vector value
def v256i8  : ValueType<2048, 34>;  //  256 x i8 vector value
def v512i8  : ValueType<4096, 35>;  //  512 x i8 vector value
def v1024i8 : ValueType<8192, 36>;  // 1024 x i8 vector value

def v1i16   : ValueType<16,   37>;  //   1 x i16 vector value
def v2i16   : ValueType<32,   38>;  //   2 x i16 vector value
def v3i16   : ValueType<48,   39>;  //   3 x i16 vector value
def v4i16   : ValueType<64,   40>;  //   4 x i16 vector value
def v8i16   : ValueType<128,  41>;  //   8 x i16 vector value
def v16i16  : ValueType<256,  42>;  //  16 x i16 vector value
def v32i16  : ValueType<512,  43>;  //  32 x i16 vector value
def v64i16  : ValueType<1024, 44>;  //  64 x i16 vector value
def v128i16 : ValueType<2048, 45>;  // 128 x i16 vector value
def v256i16 : ValueType<4096, 46>;  // 256 x i16 vector value
def v512i16 : ValueType<8192, 47>;  // 512 x i16 vector value

def v1i32    : ValueType<32,    48>;  //    1 x i32 vector value
def v2i32    : ValueType<64,    49>;  //    2 x i32 vector value
def v3i32    : ValueType<96,    50>;  //    3 x i32 vector value
def v4i32    : ValueType<128,   51>;  //    4 x i32 vector value
def v5i32    : ValueType<160,   52>;  //    5 x i32 vector value
def v6i32    : ValueType<192,   53>;  //    6 x f32 vector value
def v7i32    : ValueType<224,   54>;  //    7 x f32 vector value
def v8i32    : ValueType<256,   55>;  //    8 x i32 vector value
def v16i32   : ValueType<512,   56>;  //   16 x i32 vector value
def v32i32   : ValueType<1024,  57>;  //   32 x i32 vector value
def v64i32   : ValueType<2048,  58>;  //   64 x i32 vector value
def v128i32  : ValueType<4096,  59>;  //  128 x i32 vector value
def v256i32  : ValueType<8192,  60>;  //  256 x i32 vector value
def v512i32  : ValueType<16384, 61>;  //  512 x i32 vector value
def v1024i32 : ValueType<32768, 62>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536, 63>;  // 2048 x i32 vector value

def v1i64   : ValueType<64,    64>;  //   1 x i64 vector value
def v2i64   : ValueType<128,   65>;  //   2 x i64 vector value
def v4i64   : ValueType<256,   66>;  //   4 x i64 vector value
def v8i64   : ValueType<512,   67>;  //   8 x i64 vector value
def v16i64  : ValueType<1024,  68>;  //  16 x i64 vector value
def v32i64  : ValueType<2048,  69>;  //  32 x i64 vector value
def v64i64  : ValueType<4096,  70>;  //  64 x i64 vector value
def v128i64 : ValueType<8192,  71>;  // 128 x i64 vector value
def v256i64 : ValueType<16384, 72>;  // 256 x i64 vector value

def v1i128 : ValueType<128, 73>;  //  1 x i128 vector value

def v1f16    : ValueType<16,     74>;  //    1 x f16 vector value
def v2f16    : ValueType<32,     75>;  //    2 x f16 vector value
def v3f16    : ValueType<48,     76>;  //    3 x f16 vector value
def v4f16    : ValueType<64,     77>;  //    4 x f16 vector value
def v8f16    : ValueType<128,    78>;  //    8 x f16 vector value
def v16f16   : ValueType<256,    79>;  //   16 x f16 vector value
def v32f16   : ValueType<512,    80>;  //   32 x f16 vector value
def v64f16   : ValueType<1024,   81>;  //   64 x f16 vector value
def v128f16  : ValueType<2048,   82>;  //  128 x f16 vector value
def v256f16  : ValueType<4096,   83>;  //  256 x f16 vector value
def v512f16  : ValueType<8192,   84>;  //  512 x f16 vector value

def v2bf16   : ValueType<32,     85>;  //    2 x bf16 vector value
def v3bf16   : ValueType<48,     86>;  //    3 x bf16 vector value
def v4bf16   : ValueType<64,     87>;  //    4 x bf16 vector value
def v8bf16   : ValueType<128,    88>;  //    8 x bf16 vector value
def v16bf16  : ValueType<256,    89>;  //   16 x bf16 vector value
def v32bf16  : ValueType<512,    90>;  //   32 x bf16 vector value
def v64bf16  : ValueType<1024,   91>;  //   64 x bf16 vector value
def v128bf16 : ValueType<2048,   92>;  //  128 x bf16 vector value

def v1f32    : ValueType<32,     93>;  //    1 x f32 vector value
def v2f32    : ValueType<64,     94>;  //    2 x f32 vector value
def v3f32    : ValueType<96,     95>;  //    3 x f32 vector value
def v4f32    : ValueType<128,    96>;  //    4 x f32 vector value
def v5f32    : ValueType<160,    97>;  //    5 x f32 vector value
def v6f32    : ValueType<192,    98>;  //    6 x f32 vector value
def v7f32    : ValueType<224,    99>;  //    7 x f32 vector value
def v8f32    : ValueType<256,   100>;  //    8 x f32 vector value
def v16f32   : ValueType<512,   101>;  //   16 x f32 vector value
def v32f32   : ValueType<1024,  102>;  //   32 x f32 vector value
def v64f32   : ValueType<2048,  103>;  //   64 x f32 vector value
def v128f32  : ValueType<4096,  104>;  //  128 x f32 vector value
def v256f32  : ValueType<8192,  105>;  //  256 x f32 vector value
def v512f32  : ValueType<16384, 106>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 107>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 108>;  // 2048 x f32 vector value

def v1f64    : ValueType<64,    109>;  //    1 x f64 vector value
def v2f64    : ValueType<128,   110>;  //    2 x f64 vector value
def v4f64    : ValueType<256,   111>;  //    4 x f64 vector value
def v8f64    : ValueType<512,   112>;  //    8 x f64 vector value
def v16f64   : ValueType<1024,  113>;  //   16 x f64 vector value
def v32f64   : ValueType<2048,  114>;  //   32 x f64 vector value
def v64f64   : ValueType<4096,  115>;  //   64 x f64 vector value
def v128f64  : ValueType<8192,  116>;  //  128 x f64 vector value
def v256f64  : ValueType<16384, 117>;  //  256 x f64 vector value

def nxv1i1  : ValueType<1,  118>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,  119>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,  120>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,  121>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16, 122>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32, 123>;  // n x 32 x i1  vector value
def nxv64i1 : ValueType<64, 124>;  // n x 64 x i1  vector value

def nxv1i8  : ValueType<8,   125>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  126>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  127>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  128>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 129>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 130>;  // n x 32 x i8  vector value
def nxv64i8 : ValueType<512, 131>;  // n x 64 x i8  vector value

def nxv1i16  : ValueType<16,  132>;  // n x  1 x i16 vector value
def nxv2i16  : ValueType<32,  133>;  // n x  2 x i16 vector value
def nxv4i16  : ValueType<64,  134>;  // n x  4 x i16 vector value
def nxv8i16  : ValueType<128, 135>;  // n x  8 x i16 vector value
def nxv16i16 : ValueType<256, 136>;  // n x 16 x i16 vector value
def nxv32i16 : ValueType<512, 137>;  // n x 32 x i16 vector value

def nxv1i32  : ValueType<32,   138>;  // n x  1 x i32 vector value
def nxv2i32  : ValueType<64,   139>;  // n x  2 x i32 vector value
def nxv4i32  : ValueType<128,  140>;  // n x  4 x i32 vector value
def nxv8i32  : ValueType<256,  141>;  // n x  8 x i32 vector value
def nxv16i32 : ValueType<512,  142>;  // n x 16 x i32 vector value
def nxv32i32 : ValueType<1024, 143>;  // n x 32 x i32 vector value

def nxv1i64  : ValueType<64,   144>;  // n x  1 x i64 vector value
def nxv2i64  : ValueType<128,  145>;  // n x  2 x i64 vector value
def nxv4i64  : ValueType<256,  146>;  // n x  4 x i64 vector value
def nxv8i64  : ValueType<512,  147>;  // n x  8 x i64 vector value
def nxv16i64 : ValueType<1024, 148>;  // n x 16 x i64 vector value
def nxv32i64 : ValueType<2048, 149>;  // n x 32 x i64 vector value

def nxv1f16  : ValueType<16,  150>;  // n x  1 x  f16 vector value
def nxv2f16  : ValueType<32,  151>;  // n x  2 x  f16 vector value
def nxv4f16  : ValueType<64,  152>;  // n x  4 x  f16 vector value
def nxv8f16  : ValueType<128, 153>;  // n x  8 x  f16 vector value
def nxv16f16 : ValueType<256, 154>;  // n x 16 x  f16 vector value
def nxv32f16 : ValueType<512, 155>;  // n x 32 x  f16 vector value

def nxv1bf16 : ValueType<16,  156>;  // n x  1 x bf16 vector value
def nxv2bf16 : ValueType<32,  157>;  // n x  2 x bf16 vector value
def nxv4bf16 : ValueType<64,  158>;  // n x  4 x bf16 vector value
def nxv8bf16 : ValueType<128, 159>;  // n x  8 x bf16 vector value

def nxv1f32  : ValueType<32,  160>;  // n x  1 x  f32 vector value
def nxv2f32  : ValueType<64,  161>;  // n x  2 x  f32 vector value
def nxv4f32  : ValueType<128, 162>;  // n x  4 x  f32 vector value
def nxv8f32  : ValueType<256, 163>;  // n x  8 x  f32 vector value
def nxv16f32 : ValueType<512, 164>;  // n x 16 x  f32 vector value

def nxv1f64  : ValueType<64,  165>;  // n x  1 x  f64 vector value
def nxv2f64  : ValueType<128, 166>;  // n x  2 x  f64 vector value
def nxv4f64  : ValueType<256, 167>;  // n x  4 x  f64 vector value
def nxv8f64  : ValueType<512, 168>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   169>;  // X86 MMX value
def FlagVT    : ValueType<0,    170>;  // Pre-RA sched glue
def isVoid    : ValueType<0,    171>;  // Produces no value
def untyped   : ValueType<8,    172>;  // Produces an untyped value
def funcref   : ValueType<0,    173>;  // WebAssembly's funcref type
def externref : ValueType<0,    174>;  // WebAssembly's externref type
def x86amx    : ValueType<8192, 175>;  // X86 AMX value


def token      : ValueType<0, 248>;  // TokenTy
def MetadataVT : ValueType<0, 249>;  // Metadata

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny    : ValueType<0, 250>;

// Pseudo valuetype to represent "vector of any size"
def vAny       : ValueType<0, 251>;

// Pseudo valuetype to represent "float of any format"
def fAny       : ValueType<0, 252>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny       : ValueType<0, 253>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR       : ValueType<0, 254>;

// Pseudo valuetype to represent "any type of any size".
def Any        : ValueType<0, 255>;

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the ponter.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
