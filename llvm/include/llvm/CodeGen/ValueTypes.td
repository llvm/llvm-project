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
def v3i64   : ValueType<192,   66>;  //   3 x i64 vector value
def v4i64   : ValueType<256,   67>;  //   4 x i64 vector value
def v8i64   : ValueType<512,   68>;  //   8 x i64 vector value
def v16i64  : ValueType<1024,  69>;  //  16 x i64 vector value
def v32i64  : ValueType<2048,  70>;  //  32 x i64 vector value
def v64i64  : ValueType<4096,  71>;  //  64 x i64 vector value
def v128i64 : ValueType<8192,  72>;  // 128 x i64 vector value
def v256i64 : ValueType<16384, 73>;  // 256 x i64 vector value

def v1i128 : ValueType<128, 74>;  //  1 x i128 vector value

def v1f16    : ValueType<16,     75>;  //    1 x f16 vector value
def v2f16    : ValueType<32,     76>;  //    2 x f16 vector value
def v3f16    : ValueType<48,     77>;  //    3 x f16 vector value
def v4f16    : ValueType<64,     78>;  //    4 x f16 vector value
def v8f16    : ValueType<128,    79>;  //    8 x f16 vector value
def v16f16   : ValueType<256,    80>;  //   16 x f16 vector value
def v32f16   : ValueType<512,    81>;  //   32 x f16 vector value
def v64f16   : ValueType<1024,   82>;  //   64 x f16 vector value
def v128f16  : ValueType<2048,   83>;  //  128 x f16 vector value
def v256f16  : ValueType<4096,   84>;  //  256 x f16 vector value
def v512f16  : ValueType<8192,   85>;  //  512 x f16 vector value

def v2bf16   : ValueType<32,     86>;  //    2 x bf16 vector value
def v3bf16   : ValueType<48,     87>;  //    3 x bf16 vector value
def v4bf16   : ValueType<64,     88>;  //    4 x bf16 vector value
def v8bf16   : ValueType<128,    89>;  //    8 x bf16 vector value
def v16bf16  : ValueType<256,    90>;  //   16 x bf16 vector value
def v32bf16  : ValueType<512,    91>;  //   32 x bf16 vector value
def v64bf16  : ValueType<1024,   92>;  //   64 x bf16 vector value
def v128bf16 : ValueType<2048,   93>;  //  128 x bf16 vector value

def v1f32    : ValueType<32,     94>;  //    1 x f32 vector value
def v2f32    : ValueType<64,     95>;  //    2 x f32 vector value
def v3f32    : ValueType<96,     96>;  //    3 x f32 vector value
def v4f32    : ValueType<128,    97>;  //    4 x f32 vector value
def v5f32    : ValueType<160,    98>;  //    5 x f32 vector value
def v6f32    : ValueType<192,    99>;  //    6 x f32 vector value
def v7f32    : ValueType<224,   100>;  //    7 x f32 vector value
def v8f32    : ValueType<256,   101>;  //    8 x f32 vector value
def v16f32   : ValueType<512,   102>;  //   16 x f32 vector value
def v32f32   : ValueType<1024,  103>;  //   32 x f32 vector value
def v64f32   : ValueType<2048,  104>;  //   64 x f32 vector value
def v128f32  : ValueType<4096,  105>;  //  128 x f32 vector value
def v256f32  : ValueType<8192,  106>;  //  256 x f32 vector value
def v512f32  : ValueType<16384, 107>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 108>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 109>;  // 2048 x f32 vector value

def v1f64    : ValueType<64,    110>;  //    1 x f64 vector value
def v2f64    : ValueType<128,   111>;  //    2 x f64 vector value
def v3f64    : ValueType<192,   112>;  //    3 x f64 vector value
def v4f64    : ValueType<256,   113>;  //    4 x f64 vector value
def v8f64    : ValueType<512,   114>;  //    8 x f64 vector value
def v16f64   : ValueType<1024,  115>;  //   16 x f64 vector value
def v32f64   : ValueType<2048,  116>;  //   32 x f64 vector value
def v64f64   : ValueType<4096,  117>;  //   64 x f64 vector value
def v128f64  : ValueType<8192,  118>;  //  128 x f64 vector value
def v256f64  : ValueType<16384, 119>;  //  256 x f64 vector value

def nxv1i1  : ValueType<1,  120>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,  121>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,  122>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,  123>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16, 124>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32, 125>;  // n x 32 x i1  vector value
def nxv64i1 : ValueType<64, 126>;  // n x 64 x i1  vector value

def nxv1i8  : ValueType<8,   127>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  128>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  129>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  130>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 131>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 132>;  // n x 32 x i8  vector value
def nxv64i8 : ValueType<512, 133>;  // n x 64 x i8  vector value

def nxv1i16  : ValueType<16,  134>;  // n x  1 x i16 vector value
def nxv2i16  : ValueType<32,  135>;  // n x  2 x i16 vector value
def nxv4i16  : ValueType<64,  136>;  // n x  4 x i16 vector value
def nxv8i16  : ValueType<128, 137>;  // n x  8 x i16 vector value
def nxv16i16 : ValueType<256, 138>;  // n x 16 x i16 vector value
def nxv32i16 : ValueType<512, 139>;  // n x 32 x i16 vector value

def nxv1i32  : ValueType<32,   140>;  // n x  1 x i32 vector value
def nxv2i32  : ValueType<64,   141>;  // n x  2 x i32 vector value
def nxv4i32  : ValueType<128,  142>;  // n x  4 x i32 vector value
def nxv8i32  : ValueType<256,  143>;  // n x  8 x i32 vector value
def nxv16i32 : ValueType<512,  144>;  // n x 16 x i32 vector value
def nxv32i32 : ValueType<1024, 145>;  // n x 32 x i32 vector value

def nxv1i64  : ValueType<64,   146>;  // n x  1 x i64 vector value
def nxv2i64  : ValueType<128,  147>;  // n x  2 x i64 vector value
def nxv4i64  : ValueType<256,  148>;  // n x  4 x i64 vector value
def nxv8i64  : ValueType<512,  149>;  // n x  8 x i64 vector value
def nxv16i64 : ValueType<1024, 150>;  // n x 16 x i64 vector value
def nxv32i64 : ValueType<2048, 151>;  // n x 32 x i64 vector value

def nxv1f16  : ValueType<16,  152>;  // n x  1 x  f16 vector value
def nxv2f16  : ValueType<32,  153>;  // n x  2 x  f16 vector value
def nxv4f16  : ValueType<64,  154>;  // n x  4 x  f16 vector value
def nxv8f16  : ValueType<128, 155>;  // n x  8 x  f16 vector value
def nxv16f16 : ValueType<256, 156>;  // n x 16 x  f16 vector value
def nxv32f16 : ValueType<512, 157>;  // n x 32 x  f16 vector value

def nxv1bf16 : ValueType<16,  158>;  // n x  1 x bf16 vector value
def nxv2bf16 : ValueType<32,  159>;  // n x  2 x bf16 vector value
def nxv4bf16 : ValueType<64,  160>;  // n x  4 x bf16 vector value
def nxv8bf16 : ValueType<128, 161>;  // n x  8 x bf16 vector value

def nxv1f32  : ValueType<32,  162>;  // n x  1 x  f32 vector value
def nxv2f32  : ValueType<64,  163>;  // n x  2 x  f32 vector value
def nxv4f32  : ValueType<128, 164>;  // n x  4 x  f32 vector value
def nxv8f32  : ValueType<256, 165>;  // n x  8 x  f32 vector value
def nxv16f32 : ValueType<512, 166>;  // n x 16 x  f32 vector value

def nxv1f64  : ValueType<64,  167>;  // n x  1 x  f64 vector value
def nxv2f64  : ValueType<128, 168>;  // n x  2 x  f64 vector value
def nxv4f64  : ValueType<256, 169>;  // n x  4 x  f64 vector value
def nxv8f64  : ValueType<512, 170>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   171>;  // X86 MMX value
def FlagVT    : ValueType<0,    172>;  // Pre-RA sched glue
def isVoid    : ValueType<0,    173>;  // Produces no value
def untyped   : ValueType<8,    174>;  // Produces an untyped value
def funcref   : ValueType<0,    175>;  // WebAssembly's funcref type
def externref : ValueType<0,    176>;  // WebAssembly's externref type
def x86amx    : ValueType<8192, 177>;  // X86 AMX value
def i64x8     : ValueType<512,  178>;  // 8 Consecutive GPRs (AArch64)


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
/// integer type with the same bit size as the pointer.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
