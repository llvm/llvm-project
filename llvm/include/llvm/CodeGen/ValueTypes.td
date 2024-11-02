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
def i2      : ValueType<2,   3>;  // 2-bit integer value
def i4      : ValueType<4,   4>;  // 4-bit integer value
def i8      : ValueType<8,   5>;  // 8-bit integer value
def i16     : ValueType<16,  6>;  // 16-bit integer value
def i32     : ValueType<32,  7>;  // 32-bit integer value
def i64     : ValueType<64,  8>;  // 64-bit integer value
def i128    : ValueType<128, 9>;  // 128-bit integer value

def bf16    : ValueType<16,  10>;  // 16-bit brain floating point value
def f16     : ValueType<16,  11>;  // 16-bit floating point value
def f32     : ValueType<32,  12>;  // 32-bit floating point value
def f64     : ValueType<64,  13>;  // 64-bit floating point value
def f80     : ValueType<80,  14>;  // 80-bit floating point value
def f128    : ValueType<128, 15>;  // 128-bit floating point value
def ppcf128 : ValueType<128, 16>;  // PPC 128-bit floating point value

def v1i1    : ValueType<1,    17>;  //    1 x i1 vector value
def v2i1    : ValueType<2,    18>;  //    2 x i1 vector value
def v4i1    : ValueType<4,    19>;  //    4 x i1 vector value
def v8i1    : ValueType<8,    20>;  //    8 x i1 vector value
def v16i1   : ValueType<16,   21>;  //   16 x i1 vector value
def v32i1   : ValueType<32,   22>;  //   32 x i1 vector value
def v64i1   : ValueType<64,   23>;  //   64 x i1 vector value
def v128i1  : ValueType<128,  24>;  //  128 x i1 vector value
def v256i1  : ValueType<256,  25>;  //  256 x i1 vector value
def v512i1  : ValueType<512,  26>;  //  512 x i1 vector value
def v1024i1 : ValueType<1024, 27>;  // 1024 x i1 vector value
def v2048i1 : ValueType<2048, 28>;  // 2048 x i1 vector value

def v128i2  : ValueType<256,  29>;   //  128 x i2 vector value
def v256i2  : ValueType<512,  30>;   //  256 x i2 vector value

def v64i4   : ValueType<256,  31>;   //   64 x i4 vector value
def v128i4  : ValueType<512,  32>;   //  128 x i4 vector value

def v1i8    : ValueType<8,    33>;  //    1 x i8 vector value
def v2i8    : ValueType<16,   34>;  //    2 x i8 vector value
def v4i8    : ValueType<32,   35>;  //    4 x i8 vector value
def v8i8    : ValueType<64,   36>;  //    8 x i8 vector value
def v16i8   : ValueType<128,  37>;  //   16 x i8 vector value
def v32i8   : ValueType<256,  38>;  //   32 x i8 vector value
def v64i8   : ValueType<512,  39>;  //   64 x i8 vector value
def v128i8  : ValueType<1024, 40>;  //  128 x i8 vector value
def v256i8  : ValueType<2048, 41>;  //  256 x i8 vector value
def v512i8  : ValueType<4096, 42>;  //  512 x i8 vector value
def v1024i8 : ValueType<8192, 43>;  // 1024 x i8 vector value

def v1i16   : ValueType<16,   44>;  //   1 x i16 vector value
def v2i16   : ValueType<32,   45>;  //   2 x i16 vector value
def v3i16   : ValueType<48,   46>;  //   3 x i16 vector value
def v4i16   : ValueType<64,   47>;  //   4 x i16 vector value
def v8i16   : ValueType<128,  48>;  //   8 x i16 vector value
def v16i16  : ValueType<256,  49>;  //  16 x i16 vector value
def v32i16  : ValueType<512,  50>;  //  32 x i16 vector value
def v64i16  : ValueType<1024, 51>;  //  64 x i16 vector value
def v128i16 : ValueType<2048, 52>;  // 128 x i16 vector value
def v256i16 : ValueType<4096, 53>;  // 256 x i16 vector value
def v512i16 : ValueType<8192, 54>;  // 512 x i16 vector value

def v1i32    : ValueType<32,    55>;  //    1 x i32 vector value
def v2i32    : ValueType<64,    56>;  //    2 x i32 vector value
def v3i32    : ValueType<96,    57>;  //    3 x i32 vector value
def v4i32    : ValueType<128,   58>;  //    4 x i32 vector value
def v5i32    : ValueType<160,   59>;  //    5 x i32 vector value
def v6i32    : ValueType<192,   60>;  //    6 x f32 vector value
def v7i32    : ValueType<224,   61>;  //    7 x f32 vector value
def v8i32    : ValueType<256,   62>;  //    8 x i32 vector value
def v16i32   : ValueType<512,   63>;  //   16 x i32 vector value
def v32i32   : ValueType<1024,  64>;  //   32 x i32 vector value
def v64i32   : ValueType<2048,  65>;  //   64 x i32 vector value
def v128i32  : ValueType<4096,  66>;  //  128 x i32 vector value
def v256i32  : ValueType<8192,  67>;  //  256 x i32 vector value
def v512i32  : ValueType<16384, 68>;  //  512 x i32 vector value
def v1024i32 : ValueType<32768, 69>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536, 70>;  // 2048 x i32 vector value

def v1i64   : ValueType<64,    71>;  //   1 x i64 vector value
def v2i64   : ValueType<128,   72>;  //   2 x i64 vector value
def v3i64   : ValueType<192,   73>;  //   3 x i64 vector value
def v4i64   : ValueType<256,   74>;  //   4 x i64 vector value
def v8i64   : ValueType<512,   75>;  //   8 x i64 vector value
def v16i64  : ValueType<1024,  76>;  //  16 x i64 vector value
def v32i64  : ValueType<2048,  77>;  //  32 x i64 vector value
def v64i64  : ValueType<4096,  78>;  //  64 x i64 vector value
def v128i64 : ValueType<8192,  79>;  // 128 x i64 vector value
def v256i64 : ValueType<16384, 80>;  // 256 x i64 vector value

def v1i128 : ValueType<128, 81>;  //  1 x i128 vector value

def v1f16    : ValueType<16,     82>;  //    1 x f16 vector value
def v2f16    : ValueType<32,     83>;  //    2 x f16 vector value
def v3f16    : ValueType<48,     84>;  //    3 x f16 vector value
def v4f16    : ValueType<64,     85>;  //    4 x f16 vector value
def v8f16    : ValueType<128,    86>;  //    8 x f16 vector value
def v16f16   : ValueType<256,    87>;  //   16 x f16 vector value
def v32f16   : ValueType<512,    88>;  //   32 x f16 vector value
def v64f16   : ValueType<1024,   89>;  //   64 x f16 vector value
def v128f16  : ValueType<2048,   90>;  //  128 x f16 vector value
def v256f16  : ValueType<4096,   91>;  //  256 x f16 vector value
def v512f16  : ValueType<8192,   92>;  //  512 x f16 vector value

def v2bf16   : ValueType<32,     93>;  //    2 x bf16 vector value
def v3bf16   : ValueType<48,     94>;  //    3 x bf16 vector value
def v4bf16   : ValueType<64,     95>;  //    4 x bf16 vector value
def v8bf16   : ValueType<128,    96>;  //    8 x bf16 vector value
def v16bf16  : ValueType<256,    97>;  //   16 x bf16 vector value
def v32bf16  : ValueType<512,    98>;  //   32 x bf16 vector value
def v64bf16  : ValueType<1024,   99>;  //   64 x bf16 vector value
def v128bf16 : ValueType<2048,  100>;  //  128 x bf16 vector value

def v1f32    : ValueType<32,    101>;  //    1 x f32 vector value
def v2f32    : ValueType<64,    102>;  //    2 x f32 vector value
def v3f32    : ValueType<96,    103>;  //    3 x f32 vector value
def v4f32    : ValueType<128,   104>;  //    4 x f32 vector value
def v5f32    : ValueType<160,   105>;  //    5 x f32 vector value
def v6f32    : ValueType<192,   106>;  //    6 x f32 vector value
def v7f32    : ValueType<224,   107>;  //    7 x f32 vector value
def v8f32    : ValueType<256,   108>;  //    8 x f32 vector value
def v16f32   : ValueType<512,   109>;  //   16 x f32 vector value
def v32f32   : ValueType<1024,  110>;  //   32 x f32 vector value
def v64f32   : ValueType<2048,  111>;  //   64 x f32 vector value
def v128f32  : ValueType<4096,  112>;  //  128 x f32 vector value
def v256f32  : ValueType<8192,  113>;  //  256 x f32 vector value
def v512f32  : ValueType<16384, 114>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 115>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 116>;  // 2048 x f32 vector value

def v1f64    : ValueType<64,    117>;  //    1 x f64 vector value
def v2f64    : ValueType<128,   118>;  //    2 x f64 vector value
def v3f64    : ValueType<192,   119>;  //    3 x f64 vector value
def v4f64    : ValueType<256,   120>;  //    4 x f64 vector value
def v8f64    : ValueType<512,   121>;  //    8 x f64 vector value
def v16f64   : ValueType<1024,  122>;  //   16 x f64 vector value
def v32f64   : ValueType<2048,  123>;  //   32 x f64 vector value
def v64f64   : ValueType<4096,  124>;  //   64 x f64 vector value
def v128f64  : ValueType<8192,  125>;  //  128 x f64 vector value
def v256f64  : ValueType<16384, 126>;  //  256 x f64 vector value

def nxv1i1  : ValueType<1,  127>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,  128>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,  129>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,  130>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16, 131>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32, 132>;  // n x 32 x i1  vector value
def nxv64i1 : ValueType<64, 133>;  // n x 64 x i1  vector value

def nxv1i8  : ValueType<8,   134>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  135>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  136>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  137>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 138>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 139>;  // n x 32 x i8  vector value
def nxv64i8 : ValueType<512, 140>;  // n x 64 x i8  vector value

def nxv1i16  : ValueType<16,  141>;  // n x  1 x i16 vector value
def nxv2i16  : ValueType<32,  142>;  // n x  2 x i16 vector value
def nxv4i16  : ValueType<64,  143>;  // n x  4 x i16 vector value
def nxv8i16  : ValueType<128, 144>;  // n x  8 x i16 vector value
def nxv16i16 : ValueType<256, 145>;  // n x 16 x i16 vector value
def nxv32i16 : ValueType<512, 146>;  // n x 32 x i16 vector value

def nxv1i32  : ValueType<32,   147>;  // n x  1 x i32 vector value
def nxv2i32  : ValueType<64,   148>;  // n x  2 x i32 vector value
def nxv4i32  : ValueType<128,  149>;  // n x  4 x i32 vector value
def nxv8i32  : ValueType<256,  150>;  // n x  8 x i32 vector value
def nxv16i32 : ValueType<512,  151>;  // n x 16 x i32 vector value
def nxv32i32 : ValueType<1024, 152>;  // n x 32 x i32 vector value

def nxv1i64  : ValueType<64,   153>;  // n x  1 x i64 vector value
def nxv2i64  : ValueType<128,  154>;  // n x  2 x i64 vector value
def nxv4i64  : ValueType<256,  155>;  // n x  4 x i64 vector value
def nxv8i64  : ValueType<512,  156>;  // n x  8 x i64 vector value
def nxv16i64 : ValueType<1024, 157>;  // n x 16 x i64 vector value
def nxv32i64 : ValueType<2048, 158>;  // n x 32 x i64 vector value

def nxv1f16  : ValueType<16,  159>;  // n x  1 x  f16 vector value
def nxv2f16  : ValueType<32,  160>;  // n x  2 x  f16 vector value
def nxv4f16  : ValueType<64,  161>;  // n x  4 x  f16 vector value
def nxv8f16  : ValueType<128, 162>;  // n x  8 x  f16 vector value
def nxv16f16 : ValueType<256, 163>;  // n x 16 x  f16 vector value
def nxv32f16 : ValueType<512, 164>;  // n x 32 x  f16 vector value

def nxv1bf16 : ValueType<16,  165>;  // n x  1 x bf16 vector value
def nxv2bf16 : ValueType<32,  166>;  // n x  2 x bf16 vector value
def nxv4bf16 : ValueType<64,  167>;  // n x  4 x bf16 vector value
def nxv8bf16 : ValueType<128, 168>;  // n x  8 x bf16 vector value
def nxv16bf16 : ValueType<256, 169>;  // n x 16 x bf16 vector value
def nxv32bf16 : ValueType<512, 170>;  // n x 32 x bf16 vector value

def nxv1f32  : ValueType<32,  171>;  // n x  1 x  f32 vector value
def nxv2f32  : ValueType<64,  172>;  // n x  2 x  f32 vector value
def nxv4f32  : ValueType<128, 173>;  // n x  4 x  f32 vector value
def nxv8f32  : ValueType<256, 174>;  // n x  8 x  f32 vector value
def nxv16f32 : ValueType<512, 175>;  // n x 16 x  f32 vector value

def nxv1f64  : ValueType<64,  176>;  // n x  1 x  f64 vector value
def nxv2f64  : ValueType<128, 177>;  // n x  2 x  f64 vector value
def nxv4f64  : ValueType<256, 178>;  // n x  4 x  f64 vector value
def nxv8f64  : ValueType<512, 179>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   180>;  // X86 MMX value
def FlagVT    : ValueType<0,    181>;  // Pre-RA sched glue
def isVoid    : ValueType<0,    182>;  // Produces no value
def untyped   : ValueType<8,    183>;  // Produces an untyped value
def funcref   : ValueType<0,    184>;  // WebAssembly's funcref type
def externref : ValueType<0,    185>;  // WebAssembly's externref type
def x86amx    : ValueType<8192, 186>;  // X86 AMX value
def i64x8     : ValueType<512,  187>;  // 8 Consecutive GPRs (AArch64)

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
