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

def v128i2  : ValueType<256,  28>;   //  128 x i2 vector value

def v64i4   : ValueType<256,  29>;   //   64 x i4 vector value

def v1i8    : ValueType<8,    30>;  //    1 x i8 vector value
def v2i8    : ValueType<16,   31>;  //    2 x i8 vector value
def v4i8    : ValueType<32,   32>;  //    4 x i8 vector value
def v8i8    : ValueType<64,   33>;  //    8 x i8 vector value
def v16i8   : ValueType<128,  34>;  //   16 x i8 vector value
def v32i8   : ValueType<256,  35>;  //   32 x i8 vector value
def v64i8   : ValueType<512,  36>;  //   64 x i8 vector value
def v128i8  : ValueType<1024, 37>;  //  128 x i8 vector value
def v256i8  : ValueType<2048, 38>;  //  256 x i8 vector value
def v512i8  : ValueType<4096, 39>;  //  512 x i8 vector value
def v1024i8 : ValueType<8192, 40>;  // 1024 x i8 vector value

def v1i16   : ValueType<16,   41>;  //   1 x i16 vector value
def v2i16   : ValueType<32,   42>;  //   2 x i16 vector value
def v3i16   : ValueType<48,   43>;  //   3 x i16 vector value
def v4i16   : ValueType<64,   44>;  //   4 x i16 vector value
def v8i16   : ValueType<128,  45>;  //   8 x i16 vector value
def v16i16  : ValueType<256,  46>;  //  16 x i16 vector value
def v32i16  : ValueType<512,  47>;  //  32 x i16 vector value
def v64i16  : ValueType<1024, 48>;  //  64 x i16 vector value
def v128i16 : ValueType<2048, 49>;  // 128 x i16 vector value
def v256i16 : ValueType<4096, 50>;  // 256 x i16 vector value
def v512i16 : ValueType<8192, 51>;  // 512 x i16 vector value

def v1i32    : ValueType<32,    52>;  //    1 x i32 vector value
def v2i32    : ValueType<64,    53>;  //    2 x i32 vector value
def v3i32    : ValueType<96,    54>;  //    3 x i32 vector value
def v4i32    : ValueType<128,   55>;  //    4 x i32 vector value
def v5i32    : ValueType<160,   56>;  //    5 x i32 vector value
def v6i32    : ValueType<192,   57>;  //    6 x f32 vector value
def v7i32    : ValueType<224,   58>;  //    7 x f32 vector value
def v8i32    : ValueType<256,   59>;  //    8 x i32 vector value
def v16i32   : ValueType<512,   60>;  //   16 x i32 vector value
def v32i32   : ValueType<1024,  61>;  //   32 x i32 vector value
def v64i32   : ValueType<2048,  62>;  //   64 x i32 vector value
def v128i32  : ValueType<4096,  63>;  //  128 x i32 vector value
def v256i32  : ValueType<8192,  64>;  //  256 x i32 vector value
def v512i32  : ValueType<16384, 65>;  //  512 x i32 vector value
def v1024i32 : ValueType<32768, 66>;  // 1024 x i32 vector value
def v2048i32 : ValueType<65536, 67>;  // 2048 x i32 vector value

def v1i64   : ValueType<64,    68>;  //   1 x i64 vector value
def v2i64   : ValueType<128,   69>;  //   2 x i64 vector value
def v3i64   : ValueType<192,   70>;  //   3 x i64 vector value
def v4i64   : ValueType<256,   71>;  //   4 x i64 vector value
def v8i64   : ValueType<512,   72>;  //   8 x i64 vector value
def v16i64  : ValueType<1024,  73>;  //  16 x i64 vector value
def v32i64  : ValueType<2048,  74>;  //  32 x i64 vector value
def v64i64  : ValueType<4096,  75>;  //  64 x i64 vector value
def v128i64 : ValueType<8192,  76>;  // 128 x i64 vector value
def v256i64 : ValueType<16384, 77>;  // 256 x i64 vector value

def v1i128 : ValueType<128, 78>;  //  1 x i128 vector value

def v1f16    : ValueType<16,     79>;  //    1 x f16 vector value
def v2f16    : ValueType<32,     80>;  //    2 x f16 vector value
def v3f16    : ValueType<48,     81>;  //    3 x f16 vector value
def v4f16    : ValueType<64,     82>;  //    4 x f16 vector value
def v8f16    : ValueType<128,    83>;  //    8 x f16 vector value
def v16f16   : ValueType<256,    84>;  //   16 x f16 vector value
def v32f16   : ValueType<512,    85>;  //   32 x f16 vector value
def v64f16   : ValueType<1024,   86>;  //   64 x f16 vector value
def v128f16  : ValueType<2048,   87>;  //  128 x f16 vector value
def v256f16  : ValueType<4096,   88>;  //  256 x f16 vector value
def v512f16  : ValueType<8192,   89>;  //  512 x f16 vector value

def v2bf16   : ValueType<32,     90>;  //    2 x bf16 vector value
def v3bf16   : ValueType<48,     91>;  //    3 x bf16 vector value
def v4bf16   : ValueType<64,     92>;  //    4 x bf16 vector value
def v8bf16   : ValueType<128,    93>;  //    8 x bf16 vector value
def v16bf16  : ValueType<256,    94>;  //   16 x bf16 vector value
def v32bf16  : ValueType<512,    95>;  //   32 x bf16 vector value
def v64bf16  : ValueType<1024,   96>;  //   64 x bf16 vector value
def v128bf16 : ValueType<2048,   97>;  //  128 x bf16 vector value

def v1f32    : ValueType<32,     98>;  //    1 x f32 vector value
def v2f32    : ValueType<64,     99>;  //    2 x f32 vector value
def v3f32    : ValueType<96,    100>;  //    3 x f32 vector value
def v4f32    : ValueType<128,   101>;  //    4 x f32 vector value
def v5f32    : ValueType<160,   102>;  //    5 x f32 vector value
def v6f32    : ValueType<192,   103>;  //    6 x f32 vector value
def v7f32    : ValueType<224,   104>;  //    7 x f32 vector value
def v8f32    : ValueType<256,   105>;  //    8 x f32 vector value
def v16f32   : ValueType<512,   106>;  //   16 x f32 vector value
def v32f32   : ValueType<1024,  107>;  //   32 x f32 vector value
def v64f32   : ValueType<2048,  108>;  //   64 x f32 vector value
def v128f32  : ValueType<4096,  109>;  //  128 x f32 vector value
def v256f32  : ValueType<8192,  110>;  //  256 x f32 vector value
def v512f32  : ValueType<16384, 111>;  //  512 x f32 vector value
def v1024f32 : ValueType<32768, 112>;  // 1024 x f32 vector value
def v2048f32 : ValueType<65536, 113>;  // 2048 x f32 vector value

def v1f64    : ValueType<64,    114>;  //    1 x f64 vector value
def v2f64    : ValueType<128,   115>;  //    2 x f64 vector value
def v3f64    : ValueType<192,   116>;  //    3 x f64 vector value
def v4f64    : ValueType<256,   117>;  //    4 x f64 vector value
def v8f64    : ValueType<512,   118>;  //    8 x f64 vector value
def v16f64   : ValueType<1024,  119>;  //   16 x f64 vector value
def v32f64   : ValueType<2048,  120>;  //   32 x f64 vector value
def v64f64   : ValueType<4096,  121>;  //   64 x f64 vector value
def v128f64  : ValueType<8192,  122>;  //  128 x f64 vector value
def v256f64  : ValueType<16384, 123>;  //  256 x f64 vector value

def nxv1i1  : ValueType<1,  124>;  // n x  1 x i1  vector value
def nxv2i1  : ValueType<2,  125>;  // n x  2 x i1  vector value
def nxv4i1  : ValueType<4,  126>;  // n x  4 x i1  vector value
def nxv8i1  : ValueType<8,  127>;  // n x  8 x i1  vector value
def nxv16i1 : ValueType<16, 128>;  // n x 16 x i1  vector value
def nxv32i1 : ValueType<32, 129>;  // n x 32 x i1  vector value
def nxv64i1 : ValueType<64, 130>;  // n x 64 x i1  vector value

def nxv1i8  : ValueType<8,   131>;  // n x  1 x i8  vector value
def nxv2i8  : ValueType<16,  132>;  // n x  2 x i8  vector value
def nxv4i8  : ValueType<32,  133>;  // n x  4 x i8  vector value
def nxv8i8  : ValueType<64,  134>;  // n x  8 x i8  vector value
def nxv16i8 : ValueType<128, 135>;  // n x 16 x i8  vector value
def nxv32i8 : ValueType<256, 136>;  // n x 32 x i8  vector value
def nxv64i8 : ValueType<512, 137>;  // n x 64 x i8  vector value

def nxv1i16  : ValueType<16,  138>;  // n x  1 x i16 vector value
def nxv2i16  : ValueType<32,  139>;  // n x  2 x i16 vector value
def nxv4i16  : ValueType<64,  140>;  // n x  4 x i16 vector value
def nxv8i16  : ValueType<128, 141>;  // n x  8 x i16 vector value
def nxv16i16 : ValueType<256, 142>;  // n x 16 x i16 vector value
def nxv32i16 : ValueType<512, 143>;  // n x 32 x i16 vector value

def nxv1i32  : ValueType<32,   144>;  // n x  1 x i32 vector value
def nxv2i32  : ValueType<64,   145>;  // n x  2 x i32 vector value
def nxv4i32  : ValueType<128,  146>;  // n x  4 x i32 vector value
def nxv8i32  : ValueType<256,  147>;  // n x  8 x i32 vector value
def nxv16i32 : ValueType<512,  148>;  // n x 16 x i32 vector value
def nxv32i32 : ValueType<1024, 149>;  // n x 32 x i32 vector value

def nxv1i64  : ValueType<64,   150>;  // n x  1 x i64 vector value
def nxv2i64  : ValueType<128,  151>;  // n x  2 x i64 vector value
def nxv4i64  : ValueType<256,  152>;  // n x  4 x i64 vector value
def nxv8i64  : ValueType<512,  153>;  // n x  8 x i64 vector value
def nxv16i64 : ValueType<1024, 154>;  // n x 16 x i64 vector value
def nxv32i64 : ValueType<2048, 155>;  // n x 32 x i64 vector value

def nxv1f16  : ValueType<16,  156>;  // n x  1 x  f16 vector value
def nxv2f16  : ValueType<32,  157>;  // n x  2 x  f16 vector value
def nxv4f16  : ValueType<64,  158>;  // n x  4 x  f16 vector value
def nxv8f16  : ValueType<128, 159>;  // n x  8 x  f16 vector value
def nxv16f16 : ValueType<256, 160>;  // n x 16 x  f16 vector value
def nxv32f16 : ValueType<512, 161>;  // n x 32 x  f16 vector value

def nxv1bf16 : ValueType<16,  162>;  // n x  1 x bf16 vector value
def nxv2bf16 : ValueType<32,  163>;  // n x  2 x bf16 vector value
def nxv4bf16 : ValueType<64,  164>;  // n x  4 x bf16 vector value
def nxv8bf16 : ValueType<128, 165>;  // n x  8 x bf16 vector value
def nxv16bf16 : ValueType<256, 166>;  // n x 16 x bf16 vector value
def nxv32bf16 : ValueType<512, 167>;  // n x 32 x bf16 vector value

def nxv1f32  : ValueType<32,  168>;  // n x  1 x  f32 vector value
def nxv2f32  : ValueType<64,  169>;  // n x  2 x  f32 vector value
def nxv4f32  : ValueType<128, 170>;  // n x  4 x  f32 vector value
def nxv8f32  : ValueType<256, 171>;  // n x  8 x  f32 vector value
def nxv16f32 : ValueType<512, 172>;  // n x 16 x  f32 vector value

def nxv1f64  : ValueType<64,  173>;  // n x  1 x  f64 vector value
def nxv2f64  : ValueType<128, 174>;  // n x  2 x  f64 vector value
def nxv4f64  : ValueType<256, 175>;  // n x  4 x  f64 vector value
def nxv8f64  : ValueType<512, 176>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   177>;  // X86 MMX value
def FlagVT    : ValueType<0,    178>;  // Pre-RA sched glue
def isVoid    : ValueType<0,    179>;  // Produces no value
def untyped   : ValueType<8,    180>;  // Produces an untyped value
def funcref   : ValueType<0,    181>;  // WebAssembly's funcref type
def externref : ValueType<0,    182>;  // WebAssembly's externref type
def x86amx    : ValueType<8192, 183>;  // X86 AMX value
def i64x8     : ValueType<512,  184>;  // 8 Consecutive GPRs (AArch64)

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
