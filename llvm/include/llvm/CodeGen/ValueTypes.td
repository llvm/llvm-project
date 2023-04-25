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

defset list<ValueType> ValueTypes = {

def OtherVT : ValueType<0,   1> {  // "Other" value
  let LLVMName = "Other";
}

def i1      : VTInt<1,   2>;  // One bit boolean value
def i2      : VTInt<2,   3>;  // 2-bit integer value
def i4      : VTInt<4,   4>;  // 4-bit integer value
def i8      : VTInt<8,   5>;  // 8-bit integer value
def i16     : VTInt<16,  6>;  // 16-bit integer value
def i32     : VTInt<32,  7>;  // 32-bit integer value
def i64     : VTInt<64,  8>;  // 64-bit integer value
def i128    : VTInt<128, 9>;  // 128-bit integer value

def bf16    : VTFP<16,  10>;  // 16-bit brain floating point value
def f16     : VTFP<16,  11>;  // 16-bit floating point value
def f32     : VTFP<32,  12>;  // 32-bit floating point value
def f64     : VTFP<64,  13>;  // 64-bit floating point value
def f80     : VTFP<80,  14>;  // 80-bit floating point value
def f128    : VTFP<128, 15>;  // 128-bit floating point value
def ppcf128 : VTFP<128, 16>;  // PPC 128-bit floating point value

def v1i1    : VTVec<1,    i1, 17>;  //    1 x i1 vector value
def v2i1    : VTVec<2,    i1, 18>;  //    2 x i1 vector value
def v4i1    : VTVec<4,    i1, 19>;  //    4 x i1 vector value
def v8i1    : VTVec<8,    i1, 20>;  //    8 x i1 vector value
def v16i1   : VTVec<16,   i1, 21>;  //   16 x i1 vector value
def v32i1   : VTVec<32,   i1, 22>;  //   32 x i1 vector value
def v64i1   : VTVec<64,   i1, 23>;  //   64 x i1 vector value
def v128i1  : VTVec<128,  i1, 24>;  //  128 x i1 vector value
def v256i1  : VTVec<256,  i1, 25>;  //  256 x i1 vector value
def v512i1  : VTVec<512,  i1, 26>;  //  512 x i1 vector value
def v1024i1 : VTVec<1024, i1, 27>;  // 1024 x i1 vector value
def v2048i1 : VTVec<2048, i1, 28>;  // 2048 x i1 vector value

def v128i2  : VTVec<128,  i2, 29>;   //  128 x i2 vector value
def v256i2  : VTVec<256,  i2, 30>;   //  256 x i2 vector value

def v64i4   : VTVec<64,   i4, 31>;   //   64 x i4 vector value
def v128i4  : VTVec<128,  i4, 32>;   //  128 x i4 vector value

def v1i8    : VTVec<1,    i8, 33>;  //    1 x i8 vector value
def v2i8    : VTVec<2,    i8, 34>;  //    2 x i8 vector value
def v4i8    : VTVec<4,    i8, 35>;  //    4 x i8 vector value
def v8i8    : VTVec<8,    i8, 36>;  //    8 x i8 vector value
def v16i8   : VTVec<16,   i8, 37>;  //   16 x i8 vector value
def v32i8   : VTVec<32,   i8, 38>;  //   32 x i8 vector value
def v64i8   : VTVec<64,   i8, 39>;  //   64 x i8 vector value
def v128i8  : VTVec<128,  i8, 40>;  //  128 x i8 vector value
def v256i8  : VTVec<256,  i8, 41>;  //  256 x i8 vector value
def v512i8  : VTVec<512,  i8, 42>;  //  512 x i8 vector value
def v1024i8 : VTVec<1024, i8, 43>;  // 1024 x i8 vector value

def v1i16   : VTVec<1,   i16, 44>;  //   1 x i16 vector value
def v2i16   : VTVec<2,   i16, 45>;  //   2 x i16 vector value
def v3i16   : VTVec<3,   i16, 46>;  //   3 x i16 vector value
def v4i16   : VTVec<4,   i16, 47>;  //   4 x i16 vector value
def v8i16   : VTVec<8,   i16, 48>;  //   8 x i16 vector value
def v16i16  : VTVec<16,  i16, 49>;  //  16 x i16 vector value
def v32i16  : VTVec<32,  i16, 50>;  //  32 x i16 vector value
def v64i16  : VTVec<64,  i16, 51>;  //  64 x i16 vector value
def v128i16 : VTVec<128, i16, 52>;  // 128 x i16 vector value
def v256i16 : VTVec<256, i16, 53>;  // 256 x i16 vector value
def v512i16 : VTVec<512, i16, 54>;  // 512 x i16 vector value

def v1i32    : VTVec<1,    i32, 55>;  //    1 x i32 vector value
def v2i32    : VTVec<2,    i32, 56>;  //    2 x i32 vector value
def v3i32    : VTVec<3,    i32, 57>;  //    3 x i32 vector value
def v4i32    : VTVec<4,    i32, 58>;  //    4 x i32 vector value
def v5i32    : VTVec<5,    i32, 59>;  //    5 x i32 vector value
def v6i32    : VTVec<6,    i32, 60>;  //    6 x f32 vector value
def v7i32    : VTVec<7,    i32, 61>;  //    7 x f32 vector value
def v8i32    : VTVec<8,    i32, 62>;  //    8 x i32 vector value
def v9i32    : VTVec<9,    i32, 63>;  //    9 x i32 vector value
def v10i32   : VTVec<10,   i32, 64>;  //   10 x i32 vector value
def v11i32   : VTVec<11,   i32, 65>;  //   11 x i32 vector value
def v12i32   : VTVec<12,   i32, 66>;  //   12 x i32 vector value
def v16i32   : VTVec<16,   i32, 67>;  //   16 x i32 vector value
def v32i32   : VTVec<32,   i32, 68>;  //   32 x i32 vector value
def v64i32   : VTVec<64,   i32, 69>;  //   64 x i32 vector value
def v128i32  : VTVec<128,  i32, 70>;  //  128 x i32 vector value
def v256i32  : VTVec<256,  i32, 71>;  //  256 x i32 vector value
def v512i32  : VTVec<512,  i32, 72>;  //  512 x i32 vector value
def v1024i32 : VTVec<1024, i32, 73>;  // 1024 x i32 vector value
def v2048i32 : VTVec<2048, i32, 74>;  // 2048 x i32 vector value

def v1i64   : VTVec<1,   i64, 75>;  //   1 x i64 vector value
def v2i64   : VTVec<2,   i64, 76>;  //   2 x i64 vector value
def v3i64   : VTVec<3,   i64, 77>;  //   3 x i64 vector value
def v4i64   : VTVec<4,   i64, 78>;  //   4 x i64 vector value
def v8i64   : VTVec<8,   i64, 79>;  //   8 x i64 vector value
def v16i64  : VTVec<16,  i64, 80>;  //  16 x i64 vector value
def v32i64  : VTVec<32,  i64, 81>;  //  32 x i64 vector value
def v64i64  : VTVec<64,  i64, 82>;  //  64 x i64 vector value
def v128i64 : VTVec<128, i64, 83>;  // 128 x i64 vector value
def v256i64 : VTVec<256, i64, 84>;  // 256 x i64 vector value

def v1i128  : VTVec<1,  i128, 85>;  //  1 x i128 vector value

def v1f16    : VTVec<1,    f16,  86>;  //    1 x f16 vector value
def v2f16    : VTVec<2,    f16,  87>;  //    2 x f16 vector value
def v3f16    : VTVec<3,    f16,  88>;  //    3 x f16 vector value
def v4f16    : VTVec<4,    f16,  89>;  //    4 x f16 vector value
def v8f16    : VTVec<8,    f16,  90>;  //    8 x f16 vector value
def v16f16   : VTVec<16,   f16,  91>;  //   16 x f16 vector value
def v32f16   : VTVec<32,   f16,  92>;  //   32 x f16 vector value
def v64f16   : VTVec<64,   f16,  93>;  //   64 x f16 vector value
def v128f16  : VTVec<128,  f16,  94>;  //  128 x f16 vector value
def v256f16  : VTVec<256,  f16,  95>;  //  256 x f16 vector value
def v512f16  : VTVec<512,  f16,  96>;  //  512 x f16 vector value

def v2bf16   : VTVec<2,   bf16,  97>;  //    2 x bf16 vector value
def v3bf16   : VTVec<3,   bf16,  98>;  //    3 x bf16 vector value
def v4bf16   : VTVec<4,   bf16,  99>;  //    4 x bf16 vector value
def v8bf16   : VTVec<8,   bf16, 100>;  //    8 x bf16 vector value
def v16bf16  : VTVec<16,  bf16, 101>;  //   16 x bf16 vector value
def v32bf16  : VTVec<32,  bf16, 102>;  //   32 x bf16 vector value
def v64bf16  : VTVec<64,  bf16, 103>;  //   64 x bf16 vector value
def v128bf16 : VTVec<128, bf16, 104>;  //  128 x bf16 vector value

def v1f32    : VTVec<1,    f32, 105>;  //    1 x f32 vector value
def v2f32    : VTVec<2,    f32, 106>;  //    2 x f32 vector value
def v3f32    : VTVec<3,    f32, 107>;  //    3 x f32 vector value
def v4f32    : VTVec<4,    f32, 108>;  //    4 x f32 vector value
def v5f32    : VTVec<5,    f32, 109>;  //    5 x f32 vector value
def v6f32    : VTVec<6,    f32, 110>;  //    6 x f32 vector value
def v7f32    : VTVec<7,    f32, 111>;  //    7 x f32 vector value
def v8f32    : VTVec<8,    f32, 112>;  //    8 x f32 vector value
def v9f32    : VTVec<9,    f32, 113>;  //    9 x f32 vector value
def v10f32   : VTVec<10,   f32, 114>;  //   10 x f32 vector value
def v11f32   : VTVec<11,   f32, 115>;  //   11 x f32 vector value
def v12f32   : VTVec<12,   f32, 116>;  //   12 x f32 vector value
def v16f32   : VTVec<16,   f32, 117>;  //   16 x f32 vector value
def v32f32   : VTVec<32,   f32, 118>;  //   32 x f32 vector value
def v64f32   : VTVec<64,   f32, 119>;  //   64 x f32 vector value
def v128f32  : VTVec<128,  f32, 120>;  //  128 x f32 vector value
def v256f32  : VTVec<256,  f32, 121>;  //  256 x f32 vector value
def v512f32  : VTVec<512,  f32, 122>;  //  512 x f32 vector value
def v1024f32 : VTVec<1024, f32, 123>;  // 1024 x f32 vector value
def v2048f32 : VTVec<2048, f32, 124>;  // 2048 x f32 vector value

def v1f64    : VTVec<1,    f64, 125>;  //    1 x f64 vector value
def v2f64    : VTVec<2,    f64, 126>;  //    2 x f64 vector value
def v3f64    : VTVec<3,    f64, 127>;  //    3 x f64 vector value
def v4f64    : VTVec<4,    f64, 128>;  //    4 x f64 vector value
def v8f64    : VTVec<8,    f64, 129>;  //    8 x f64 vector value
def v16f64   : VTVec<16,   f64, 130>;  //   16 x f64 vector value
def v32f64   : VTVec<32,   f64, 131>;  //   32 x f64 vector value
def v64f64   : VTVec<64,   f64, 132>;  //   64 x f64 vector value
def v128f64  : VTVec<128,  f64, 133>;  //  128 x f64 vector value
def v256f64  : VTVec<256,  f64, 134>;  //  256 x f64 vector value

def nxv1i1  : VTScalableVec<1,  i1, 135>;  // n x  1 x i1  vector value
def nxv2i1  : VTScalableVec<2,  i1, 136>;  // n x  2 x i1  vector value
def nxv4i1  : VTScalableVec<4,  i1, 137>;  // n x  4 x i1  vector value
def nxv8i1  : VTScalableVec<8,  i1, 138>;  // n x  8 x i1  vector value
def nxv16i1 : VTScalableVec<16, i1, 139>;  // n x 16 x i1  vector value
def nxv32i1 : VTScalableVec<32, i1, 140>;  // n x 32 x i1  vector value
def nxv64i1 : VTScalableVec<64, i1, 141>;  // n x 64 x i1  vector value

def nxv1i8  : VTScalableVec<1,  i8, 142>;  // n x  1 x i8  vector value
def nxv2i8  : VTScalableVec<2,  i8, 143>;  // n x  2 x i8  vector value
def nxv4i8  : VTScalableVec<4,  i8, 144>;  // n x  4 x i8  vector value
def nxv8i8  : VTScalableVec<8,  i8, 145>;  // n x  8 x i8  vector value
def nxv16i8 : VTScalableVec<16, i8, 146>;  // n x 16 x i8  vector value
def nxv32i8 : VTScalableVec<32, i8, 147>;  // n x 32 x i8  vector value
def nxv64i8 : VTScalableVec<64, i8, 148>;  // n x 64 x i8  vector value

def nxv1i16  : VTScalableVec<1,  i16, 149>;  // n x  1 x i16 vector value
def nxv2i16  : VTScalableVec<2,  i16, 150>;  // n x  2 x i16 vector value
def nxv4i16  : VTScalableVec<4,  i16, 151>;  // n x  4 x i16 vector value
def nxv8i16  : VTScalableVec<8,  i16, 152>;  // n x  8 x i16 vector value
def nxv16i16 : VTScalableVec<16, i16, 153>;  // n x 16 x i16 vector value
def nxv32i16 : VTScalableVec<32, i16, 154>;  // n x 32 x i16 vector value

def nxv1i32  : VTScalableVec<1,  i32, 155>;  // n x  1 x i32 vector value
def nxv2i32  : VTScalableVec<2,  i32, 156>;  // n x  2 x i32 vector value
def nxv4i32  : VTScalableVec<4,  i32, 157>;  // n x  4 x i32 vector value
def nxv8i32  : VTScalableVec<8,  i32, 158>;  // n x  8 x i32 vector value
def nxv16i32 : VTScalableVec<16, i32, 159>;  // n x 16 x i32 vector value
def nxv32i32 : VTScalableVec<32, i32, 160>;  // n x 32 x i32 vector value

def nxv1i64  : VTScalableVec<1,  i64, 161>;  // n x  1 x i64 vector value
def nxv2i64  : VTScalableVec<2,  i64, 162>;  // n x  2 x i64 vector value
def nxv4i64  : VTScalableVec<4,  i64, 163>;  // n x  4 x i64 vector value
def nxv8i64  : VTScalableVec<8,  i64, 164>;  // n x  8 x i64 vector value
def nxv16i64 : VTScalableVec<16, i64, 165>;  // n x 16 x i64 vector value
def nxv32i64 : VTScalableVec<32, i64, 166>;  // n x 32 x i64 vector value

def nxv1f16  : VTScalableVec<1,  f16, 167>;  // n x  1 x  f16 vector value
def nxv2f16  : VTScalableVec<2,  f16, 168>;  // n x  2 x  f16 vector value
def nxv4f16  : VTScalableVec<4,  f16, 169>;  // n x  4 x  f16 vector value
def nxv8f16  : VTScalableVec<8,  f16, 170>;  // n x  8 x  f16 vector value
def nxv16f16 : VTScalableVec<16, f16, 171>;  // n x 16 x  f16 vector value
def nxv32f16 : VTScalableVec<32, f16, 172>;  // n x 32 x  f16 vector value

def nxv1bf16  : VTScalableVec<1,  bf16, 173>;  // n x  1 x bf16 vector value
def nxv2bf16  : VTScalableVec<2,  bf16, 174>;  // n x  2 x bf16 vector value
def nxv4bf16  : VTScalableVec<4,  bf16, 175>;  // n x  4 x bf16 vector value
def nxv8bf16  : VTScalableVec<8,  bf16, 176>;  // n x  8 x bf16 vector value
def nxv16bf16 : VTScalableVec<16, bf16, 177>;  // n x 16 x bf16 vector value
def nxv32bf16 : VTScalableVec<32, bf16, 178>;  // n x 32 x bf16 vector value

def nxv1f32  : VTScalableVec<1,  f32, 179>;  // n x  1 x  f32 vector value
def nxv2f32  : VTScalableVec<2,  f32, 180>;  // n x  2 x  f32 vector value
def nxv4f32  : VTScalableVec<4,  f32, 181>;  // n x  4 x  f32 vector value
def nxv8f32  : VTScalableVec<8,  f32, 182>;  // n x  8 x  f32 vector value
def nxv16f32 : VTScalableVec<16, f32, 183>;  // n x 16 x  f32 vector value

def nxv1f64  : VTScalableVec<1,  f64, 184>;  // n x  1 x  f64 vector value
def nxv2f64  : VTScalableVec<2,  f64, 185>;  // n x  2 x  f64 vector value
def nxv4f64  : VTScalableVec<4,  f64, 186>;  // n x  4 x  f64 vector value
def nxv8f64  : VTScalableVec<8,  f64, 187>;  // n x  8 x  f64 vector value

def x86mmx    : ValueType<64,   188>;  // X86 MMX value
def FlagVT    : ValueType<0,    189> { // Pre-RA sched glue
  let LLVMName = "Glue";
}
def isVoid    : ValueType<0,    190>;  // Produces no value
def untyped   : ValueType<8,    191> { // Produces an untyped value
  let LLVMName = "Untyped";
}
def funcref   : ValueType<0,    192>;  // WebAssembly's funcref type
def externref : ValueType<0,    193>;  // WebAssembly's externref type
def x86amx    : ValueType<8192, 194>;  // X86 AMX value
def i64x8     : ValueType<512,  195>;  // 8 Consecutive GPRs (AArch64)
def aarch64svcount
              : ValueType<16,   196>;  // AArch64 predicate-as-counter
def spirvbuiltin : ValueType<0,  197>; // SPIR-V's builtin type

def token      : ValueType<0, 248>;  // TokenTy
def MetadataVT : ValueType<0, 249> { // Metadata
  let LLVMName = "Metadata";
}

// Pseudo valuetype mapped to the current pointer size to any address space.
// Should only be used in TableGen.
def iPTRAny    : VTAny<250>;

// Pseudo valuetype to represent "vector of any size"
def vAny       : VTAny<251>;

// Pseudo valuetype to represent "float of any format"
def fAny       : VTAny<252>;

// Pseudo valuetype to represent "integer of any bit width"
def iAny       : VTAny<253>;

// Pseudo valuetype mapped to the current pointer size.
def iPTR       : ValueType<0, 254>;

// Pseudo valuetype to represent "any type of any size".
def Any        : VTAny<255>;

} // end defset ValueTypes

/// This class is for targets that want to use pointer types in patterns
/// with the GlobalISelEmitter.  Targets must define their own pointer
/// derived from this class.  The scalar argument should be an
/// integer type with the same bit size as the pointer.
/// e.g. def p0 : PtrValueType <i64, 0>;

class PtrValueType <ValueType scalar, int addrspace> :
    ValueType<scalar.Size, scalar.Value> {
  int AddrSpace = addrspace;
}
