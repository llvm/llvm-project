//===-- ConnexConfig.h ---------------------*---------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
///
//===----------------------------------------------------------------------===//

#ifndef CONNEX_CONFIG_H
#define CONNEX_CONFIG_H

// This file is used by ConnexISelDAGToDAG.cpp, ConnexISelLowering.h,
//    ReplaceLoopsWithOpincaaKernels.cpp.

// The macros in this header file are strategic, in the sense that the back end
//  could target a Connex vector processor of different vector length.
// There are also some other important macros like: CONNEX_MEM_NUM_ROWS_EXTRA
//  (used to keep spilled registers, or tables for f16 operations like sqrt
//  or div, etc), STR_OPINCAA, etc.

// These 2 types are defined also in OPINCAA lib, in include/Architecture.h
typedef short TypeElement;
typedef unsigned short UnsignedTypeElement;

// The vector length of the Connex-S back end, which could be different
//   from the actual vector length of the Connex-S processor.
#define CONNEX_VECTOR_LENGTH 8

// TYPE is the type of an element of a Connex-S vector register
#define TYPE_SIZEOF 2
#define CONNEX_LINE_SIZE (CONNEX_VECTOR_LENGTH * TYPE_SIZEOF)

//#define STR_LOOP_SYMBOLIC_INDEX "indexLLVM_LV / CONNEX_VECTOR_LENGTH"
// NOTE: make sure it is equiavlent to the above commented macro
// NOTE: keep the paranthesis since >> has low operator priority
#define STR_LOOP_SYMBOLIC_INDEX "(indexLLVM_LV >> 7)"

// This is the type of the scalar processor (normally the BPF processor) operand
// TODO_CHANGE_BACKEND:
#define TYPE_SCALAR_ELEMENT MVT::i64
// #define TYPE_ELEMENT MVT::i32

// #define TYPE_VECTOR MVT::v8i64
// #define TYPE_VECTOR MVT::v16i32
// #define TYPE_VECTOR MVT::v32i16
// #define TYPE_VECTOR_I16 MVT::v128i16
#define TYPE_VECTOR_I16 MVT::v8i16
#define TYPE_VECTOR_I16_EXT_I64 MVT::v8i64
// #define TYPE_VECTOR_ELEMENT MVT::i64
#define TYPE_VECTOR_I16_ELEMENT MVT::i16

// #define TYPE_VECTOR_I32 MVT::v64i32
#define TYPE_VECTOR_I32 MVT::v4i32
#define TYPE_VECTOR_I32_EXT_I64 MVT::v4i64
#define TYPE_VECTOR_I32_ELEMENT MVT::i32

#define TYPE_VECTOR_I64 MVT::v2i64
// #define TYPE_VECTOR_I64_EXT_I64 MVT::v2i64
#define TYPE_VECTOR_I64_ELEMENT MVT::i64

// #define TYPE_VECTOR_F16 MVT::v128f16
#define TYPE_VECTOR_F16 MVT::v8f16
#define TYPE_VECTOR_F16_ELEMENT MVT::f16

#define TYPE_VECTOR_I16_ELEMENT_BITSIZE 16
#define TYPE_VECTOR_I32_ELEMENT_BITSIZE 32
#define TYPE_VECTOR_F16_ELEMENT_BITSIZE 16

// This constant is used as an offset to inform from LoopVectorize pass to
//   the ConnexInstPrinter that the respective address of the LD_H or ST_H
//   Connex-S instruction is actually symbolic (and the symbolic value
//   can be found in the associated InlineAsm expression for it).
#define CONNEX_MEM_CONSTANT_OFFSET 1000

#define CONNEX_MEM_NUM_ROWS 1024
// For 64 lanes: #define CONNEX_MEM_NUM_ROWS 2048
// Extra LS memory for spills and LUTs for div/sqrt.f16, etc
#define CONNEX_MEM_NUM_ROWS_EXTRA 200
#define CONNEX_MEM_NUM_ROWS_EXTRA_FOR_SPILL 50

// NOTE: normally REPEAT accepts immediates in interval 0..1023
#define VALUE_BOGUS_REPEAT_X_TIMES 32761

// #ifndef MAXLEN_STR
#define MAXLEN_STR 8192
// #endif

// Used in ConnexAsmPrinter.cpp and LoopVectorize.cpp
#define STR_OPINCAA_CODE_BEGIN "// START_OPINCAA_HOST_DEVICE_CODE"
#define STR_OPINCAA_CODE_END "// END_OPINCAA_HOST_DEVICE_CODE"
//
#define STR_OPINCAA_KERNEL_REDUCE_BEFORE_END                                   \
  "REDUCE R(0); // We add a 'bogus' REDUCE to wait for it"

#define FILENAME_LOOPNESTS_LOCATIONS "loopsLoc.txt"

#endif
