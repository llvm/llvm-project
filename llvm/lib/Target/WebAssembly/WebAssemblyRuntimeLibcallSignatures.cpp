// CodeGen/RuntimeLibcallSignatures.cpp - R.T. Lib. Call Signatures -*- C++ -*--
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains signature information for runtime libcalls.
///
/// CodeGen uses external symbols, which it refers to by name. The WebAssembly
/// target needs type information for all functions. This file contains a big
/// table providing type signatures for all runtime library functions that LLVM
/// uses.
///
/// This is currently a fairly heavy-handed solution.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyRuntimeLibcallSignatures.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/RuntimeLibcalls.h"

using namespace llvm;

namespace {

enum RuntimeLibcallSignature {
  func,
  f32_func_f32,
  f32_func_f64,
  f32_func_i32,
  f32_func_i64,
  f32_func_i16,
  f64_func_f32,
  f64_func_f64,
  f64_func_i32,
  f64_func_i64,
  i32_func_f32,
  i32_func_f64,
  i32_func_i32,
  i64_func_f32,
  i64_func_f64,
  i64_func_i64,
  f32_func_f32_f32,
  f32_func_f32_i32,
  f32_func_i64_i64,
  f64_func_f64_f64,
  f64_func_f64_i32,
  f64_func_i64_i64,
  i16_func_f32,
  i8_func_i8_i8,
  func_f32_iPTR_iPTR,
  func_f64_iPTR_iPTR,
  i16_func_i16_i16,
  i32_func_f32_f32,
  i32_func_f64_f64,
  i32_func_i32_i32,
  i64_func_i64_i64,
  i64_i64_func_f32,
  i64_i64_func_f64,
  i16_i16_func_i16_i16,
  i32_i32_func_i32_i32,
  i64_i64_func_i64_i64,
  i64_i64_func_i64_i64_i64_i64,
  i64_i64_i64_i64_func_i64_i64_i64_i64,
  i64_i64_func_i64_i64_i32,
  iPTR_func_iPTR_i32_iPTR,
  iPTR_func_iPTR_iPTR_iPTR,
  f32_func_f32_f32_f32,
  f64_func_f64_f64_f64,
  func_i64_i64_iPTR_iPTR,
  func_iPTR_f32,
  func_iPTR_f64,
  func_iPTR_i32,
  func_iPTR_i64,
  func_iPTR_i64_i64,
  func_iPTR_i64_i64_i64_i64,
  func_iPTR_i64_i64_i64_i64_i64_i64,
  i32_func_i64_i64,
  i32_func_i64_i64_i64_i64,
  unsupported
};

} // end anonymous namespace

static const RuntimeLibcallSignature
RuntimeLibcallSignatures[RTLIB::UNKNOWN_LIBCALL] = {
// Integer
/* SHL_I16 */ i16_func_i16_i16,
/* SHL_I32 */ i32_func_i32_i32,
/* SHL_I64 */ i64_func_i64_i64,
/* SHL_I128 */ i64_i64_func_i64_i64_i32,
/* SRL_I16 */ i16_func_i16_i16,
/* SRL_I32 */ i32_func_i32_i32,
/* SRL_I64 */ i64_func_i64_i64,
/* SRL_I128 */ i64_i64_func_i64_i64_i32,
/* SRA_I16 */ i16_func_i16_i16,
/* SRA_I32 */ i32_func_i32_i32,
/* SRA_I64 */ i64_func_i64_i64,
/* SRA_I128 */ i64_i64_func_i64_i64_i32,
/* MUL_I8 */ i8_func_i8_i8,
/* MUL_I16 */ i16_func_i16_i16,
/* MUL_I32 */ i32_func_i32_i32,
/* MUL_I64 */ i64_func_i64_i64,
/* MUL_I128 */ i64_i64_func_i64_i64_i64_i64,
/* MULO_I32 */ i32_func_i32_i32,
/* MULO_I64 */ i64_func_i64_i64,
/* MULO_I128 */ i64_i64_func_i64_i64_i64_i64,
/* SDIV_I8 */ i8_func_i8_i8,
/* SDIV_I16 */ i16_func_i16_i16,
/* SDIV_I32 */ i32_func_i32_i32,
/* SDIV_I64 */ i64_func_i64_i64,
/* SDIV_I128 */ i64_i64_func_i64_i64_i64_i64,
/* UDIV_I8 */ i8_func_i8_i8,
/* UDIV_I16 */ i16_func_i16_i16,
/* UDIV_I32 */ i32_func_i32_i32,
/* UDIV_I64 */ i64_func_i64_i64,
/* UDIV_I128 */ i64_i64_func_i64_i64_i64_i64,
/* SREM_I8 */ i8_func_i8_i8,
/* SREM_I16 */ i16_func_i16_i16,
/* SREM_I32 */ i32_func_i32_i32,
/* SREM_I64 */ i64_func_i64_i64,
/* SREM_I128 */ i64_i64_func_i64_i64_i64_i64,
/* UREM_I8 */ i8_func_i8_i8,
/* UREM_I16 */ i16_func_i16_i16,
/* UREM_I32 */ i32_func_i32_i32,
/* UREM_I64 */ i64_func_i64_i64,
/* UREM_I128 */ i64_i64_func_i64_i64_i64_i64,
/* SDIVREM_I8 */ i8_func_i8_i8,
/* SDIVREM_I16 */ i16_i16_func_i16_i16,
/* SDIVREM_I32 */ i32_i32_func_i32_i32,
/* SDIVREM_I64 */ i64_func_i64_i64,
/* SDIVREM_I128 */ i64_i64_i64_i64_func_i64_i64_i64_i64,
/* UDIVREM_I8 */ i8_func_i8_i8,
/* UDIVREM_I16 */ i16_i16_func_i16_i16,
/* UDIVREM_I32 */ i32_i32_func_i32_i32,
/* UDIVREM_I64 */ i64_i64_func_i64_i64,
/* UDIVREM_I128 */ i64_i64_i64_i64_func_i64_i64_i64_i64,
/* NEG_I32 */ i32_func_i32,
/* NEG_I64 */ i64_func_i64,

// FLOATING POINT
/* ADD_F32 */ f32_func_f32_f32,
/* ADD_F64 */ f64_func_f64_f64,
/* ADD_F80 */ unsupported,
/* ADD_F128 */ func_iPTR_i64_i64_i64_i64,
/* ADD_PPCF128 */ unsupported,
/* SUB_F32 */ f32_func_f32_f32,
/* SUB_F64 */ f64_func_f64_f64,
/* SUB_F80 */ unsupported,
/* SUB_F128 */ func_iPTR_i64_i64_i64_i64,
/* SUB_PPCF128 */ unsupported,
/* MUL_F32 */ f32_func_f32_f32,
/* MUL_F64 */ f64_func_f64_f64,
/* MUL_F80 */ unsupported,
/* MUL_F128 */ func_iPTR_i64_i64_i64_i64,
/* MUL_PPCF128 */ unsupported,
/* DIV_F32 */ f32_func_f32_f32,
/* DIV_F64 */ f64_func_f64_f64,
/* DIV_F80 */ unsupported,
/* DIV_F128 */ func_iPTR_i64_i64_i64_i64,
/* DIV_PPCF128 */ unsupported,
/* REM_F32 */ f32_func_f32_f32,
/* REM_F64 */ f64_func_f64_f64,
/* REM_F80 */ unsupported,
/* REM_F128 */ func_iPTR_i64_i64_i64_i64,
/* REM_PPCF128 */ unsupported,
/* FMA_F32 */ f32_func_f32_f32_f32,
/* FMA_F64 */ f64_func_f64_f64_f64,
/* FMA_F80 */ unsupported,
/* FMA_F128 */ func_iPTR_i64_i64_i64_i64_i64_i64,
/* FMA_PPCF128 */ unsupported,
/* POWI_F32 */ f32_func_f32_i32,
/* POWI_F64 */ f64_func_f64_i32,
/* POWI_F80 */ unsupported,
/* POWI_F128 */ func_iPTR_i64_i64_i64_i64,
/* POWI_PPCF128 */ unsupported,
/* SQRT_F32 */ f32_func_f32,
/* SQRT_F64 */ f64_func_f64,
/* SQRT_F80 */ unsupported,
/* SQRT_F128 */ func_iPTR_i64_i64,
/* SQRT_PPCF128 */ unsupported,
/* LOG_F32 */ f32_func_f32,
/* LOG_F64 */ f64_func_f64,
/* LOG_F80 */ unsupported,
/* LOG_F128 */ func_iPTR_i64_i64,
/* LOG_PPCF128 */ unsupported,
/* LOG_FINITE_F32 */ unsupported,
/* LOG_FINITE_F64 */ unsupported,
/* LOG_FINITE_F80 */ unsupported,
/* LOG_FINITE_F128 */ unsupported,
/* LOG_FINITE_PPCF128 */ unsupported,
/* LOG2_F32 */ f32_func_f32,
/* LOG2_F64 */ f64_func_f64,
/* LOG2_F80 */ unsupported,
/* LOG2_F128 */ func_iPTR_i64_i64,
/* LOG2_PPCF128 */ unsupported,
/* LOG2_FINITE_F32 */ unsupported,
/* LOG2_FINITE_F64 */ unsupported,
/* LOG2_FINITE_F80 */ unsupported,
/* LOG2_FINITE_F128 */ unsupported,
/* LOG2_FINITE_PPCF128 */ unsupported,
/* LOG10_F32 */ f32_func_f32,
/* LOG10_F64 */ f64_func_f64,
/* LOG10_F80 */ unsupported,
/* LOG10_F128 */ func_iPTR_i64_i64,
/* LOG10_PPCF128 */ unsupported,
/* LOG10_FINITE_F32 */ unsupported,
/* LOG10_FINITE_F64 */ unsupported,
/* LOG10_FINITE_F80 */ unsupported,
/* LOG10_FINITE_F128 */ unsupported,
/* LOG10_FINITE_PPCF128 */ unsupported,
/* EXP_F32 */ f32_func_f32,
/* EXP_F64 */ f64_func_f64,
/* EXP_F80 */ unsupported,
/* EXP_F128 */ func_iPTR_i64_i64,
/* EXP_PPCF128 */ unsupported,
/* EXP_FINITE_F32 */ unsupported,
/* EXP_FINITE_F64 */ unsupported,
/* EXP_FINITE_F80 */ unsupported,
/* EXP_FINITE_F128 */ unsupported,
/* EXP_FINITE_PPCF128 */ unsupported,
/* EXP2_F32 */ f32_func_f32,
/* EXP2_F64 */ f64_func_f64,
/* EXP2_F80 */ unsupported,
/* EXP2_F128 */ func_iPTR_i64_i64,
/* EXP2_PPCF128 */ unsupported,
/* EXP2_FINITE_F32 */ unsupported,
/* EXP2_FINITE_F64 */ unsupported,
/* EXP2_FINITE_F80 */ unsupported,
/* EXP2_FINITE_F128 */ unsupported,
/* EXP2_FINITE_PPCF128 */ unsupported,
/* SIN_F32 */ f32_func_f32,
/* SIN_F64 */ f64_func_f64,
/* SIN_F80 */ unsupported,
/* SIN_F128 */ func_iPTR_i64_i64,
/* SIN_PPCF128 */ unsupported,
/* COS_F32 */ f32_func_f32,
/* COS_F64 */ f64_func_f64,
/* COS_F80 */ unsupported,
/* COS_F128 */ func_iPTR_i64_i64,
/* COS_PPCF128 */ unsupported,
/* SINCOS_F32 */ func_f32_iPTR_iPTR,
/* SINCOS_F64 */ func_f64_iPTR_iPTR,
/* SINCOS_F80 */ unsupported,
/* SINCOS_F128 */ func_i64_i64_iPTR_iPTR,
/* SINCOS_PPCF128 */ unsupported,
/* SINCOS_STRET_F32 */ unsupported,
/* SINCOS_STRET_F64 */ unsupported,
/* POW_F32 */ f32_func_f32_f32,
/* POW_F64 */ f64_func_f64_f64,
/* POW_F80 */ unsupported,
/* POW_F128 */ func_iPTR_i64_i64_i64_i64,
/* POW_PPCF128 */ unsupported,
/* POW_FINITE_F32 */ unsupported,
/* POW_FINITE_F64 */ unsupported,
/* POW_FINITE_F80 */ unsupported,
/* POW_FINITE_F128 */ unsupported,
/* POW_FINITE_PPCF128 */ unsupported,
/* CEIL_F32 */ f32_func_f32,
/* CEIL_F64 */ f64_func_f64,
/* CEIL_F80 */ unsupported,
/* CEIL_F128 */ func_iPTR_i64_i64,
/* CEIL_PPCF128 */ unsupported,
/* TRUNC_F32 */ f32_func_f32,
/* TRUNC_F64 */ f64_func_f64,
/* TRUNC_F80 */ unsupported,
/* TRUNC_F128 */ func_iPTR_i64_i64,
/* TRUNC_PPCF128 */ unsupported,
/* RINT_F32 */ f32_func_f32,
/* RINT_F64 */ f64_func_f64,
/* RINT_F80 */ unsupported,
/* RINT_F128 */ func_iPTR_i64_i64,
/* RINT_PPCF128 */ unsupported,
/* NEARBYINT_F32 */ f32_func_f32,
/* NEARBYINT_F64 */ f64_func_f64,
/* NEARBYINT_F80 */ unsupported,
/* NEARBYINT_F128 */ func_iPTR_i64_i64,
/* NEARBYINT_PPCF128 */ unsupported,
/* ROUND_F32 */ f32_func_f32,
/* ROUND_F64 */ f64_func_f64,
/* ROUND_F80 */ unsupported,
/* ROUND_F128 */ func_iPTR_i64_i64,
/* ROUND_PPCF128 */ unsupported,
/* FLOOR_F32 */ f32_func_f32,
/* FLOOR_F64 */ f64_func_f64,
/* FLOOR_F80 */ unsupported,
/* FLOOR_F128 */ func_iPTR_i64_i64,
/* FLOOR_PPCF128 */ unsupported,
/* COPYSIGN_F32 */ f32_func_f32_f32,
/* COPYSIGN_F64 */ f64_func_f64_f64,
/* COPYSIGN_F80 */ unsupported,
/* COPYSIGN_F128 */ func_iPTR_i64_i64_i64_i64,
/* COPYSIGN_PPCF128 */ unsupported,
/* FMIN_F32 */ f32_func_f32_f32,
/* FMIN_F64 */ f64_func_f64_f64,
/* FMIN_F80 */ unsupported,
/* FMIN_F128 */ func_iPTR_i64_i64_i64_i64,
/* FMIN_PPCF128 */ unsupported,
/* FMAX_F32 */ f32_func_f32_f32,
/* FMAX_F64 */ f64_func_f64_f64,
/* FMAX_F80 */ unsupported,
/* FMAX_F128 */ func_iPTR_i64_i64_i64_i64,
/* FMAX_PPCF128 */ unsupported,

// CONVERSION
/* FPEXT_F32_PPCF128 */ unsupported,
/* FPEXT_F64_PPCF128 */ unsupported,
/* FPEXT_F64_F128 */ func_iPTR_f64,
/* FPEXT_F32_F128 */ func_iPTR_f32,
/* FPEXT_F32_F64 */ f64_func_f32,
/* FPEXT_F16_F32 */ f32_func_i16,
/* FPROUND_F32_F16 */ i16_func_f32,
/* FPROUND_F64_F16 */ unsupported,
/* FPROUND_F80_F16 */ unsupported,
/* FPROUND_F128_F16 */ unsupported,
/* FPROUND_PPCF128_F16 */ unsupported,
/* FPROUND_F64_F32 */ f32_func_f64,
/* FPROUND_F80_F32 */ unsupported,
/* FPROUND_F128_F32 */ f32_func_i64_i64,
/* FPROUND_PPCF128_F32 */ unsupported,
/* FPROUND_F80_F64 */ unsupported,
/* FPROUND_F128_F64 */ f64_func_i64_i64,
/* FPROUND_PPCF128_F64 */ unsupported,
/* FPTOSINT_F32_I32 */ i32_func_f32,
/* FPTOSINT_F32_I64 */ i64_func_f32,
/* FPTOSINT_F32_I128 */ i64_i64_func_f32,
/* FPTOSINT_F64_I32 */ i32_func_f64,
/* FPTOSINT_F64_I64 */ i64_func_f64,
/* FPTOSINT_F64_I128 */ i64_i64_func_f64,
/* FPTOSINT_F80_I32 */ unsupported,
/* FPTOSINT_F80_I64 */ unsupported,
/* FPTOSINT_F80_I128 */ unsupported,
/* FPTOSINT_F128_I32 */ i32_func_i64_i64,
/* FPTOSINT_F128_I64 */ i64_func_i64_i64,
/* FPTOSINT_F128_I128 */ i64_i64_func_i64_i64,
/* FPTOSINT_PPCF128_I32 */ unsupported,
/* FPTOSINT_PPCF128_I64 */ unsupported,
/* FPTOSINT_PPCF128_I128 */ unsupported,
/* FPTOUINT_F32_I32 */ i32_func_f32,
/* FPTOUINT_F32_I64 */ i64_func_f32,
/* FPTOUINT_F32_I128 */ i64_i64_func_f32,
/* FPTOUINT_F64_I32 */ i32_func_f64,
/* FPTOUINT_F64_I64 */ i64_func_f64,
/* FPTOUINT_F64_I128 */ i64_i64_func_f64,
/* FPTOUINT_F80_I32 */ unsupported,
/* FPTOUINT_F80_I64 */ unsupported,
/* FPTOUINT_F80_I128 */ unsupported,
/* FPTOUINT_F128_I32 */ i32_func_i64_i64,
/* FPTOUINT_F128_I64 */ i64_func_i64_i64,
/* FPTOUINT_F128_I128 */ i64_i64_func_i64_i64,
/* FPTOUINT_PPCF128_I32 */ unsupported,
/* FPTOUINT_PPCF128_I64 */ unsupported,
/* FPTOUINT_PPCF128_I128 */ unsupported,
/* SINTTOFP_I32_F32 */ f32_func_i32,
/* SINTTOFP_I32_F64 */ f64_func_i32,
/* SINTTOFP_I32_F80 */ unsupported,
/* SINTTOFP_I32_F128 */ func_iPTR_i32,
/* SINTTOFP_I32_PPCF128 */ unsupported,
/* SINTTOFP_I64_F32 */ f32_func_i64,
/* SINTTOFP_I64_F64 */ f64_func_i64,
/* SINTTOFP_I64_F80 */ unsupported,
/* SINTTOFP_I64_F128 */ func_iPTR_i64,
/* SINTTOFP_I64_PPCF128 */ unsupported,
/* SINTTOFP_I128_F32 */ f32_func_i64_i64,
/* SINTTOFP_I128_F64 */ f64_func_i64_i64,
/* SINTTOFP_I128_F80 */ unsupported,
/* SINTTOFP_I128_F128 */ func_iPTR_i64_i64,
/* SINTTOFP_I128_PPCF128 */ unsupported,
/* UINTTOFP_I32_F32 */ f32_func_i32,
/* UINTTOFP_I32_F64 */ f64_func_i64,
/* UINTTOFP_I32_F80 */ unsupported,
/* UINTTOFP_I32_F128 */ func_iPTR_i32,
/* UINTTOFP_I32_PPCF128 */ unsupported,
/* UINTTOFP_I64_F32 */ f32_func_i64,
/* UINTTOFP_I64_F64 */ f64_func_i64,
/* UINTTOFP_I64_F80 */ unsupported,
/* UINTTOFP_I64_F128 */ func_iPTR_i64,
/* UINTTOFP_I64_PPCF128 */ unsupported,
/* UINTTOFP_I128_F32 */ f32_func_i64_i64,
/* UINTTOFP_I128_F64 */ f64_func_i64_i64,
/* UINTTOFP_I128_F80 */ unsupported,
/* UINTTOFP_I128_F128 */ func_iPTR_i64_i64,
/* UINTTOFP_I128_PPCF128 */ unsupported,

// COMPARISON
/* OEQ_F32 */ i32_func_f32_f32,
/* OEQ_F64 */ i32_func_f64_f64,
/* OEQ_F128 */ i32_func_i64_i64_i64_i64,
/* OEQ_PPCF128 */ unsupported,
/* UNE_F32 */ i32_func_f32_f32,
/* UNE_F64 */ i32_func_f64_f64,
/* UNE_F128 */ i32_func_i64_i64_i64_i64,
/* UNE_PPCF128 */ unsupported,
/* OGE_F32 */ i32_func_f32_f32,
/* OGE_F64 */ i32_func_f64_f64,
/* OGE_F128 */ i32_func_i64_i64_i64_i64,
/* OGE_PPCF128 */ unsupported,
/* OLT_F32 */ i32_func_f32_f32,
/* OLT_F64 */ i32_func_f64_f64,
/* OLT_F128 */ i32_func_i64_i64_i64_i64,
/* OLT_PPCF128 */ unsupported,
/* OLE_F32 */ i32_func_f32_f32,
/* OLE_F64 */ i32_func_f64_f64,
/* OLE_F128 */ i32_func_i64_i64_i64_i64,
/* OLE_PPCF128 */ unsupported,
/* OGT_F32 */ i32_func_f32_f32,
/* OGT_F64 */ i32_func_f64_f64,
/* OGT_F128 */ i32_func_i64_i64_i64_i64,
/* OGT_PPCF128 */ unsupported,
/* UO_F32 */ i32_func_f32_f32,
/* UO_F64 */ i32_func_f64_f64,
/* UO_F128 */ i32_func_i64_i64_i64_i64,
/* UO_PPCF128 */ unsupported,
/* O_F32 */ i32_func_f32_f32,
/* O_F64 */ i32_func_f64_f64,
/* O_F128 */ i32_func_i64_i64_i64_i64,
/* O_PPCF128 */ unsupported,

// MEMORY
/* MEMCPY */ iPTR_func_iPTR_iPTR_iPTR,
/* MEMMOVE */ iPTR_func_iPTR_iPTR_iPTR,
/* MEMSET */ iPTR_func_iPTR_i32_iPTR,
/* BZERO */ unsupported,

// ELEMENT-WISE ATOMIC MEMORY
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_1 */ unsupported,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_2 */ unsupported,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_4 */ unsupported,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_8 */ unsupported,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_16 */ unsupported,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_1 */ unsupported,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_2 */ unsupported,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_4 */ unsupported,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_8 */ unsupported,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_16 */ unsupported,

/* MEMSET_ELEMENT_UNORDERED_ATOMIC_1 */ unsupported,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_2 */ unsupported,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_4 */ unsupported,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_8 */ unsupported,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_16 */ unsupported,

// EXCEPTION HANDLING
/* UNWIND_RESUME */ unsupported,

// Note: there's two sets of atomics libcalls; see
// <http://llvm.org/docs/Atomics.html> for more info on the
// difference between them.

// Atomic '__sync_*' libcalls.
/* SYNC_VAL_COMPARE_AND_SWAP_1 */ unsupported,
/* SYNC_VAL_COMPARE_AND_SWAP_2 */ unsupported,
/* SYNC_VAL_COMPARE_AND_SWAP_4 */ unsupported,
/* SYNC_VAL_COMPARE_AND_SWAP_8 */ unsupported,
/* SYNC_VAL_COMPARE_AND_SWAP_16 */ unsupported,
/* SYNC_LOCK_TEST_AND_SET_1 */ unsupported,
/* SYNC_LOCK_TEST_AND_SET_2 */ unsupported,
/* SYNC_LOCK_TEST_AND_SET_4 */ unsupported,
/* SYNC_LOCK_TEST_AND_SET_8 */ unsupported,
/* SYNC_LOCK_TEST_AND_SET_16 */ unsupported,
/* SYNC_FETCH_AND_ADD_1 */ unsupported,
/* SYNC_FETCH_AND_ADD_2 */ unsupported,
/* SYNC_FETCH_AND_ADD_4 */ unsupported,
/* SYNC_FETCH_AND_ADD_8 */ unsupported,
/* SYNC_FETCH_AND_ADD_16 */ unsupported,
/* SYNC_FETCH_AND_SUB_1 */ unsupported,
/* SYNC_FETCH_AND_SUB_2 */ unsupported,
/* SYNC_FETCH_AND_SUB_4 */ unsupported,
/* SYNC_FETCH_AND_SUB_8 */ unsupported,
/* SYNC_FETCH_AND_SUB_16 */ unsupported,
/* SYNC_FETCH_AND_AND_1 */ unsupported,
/* SYNC_FETCH_AND_AND_2 */ unsupported,
/* SYNC_FETCH_AND_AND_4 */ unsupported,
/* SYNC_FETCH_AND_AND_8 */ unsupported,
/* SYNC_FETCH_AND_AND_16 */ unsupported,
/* SYNC_FETCH_AND_OR_1 */ unsupported,
/* SYNC_FETCH_AND_OR_2 */ unsupported,
/* SYNC_FETCH_AND_OR_4 */ unsupported,
/* SYNC_FETCH_AND_OR_8 */ unsupported,
/* SYNC_FETCH_AND_OR_16 */ unsupported,
/* SYNC_FETCH_AND_XOR_1 */ unsupported,
/* SYNC_FETCH_AND_XOR_2 */ unsupported,
/* SYNC_FETCH_AND_XOR_4 */ unsupported,
/* SYNC_FETCH_AND_XOR_8 */ unsupported,
/* SYNC_FETCH_AND_XOR_16 */ unsupported,
/* SYNC_FETCH_AND_NAND_1 */ unsupported,
/* SYNC_FETCH_AND_NAND_2 */ unsupported,
/* SYNC_FETCH_AND_NAND_4 */ unsupported,
/* SYNC_FETCH_AND_NAND_8 */ unsupported,
/* SYNC_FETCH_AND_NAND_16 */ unsupported,
/* SYNC_FETCH_AND_MAX_1 */ unsupported,
/* SYNC_FETCH_AND_MAX_2 */ unsupported,
/* SYNC_FETCH_AND_MAX_4 */ unsupported,
/* SYNC_FETCH_AND_MAX_8 */ unsupported,
/* SYNC_FETCH_AND_MAX_16 */ unsupported,
/* SYNC_FETCH_AND_UMAX_1 */ unsupported,
/* SYNC_FETCH_AND_UMAX_2 */ unsupported,
/* SYNC_FETCH_AND_UMAX_4 */ unsupported,
/* SYNC_FETCH_AND_UMAX_8 */ unsupported,
/* SYNC_FETCH_AND_UMAX_16 */ unsupported,
/* SYNC_FETCH_AND_MIN_1 */ unsupported,
/* SYNC_FETCH_AND_MIN_2 */ unsupported,
/* SYNC_FETCH_AND_MIN_4 */ unsupported,
/* SYNC_FETCH_AND_MIN_8 */ unsupported,
/* SYNC_FETCH_AND_MIN_16 */ unsupported,
/* SYNC_FETCH_AND_UMIN_1 */ unsupported,
/* SYNC_FETCH_AND_UMIN_2 */ unsupported,
/* SYNC_FETCH_AND_UMIN_4 */ unsupported,
/* SYNC_FETCH_AND_UMIN_8 */ unsupported,
/* SYNC_FETCH_AND_UMIN_16 */ unsupported,

// Atomic '__atomic_*' libcalls.
/* ATOMIC_LOAD */ unsupported,
/* ATOMIC_LOAD_1 */ unsupported,
/* ATOMIC_LOAD_2 */ unsupported,
/* ATOMIC_LOAD_4 */ unsupported,
/* ATOMIC_LOAD_8 */ unsupported,
/* ATOMIC_LOAD_16 */ unsupported,

/* ATOMIC_STORE */ unsupported,
/* ATOMIC_STORE_1 */ unsupported,
/* ATOMIC_STORE_2 */ unsupported,
/* ATOMIC_STORE_4 */ unsupported,
/* ATOMIC_STORE_8 */ unsupported,
/* ATOMIC_STORE_16 */ unsupported,

/* ATOMIC_EXCHANGE */ unsupported,
/* ATOMIC_EXCHANGE_1 */ unsupported,
/* ATOMIC_EXCHANGE_2 */ unsupported,
/* ATOMIC_EXCHANGE_4 */ unsupported,
/* ATOMIC_EXCHANGE_8 */ unsupported,
/* ATOMIC_EXCHANGE_16 */ unsupported,

/* ATOMIC_COMPARE_EXCHANGE */ unsupported,
/* ATOMIC_COMPARE_EXCHANGE_1 */ unsupported,
/* ATOMIC_COMPARE_EXCHANGE_2 */ unsupported,
/* ATOMIC_COMPARE_EXCHANGE_4 */ unsupported,
/* ATOMIC_COMPARE_EXCHANGE_8 */ unsupported,
/* ATOMIC_COMPARE_EXCHANGE_16 */ unsupported,

/* ATOMIC_FETCH_ADD_1 */ unsupported,
/* ATOMIC_FETCH_ADD_2 */ unsupported,
/* ATOMIC_FETCH_ADD_4 */ unsupported,
/* ATOMIC_FETCH_ADD_8 */ unsupported,
/* ATOMIC_FETCH_ADD_16 */ unsupported,

/* ATOMIC_FETCH_SUB_1 */ unsupported,
/* ATOMIC_FETCH_SUB_2 */ unsupported,
/* ATOMIC_FETCH_SUB_4 */ unsupported,
/* ATOMIC_FETCH_SUB_8 */ unsupported,
/* ATOMIC_FETCH_SUB_16 */ unsupported,

/* ATOMIC_FETCH_AND_1 */ unsupported,
/* ATOMIC_FETCH_AND_2 */ unsupported,
/* ATOMIC_FETCH_AND_4 */ unsupported,
/* ATOMIC_FETCH_AND_8 */ unsupported,
/* ATOMIC_FETCH_AND_16 */ unsupported,

/* ATOMIC_FETCH_OR_1 */ unsupported,
/* ATOMIC_FETCH_OR_2 */ unsupported,
/* ATOMIC_FETCH_OR_4 */ unsupported,
/* ATOMIC_FETCH_OR_8 */ unsupported,
/* ATOMIC_FETCH_OR_16 */ unsupported,

/* ATOMIC_FETCH_XOR_1 */ unsupported,
/* ATOMIC_FETCH_XOR_2 */ unsupported,
/* ATOMIC_FETCH_XOR_4 */ unsupported,
/* ATOMIC_FETCH_XOR_8 */ unsupported,
/* ATOMIC_FETCH_XOR_16 */ unsupported,

/* ATOMIC_FETCH_NAND_1 */ unsupported,
/* ATOMIC_FETCH_NAND_2 */ unsupported,
/* ATOMIC_FETCH_NAND_4 */ unsupported,
/* ATOMIC_FETCH_NAND_8 */ unsupported,
/* ATOMIC_FETCH_NAND_16 */ unsupported,

// Stack Protector Fail.
/* STACKPROTECTOR_CHECK_FAIL */ func,

// Deoptimization.
/* DEOPTIMIZE */ unsupported,

};

static const char *
RuntimeLibcallNames[RTLIB::UNKNOWN_LIBCALL] = {
/* SHL_I16 */ "__ashlhi3",
/* SHL_I32 */ "__ashlsi3",
/* SHL_I64 */ "__ashldi3",
/* SHL_I128 */ "__ashlti3",
/* SRL_I16 */ "__lshrhi3",
/* SRL_I32 */ "__lshrsi3",
/* SRL_I64 */ "__lshrdi3",
/* SRL_I128 */ "__lshrti3",
/* SRA_I16 */ "__ashrhi3",
/* SRA_I32 */ "__ashrsi3",
/* SRA_I64 */ "__ashrdi3",
/* SRA_I128 */ "__ashrti3",
/* MUL_I8 */ "__mulqi3",
/* MUL_I16 */ "__mulhi3",
/* MUL_I32 */ "__mulsi3",
/* MUL_I64 */ "__muldi3",
/* MUL_I128 */ "__multi3",
/* MULO_I32 */ "__mulosi4",
/* MULO_I64 */ "__mulodi4",
/* MULO_I128 */ "__muloti4",
/* SDIV_I8 */ "__divqi3",
/* SDIV_I16 */ "__divhi3",
/* SDIV_I32 */ "__divsi3",
/* SDIV_I64 */ "__divdi3",
/* SDIV_I128 */ "__divti3",
/* UDIV_I8 */ "__udivqi3",
/* UDIV_I16 */ "__udivhi3",
/* UDIV_I32 */ "__udivsi3",
/* UDIV_I64 */ "__udivdi3",
/* UDIV_I128 */ "__udivti3",
/* SREM_I8 */ "__modqi3",
/* SREM_I16 */ "__modhi3",
/* SREM_I32 */ "__modsi3",
/* SREM_I64 */ "__moddi3",
/* SREM_I128 */ "__modti3",
/* UREM_I8 */ "__umodqi3",
/* UREM_I16 */ "__umodhi3",
/* UREM_I32 */ "__umodsi3",
/* UREM_I64 */ "__umoddi3",
/* UREM_I128 */ "__umodti3",
/* SDIVREM_I8 */ nullptr,
/* SDIVREM_I16 */ nullptr,
/* SDIVREM_I32 */ nullptr,
/* SDIVREM_I64 */ nullptr,
/* SDIVREM_I128 */ nullptr,
/* UDIVREM_I8 */ nullptr,
/* UDIVREM_I16 */ nullptr,
/* UDIVREM_I32 */ nullptr,
/* UDIVREM_I64 */ nullptr,
/* UDIVREM_I128 */ nullptr,
/* NEG_I32 */ "__negsi2",
/* NEG_I64 */ "__negdi2",
/* ADD_F32 */ "__addsf3",
/* ADD_F64 */ "__adddf3",
/* ADD_F80 */ nullptr,
/* ADD_F128 */ "__addtf3",
/* ADD_PPCF128 */ nullptr,
/* SUB_F32 */ "__subsf3",
/* SUB_F64 */ "__subdf3",
/* SUB_F80 */ nullptr,
/* SUB_F128 */ "__subtf3",
/* SUB_PPCF128 */ nullptr,
/* MUL_F32 */ "__mulsf3",
/* MUL_F64 */ "__muldf3",
/* MUL_F80 */ nullptr,
/* MUL_F128 */ "__multf3",
/* MUL_PPCF128 */ nullptr,
/* DIV_F32 */ "__divsf3",
/* DIV_F64 */ "__divdf3",
/* DIV_F80 */ nullptr,
/* DIV_F128 */ "__divtf3",
/* DIV_PPCF128 */ nullptr,
/* REM_F32 */ "fmodf",
/* REM_F64 */ "fmod",
/* REM_F80 */ nullptr,
/* REM_F128 */ "fmodl",
/* REM_PPCF128 */ nullptr,
/* FMA_F32 */ "fmaf",
/* FMA_F64 */ "fma",
/* FMA_F80 */ nullptr,
/* FMA_F128 */ "fmal",
/* FMA_PPCF128 */ nullptr,
/* POWI_F32 */ "__powisf2",
/* POWI_F64 */ "__powidf2",
/* POWI_F80 */ nullptr,
/* POWI_F128 */ "__powitf2",
/* POWI_PPCF128 */ nullptr,
/* SQRT_F32 */ "sqrtf",
/* SQRT_F64 */ "sqrt",
/* SQRT_F80 */ nullptr,
/* SQRT_F128 */ "sqrtl",
/* SQRT_PPCF128 */ nullptr,
/* LOG_F32 */ "logf",
/* LOG_F64 */ "log",
/* LOG_F80 */ nullptr,
/* LOG_F128 */ "logl",
/* LOG_PPCF128 */ nullptr,
/* LOG_FINITE_F32 */ nullptr,
/* LOG_FINITE_F64 */ nullptr,
/* LOG_FINITE_F80 */ nullptr,
/* LOG_FINITE_F128 */ nullptr,
/* LOG_FINITE_PPCF128 */ nullptr,
/* LOG2_F32 */ "log2f",
/* LOG2_F64 */ "log2",
/* LOG2_F80 */ nullptr,
/* LOG2_F128 */ "log2l",
/* LOG2_PPCF128 */ nullptr,
/* LOG2_FINITE_F32 */ nullptr,
/* LOG2_FINITE_F64 */ nullptr,
/* LOG2_FINITE_F80 */ nullptr,
/* LOG2_FINITE_F128 */ nullptr,
/* LOG2_FINITE_PPCF128 */ nullptr,
/* LOG10_F32 */ "log10f",
/* LOG10_F64 */ "log10",
/* LOG10_F80 */ nullptr,
/* LOG10_F128 */ "log10l",
/* LOG10_PPCF128 */ nullptr,
/* LOG10_FINITE_F32 */ nullptr,
/* LOG10_FINITE_F64 */ nullptr,
/* LOG10_FINITE_F80 */ nullptr,
/* LOG10_FINITE_F128 */ nullptr,
/* LOG10_FINITE_PPCF128 */ nullptr,
/* EXP_F32 */ "expf",
/* EXP_F64 */ "exp",
/* EXP_F80 */ nullptr,
/* EXP_F128 */ "expl",
/* EXP_PPCF128 */ nullptr,
/* EXP_FINITE_F32 */ nullptr,
/* EXP_FINITE_F64 */ nullptr,
/* EXP_FINITE_F80 */ nullptr,
/* EXP_FINITE_F128 */ nullptr,
/* EXP_FINITE_PPCF128 */ nullptr,
/* EXP2_F32 */ "exp2f",
/* EXP2_F64 */ "exp2",
/* EXP2_F80 */ nullptr,
/* EXP2_F128 */ "exp2l",
/* EXP2_PPCF128 */ nullptr,
/* EXP2_FINITE_F32 */ nullptr,
/* EXP2_FINITE_F64 */ nullptr,
/* EXP2_FINITE_F80 */ nullptr,
/* EXP2_FINITE_F128 */ nullptr,
/* EXP2_FINITE_PPCF128 */ nullptr,
/* SIN_F32 */ "sinf",
/* SIN_F64 */ "sin",
/* SIN_F80 */ nullptr,
/* SIN_F128 */ "sinl",
/* SIN_PPCF128 */ nullptr,
/* COS_F32 */ "cosf",
/* COS_F64 */ "cos",
/* COS_F80 */ nullptr,
/* COS_F128 */ "cosl",
/* COS_PPCF128 */ nullptr,
/* SINCOS_F32 */ "sincosf",
/* SINCOS_F64 */ "sincos",
/* SINCOS_F80 */ nullptr,
/* SINCOS_F128 */ "sincosl",
/* SINCOS_PPCF128 */ nullptr,
/* SINCOS_STRET_F32 */ nullptr,
/* SINCOS_STRET_F64 */ nullptr,
/* POW_F32 */ "powf",
/* POW_F64 */ "pow",
/* POW_F80 */ nullptr,
/* POW_F128 */ "powl",
/* POW_PPCF128 */ nullptr,
/* POW_FINITE_F32 */ nullptr,
/* POW_FINITE_F64 */ nullptr,
/* POW_FINITE_F80 */ nullptr,
/* POW_FINITE_F128 */ nullptr,
/* POW_FINITE_PPCF128 */ nullptr,
/* CEIL_F32 */ "ceilf",
/* CEIL_F64 */ "ceil",
/* CEIL_F80 */ nullptr,
/* CEIL_F128 */ "ceill",
/* CEIL_PPCF128 */ nullptr,
/* TRUNC_F32 */ "truncf",
/* TRUNC_F64 */ "trunc",
/* TRUNC_F80 */ nullptr,
/* TRUNC_F128 */ "truncl",
/* TRUNC_PPCF128 */ nullptr,
/* RINT_F32 */ "rintf",
/* RINT_F64 */ "rint",
/* RINT_F80 */ nullptr,
/* RINT_F128 */ "rintl",
/* RINT_PPCF128 */ nullptr,
/* NEARBYINT_F32 */ "nearbyintf",
/* NEARBYINT_F64 */ "nearbyint",
/* NEARBYINT_F80 */ nullptr,
/* NEARBYINT_F128 */ "nearbyintl",
/* NEARBYINT_PPCF128 */ nullptr,
/* ROUND_F32 */ "roundf",
/* ROUND_F64 */ "round",
/* ROUND_F80 */ nullptr,
/* ROUND_F128 */ "roundl",
/* ROUND_PPCF128 */ nullptr,
/* FLOOR_F32 */ "floorf",
/* FLOOR_F64 */ "floor",
/* FLOOR_F80 */ nullptr,
/* FLOOR_F128 */ "floorl",
/* FLOOR_PPCF128 */ nullptr,
/* COPYSIGN_F32 */ "copysignf",
/* COPYSIGN_F64 */ "copysign",
/* COPYSIGN_F80 */ nullptr,
/* COPYSIGN_F128 */ "copysignl",
/* COPYSIGN_PPCF128 */ nullptr,
/* FMIN_F32 */ "fminf",
/* FMIN_F64 */ "fmin",
/* FMIN_F80 */ nullptr,
/* FMIN_F128 */ "fminl",
/* FMIN_PPCF128 */ nullptr,
/* FMAX_F32 */ "fmaxf",
/* FMAX_F64 */ "fmax",
/* FMAX_F80 */ nullptr,
/* FMAX_F128 */ "fmaxl",
/* FMAX_PPCF128 */ nullptr,
/* FPEXT_F32_PPCF128 */ nullptr,
/* FPEXT_F64_PPCF128 */ nullptr,
/* FPEXT_F64_F128 */ "__extenddftf2",
/* FPEXT_F32_F128 */ "__extendsftf2",
/* FPEXT_F32_F64 */ "__extendsfdf2",
/* FPEXT_F16_F32 */ "__gnu_h2f_ieee",
/* FPROUND_F32_F16 */ "__gnu_f2h_ieee",
/* FPROUND_F64_F16 */ nullptr,
/* FPROUND_F80_F16 */ nullptr,
/* FPROUND_F128_F16 */ nullptr,
/* FPROUND_PPCF128_F16 */ nullptr,
/* FPROUND_F64_F32 */ "__truncdfsf2",
/* FPROUND_F80_F32 */ "__truncxfsf2",
/* FPROUND_F128_F32 */ "__trunctfsf2",
/* FPROUND_PPCF128_F32 */ nullptr,
/* FPROUND_F80_F64 */ "__truncxfdf2",
/* FPROUND_F128_F64 */ "__trunctfdf2",
/* FPROUND_PPCF128_F64 */ nullptr,
/* FPTOSINT_F32_I32 */ "__fixsfsi",
/* FPTOSINT_F32_I64 */ "__fixsfdi",
/* FPTOSINT_F32_I128 */ "__fixsfti",
/* FPTOSINT_F64_I32 */ "__fixdfsi",
/* FPTOSINT_F64_I64 */ "__fixdfdi",
/* FPTOSINT_F64_I128 */ "__fixdfti",
/* FPTOSINT_F80_I32 */ "__fixxfsi",
/* FPTOSINT_F80_I64 */ "__fixxfdi",
/* FPTOSINT_F80_I128 */ "__fixxfti",
/* FPTOSINT_F128_I32 */ "__fixtfsi",
/* FPTOSINT_F128_I64 */ "__fixtfdi",
/* FPTOSINT_F128_I128 */ "__fixtfti",
/* FPTOSINT_PPCF128_I32 */ nullptr,
/* FPTOSINT_PPCF128_I64 */ nullptr,
/* FPTOSINT_PPCF128_I128 */ nullptr,
/* FPTOUINT_F32_I32 */ "__fixunssfsi",
/* FPTOUINT_F32_I64 */ "__fixunssfdi",
/* FPTOUINT_F32_I128 */ "__fixunssfti",
/* FPTOUINT_F64_I32 */ "__fixunsdfsi",
/* FPTOUINT_F64_I64 */ "__fixunsdfdi",
/* FPTOUINT_F64_I128 */ "__fixunsdfti",
/* FPTOUINT_F80_I32 */ "__fixunsxfsi",
/* FPTOUINT_F80_I64 */ "__fixunsxfdi",
/* FPTOUINT_F80_I128 */ "__fixunsxfti",
/* FPTOUINT_F128_I32 */ "__fixunstfsi",
/* FPTOUINT_F128_I64 */ "__fixunstfdi",
/* FPTOUINT_F128_I128 */ "__fixunstfti",
/* FPTOUINT_PPCF128_I32 */ nullptr,
/* FPTOUINT_PPCF128_I64 */ nullptr,
/* FPTOUINT_PPCF128_I128 */ nullptr,
/* SINTTOFP_I32_F32 */ "__floatsisf",
/* SINTTOFP_I32_F64 */ "__floatsidf",
/* SINTTOFP_I32_F80 */ nullptr,
/* SINTTOFP_I32_F128 */ "__floatsitf",
/* SINTTOFP_I32_PPCF128 */ nullptr,
/* SINTTOFP_I64_F32 */ "__floatdisf",
/* SINTTOFP_I64_F64 */ "__floatdidf",
/* SINTTOFP_I64_F80 */ nullptr,
/* SINTTOFP_I64_F128 */ "__floatditf",
/* SINTTOFP_I64_PPCF128 */ nullptr,
/* SINTTOFP_I128_F32 */ "__floattisf",
/* SINTTOFP_I128_F64 */ "__floattidf",
/* SINTTOFP_I128_F80 */ nullptr,
/* SINTTOFP_I128_F128 */ "__floattitf",
/* SINTTOFP_I128_PPCF128 */ nullptr,
/* UINTTOFP_I32_F32 */ "__floatunsisf",
/* UINTTOFP_I32_F64 */ "__floatunsidf",
/* UINTTOFP_I32_F80 */ nullptr,
/* UINTTOFP_I32_F128 */ "__floatunsitf",
/* UINTTOFP_I32_PPCF128 */ nullptr,
/* UINTTOFP_I64_F32 */ "__floatundisf",
/* UINTTOFP_I64_F64 */ "__floatundidf",
/* UINTTOFP_I64_F80 */ nullptr,
/* UINTTOFP_I64_F128 */ "__floatunditf",
/* UINTTOFP_I64_PPCF128 */ nullptr,
/* UINTTOFP_I128_F32 */ "__floatuntisf",
/* UINTTOFP_I128_F64 */ "__floatuntidf",
/* UINTTOFP_I128_F80 */ nullptr,
/* UINTTOFP_I128_F128 */ "__floatuntitf",
/* UINTTOFP_I128_PPCF128 */ nullptr,
/* OEQ_F32 */ "__eqsf2",
/* OEQ_F64 */ "__eqdf2",
/* OEQ_F128 */ "__eqtf2",
/* OEQ_PPCF128 */ nullptr,
/* UNE_F32 */ "__nesf2",
/* UNE_F64 */ "__nedf2",
/* UNE_F128 */ "__netf2",
/* UNE_PPCF128 */ nullptr,
/* OGE_F32 */ "__gesf2",
/* OGE_F64 */ "__gedf2",
/* OGE_F128 */ "__getf2",
/* OGE_PPCF128 */ nullptr,
/* OLT_F32 */ "__ltsf2",
/* OLT_F64 */ "__ltdf2",
/* OLT_F128 */ "__lttf2",
/* OLT_PPCF128 */ nullptr,
/* OLE_F32 */ "__lesf2",
/* OLE_F64 */ "__ledf2",
/* OLE_F128 */ "__letf2",
/* OLE_PPCF128 */ nullptr,
/* OGT_F32 */ "__gtsf2",
/* OGT_F64 */ "__gtdf2",
/* OGT_F128 */ "__gttf2",
/* OGT_PPCF128 */ nullptr,
/* UO_F32 */ "__unordsf2",
/* UO_F64 */ "__unorddf2",
/* UO_F128 */ "__unordtf2",
/* UO_PPCF128 */ nullptr,
/* O_F32 */ "__unordsf2",
/* O_F64 */ "__unorddf2",
/* O_F128 */ "__unordtf2",
/* O_PPCF128 */ nullptr,
/* MEMCPY */ "memcpy",
/* MEMMOVE */ "memset",
/* MEMSET */ "memmove",
/* BZERO */ nullptr,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_1 */ nullptr,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_2 */ nullptr,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_4 */ nullptr,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_8 */ nullptr,
/* MEMCPY_ELEMENT_UNORDERED_ATOMIC_16 */ nullptr,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_1 */ nullptr,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_2 */ nullptr,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_4 */ nullptr,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_8 */ nullptr,
/* MEMMOVE_ELEMENT_UNORDERED_ATOMIC_16 */ nullptr,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_1 */ nullptr,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_2 */ nullptr,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_4 */ nullptr,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_8 */ nullptr,
/* MEMSET_ELEMENT_UNORDERED_ATOMIC_16 */ nullptr,
/* UNWIND_RESUME */ "_Unwind_Resume",
/* SYNC_VAL_COMPARE_AND_SWAP_1 */ "__sync_val_compare_and_swap_1",
/* SYNC_VAL_COMPARE_AND_SWAP_2 */ "__sync_val_compare_and_swap_2",
/* SYNC_VAL_COMPARE_AND_SWAP_4 */ "__sync_val_compare_and_swap_4",
/* SYNC_VAL_COMPARE_AND_SWAP_8 */ "__sync_val_compare_and_swap_8",
/* SYNC_VAL_COMPARE_AND_SWAP_16 */ "__sync_val_compare_and_swap_16",
/* SYNC_LOCK_TEST_AND_SET_1 */ "__sync_lock_test_and_set_1",
/* SYNC_LOCK_TEST_AND_SET_2 */ "__sync_lock_test_and_set_2",
/* SYNC_LOCK_TEST_AND_SET_4 */ "__sync_lock_test_and_set_4",
/* SYNC_LOCK_TEST_AND_SET_8 */ "__sync_lock_test_and_set_8",
/* SYNC_LOCK_TEST_AND_SET_16 */ "__sync_lock_test_and_set_16",
/* SYNC_FETCH_AND_ADD_1 */ "__sync_fetch_and_add_1",
/* SYNC_FETCH_AND_ADD_2 */ "__sync_fetch_and_add_2",
/* SYNC_FETCH_AND_ADD_4 */ "__sync_fetch_and_add_4",
/* SYNC_FETCH_AND_ADD_8 */ "__sync_fetch_and_add_8",
/* SYNC_FETCH_AND_ADD_16 */ "__sync_fetch_and_add_16",
/* SYNC_FETCH_AND_SUB_1 */ "__sync_fetch_and_sub_1",
/* SYNC_FETCH_AND_SUB_2 */ "__sync_fetch_and_sub_2",
/* SYNC_FETCH_AND_SUB_4 */ "__sync_fetch_and_sub_4",
/* SYNC_FETCH_AND_SUB_8 */ "__sync_fetch_and_sub_8",
/* SYNC_FETCH_AND_SUB_16 */ "__sync_fetch_and_sub_16",
/* SYNC_FETCH_AND_AND_1 */ "__sync_fetch_and_and_1",
/* SYNC_FETCH_AND_AND_2 */ "__sync_fetch_and_and_2",
/* SYNC_FETCH_AND_AND_4 */ "__sync_fetch_and_and_4",
/* SYNC_FETCH_AND_AND_8 */ "__sync_fetch_and_and_8",
/* SYNC_FETCH_AND_AND_16 */ "__sync_fetch_and_and_16",
/* SYNC_FETCH_AND_OR_1 */ "__sync_fetch_and_or_1",
/* SYNC_FETCH_AND_OR_2 */ "__sync_fetch_and_or_2",
/* SYNC_FETCH_AND_OR_4 */ "__sync_fetch_and_or_4",
/* SYNC_FETCH_AND_OR_8 */ "__sync_fetch_and_or_8",
/* SYNC_FETCH_AND_OR_16 */ "__sync_fetch_and_or_16",
/* SYNC_FETCH_AND_XOR_1 */ "__sync_fetch_and_xor_1",
/* SYNC_FETCH_AND_XOR_2 */ "__sync_fetch_and_xor_2",
/* SYNC_FETCH_AND_XOR_4 */ "__sync_fetch_and_xor_4",
/* SYNC_FETCH_AND_XOR_8 */ "__sync_fetch_and_xor_8",
/* SYNC_FETCH_AND_XOR_16 */ "__sync_fetch_and_xor_16",
/* SYNC_FETCH_AND_NAND_1 */ "__sync_fetch_and_nand_1",
/* SYNC_FETCH_AND_NAND_2 */ "__sync_fetch_and_nand_2",
/* SYNC_FETCH_AND_NAND_4 */ "__sync_fetch_and_nand_4",
/* SYNC_FETCH_AND_NAND_8 */ "__sync_fetch_and_nand_8",
/* SYNC_FETCH_AND_NAND_16 */ "__sync_fetch_and_nand_16",
/* SYNC_FETCH_AND_MAX_1 */ "__sync_fetch_and_max_1",
/* SYNC_FETCH_AND_MAX_2 */ "__sync_fetch_and_max_2",
/* SYNC_FETCH_AND_MAX_4 */ "__sync_fetch_and_max_4",
/* SYNC_FETCH_AND_MAX_8 */ "__sync_fetch_and_max_8",
/* SYNC_FETCH_AND_MAX_16 */ "__sync_fetch_and_max_16",
/* SYNC_FETCH_AND_UMAX_1 */ "__sync_fetch_and_umax_1",
/* SYNC_FETCH_AND_UMAX_2 */ "__sync_fetch_and_umax_2",
/* SYNC_FETCH_AND_UMAX_4 */ "__sync_fetch_and_umax_4",
/* SYNC_FETCH_AND_UMAX_8 */ "__sync_fetch_and_umax_8",
/* SYNC_FETCH_AND_UMAX_16 */ "__sync_fetch_and_umax_16",
/* SYNC_FETCH_AND_MIN_1 */ "__sync_fetch_and_min_1",
/* SYNC_FETCH_AND_MIN_2 */ "__sync_fetch_and_min_2",
/* SYNC_FETCH_AND_MIN_4 */ "__sync_fetch_and_min_4",
/* SYNC_FETCH_AND_MIN_8 */ "__sync_fetch_and_min_8",
/* SYNC_FETCH_AND_MIN_16 */ "__sync_fetch_and_min_16",
/* SYNC_FETCH_AND_UMIN_1 */ "__sync_fetch_and_umin_1",
/* SYNC_FETCH_AND_UMIN_2 */ "__sync_fetch_and_umin_2",
/* SYNC_FETCH_AND_UMIN_4 */ "__sync_fetch_and_umin_4",
/* SYNC_FETCH_AND_UMIN_8 */ "__sync_fetch_and_umin_8",
/* SYNC_FETCH_AND_UMIN_16 */ "__sync_fetch_and_umin_16",

/* ATOMIC_LOAD */ "__atomic_load",
/* ATOMIC_LOAD_1 */ "__atomic_load_1",
/* ATOMIC_LOAD_2 */ "__atomic_load_2",
/* ATOMIC_LOAD_4 */ "__atomic_load_4",
/* ATOMIC_LOAD_8 */ "__atomic_load_8",
/* ATOMIC_LOAD_16 */ "__atomic_load_16",

/* ATOMIC_STORE */ "__atomic_store",
/* ATOMIC_STORE_1 */ "__atomic_store_1",
/* ATOMIC_STORE_2 */ "__atomic_store_2",
/* ATOMIC_STORE_4 */ "__atomic_store_4",
/* ATOMIC_STORE_8 */ "__atomic_store_8",
/* ATOMIC_STORE_16 */ "__atomic_store_16",

/* ATOMIC_EXCHANGE */ "__atomic_exchange",
/* ATOMIC_EXCHANGE_1 */ "__atomic_exchange_1",
/* ATOMIC_EXCHANGE_2 */ "__atomic_exchange_2",
/* ATOMIC_EXCHANGE_4 */ "__atomic_exchange_4",
/* ATOMIC_EXCHANGE_8 */ "__atomic_exchange_8",
/* ATOMIC_EXCHANGE_16 */ "__atomic_exchange_16",

/* ATOMIC_COMPARE_EXCHANGE */ "__atomic_compare_exchange",
/* ATOMIC_COMPARE_EXCHANGE_1 */ "__atomic_compare_exchange_1",
/* ATOMIC_COMPARE_EXCHANGE_2 */ "__atomic_compare_exchange_2",
/* ATOMIC_COMPARE_EXCHANGE_4 */ "__atomic_compare_exchange_4",
/* ATOMIC_COMPARE_EXCHANGE_8 */ "__atomic_compare_exchange_8",
/* ATOMIC_COMPARE_EXCHANGE_16 */ "__atomic_compare_exchange_16",

/* ATOMIC_FETCH_ADD_1 */ "__atomic_fetch_add_1",
/* ATOMIC_FETCH_ADD_2 */ "__atomic_fetch_add_2",
/* ATOMIC_FETCH_ADD_4 */ "__atomic_fetch_add_4",
/* ATOMIC_FETCH_ADD_8 */ "__atomic_fetch_add_8",
/* ATOMIC_FETCH_ADD_16 */ "__atomic_fetch_add_16",
/* ATOMIC_FETCH_SUB_1 */ "__atomic_fetch_sub_1",
/* ATOMIC_FETCH_SUB_2 */ "__atomic_fetch_sub_2",
/* ATOMIC_FETCH_SUB_4 */ "__atomic_fetch_sub_4",
/* ATOMIC_FETCH_SUB_8 */ "__atomic_fetch_sub_8",
/* ATOMIC_FETCH_SUB_16 */ "__atomic_fetch_sub_16",
/* ATOMIC_FETCH_AND_1 */ "__atomic_fetch_and_1",
/* ATOMIC_FETCH_AND_2 */ "__atomic_fetch_and_2",
/* ATOMIC_FETCH_AND_4 */ "__atomic_fetch_and_4",
/* ATOMIC_FETCH_AND_8 */ "__atomic_fetch_and_8",
/* ATOMIC_FETCH_AND_16 */ "__atomic_fetch_and_16",
/* ATOMIC_FETCH_OR_1 */ "__atomic_fetch_or_1",
/* ATOMIC_FETCH_OR_2 */ "__atomic_fetch_or_2",
/* ATOMIC_FETCH_OR_4 */ "__atomic_fetch_or_4",
/* ATOMIC_FETCH_OR_8 */ "__atomic_fetch_or_8",
/* ATOMIC_FETCH_OR_16 */ "__atomic_fetch_or_16",
/* ATOMIC_FETCH_XOR_1 */ "__atomic_fetch_xor_1",
/* ATOMIC_FETCH_XOR_2 */ "__atomic_fetch_xor_2",
/* ATOMIC_FETCH_XOR_4 */ "__atomic_fetch_xor_4",
/* ATOMIC_FETCH_XOR_8 */ "__atomic_fetch_xor_8",
/* ATOMIC_FETCH_XOR_16 */ "__atomic_fetch_xor_16",
/* ATOMIC_FETCH_NAND_1 */ "__atomic_fetch_nand_1",
/* ATOMIC_FETCH_NAND_2 */ "__atomic_fetch_nand_2",
/* ATOMIC_FETCH_NAND_4 */ "__atomic_fetch_nand_4",
/* ATOMIC_FETCH_NAND_8 */ "__atomic_fetch_nand_8",
/* ATOMIC_FETCH_NAND_16 */ "__atomic_fetch_nand_16",

/* STACKPROTECTOR_CHECK_FAIL */ "__stack_chk_fail",

/* DEOPTIMIZE */ "__llvm_deoptimize",
};

void llvm::GetSignature(const WebAssemblySubtarget &Subtarget,
                        RTLIB::Libcall LC, SmallVectorImpl<wasm::ValType> &Rets,
                        SmallVectorImpl<wasm::ValType> &Params) {
  assert(Rets.empty());
  assert(Params.empty());

  WebAssembly::ExprType iPTR = Subtarget.hasAddr64() ?
                               WebAssembly::ExprType::I64 :
                               WebAssembly::ExprType::I32;

  switch (RuntimeLibcallSignatures[LC]) {
  case func:
    break;
  case f32_func_f32:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    break;
  case f32_func_f64:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F64);
    break;
  case f32_func_i32:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::I32);
    break;
  case f32_func_i64:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::I64);
    break;
  case f32_func_i16:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::I32);
    break;
  case f64_func_f32:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F32);
    break;
  case f64_func_f64:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    break;
  case f64_func_i32:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::I32);
    break;
  case f64_func_i64:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i32_func_f32:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::F32);
    break;
  case i32_func_f64:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::F64);
    break;
  case i32_func_i32:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    break;
  case i64_func_f32:
    Rets.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::F32);
    break;
  case i64_func_f64:
    Rets.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::F64);
    break;
  case i64_func_i64:
    Rets.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case f32_func_f32_f32:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    break;
  case f32_func_f32_i32:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::I32);
    break;
  case f32_func_i64_i64:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case f64_func_f64_f64:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    break;
  case f64_func_f64_i32:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::I32);
    break;
  case f64_func_i64_i64:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i16_func_f32:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::F32);
    break;
  case i8_func_i8_i8:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    break;
  case func_f32_iPTR_iPTR:
    Params.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    break;
  case func_f64_iPTR_iPTR:
    Params.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    break;
  case i16_func_i16_i16:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    break;
  case i32_func_f32_f32:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    break;
  case i32_func_f64_f64:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    break;
  case i32_func_i32_i32:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    break;
  case i64_func_i64_i64:
    Rets.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i64_i64_func_f32:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::F32);
    break;
  case i64_i64_func_f64:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::F64);
    break;
  case i16_i16_func_i16_i16:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I32);
    Rets.push_back(wasm::ValType::I32);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    break;
  case i32_i32_func_i32_i32:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I32);
    Rets.push_back(wasm::ValType::I32);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I32);
    break;
  case i64_i64_func_i64_i64:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i64_i64_func_i64_i64_i64_i64:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i64_i64_i64_i64_func_i64_i64_i64_i64:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i64_i64_func_i64_i64_i32:
#if 0 // TODO: Enable this when wasm gets multiple-return-value support.
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
    Rets.push_back(wasm::ValType::I64);
#else
    Params.push_back(wasm::ValType(iPTR));
#endif
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I32);
    break;
  case iPTR_func_iPTR_i32_iPTR:
    Rets.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType(iPTR));
    break;
  case iPTR_func_iPTR_iPTR_iPTR:
    Rets.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    break;
  case f32_func_f32_f32_f32:
    Rets.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    Params.push_back(wasm::ValType::F32);
    break;
  case f64_func_f64_f64_f64:
    Rets.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    Params.push_back(wasm::ValType::F64);
    break;
  case func_i64_i64_iPTR_iPTR:
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType(iPTR));
    break;
  case func_iPTR_f32:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::F32);
    break;
  case func_iPTR_f64:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::F64);
    break;
  case func_iPTR_i32:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::I32);
    break;
  case func_iPTR_i64:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::I64);
    break;
  case func_iPTR_i64_i64:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case func_iPTR_i64_i64_i64_i64:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case func_iPTR_i64_i64_i64_i64_i64_i64:
    Params.push_back(wasm::ValType(iPTR));
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i32_func_i64_i64:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case i32_func_i64_i64_i64_i64:
    Rets.push_back(wasm::ValType::I32);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    Params.push_back(wasm::ValType::I64);
    break;
  case unsupported:
    llvm_unreachable("unsupported runtime library signature");
  }
}

void llvm::GetSignature(const WebAssemblySubtarget &Subtarget, const char *Name,
                        SmallVectorImpl<wasm::ValType> &Rets,
                        SmallVectorImpl<wasm::ValType> &Params) {
  assert(strcmp(RuntimeLibcallNames[RTLIB::DEOPTIMIZE], "__llvm_deoptimize") ==
         0);

  for (size_t i = 0, e = RTLIB::UNKNOWN_LIBCALL; i < e; ++i)
    if (RuntimeLibcallNames[i] && strcmp(RuntimeLibcallNames[i], Name) == 0)
      return GetSignature(Subtarget, RTLIB::Libcall(i), Rets, Params);

  llvm_unreachable("unexpected runtime library name");
}
