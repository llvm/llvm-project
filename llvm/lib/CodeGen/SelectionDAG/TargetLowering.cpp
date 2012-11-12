//===-- TargetLowering.cpp - Implement the TargetLowering class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the TargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetLowering.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/GlobalVariable.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <cctype>
using namespace llvm;

/// InitLibcallNames - Set default libcall names.
///
static void InitLibcallNames(const char **Names) {
  Names[RTLIB::SHL_I16] = "__ashlhi3";
  Names[RTLIB::SHL_I32] = "__ashlsi3";
  Names[RTLIB::SHL_I64] = "__ashldi3";
  Names[RTLIB::SHL_I128] = "__ashlti3";
  Names[RTLIB::SRL_I16] = "__lshrhi3";
  Names[RTLIB::SRL_I32] = "__lshrsi3";
  Names[RTLIB::SRL_I64] = "__lshrdi3";
  Names[RTLIB::SRL_I128] = "__lshrti3";
  Names[RTLIB::SRA_I16] = "__ashrhi3";
  Names[RTLIB::SRA_I32] = "__ashrsi3";
  Names[RTLIB::SRA_I64] = "__ashrdi3";
  Names[RTLIB::SRA_I128] = "__ashrti3";
  Names[RTLIB::MUL_I8] = "__mulqi3";
  Names[RTLIB::MUL_I16] = "__mulhi3";
  Names[RTLIB::MUL_I32] = "__mulsi3";
  Names[RTLIB::MUL_I64] = "__muldi3";
  Names[RTLIB::MUL_I128] = "__multi3";
  Names[RTLIB::MULO_I32] = "__mulosi4";
  Names[RTLIB::MULO_I64] = "__mulodi4";
  Names[RTLIB::MULO_I128] = "__muloti4";
  Names[RTLIB::SDIV_I8] = "__divqi3";
  Names[RTLIB::SDIV_I16] = "__divhi3";
  Names[RTLIB::SDIV_I32] = "__divsi3";
  Names[RTLIB::SDIV_I64] = "__divdi3";
  Names[RTLIB::SDIV_I128] = "__divti3";
  Names[RTLIB::UDIV_I8] = "__udivqi3";
  Names[RTLIB::UDIV_I16] = "__udivhi3";
  Names[RTLIB::UDIV_I32] = "__udivsi3";
  Names[RTLIB::UDIV_I64] = "__udivdi3";
  Names[RTLIB::UDIV_I128] = "__udivti3";
  Names[RTLIB::SREM_I8] = "__modqi3";
  Names[RTLIB::SREM_I16] = "__modhi3";
  Names[RTLIB::SREM_I32] = "__modsi3";
  Names[RTLIB::SREM_I64] = "__moddi3";
  Names[RTLIB::SREM_I128] = "__modti3";
  Names[RTLIB::UREM_I8] = "__umodqi3";
  Names[RTLIB::UREM_I16] = "__umodhi3";
  Names[RTLIB::UREM_I32] = "__umodsi3";
  Names[RTLIB::UREM_I64] = "__umoddi3";
  Names[RTLIB::UREM_I128] = "__umodti3";

  // These are generally not available.
  Names[RTLIB::SDIVREM_I8] = 0;
  Names[RTLIB::SDIVREM_I16] = 0;
  Names[RTLIB::SDIVREM_I32] = 0;
  Names[RTLIB::SDIVREM_I64] = 0;
  Names[RTLIB::SDIVREM_I128] = 0;
  Names[RTLIB::UDIVREM_I8] = 0;
  Names[RTLIB::UDIVREM_I16] = 0;
  Names[RTLIB::UDIVREM_I32] = 0;
  Names[RTLIB::UDIVREM_I64] = 0;
  Names[RTLIB::UDIVREM_I128] = 0;

  Names[RTLIB::NEG_I32] = "__negsi2";
  Names[RTLIB::NEG_I64] = "__negdi2";
  Names[RTLIB::ADD_F32] = "__addsf3";
  Names[RTLIB::ADD_F64] = "__adddf3";
  Names[RTLIB::ADD_F80] = "__addxf3";
  Names[RTLIB::ADD_PPCF128] = "__gcc_qadd";
  Names[RTLIB::SUB_F32] = "__subsf3";
  Names[RTLIB::SUB_F64] = "__subdf3";
  Names[RTLIB::SUB_F80] = "__subxf3";
  Names[RTLIB::SUB_PPCF128] = "__gcc_qsub";
  Names[RTLIB::MUL_F32] = "__mulsf3";
  Names[RTLIB::MUL_F64] = "__muldf3";
  Names[RTLIB::MUL_F80] = "__mulxf3";
  Names[RTLIB::MUL_PPCF128] = "__gcc_qmul";
  Names[RTLIB::DIV_F32] = "__divsf3";
  Names[RTLIB::DIV_F64] = "__divdf3";
  Names[RTLIB::DIV_F80] = "__divxf3";
  Names[RTLIB::DIV_PPCF128] = "__gcc_qdiv";
  Names[RTLIB::REM_F32] = "fmodf";
  Names[RTLIB::REM_F64] = "fmod";
  Names[RTLIB::REM_F80] = "fmodl";
  Names[RTLIB::REM_PPCF128] = "fmodl";
  Names[RTLIB::FMA_F32] = "fmaf";
  Names[RTLIB::FMA_F64] = "fma";
  Names[RTLIB::FMA_F80] = "fmal";
  Names[RTLIB::FMA_PPCF128] = "fmal";
  Names[RTLIB::POWI_F32] = "__powisf2";
  Names[RTLIB::POWI_F64] = "__powidf2";
  Names[RTLIB::POWI_F80] = "__powixf2";
  Names[RTLIB::POWI_PPCF128] = "__powitf2";
  Names[RTLIB::SQRT_F32] = "sqrtf";
  Names[RTLIB::SQRT_F64] = "sqrt";
  Names[RTLIB::SQRT_F80] = "sqrtl";
  Names[RTLIB::SQRT_PPCF128] = "sqrtl";
  Names[RTLIB::LOG_F32] = "logf";
  Names[RTLIB::LOG_F64] = "log";
  Names[RTLIB::LOG_F80] = "logl";
  Names[RTLIB::LOG_PPCF128] = "logl";
  Names[RTLIB::LOG2_F32] = "log2f";
  Names[RTLIB::LOG2_F64] = "log2";
  Names[RTLIB::LOG2_F80] = "log2l";
  Names[RTLIB::LOG2_PPCF128] = "log2l";
  Names[RTLIB::LOG10_F32] = "log10f";
  Names[RTLIB::LOG10_F64] = "log10";
  Names[RTLIB::LOG10_F80] = "log10l";
  Names[RTLIB::LOG10_PPCF128] = "log10l";
  Names[RTLIB::EXP_F32] = "expf";
  Names[RTLIB::EXP_F64] = "exp";
  Names[RTLIB::EXP_F80] = "expl";
  Names[RTLIB::EXP_PPCF128] = "expl";
  Names[RTLIB::EXP2_F32] = "exp2f";
  Names[RTLIB::EXP2_F64] = "exp2";
  Names[RTLIB::EXP2_F80] = "exp2l";
  Names[RTLIB::EXP2_PPCF128] = "exp2l";
  Names[RTLIB::SIN_F32] = "sinf";
  Names[RTLIB::SIN_F64] = "sin";
  Names[RTLIB::SIN_F80] = "sinl";
  Names[RTLIB::SIN_PPCF128] = "sinl";
  Names[RTLIB::COS_F32] = "cosf";
  Names[RTLIB::COS_F64] = "cos";
  Names[RTLIB::COS_F80] = "cosl";
  Names[RTLIB::COS_PPCF128] = "cosl";
  Names[RTLIB::POW_F32] = "powf";
  Names[RTLIB::POW_F64] = "pow";
  Names[RTLIB::POW_F80] = "powl";
  Names[RTLIB::POW_PPCF128] = "powl";
  Names[RTLIB::CEIL_F32] = "ceilf";
  Names[RTLIB::CEIL_F64] = "ceil";
  Names[RTLIB::CEIL_F80] = "ceill";
  Names[RTLIB::CEIL_PPCF128] = "ceill";
  Names[RTLIB::TRUNC_F32] = "truncf";
  Names[RTLIB::TRUNC_F64] = "trunc";
  Names[RTLIB::TRUNC_F80] = "truncl";
  Names[RTLIB::TRUNC_PPCF128] = "truncl";
  Names[RTLIB::RINT_F32] = "rintf";
  Names[RTLIB::RINT_F64] = "rint";
  Names[RTLIB::RINT_F80] = "rintl";
  Names[RTLIB::RINT_PPCF128] = "rintl";
  Names[RTLIB::NEARBYINT_F32] = "nearbyintf";
  Names[RTLIB::NEARBYINT_F64] = "nearbyint";
  Names[RTLIB::NEARBYINT_F80] = "nearbyintl";
  Names[RTLIB::NEARBYINT_PPCF128] = "nearbyintl";
  Names[RTLIB::FLOOR_F32] = "floorf";
  Names[RTLIB::FLOOR_F64] = "floor";
  Names[RTLIB::FLOOR_F80] = "floorl";
  Names[RTLIB::FLOOR_PPCF128] = "floorl";
  Names[RTLIB::COPYSIGN_F32] = "copysignf";
  Names[RTLIB::COPYSIGN_F64] = "copysign";
  Names[RTLIB::COPYSIGN_F80] = "copysignl";
  Names[RTLIB::COPYSIGN_PPCF128] = "copysignl";
  Names[RTLIB::FPEXT_F32_F64] = "__extendsfdf2";
  Names[RTLIB::FPEXT_F16_F32] = "__gnu_h2f_ieee";
  Names[RTLIB::FPROUND_F32_F16] = "__gnu_f2h_ieee";
  Names[RTLIB::FPROUND_F64_F32] = "__truncdfsf2";
  Names[RTLIB::FPROUND_F80_F32] = "__truncxfsf2";
  Names[RTLIB::FPROUND_PPCF128_F32] = "__trunctfsf2";
  Names[RTLIB::FPROUND_F80_F64] = "__truncxfdf2";
  Names[RTLIB::FPROUND_PPCF128_F64] = "__trunctfdf2";
  Names[RTLIB::FPTOSINT_F32_I8] = "__fixsfqi";
  Names[RTLIB::FPTOSINT_F32_I16] = "__fixsfhi";
  Names[RTLIB::FPTOSINT_F32_I32] = "__fixsfsi";
  Names[RTLIB::FPTOSINT_F32_I64] = "__fixsfdi";
  Names[RTLIB::FPTOSINT_F32_I128] = "__fixsfti";
  Names[RTLIB::FPTOSINT_F64_I8] = "__fixdfqi";
  Names[RTLIB::FPTOSINT_F64_I16] = "__fixdfhi";
  Names[RTLIB::FPTOSINT_F64_I32] = "__fixdfsi";
  Names[RTLIB::FPTOSINT_F64_I64] = "__fixdfdi";
  Names[RTLIB::FPTOSINT_F64_I128] = "__fixdfti";
  Names[RTLIB::FPTOSINT_F80_I32] = "__fixxfsi";
  Names[RTLIB::FPTOSINT_F80_I64] = "__fixxfdi";
  Names[RTLIB::FPTOSINT_F80_I128] = "__fixxfti";
  Names[RTLIB::FPTOSINT_PPCF128_I32] = "__fixtfsi";
  Names[RTLIB::FPTOSINT_PPCF128_I64] = "__fixtfdi";
  Names[RTLIB::FPTOSINT_PPCF128_I128] = "__fixtfti";
  Names[RTLIB::FPTOUINT_F32_I8] = "__fixunssfqi";
  Names[RTLIB::FPTOUINT_F32_I16] = "__fixunssfhi";
  Names[RTLIB::FPTOUINT_F32_I32] = "__fixunssfsi";
  Names[RTLIB::FPTOUINT_F32_I64] = "__fixunssfdi";
  Names[RTLIB::FPTOUINT_F32_I128] = "__fixunssfti";
  Names[RTLIB::FPTOUINT_F64_I8] = "__fixunsdfqi";
  Names[RTLIB::FPTOUINT_F64_I16] = "__fixunsdfhi";
  Names[RTLIB::FPTOUINT_F64_I32] = "__fixunsdfsi";
  Names[RTLIB::FPTOUINT_F64_I64] = "__fixunsdfdi";
  Names[RTLIB::FPTOUINT_F64_I128] = "__fixunsdfti";
  Names[RTLIB::FPTOUINT_F80_I32] = "__fixunsxfsi";
  Names[RTLIB::FPTOUINT_F80_I64] = "__fixunsxfdi";
  Names[RTLIB::FPTOUINT_F80_I128] = "__fixunsxfti";
  Names[RTLIB::FPTOUINT_PPCF128_I32] = "__fixunstfsi";
  Names[RTLIB::FPTOUINT_PPCF128_I64] = "__fixunstfdi";
  Names[RTLIB::FPTOUINT_PPCF128_I128] = "__fixunstfti";
  Names[RTLIB::SINTTOFP_I32_F32] = "__floatsisf";
  Names[RTLIB::SINTTOFP_I32_F64] = "__floatsidf";
  Names[RTLIB::SINTTOFP_I32_F80] = "__floatsixf";
  Names[RTLIB::SINTTOFP_I32_PPCF128] = "__floatsitf";
  Names[RTLIB::SINTTOFP_I64_F32] = "__floatdisf";
  Names[RTLIB::SINTTOFP_I64_F64] = "__floatdidf";
  Names[RTLIB::SINTTOFP_I64_F80] = "__floatdixf";
  Names[RTLIB::SINTTOFP_I64_PPCF128] = "__floatditf";
  Names[RTLIB::SINTTOFP_I128_F32] = "__floattisf";
  Names[RTLIB::SINTTOFP_I128_F64] = "__floattidf";
  Names[RTLIB::SINTTOFP_I128_F80] = "__floattixf";
  Names[RTLIB::SINTTOFP_I128_PPCF128] = "__floattitf";
  Names[RTLIB::UINTTOFP_I32_F32] = "__floatunsisf";
  Names[RTLIB::UINTTOFP_I32_F64] = "__floatunsidf";
  Names[RTLIB::UINTTOFP_I32_F80] = "__floatunsixf";
  Names[RTLIB::UINTTOFP_I32_PPCF128] = "__floatunsitf";
  Names[RTLIB::UINTTOFP_I64_F32] = "__floatundisf";
  Names[RTLIB::UINTTOFP_I64_F64] = "__floatundidf";
  Names[RTLIB::UINTTOFP_I64_F80] = "__floatundixf";
  Names[RTLIB::UINTTOFP_I64_PPCF128] = "__floatunditf";
  Names[RTLIB::UINTTOFP_I128_F32] = "__floatuntisf";
  Names[RTLIB::UINTTOFP_I128_F64] = "__floatuntidf";
  Names[RTLIB::UINTTOFP_I128_F80] = "__floatuntixf";
  Names[RTLIB::UINTTOFP_I128_PPCF128] = "__floatuntitf";
  Names[RTLIB::OEQ_F32] = "__eqsf2";
  Names[RTLIB::OEQ_F64] = "__eqdf2";
  Names[RTLIB::UNE_F32] = "__nesf2";
  Names[RTLIB::UNE_F64] = "__nedf2";
  Names[RTLIB::OGE_F32] = "__gesf2";
  Names[RTLIB::OGE_F64] = "__gedf2";
  Names[RTLIB::OLT_F32] = "__ltsf2";
  Names[RTLIB::OLT_F64] = "__ltdf2";
  Names[RTLIB::OLE_F32] = "__lesf2";
  Names[RTLIB::OLE_F64] = "__ledf2";
  Names[RTLIB::OGT_F32] = "__gtsf2";
  Names[RTLIB::OGT_F64] = "__gtdf2";
  Names[RTLIB::UO_F32] = "__unordsf2";
  Names[RTLIB::UO_F64] = "__unorddf2";
  Names[RTLIB::O_F32] = "__unordsf2";
  Names[RTLIB::O_F64] = "__unorddf2";
  Names[RTLIB::MEMCPY] = "memcpy";
  Names[RTLIB::MEMMOVE] = "memmove";
  Names[RTLIB::MEMSET] = "memset";
  Names[RTLIB::UNWIND_RESUME] = "_Unwind_Resume";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_1] = "__sync_val_compare_and_swap_1";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_2] = "__sync_val_compare_and_swap_2";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_4] = "__sync_val_compare_and_swap_4";
  Names[RTLIB::SYNC_VAL_COMPARE_AND_SWAP_8] = "__sync_val_compare_and_swap_8";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_1] = "__sync_lock_test_and_set_1";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_2] = "__sync_lock_test_and_set_2";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_4] = "__sync_lock_test_and_set_4";
  Names[RTLIB::SYNC_LOCK_TEST_AND_SET_8] = "__sync_lock_test_and_set_8";
  Names[RTLIB::SYNC_FETCH_AND_ADD_1] = "__sync_fetch_and_add_1";
  Names[RTLIB::SYNC_FETCH_AND_ADD_2] = "__sync_fetch_and_add_2";
  Names[RTLIB::SYNC_FETCH_AND_ADD_4] = "__sync_fetch_and_add_4";
  Names[RTLIB::SYNC_FETCH_AND_ADD_8] = "__sync_fetch_and_add_8";
  Names[RTLIB::SYNC_FETCH_AND_SUB_1] = "__sync_fetch_and_sub_1";
  Names[RTLIB::SYNC_FETCH_AND_SUB_2] = "__sync_fetch_and_sub_2";
  Names[RTLIB::SYNC_FETCH_AND_SUB_4] = "__sync_fetch_and_sub_4";
  Names[RTLIB::SYNC_FETCH_AND_SUB_8] = "__sync_fetch_and_sub_8";
  Names[RTLIB::SYNC_FETCH_AND_AND_1] = "__sync_fetch_and_and_1";
  Names[RTLIB::SYNC_FETCH_AND_AND_2] = "__sync_fetch_and_and_2";
  Names[RTLIB::SYNC_FETCH_AND_AND_4] = "__sync_fetch_and_and_4";
  Names[RTLIB::SYNC_FETCH_AND_AND_8] = "__sync_fetch_and_and_8";
  Names[RTLIB::SYNC_FETCH_AND_OR_1] = "__sync_fetch_and_or_1";
  Names[RTLIB::SYNC_FETCH_AND_OR_2] = "__sync_fetch_and_or_2";
  Names[RTLIB::SYNC_FETCH_AND_OR_4] = "__sync_fetch_and_or_4";
  Names[RTLIB::SYNC_FETCH_AND_OR_8] = "__sync_fetch_and_or_8";
  Names[RTLIB::SYNC_FETCH_AND_XOR_1] = "__sync_fetch_and_xor_1";
  Names[RTLIB::SYNC_FETCH_AND_XOR_2] = "__sync_fetch_and_xor_2";
  Names[RTLIB::SYNC_FETCH_AND_XOR_4] = "__sync_fetch_and_xor_4";
  Names[RTLIB::SYNC_FETCH_AND_XOR_8] = "__sync_fetch_and_xor_8";
  Names[RTLIB::SYNC_FETCH_AND_NAND_1] = "__sync_fetch_and_nand_1";
  Names[RTLIB::SYNC_FETCH_AND_NAND_2] = "__sync_fetch_and_nand_2";
  Names[RTLIB::SYNC_FETCH_AND_NAND_4] = "__sync_fetch_and_nand_4";
  Names[RTLIB::SYNC_FETCH_AND_NAND_8] = "__sync_fetch_and_nand_8";
}

/// InitLibcallCallingConvs - Set default libcall CallingConvs.
///
static void InitLibcallCallingConvs(CallingConv::ID *CCs) {
  for (int i = 0; i < RTLIB::UNKNOWN_LIBCALL; ++i) {
    CCs[i] = CallingConv::C;
  }
}

/// getFPEXT - Return the FPEXT_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPEXT(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::f32) {
    if (RetVT == MVT::f64)
      return FPEXT_F32_F64;
  }

  return UNKNOWN_LIBCALL;
}

/// getFPROUND - Return the FPROUND_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPROUND(EVT OpVT, EVT RetVT) {
  if (RetVT == MVT::f32) {
    if (OpVT == MVT::f64)
      return FPROUND_F64_F32;
    if (OpVT == MVT::f80)
      return FPROUND_F80_F32;
    if (OpVT == MVT::ppcf128)
      return FPROUND_PPCF128_F32;
  } else if (RetVT == MVT::f64) {
    if (OpVT == MVT::f80)
      return FPROUND_F80_F64;
    if (OpVT == MVT::ppcf128)
      return FPROUND_PPCF128_F64;
  }

  return UNKNOWN_LIBCALL;
}

/// getFPTOSINT - Return the FPTOSINT_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPTOSINT(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::f32) {
    if (RetVT == MVT::i8)
      return FPTOSINT_F32_I8;
    if (RetVT == MVT::i16)
      return FPTOSINT_F32_I16;
    if (RetVT == MVT::i32)
      return FPTOSINT_F32_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F32_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F32_I128;
  } else if (OpVT == MVT::f64) {
    if (RetVT == MVT::i8)
      return FPTOSINT_F64_I8;
    if (RetVT == MVT::i16)
      return FPTOSINT_F64_I16;
    if (RetVT == MVT::i32)
      return FPTOSINT_F64_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F64_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F64_I128;
  } else if (OpVT == MVT::f80) {
    if (RetVT == MVT::i32)
      return FPTOSINT_F80_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_F80_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_F80_I128;
  } else if (OpVT == MVT::ppcf128) {
    if (RetVT == MVT::i32)
      return FPTOSINT_PPCF128_I32;
    if (RetVT == MVT::i64)
      return FPTOSINT_PPCF128_I64;
    if (RetVT == MVT::i128)
      return FPTOSINT_PPCF128_I128;
  }
  return UNKNOWN_LIBCALL;
}

/// getFPTOUINT - Return the FPTOUINT_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getFPTOUINT(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::f32) {
    if (RetVT == MVT::i8)
      return FPTOUINT_F32_I8;
    if (RetVT == MVT::i16)
      return FPTOUINT_F32_I16;
    if (RetVT == MVT::i32)
      return FPTOUINT_F32_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F32_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F32_I128;
  } else if (OpVT == MVT::f64) {
    if (RetVT == MVT::i8)
      return FPTOUINT_F64_I8;
    if (RetVT == MVT::i16)
      return FPTOUINT_F64_I16;
    if (RetVT == MVT::i32)
      return FPTOUINT_F64_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F64_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F64_I128;
  } else if (OpVT == MVT::f80) {
    if (RetVT == MVT::i32)
      return FPTOUINT_F80_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_F80_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_F80_I128;
  } else if (OpVT == MVT::ppcf128) {
    if (RetVT == MVT::i32)
      return FPTOUINT_PPCF128_I32;
    if (RetVT == MVT::i64)
      return FPTOUINT_PPCF128_I64;
    if (RetVT == MVT::i128)
      return FPTOUINT_PPCF128_I128;
  }
  return UNKNOWN_LIBCALL;
}

/// getSINTTOFP - Return the SINTTOFP_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getSINTTOFP(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::i32) {
    if (RetVT == MVT::f32)
      return SINTTOFP_I32_F32;
    else if (RetVT == MVT::f64)
      return SINTTOFP_I32_F64;
    else if (RetVT == MVT::f80)
      return SINTTOFP_I32_F80;
    else if (RetVT == MVT::ppcf128)
      return SINTTOFP_I32_PPCF128;
  } else if (OpVT == MVT::i64) {
    if (RetVT == MVT::f32)
      return SINTTOFP_I64_F32;
    else if (RetVT == MVT::f64)
      return SINTTOFP_I64_F64;
    else if (RetVT == MVT::f80)
      return SINTTOFP_I64_F80;
    else if (RetVT == MVT::ppcf128)
      return SINTTOFP_I64_PPCF128;
  } else if (OpVT == MVT::i128) {
    if (RetVT == MVT::f32)
      return SINTTOFP_I128_F32;
    else if (RetVT == MVT::f64)
      return SINTTOFP_I128_F64;
    else if (RetVT == MVT::f80)
      return SINTTOFP_I128_F80;
    else if (RetVT == MVT::ppcf128)
      return SINTTOFP_I128_PPCF128;
  }
  return UNKNOWN_LIBCALL;
}

/// getUINTTOFP - Return the UINTTOFP_*_* value for the given types, or
/// UNKNOWN_LIBCALL if there is none.
RTLIB::Libcall RTLIB::getUINTTOFP(EVT OpVT, EVT RetVT) {
  if (OpVT == MVT::i32) {
    if (RetVT == MVT::f32)
      return UINTTOFP_I32_F32;
    else if (RetVT == MVT::f64)
      return UINTTOFP_I32_F64;
    else if (RetVT == MVT::f80)
      return UINTTOFP_I32_F80;
    else if (RetVT == MVT::ppcf128)
      return UINTTOFP_I32_PPCF128;
  } else if (OpVT == MVT::i64) {
    if (RetVT == MVT::f32)
      return UINTTOFP_I64_F32;
    else if (RetVT == MVT::f64)
      return UINTTOFP_I64_F64;
    else if (RetVT == MVT::f80)
      return UINTTOFP_I64_F80;
    else if (RetVT == MVT::ppcf128)
      return UINTTOFP_I64_PPCF128;
  } else if (OpVT == MVT::i128) {
    if (RetVT == MVT::f32)
      return UINTTOFP_I128_F32;
    else if (RetVT == MVT::f64)
      return UINTTOFP_I128_F64;
    else if (RetVT == MVT::f80)
      return UINTTOFP_I128_F80;
    else if (RetVT == MVT::ppcf128)
      return UINTTOFP_I128_PPCF128;
  }
  return UNKNOWN_LIBCALL;
}

/// InitCmpLibcallCCs - Set default comparison libcall CC.
///
static void InitCmpLibcallCCs(ISD::CondCode *CCs) {
  memset(CCs, ISD::SETCC_INVALID, sizeof(ISD::CondCode)*RTLIB::UNKNOWN_LIBCALL);
  CCs[RTLIB::OEQ_F32] = ISD::SETEQ;
  CCs[RTLIB::OEQ_F64] = ISD::SETEQ;
  CCs[RTLIB::UNE_F32] = ISD::SETNE;
  CCs[RTLIB::UNE_F64] = ISD::SETNE;
  CCs[RTLIB::OGE_F32] = ISD::SETGE;
  CCs[RTLIB::OGE_F64] = ISD::SETGE;
  CCs[RTLIB::OLT_F32] = ISD::SETLT;
  CCs[RTLIB::OLT_F64] = ISD::SETLT;
  CCs[RTLIB::OLE_F32] = ISD::SETLE;
  CCs[RTLIB::OLE_F64] = ISD::SETLE;
  CCs[RTLIB::OGT_F32] = ISD::SETGT;
  CCs[RTLIB::OGT_F64] = ISD::SETGT;
  CCs[RTLIB::UO_F32] = ISD::SETNE;
  CCs[RTLIB::UO_F64] = ISD::SETNE;
  CCs[RTLIB::O_F32] = ISD::SETEQ;
  CCs[RTLIB::O_F64] = ISD::SETEQ;
}

/// NOTE: The constructor takes ownership of TLOF.
TargetLowering::TargetLowering(const TargetMachine &tm,
                               const TargetLoweringObjectFile *tlof)
  : TM(tm), TD(TM.getDataLayout()), TLOF(*tlof) {
  // All operations default to being supported.
  memset(OpActions, 0, sizeof(OpActions));
  memset(LoadExtActions, 0, sizeof(LoadExtActions));
  memset(TruncStoreActions, 0, sizeof(TruncStoreActions));
  memset(IndexedModeActions, 0, sizeof(IndexedModeActions));
  memset(CondCodeActions, 0, sizeof(CondCodeActions));

  // Set default actions for various operations.
  for (unsigned VT = 0; VT != (unsigned)MVT::LAST_VALUETYPE; ++VT) {
    // Default all indexed load / store to expand.
    for (unsigned IM = (unsigned)ISD::PRE_INC;
         IM != (unsigned)ISD::LAST_INDEXED_MODE; ++IM) {
      setIndexedLoadAction(IM, (MVT::SimpleValueType)VT, Expand);
      setIndexedStoreAction(IM, (MVT::SimpleValueType)VT, Expand);
    }

    // These operations default to expand.
    setOperationAction(ISD::FGETSIGN, (MVT::SimpleValueType)VT, Expand);
    setOperationAction(ISD::CONCAT_VECTORS, (MVT::SimpleValueType)VT, Expand);
  }

  // Most targets ignore the @llvm.prefetch intrinsic.
  setOperationAction(ISD::PREFETCH, MVT::Other, Expand);

  // ConstantFP nodes default to expand.  Targets can either change this to
  // Legal, in which case all fp constants are legal, or use isFPImmLegal()
  // to optimize expansions for certain constants.
  setOperationAction(ISD::ConstantFP, MVT::f16, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f80, Expand);

  // These library functions default to expand.
  setOperationAction(ISD::FLOG ,  MVT::f16, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f16, Expand);
  setOperationAction(ISD::FLOG10, MVT::f16, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f16, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f16, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f16, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f16, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f16, Expand);
  setOperationAction(ISD::FRINT,  MVT::f16, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f16, Expand);
  setOperationAction(ISD::FLOG ,  MVT::f32, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f32, Expand);
  setOperationAction(ISD::FLOG10, MVT::f32, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f32, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f32, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f32, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f32, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f32, Expand);
  setOperationAction(ISD::FRINT,  MVT::f32, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f32, Expand);
  setOperationAction(ISD::FLOG ,  MVT::f64, Expand);
  setOperationAction(ISD::FLOG2,  MVT::f64, Expand);
  setOperationAction(ISD::FLOG10, MVT::f64, Expand);
  setOperationAction(ISD::FEXP ,  MVT::f64, Expand);
  setOperationAction(ISD::FEXP2,  MVT::f64, Expand);
  setOperationAction(ISD::FFLOOR, MVT::f64, Expand);
  setOperationAction(ISD::FNEARBYINT, MVT::f64, Expand);
  setOperationAction(ISD::FCEIL,  MVT::f64, Expand);
  setOperationAction(ISD::FRINT,  MVT::f64, Expand);
  setOperationAction(ISD::FTRUNC, MVT::f64, Expand);

  // Default ISD::TRAP to expand (which turns it into abort).
  setOperationAction(ISD::TRAP, MVT::Other, Expand);

  // On most systems, DEBUGTRAP and TRAP have no difference. The "Expand"
  // here is to inform DAG Legalizer to replace DEBUGTRAP with TRAP.
  //
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Expand);

  IsLittleEndian = TD->isLittleEndian();
  PointerTy = MVT::getIntegerVT(8*TD->getPointerSize(0));
  memset(RegClassForVT, 0,MVT::LAST_VALUETYPE*sizeof(TargetRegisterClass*));
  memset(TargetDAGCombineArray, 0, array_lengthof(TargetDAGCombineArray));
  maxStoresPerMemset = maxStoresPerMemcpy = maxStoresPerMemmove = 8;
  maxStoresPerMemsetOptSize = maxStoresPerMemcpyOptSize
    = maxStoresPerMemmoveOptSize = 4;
  benefitFromCodePlacementOpt = false;
  UseUnderscoreSetJmp = false;
  UseUnderscoreLongJmp = false;
  SelectIsExpensive = false;
  IntDivIsCheap = false;
  Pow2DivIsCheap = false;
  JumpIsExpensive = false;
  predictableSelectIsExpensive = false;
  StackPointerRegisterToSaveRestore = 0;
  ExceptionPointerRegister = 0;
  ExceptionSelectorRegister = 0;
  BooleanContents = UndefinedBooleanContent;
  BooleanVectorContents = UndefinedBooleanContent;
  SchedPreferenceInfo = Sched::ILP;
  JumpBufSize = 0;
  JumpBufAlignment = 0;
  MinFunctionAlignment = 0;
  PrefFunctionAlignment = 0;
  PrefLoopAlignment = 0;
  MinStackArgumentAlignment = 1;
  ShouldFoldAtomicFences = false;
  InsertFencesForAtomic = false;
  SupportJumpTables = true;
  MinimumJumpTableEntries = 4;

  InitLibcallNames(LibcallRoutineNames);
  InitCmpLibcallCCs(CmpLibcallCCs);
  InitLibcallCallingConvs(LibcallCallingConvs);
}

TargetLowering::~TargetLowering() {
  delete &TLOF;
}

MVT TargetLowering::getShiftAmountTy(EVT LHSTy) const {
  return MVT::getIntegerVT(8*TD->getPointerSize(0));
}

/// canOpTrap - Returns true if the operation can trap for the value type.
/// VT must be a legal type.
bool TargetLowering::canOpTrap(unsigned Op, EVT VT) const {
  assert(isTypeLegal(VT));
  switch (Op) {
  default:
    return false;
  case ISD::FDIV:
  case ISD::FREM:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM:
    return true;
  }
}


static unsigned getVectorTypeBreakdownMVT(MVT VT, MVT &IntermediateVT,
                                          unsigned &NumIntermediates,
                                          EVT &RegisterVT,
                                          TargetLowering *TLI) {
  // Figure out the right, legal destination reg to copy into.
  unsigned NumElts = VT.getVectorNumElements();
  MVT EltTy = VT.getVectorElementType();

  unsigned NumVectorRegs = 1;

  // FIXME: We don't support non-power-of-2-sized vectors for now.  Ideally we
  // could break down into LHS/RHS like LegalizeDAG does.
  if (!isPowerOf2_32(NumElts)) {
    NumVectorRegs = NumElts;
    NumElts = 1;
  }

  // Divide the input until we get to a supported size.  This will always
  // end with a scalar if the target doesn't support vectors.
  while (NumElts > 1 && !TLI->isTypeLegal(MVT::getVectorVT(EltTy, NumElts))) {
    NumElts >>= 1;
    NumVectorRegs <<= 1;
  }

  NumIntermediates = NumVectorRegs;

  MVT NewVT = MVT::getVectorVT(EltTy, NumElts);
  if (!TLI->isTypeLegal(NewVT))
    NewVT = EltTy;
  IntermediateVT = NewVT;

  unsigned NewVTSize = NewVT.getSizeInBits();

  // Convert sizes such as i33 to i64.
  if (!isPowerOf2_32(NewVTSize))
    NewVTSize = NextPowerOf2(NewVTSize);

  EVT DestVT = TLI->getRegisterType(NewVT);
  RegisterVT = DestVT;
  if (EVT(DestVT).bitsLT(NewVT))    // Value is expanded, e.g. i64 -> i16.
    return NumVectorRegs*(NewVTSize/DestVT.getSizeInBits());

  // Otherwise, promotion or legal types use the same number of registers as
  // the vector decimated to the appropriate level.
  return NumVectorRegs;
}

/// isLegalRC - Return true if the value types that can be represented by the
/// specified register class are all legal.
bool TargetLowering::isLegalRC(const TargetRegisterClass *RC) const {
  for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
       I != E; ++I) {
    if (isTypeLegal(*I))
      return true;
  }
  return false;
}

/// findRepresentativeClass - Return the largest legal super-reg register class
/// of the register class for the specified type and its associated "cost".
std::pair<const TargetRegisterClass*, uint8_t>
TargetLowering::findRepresentativeClass(EVT VT) const {
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const TargetRegisterClass *RC = RegClassForVT[VT.getSimpleVT().SimpleTy];
  if (!RC)
    return std::make_pair(RC, 0);

  // Compute the set of all super-register classes.
  BitVector SuperRegRC(TRI->getNumRegClasses());
  for (SuperRegClassIterator RCI(RC, TRI); RCI.isValid(); ++RCI)
    SuperRegRC.setBitsInMask(RCI.getMask());

  // Find the first legal register class with the largest spill size.
  const TargetRegisterClass *BestRC = RC;
  for (int i = SuperRegRC.find_first(); i >= 0; i = SuperRegRC.find_next(i)) {
    const TargetRegisterClass *SuperRC = TRI->getRegClass(i);
    // We want the largest possible spill size.
    if (SuperRC->getSize() <= BestRC->getSize())
      continue;
    if (!isLegalRC(SuperRC))
      continue;
    BestRC = SuperRC;
  }
  return std::make_pair(BestRC, 1);
}

/// computeRegisterProperties - Once all of the register classes are added,
/// this allows us to compute derived properties we expose.
void TargetLowering::computeRegisterProperties() {
  assert(MVT::LAST_VALUETYPE <= MVT::MAX_ALLOWED_VALUETYPE &&
         "Too many value types for ValueTypeActions to hold!");

  // Everything defaults to needing one register.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i) {
    NumRegistersForVT[i] = 1;
    RegisterTypeForVT[i] = TransformToType[i] = (MVT::SimpleValueType)i;
  }
  // ...except isVoid, which doesn't need any registers.
  NumRegistersForVT[MVT::isVoid] = 0;

  // Find the largest integer register class.
  unsigned LargestIntReg = MVT::LAST_INTEGER_VALUETYPE;
  for (; RegClassForVT[LargestIntReg] == 0; --LargestIntReg)
    assert(LargestIntReg != MVT::i1 && "No integer registers defined!");

  // Every integer value type larger than this largest register takes twice as
  // many registers to represent as the previous ValueType.
  for (unsigned ExpandedReg = LargestIntReg + 1; ; ++ExpandedReg) {
    EVT ExpandedVT = (MVT::SimpleValueType)ExpandedReg;
    if (!ExpandedVT.isInteger())
      break;
    NumRegistersForVT[ExpandedReg] = 2*NumRegistersForVT[ExpandedReg-1];
    RegisterTypeForVT[ExpandedReg] = (MVT::SimpleValueType)LargestIntReg;
    TransformToType[ExpandedReg] = (MVT::SimpleValueType)(ExpandedReg - 1);
    ValueTypeActions.setTypeAction(ExpandedVT, TypeExpandInteger);
  }

  // Inspect all of the ValueType's smaller than the largest integer
  // register to see which ones need promotion.
  unsigned LegalIntReg = LargestIntReg;
  for (unsigned IntReg = LargestIntReg - 1;
       IntReg >= (unsigned)MVT::i1; --IntReg) {
    EVT IVT = (MVT::SimpleValueType)IntReg;
    if (isTypeLegal(IVT)) {
      LegalIntReg = IntReg;
    } else {
      RegisterTypeForVT[IntReg] = TransformToType[IntReg] =
        (const MVT::SimpleValueType)LegalIntReg;
      ValueTypeActions.setTypeAction(IVT, TypePromoteInteger);
    }
  }

  // ppcf128 type is really two f64's.
  if (!isTypeLegal(MVT::ppcf128)) {
    NumRegistersForVT[MVT::ppcf128] = 2*NumRegistersForVT[MVT::f64];
    RegisterTypeForVT[MVT::ppcf128] = MVT::f64;
    TransformToType[MVT::ppcf128] = MVT::f64;
    ValueTypeActions.setTypeAction(MVT::ppcf128, TypeExpandFloat);
  }

  // Decide how to handle f64. If the target does not have native f64 support,
  // expand it to i64 and we will be generating soft float library calls.
  if (!isTypeLegal(MVT::f64)) {
    NumRegistersForVT[MVT::f64] = NumRegistersForVT[MVT::i64];
    RegisterTypeForVT[MVT::f64] = RegisterTypeForVT[MVT::i64];
    TransformToType[MVT::f64] = MVT::i64;
    ValueTypeActions.setTypeAction(MVT::f64, TypeSoftenFloat);
  }

  // Decide how to handle f32. If the target does not have native support for
  // f32, promote it to f64 if it is legal. Otherwise, expand it to i32.
  if (!isTypeLegal(MVT::f32)) {
    if (isTypeLegal(MVT::f64)) {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::f64];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::f64];
      TransformToType[MVT::f32] = MVT::f64;
      ValueTypeActions.setTypeAction(MVT::f32, TypePromoteInteger);
    } else {
      NumRegistersForVT[MVT::f32] = NumRegistersForVT[MVT::i32];
      RegisterTypeForVT[MVT::f32] = RegisterTypeForVT[MVT::i32];
      TransformToType[MVT::f32] = MVT::i32;
      ValueTypeActions.setTypeAction(MVT::f32, TypeSoftenFloat);
    }
  }

  // Loop over all of the vector value types to see which need transformations.
  for (unsigned i = MVT::FIRST_VECTOR_VALUETYPE;
       i <= (unsigned)MVT::LAST_VECTOR_VALUETYPE; ++i) {
    MVT VT = (MVT::SimpleValueType)i;
    if (isTypeLegal(VT)) continue;

    // Determine if there is a legal wider type.  If so, we should promote to
    // that wider vector type.
    EVT EltVT = VT.getVectorElementType();
    unsigned NElts = VT.getVectorNumElements();
    if (NElts != 1) {
      bool IsLegalWiderType = false;
      // First try to promote the elements of integer vectors. If no legal
      // promotion was found, fallback to the widen-vector method.
      for (unsigned nVT = i+1; nVT <= MVT::LAST_VECTOR_VALUETYPE; ++nVT) {
        EVT SVT = (MVT::SimpleValueType)nVT;
        // Promote vectors of integers to vectors with the same number
        // of elements, with a wider element type.
        if (SVT.getVectorElementType().getSizeInBits() > EltVT.getSizeInBits()
            && SVT.getVectorNumElements() == NElts &&
            isTypeLegal(SVT) && SVT.getScalarType().isInteger()) {
          TransformToType[i] = SVT;
          RegisterTypeForVT[i] = SVT;
          NumRegistersForVT[i] = 1;
          ValueTypeActions.setTypeAction(VT, TypePromoteInteger);
          IsLegalWiderType = true;
          break;
        }
      }

      if (IsLegalWiderType) continue;

      // Try to widen the vector.
      for (unsigned nVT = i+1; nVT <= MVT::LAST_VECTOR_VALUETYPE; ++nVT) {
        EVT SVT = (MVT::SimpleValueType)nVT;
        if (SVT.getVectorElementType() == EltVT &&
            SVT.getVectorNumElements() > NElts &&
            isTypeLegal(SVT)) {
          TransformToType[i] = SVT;
          RegisterTypeForVT[i] = SVT;
          NumRegistersForVT[i] = 1;
          ValueTypeActions.setTypeAction(VT, TypeWidenVector);
          IsLegalWiderType = true;
          break;
        }
      }
      if (IsLegalWiderType) continue;
    }

    MVT IntermediateVT;
    EVT RegisterVT;
    unsigned NumIntermediates;
    NumRegistersForVT[i] =
      getVectorTypeBreakdownMVT(VT, IntermediateVT, NumIntermediates,
                                RegisterVT, this);
    RegisterTypeForVT[i] = RegisterVT;

    EVT NVT = VT.getPow2VectorType();
    if (NVT == VT) {
      // Type is already a power of 2.  The default action is to split.
      TransformToType[i] = MVT::Other;
      unsigned NumElts = VT.getVectorNumElements();
      ValueTypeActions.setTypeAction(VT,
            NumElts > 1 ? TypeSplitVector : TypeScalarizeVector);
    } else {
      TransformToType[i] = NVT;
      ValueTypeActions.setTypeAction(VT, TypeWidenVector);
    }
  }

  // Determine the 'representative' register class for each value type.
  // An representative register class is the largest (meaning one which is
  // not a sub-register class / subreg register class) legal register class for
  // a group of value types. For example, on i386, i8, i16, and i32
  // representative would be GR32; while on x86_64 it's GR64.
  for (unsigned i = 0; i != MVT::LAST_VALUETYPE; ++i) {
    const TargetRegisterClass* RRC;
    uint8_t Cost;
    tie(RRC, Cost) =  findRepresentativeClass((MVT::SimpleValueType)i);
    RepRegClassForVT[i] = RRC;
    RepRegClassCostForVT[i] = Cost;
  }
}

const char *TargetLowering::getTargetNodeName(unsigned Opcode) const {
  return NULL;
}

EVT TargetLowering::getSetCCResultType(EVT VT) const {
  assert(!VT.isVector() && "No default SetCC type for vectors!");
  return getPointerTy(0).SimpleTy;
}

MVT::SimpleValueType TargetLowering::getCmpLibcallReturnType() const {
  return MVT::i32; // return the default value
}

/// getVectorTypeBreakdown - Vector types are broken down into some number of
/// legal first class types.  For example, MVT::v8f32 maps to 2 MVT::v4f32
/// with Altivec or SSE1, or 8 promoted MVT::f64 values with the X86 FP stack.
/// Similarly, MVT::v2i64 turns into 4 MVT::i32 values with both PPC and X86.
///
/// This method returns the number of registers needed, and the VT for each
/// register.  It also returns the VT and quantity of the intermediate values
/// before they are promoted/expanded.
///
unsigned TargetLowering::getVectorTypeBreakdown(LLVMContext &Context, EVT VT,
                                                EVT &IntermediateVT,
                                                unsigned &NumIntermediates,
                                                EVT &RegisterVT) const {
  unsigned NumElts = VT.getVectorNumElements();

  // If there is a wider vector type with the same element type as this one,
  // or a promoted vector type that has the same number of elements which
  // are wider, then we should convert to that legal vector type.
  // This handles things like <2 x float> -> <4 x float> and
  // <4 x i1> -> <4 x i32>.
  LegalizeTypeAction TA = getTypeAction(Context, VT);
  if (NumElts != 1 && (TA == TypeWidenVector || TA == TypePromoteInteger)) {
    RegisterVT = getTypeToTransformTo(Context, VT);
    if (isTypeLegal(RegisterVT)) {
      IntermediateVT = RegisterVT;
      NumIntermediates = 1;
      return 1;
    }
  }

  // Figure out the right, legal destination reg to copy into.
  EVT EltTy = VT.getVectorElementType();

  unsigned NumVectorRegs = 1;

  // FIXME: We don't support non-power-of-2-sized vectors for now.  Ideally we
  // could break down into LHS/RHS like LegalizeDAG does.
  if (!isPowerOf2_32(NumElts)) {
    NumVectorRegs = NumElts;
    NumElts = 1;
  }

  // Divide the input until we get to a supported size.  This will always
  // end with a scalar if the target doesn't support vectors.
  while (NumElts > 1 && !isTypeLegal(
                                   EVT::getVectorVT(Context, EltTy, NumElts))) {
    NumElts >>= 1;
    NumVectorRegs <<= 1;
  }

  NumIntermediates = NumVectorRegs;

  EVT NewVT = EVT::getVectorVT(Context, EltTy, NumElts);
  if (!isTypeLegal(NewVT))
    NewVT = EltTy;
  IntermediateVT = NewVT;

  EVT DestVT = getRegisterType(Context, NewVT);
  RegisterVT = DestVT;
  unsigned NewVTSize = NewVT.getSizeInBits();

  // Convert sizes such as i33 to i64.
  if (!isPowerOf2_32(NewVTSize))
    NewVTSize = NextPowerOf2(NewVTSize);

  if (DestVT.bitsLT(NewVT))   // Value is expanded, e.g. i64 -> i16.
    return NumVectorRegs*(NewVTSize/DestVT.getSizeInBits());

  // Otherwise, promotion or legal types use the same number of registers as
  // the vector decimated to the appropriate level.
  return NumVectorRegs;
}

/// Get the EVTs and ArgFlags collections that represent the legalized return
/// type of the given function.  This does not require a DAG or a return value,
/// and is suitable for use before any DAGs for the function are constructed.
/// TODO: Move this out of TargetLowering.cpp.
void llvm::GetReturnInfo(Type* ReturnType, Attributes attr,
                         SmallVectorImpl<ISD::OutputArg> &Outs,
                         const TargetLowering &TLI) {
  SmallVector<EVT, 4> ValueVTs;
  ComputeValueVTs(TLI, ReturnType, ValueVTs);
  unsigned NumValues = ValueVTs.size();
  if (NumValues == 0) return;

  for (unsigned j = 0, f = NumValues; j != f; ++j) {
    EVT VT = ValueVTs[j];
    ISD::NodeType ExtendKind = ISD::ANY_EXTEND;

    if (attr.hasAttribute(Attributes::SExt))
      ExtendKind = ISD::SIGN_EXTEND;
    else if (attr.hasAttribute(Attributes::ZExt))
      ExtendKind = ISD::ZERO_EXTEND;

    // FIXME: C calling convention requires the return type to be promoted to
    // at least 32-bit. But this is not necessary for non-C calling
    // conventions. The frontend should mark functions whose return values
    // require promoting with signext or zeroext attributes.
    if (ExtendKind != ISD::ANY_EXTEND && VT.isInteger()) {
      EVT MinVT = TLI.getRegisterType(ReturnType->getContext(), MVT::i32);
      if (VT.bitsLT(MinVT))
        VT = MinVT;
    }

    unsigned NumParts = TLI.getNumRegisters(ReturnType->getContext(), VT);
    EVT PartVT = TLI.getRegisterType(ReturnType->getContext(), VT);

    // 'inreg' on function refers to return value
    ISD::ArgFlagsTy Flags = ISD::ArgFlagsTy();
    if (attr.hasAttribute(Attributes::InReg))
      Flags.setInReg();

    // Propagate extension type if any
    if (attr.hasAttribute(Attributes::SExt))
      Flags.setSExt();
    else if (attr.hasAttribute(Attributes::ZExt))
      Flags.setZExt();

    for (unsigned i = 0; i < NumParts; ++i)
      Outs.push_back(ISD::OutputArg(Flags, PartVT, /*isFixed=*/true, 0, 0));
  }
}

/// getByValTypeAlignment - Return the desired alignment for ByVal aggregate
/// function arguments in the caller parameter area.  This is the actual
/// alignment, not its logarithm.
unsigned TargetLowering::getByValTypeAlignment(Type *Ty) const {
  return TD->getCallFrameTypeAlignment(Ty);
}

/// getJumpTableEncoding - Return the entry encoding for a jump table in the
/// current function.  The returned value is a member of the
/// MachineJumpTableInfo::JTEntryKind enum.
unsigned TargetLowering::getJumpTableEncoding() const {
  // In non-pic modes, just use the address of a block.
  if (getTargetMachine().getRelocationModel() != Reloc::PIC_)
    return MachineJumpTableInfo::EK_BlockAddress;

  // In PIC mode, if the target supports a GPRel32 directive, use it.
  if (getTargetMachine().getMCAsmInfo()->getGPRel32Directive() != 0)
    return MachineJumpTableInfo::EK_GPRel32BlockAddress;

  // Otherwise, use a label difference.
  return MachineJumpTableInfo::EK_LabelDifference32;
}

SDValue TargetLowering::getPICJumpTableRelocBase(SDValue Table,
                                                 SelectionDAG &DAG) const {
  // If our PIC model is GP relative, use the global offset table as the base.
  unsigned JTEncoding = getJumpTableEncoding();

  if ((JTEncoding == MachineJumpTableInfo::EK_GPRel64BlockAddress) ||
      (JTEncoding == MachineJumpTableInfo::EK_GPRel32BlockAddress))
    return DAG.getGLOBAL_OFFSET_TABLE(getPointerTy(0));

  return Table;
}

/// getPICJumpTableRelocBaseExpr - This returns the relocation base for the
/// given PIC jumptable, the same as getPICJumpTableRelocBase, but as an
/// MCExpr.
const MCExpr *
TargetLowering::getPICJumpTableRelocBaseExpr(const MachineFunction *MF,
                                             unsigned JTI,MCContext &Ctx) const{
  // The normal PIC reloc base is the label at the start of the jump table.
  return MCSymbolRefExpr::Create(MF->getJTISymbol(JTI, Ctx), Ctx);
}

bool
TargetLowering::isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // Assume that everything is safe in static mode.
  if (getTargetMachine().getRelocationModel() == Reloc::Static)
    return true;

  // In dynamic-no-pic mode, assume that known defined values are safe.
  if (getTargetMachine().getRelocationModel() == Reloc::DynamicNoPIC &&
      GA &&
      !GA->getGlobal()->isDeclaration() &&
      !GA->getGlobal()->isWeakForLinker())
    return true;

  // Otherwise assume nothing is safe.
  return false;
}

//===----------------------------------------------------------------------===//
//  Optimization Methods
//===----------------------------------------------------------------------===//

/// ShrinkDemandedConstant - Check to see if the specified operand of the
/// specified instruction is a constant integer.  If so, check to see if there
/// are any bits set in the constant that are not demanded.  If so, shrink the
/// constant and return true.
bool TargetLowering::TargetLoweringOpt::ShrinkDemandedConstant(SDValue Op,
                                                        const APInt &Demanded) {
  DebugLoc dl = Op.getDebugLoc();

  // FIXME: ISD::SELECT, ISD::SELECT_CC
  switch (Op.getOpcode()) {
  default: break;
  case ISD::XOR:
  case ISD::AND:
  case ISD::OR: {
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
    if (!C) return false;

    if (Op.getOpcode() == ISD::XOR &&
        (C->getAPIntValue() | (~Demanded)).isAllOnesValue())
      return false;

    // if we can expand it to have all bits set, do it
    if (C->getAPIntValue().intersects(~Demanded)) {
      EVT VT = Op.getValueType();
      SDValue New = DAG.getNode(Op.getOpcode(), dl, VT, Op.getOperand(0),
                                DAG.getConstant(Demanded &
                                                C->getAPIntValue(),
                                                VT));
      return CombineTo(Op, New);
    }

    break;
  }
  }

  return false;
}

/// ShrinkDemandedOp - Convert x+y to (VT)((SmallVT)x+(SmallVT)y) if the
/// casts are free.  This uses isZExtFree and ZERO_EXTEND for the widening
/// cast, but it could be generalized for targets with other types of
/// implicit widening casts.
bool
TargetLowering::TargetLoweringOpt::ShrinkDemandedOp(SDValue Op,
                                                    unsigned BitWidth,
                                                    const APInt &Demanded,
                                                    DebugLoc dl) {
  assert(Op.getNumOperands() == 2 &&
         "ShrinkDemandedOp only supports binary operators!");
  assert(Op.getNode()->getNumValues() == 1 &&
         "ShrinkDemandedOp only supports nodes with one result!");

  // Don't do this if the node has another user, which may require the
  // full value.
  if (!Op.getNode()->hasOneUse())
    return false;

  // Search for the smallest integer type with free casts to and from
  // Op's type. For expedience, just check power-of-2 integer types.
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  unsigned SmallVTBits = BitWidth - Demanded.countLeadingZeros();
  if (!isPowerOf2_32(SmallVTBits))
    SmallVTBits = NextPowerOf2(SmallVTBits);
  for (; SmallVTBits < BitWidth; SmallVTBits = NextPowerOf2(SmallVTBits)) {
    EVT SmallVT = EVT::getIntegerVT(*DAG.getContext(), SmallVTBits);
    if (TLI.isTruncateFree(Op.getValueType(), SmallVT) &&
        TLI.isZExtFree(SmallVT, Op.getValueType())) {
      // We found a type with free casts.
      SDValue X = DAG.getNode(Op.getOpcode(), dl, SmallVT,
                              DAG.getNode(ISD::TRUNCATE, dl, SmallVT,
                                          Op.getNode()->getOperand(0)),
                              DAG.getNode(ISD::TRUNCATE, dl, SmallVT,
                                          Op.getNode()->getOperand(1)));
      SDValue Z = DAG.getNode(ISD::ZERO_EXTEND, dl, Op.getValueType(), X);
      return CombineTo(Op, Z);
    }
  }
  return false;
}

/// SimplifyDemandedBits - Look at Op.  At this point, we know that only the
/// DemandedMask bits of the result of Op are ever used downstream.  If we can
/// use this information to simplify Op, create a new simplified DAG node and
/// return true, returning the original and new nodes in Old and New. Otherwise,
/// analyze the expression and return a mask of KnownOne and KnownZero bits for
/// the expression (used to simplify the caller).  The KnownZero/One bits may
/// only be accurate for those bits in the DemandedMask.
bool TargetLowering::SimplifyDemandedBits(SDValue Op,
                                          const APInt &DemandedMask,
                                          APInt &KnownZero,
                                          APInt &KnownOne,
                                          TargetLoweringOpt &TLO,
                                          unsigned Depth) const {
  unsigned BitWidth = DemandedMask.getBitWidth();
  assert(Op.getValueType().getScalarType().getSizeInBits() == BitWidth &&
         "Mask size mismatches value type size!");
  APInt NewMask = DemandedMask;
  DebugLoc dl = Op.getDebugLoc();

  // Don't know anything.
  KnownZero = KnownOne = APInt(BitWidth, 0);

  // Other users may use these bits.
  if (!Op.getNode()->hasOneUse()) {
    if (Depth != 0) {
      // If not at the root, Just compute the KnownZero/KnownOne bits to
      // simplify things downstream.
      TLO.DAG.ComputeMaskedBits(Op, KnownZero, KnownOne, Depth);
      return false;
    }
    // If this is the root being simplified, allow it to have multiple uses,
    // just set the NewMask to all bits.
    NewMask = APInt::getAllOnesValue(BitWidth);
  } else if (DemandedMask == 0) {
    // Not demanding any bits from Op.
    if (Op.getOpcode() != ISD::UNDEF)
      return TLO.CombineTo(Op, TLO.DAG.getUNDEF(Op.getValueType()));
    return false;
  } else if (Depth == 6) {        // Limit search depth.
    return false;
  }

  APInt KnownZero2, KnownOne2, KnownZeroOut, KnownOneOut;
  switch (Op.getOpcode()) {
  case ISD::Constant:
    // We know all of the bits for a constant!
    KnownOne = cast<ConstantSDNode>(Op)->getAPIntValue();
    KnownZero = ~KnownOne;
    return false;   // Don't fall through, will infinitely loop.
  case ISD::AND:
    // If the RHS is a constant, check to see if the LHS would be zero without
    // using the bits from the RHS.  Below, we use knowledge about the RHS to
    // simplify the LHS, here we're using information from the LHS to simplify
    // the RHS.
    if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt LHSZero, LHSOne;
      // Do not increment Depth here; that can cause an infinite loop.
      TLO.DAG.ComputeMaskedBits(Op.getOperand(0), LHSZero, LHSOne, Depth);
      // If the LHS already has zeros where RHSC does, this and is dead.
      if ((LHSZero & NewMask) == (~RHSC->getAPIntValue() & NewMask))
        return TLO.CombineTo(Op, Op.getOperand(0));
      // If any of the set bits in the RHS are known zero on the LHS, shrink
      // the constant.
      if (TLO.ShrinkDemandedConstant(Op, ~LHSZero & NewMask))
        return true;
    }

    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    if (SimplifyDemandedBits(Op.getOperand(0), ~KnownZero & NewMask,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If all of the demanded bits are known one on one side, return the other.
    // These bits cannot contribute to the result of the 'and'.
    if ((NewMask & ~KnownZero2 & KnownOne) == (~KnownZero2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownZero & KnownOne2) == (~KnownZero & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the demanded bits in the inputs are known zeros, return zero.
    if ((NewMask & (KnownZero|KnownZero2)) == NewMask)
      return TLO.CombineTo(Op, TLO.DAG.getConstant(0, Op.getValueType()));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, ~KnownZero2 & NewMask))
      return true;
    // If the operation can be done in a smaller type, do so.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;

    // Output known-1 bits are only known if set in both the LHS & RHS.
    KnownOne &= KnownOne2;
    // Output known-0 are known to be clear if zero in either the LHS | RHS.
    KnownZero |= KnownZero2;
    break;
  case ISD::OR:
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    if (SimplifyDemandedBits(Op.getOperand(0), ~KnownOne & NewMask,
                             KnownZero2, KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'or'.
    if ((NewMask & ~KnownOne2 & KnownZero) == (~KnownOne2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownOne & KnownZero2) == (~KnownOne & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If all of the potentially set bits on one side are known to be set on
    // the other side, just use the 'other' side.
    if ((NewMask & ~KnownZero & KnownOne2) == (~KnownZero & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((NewMask & ~KnownZero2 & KnownOne) == (~KnownZero2 & NewMask))
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the RHS is a constant, see if we can simplify it.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;
    // If the operation can be done in a smaller type, do so.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;

    // Output known-0 bits are only known if clear in both the LHS & RHS.
    KnownZero &= KnownZero2;
    // Output known-1 are known to be set if set in either the LHS | RHS.
    KnownOne |= KnownOne2;
    break;
  case ISD::XOR:
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    if (SimplifyDemandedBits(Op.getOperand(0), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If all of the demanded bits are known zero on one side, return the other.
    // These bits cannot contribute to the result of the 'xor'.
    if ((KnownZero & NewMask) == NewMask)
      return TLO.CombineTo(Op, Op.getOperand(0));
    if ((KnownZero2 & NewMask) == NewMask)
      return TLO.CombineTo(Op, Op.getOperand(1));
    // If the operation can be done in a smaller type, do so.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;

    // If all of the unknown bits are known to be zero on one side or the other
    // (but not both) turn this into an *inclusive* or.
    //    e.g. (A & C1)^(B & C2) -> (A & C1)|(B & C2) iff C1&C2 == 0
    if ((NewMask & ~KnownZero & ~KnownZero2) == 0)
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::OR, dl, Op.getValueType(),
                                               Op.getOperand(0),
                                               Op.getOperand(1)));

    // Output known-0 bits are known if clear or set in both the LHS & RHS.
    KnownZeroOut = (KnownZero & KnownZero2) | (KnownOne & KnownOne2);
    // Output known-1 are known to be set if set in only one of the LHS, RHS.
    KnownOneOut = (KnownZero & KnownOne2) | (KnownOne & KnownZero2);

    // If all of the demanded bits on one side are known, and all of the set
    // bits on that side are also known to be set on the other side, turn this
    // into an AND, as we know the bits will be cleared.
    //    e.g. (X | C1) ^ C2 --> (X | C1) & ~C2 iff (C1&C2) == C2
    // NB: it is okay if more bits are known than are requested
    if ((NewMask & (KnownZero|KnownOne)) == NewMask) { // all known on one side 
      if (KnownOne == KnownOne2) { // set bits are the same on both sides
        EVT VT = Op.getValueType();
        SDValue ANDC = TLO.DAG.getConstant(~KnownOne & NewMask, VT);
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::AND, dl, VT,
                                                 Op.getOperand(0), ANDC));
      }
    }

    // If the RHS is a constant, see if we can simplify it.
    // for XOR, we prefer to force bits to 1 if they will make a -1.
    // if we can't force bits, try to shrink constant
    if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      APInt Expanded = C->getAPIntValue() | (~NewMask);
      // if we can expand it to have all bits set, do it
      if (Expanded.isAllOnesValue()) {
        if (Expanded != C->getAPIntValue()) {
          EVT VT = Op.getValueType();
          SDValue New = TLO.DAG.getNode(Op.getOpcode(), dl,VT, Op.getOperand(0),
                                          TLO.DAG.getConstant(Expanded, VT));
          return TLO.CombineTo(Op, New);
        }
        // if it already has all the bits set, nothing to change
        // but don't shrink either!
      } else if (TLO.ShrinkDemandedConstant(Op, NewMask)) {
        return true;
      }
    }

    KnownZero = KnownZeroOut;
    KnownOne  = KnownOneOut;
    break;
  case ISD::SELECT:
    if (SimplifyDemandedBits(Op.getOperand(2), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SELECT_CC:
    if (SimplifyDemandedBits(Op.getOperand(3), NewMask, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(2), NewMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    assert((KnownZero2 & KnownOne2) == 0 && "Bits known to be one AND zero?");

    // If the operands are constants, see if we can simplify them.
    if (TLO.ShrinkDemandedConstant(Op, NewMask))
      return true;

    // Only known if known in both the LHS and RHS.
    KnownOne &= KnownOne2;
    KnownZero &= KnownZero2;
    break;
  case ISD::SHL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      unsigned ShAmt = SA->getZExtValue();
      SDValue InOp = Op.getOperand(0);

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      // If this is ((X >>u C1) << ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the bottom bits (which are shifted
      // out) are never demanded.
      if (InOp.getOpcode() == ISD::SRL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (NewMask & APInt::getLowBitsSet(BitWidth, ShAmt)) == 0) {
          unsigned C1= cast<ConstantSDNode>(InOp.getOperand(1))->getZExtValue();
          unsigned Opc = ISD::SHL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SRL;
          }

          SDValue NewSA =
            TLO.DAG.getConstant(Diff, Op.getOperand(1).getValueType());
          EVT VT = Op.getValueType();
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, dl, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }

      if (SimplifyDemandedBits(InOp, NewMask.lshr(ShAmt),
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;

      // Convert (shl (anyext x, c)) to (anyext (shl x, c)) if the high bits
      // are not demanded. This will likely allow the anyext to be folded away.
      if (InOp.getNode()->getOpcode() == ISD::ANY_EXTEND) {
        SDValue InnerOp = InOp.getNode()->getOperand(0);
        EVT InnerVT = InnerOp.getValueType();
        unsigned InnerBits = InnerVT.getSizeInBits();
        if (ShAmt < InnerBits && NewMask.lshr(InnerBits) == 0 &&
            isTypeDesirableForOp(ISD::SHL, InnerVT)) {
          EVT ShTy = getShiftAmountTy(InnerVT);
          if (!APInt(BitWidth, ShAmt).isIntN(ShTy.getSizeInBits()))
            ShTy = InnerVT;
          SDValue NarrowShl =
            TLO.DAG.getNode(ISD::SHL, dl, InnerVT, InnerOp,
                            TLO.DAG.getConstant(ShAmt, ShTy));
          return
            TLO.CombineTo(Op,
                          TLO.DAG.getNode(ISD::ANY_EXTEND, dl, Op.getValueType(),
                                          NarrowShl));
        }
      }

      KnownZero <<= SA->getZExtValue();
      KnownOne  <<= SA->getZExtValue();
      // low bits known zero.
      KnownZero |= APInt::getLowBitsSet(BitWidth, SA->getZExtValue());
    }
    break;
  case ISD::SRL:
    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      EVT VT = Op.getValueType();
      unsigned ShAmt = SA->getZExtValue();
      unsigned VTSize = VT.getSizeInBits();
      SDValue InOp = Op.getOperand(0);

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      // If this is ((X << C1) >>u ShAmt), see if we can simplify this into a
      // single shift.  We can do this if the top bits (which are shifted out)
      // are never demanded.
      if (InOp.getOpcode() == ISD::SHL &&
          isa<ConstantSDNode>(InOp.getOperand(1))) {
        if (ShAmt && (NewMask & APInt::getHighBitsSet(VTSize, ShAmt)) == 0) {
          unsigned C1= cast<ConstantSDNode>(InOp.getOperand(1))->getZExtValue();
          unsigned Opc = ISD::SRL;
          int Diff = ShAmt-C1;
          if (Diff < 0) {
            Diff = -Diff;
            Opc = ISD::SHL;
          }

          SDValue NewSA =
            TLO.DAG.getConstant(Diff, Op.getOperand(1).getValueType());
          return TLO.CombineTo(Op, TLO.DAG.getNode(Opc, dl, VT,
                                                   InOp.getOperand(0), NewSA));
        }
      }

      // Compute the new bits that are at the top now.
      if (SimplifyDemandedBits(InOp, (NewMask << ShAmt),
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      KnownZero |= HighBits;  // High bits known zero.
    }
    break;
  case ISD::SRA:
    // If this is an arithmetic shift right and only the low-bit is set, we can
    // always convert this into a logical shr, even if the shift amount is
    // variable.  The low bit of the shift cannot be an input sign bit unless
    // the shift amount is >= the size of the datatype, which is undefined.
    if (NewMask == 1)
      return TLO.CombineTo(Op,
                           TLO.DAG.getNode(ISD::SRL, dl, Op.getValueType(),
                                           Op.getOperand(0), Op.getOperand(1)));

    if (ConstantSDNode *SA = dyn_cast<ConstantSDNode>(Op.getOperand(1))) {
      EVT VT = Op.getValueType();
      unsigned ShAmt = SA->getZExtValue();

      // If the shift count is an invalid immediate, don't do anything.
      if (ShAmt >= BitWidth)
        break;

      APInt InDemandedMask = (NewMask << ShAmt);

      // If any of the demanded bits are produced by the sign extension, we also
      // demand the input sign bit.
      APInt HighBits = APInt::getHighBitsSet(BitWidth, ShAmt);
      if (HighBits.intersects(NewMask))
        InDemandedMask |= APInt::getSignBit(VT.getScalarType().getSizeInBits());

      if (SimplifyDemandedBits(Op.getOperand(0), InDemandedMask,
                               KnownZero, KnownOne, TLO, Depth+1))
        return true;
      assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
      KnownZero = KnownZero.lshr(ShAmt);
      KnownOne  = KnownOne.lshr(ShAmt);

      // Handle the sign bit, adjusted to where it is now in the mask.
      APInt SignBit = APInt::getSignBit(BitWidth).lshr(ShAmt);

      // If the input sign bit is known to be zero, or if none of the top bits
      // are demanded, turn this into an unsigned shift right.
      if (KnownZero.intersects(SignBit) || (HighBits & ~NewMask) == HighBits) {
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, dl, VT,
                                                 Op.getOperand(0),
                                                 Op.getOperand(1)));
      } else if (KnownOne.intersects(SignBit)) { // New bits are known one.
        KnownOne |= HighBits;
      }
    }
    break;
  case ISD::SIGN_EXTEND_INREG: {
    EVT ExVT = cast<VTSDNode>(Op.getOperand(1))->getVT();

    APInt MsbMask = APInt::getHighBitsSet(BitWidth, 1);
    // If we only care about the highest bit, don't bother shifting right.
    if (MsbMask == DemandedMask) {
      unsigned ShAmt = ExVT.getScalarType().getSizeInBits();
      SDValue InOp = Op.getOperand(0);

      // Compute the correct shift amount type, which must be getShiftAmountTy
      // for scalar types after legalization.
      EVT ShiftAmtTy = Op.getValueType();
      if (TLO.LegalTypes() && !ShiftAmtTy.isVector())
        ShiftAmtTy = getShiftAmountTy(ShiftAmtTy);

      SDValue ShiftAmt = TLO.DAG.getConstant(BitWidth - ShAmt, ShiftAmtTy);
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SHL, dl,
                                            Op.getValueType(), InOp, ShiftAmt));
    }

    // Sign extension.  Compute the demanded bits in the result that are not
    // present in the input.
    APInt NewBits =
      APInt::getHighBitsSet(BitWidth,
                            BitWidth - ExVT.getScalarType().getSizeInBits());

    // If none of the extended bits are demanded, eliminate the sextinreg.
    if ((NewBits & NewMask) == 0)
      return TLO.CombineTo(Op, Op.getOperand(0));

    APInt InSignBit =
      APInt::getSignBit(ExVT.getScalarType().getSizeInBits()).zext(BitWidth);
    APInt InputDemandedBits =
      APInt::getLowBitsSet(BitWidth,
                           ExVT.getScalarType().getSizeInBits()) &
      NewMask;

    // Since the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    InputDemandedBits |= InSignBit;

    if (SimplifyDemandedBits(Op.getOperand(0), InputDemandedBits,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");

    // If the sign bit of the input is known set or clear, then we know the
    // top bits of the result.

    // If the input sign bit is known zero, convert this into a zero extension.
    if (KnownZero.intersects(InSignBit))
      return TLO.CombineTo(Op,
                          TLO.DAG.getZeroExtendInReg(Op.getOperand(0),dl,ExVT));

    if (KnownOne.intersects(InSignBit)) {    // Input sign bit known set
      KnownOne |= NewBits;
      KnownZero &= ~NewBits;
    } else {                       // Input sign bit unknown
      KnownZero &= ~NewBits;
      KnownOne &= ~NewBits;
    }
    break;
  }
  case ISD::ZERO_EXTEND: {
    unsigned OperandBitWidth =
      Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    APInt InMask = NewMask.trunc(OperandBitWidth);

    // If none of the top bits are demanded, convert this into an any_extend.
    APInt NewBits =
      APInt::getHighBitsSet(BitWidth, BitWidth - OperandBitWidth) & NewMask;
    if (!NewBits.intersects(NewMask))
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ANY_EXTEND, dl,
                                               Op.getValueType(),
                                               Op.getOperand(0)));

    if (SimplifyDemandedBits(Op.getOperand(0), InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    KnownZero |= NewBits;
    break;
  }
  case ISD::SIGN_EXTEND: {
    EVT InVT = Op.getOperand(0).getValueType();
    unsigned InBits = InVT.getScalarType().getSizeInBits();
    APInt InMask    = APInt::getLowBitsSet(BitWidth, InBits);
    APInt InSignBit = APInt::getBitsSet(BitWidth, InBits - 1, InBits);
    APInt NewBits   = ~InMask & NewMask;

    // If none of the top bits are demanded, convert this into an any_extend.
    if (NewBits == 0)
      return TLO.CombineTo(Op,TLO.DAG.getNode(ISD::ANY_EXTEND, dl,
                                              Op.getValueType(),
                                              Op.getOperand(0)));

    // Since some of the sign extended bits are demanded, we know that the sign
    // bit is demanded.
    APInt InDemandedBits = InMask & NewMask;
    InDemandedBits |= InSignBit;
    InDemandedBits = InDemandedBits.trunc(InBits);

    if (SimplifyDemandedBits(Op.getOperand(0), InDemandedBits, KnownZero,
                             KnownOne, TLO, Depth+1))
      return true;
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);

    // If the sign bit is known zero, convert this to a zero extend.
    if (KnownZero.intersects(InSignBit))
      return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::ZERO_EXTEND, dl,
                                               Op.getValueType(),
                                               Op.getOperand(0)));

    // If the sign bit is known one, the top bits match.
    if (KnownOne.intersects(InSignBit)) {
      KnownOne |= NewBits;
      assert((KnownZero & NewBits) == 0);
    } else {   // Otherwise, top bits aren't known.
      assert((KnownOne & NewBits) == 0);
      assert((KnownZero & NewBits) == 0);
    }
    break;
  }
  case ISD::ANY_EXTEND: {
    unsigned OperandBitWidth =
      Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    APInt InMask = NewMask.trunc(OperandBitWidth);
    if (SimplifyDemandedBits(Op.getOperand(0), InMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    KnownZero = KnownZero.zext(BitWidth);
    KnownOne = KnownOne.zext(BitWidth);
    break;
  }
  case ISD::TRUNCATE: {
    // Simplify the input, using demanded bit information, and compute the known
    // zero/one bits live out.
    unsigned OperandBitWidth =
      Op.getOperand(0).getValueType().getScalarType().getSizeInBits();
    APInt TruncMask = NewMask.zext(OperandBitWidth);
    if (SimplifyDemandedBits(Op.getOperand(0), TruncMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    KnownZero = KnownZero.trunc(BitWidth);
    KnownOne = KnownOne.trunc(BitWidth);

    // If the input is only used by this truncate, see if we can shrink it based
    // on the known demanded bits.
    if (Op.getOperand(0).getNode()->hasOneUse()) {
      SDValue In = Op.getOperand(0);
      switch (In.getOpcode()) {
      default: break;
      case ISD::SRL:
        // Shrink SRL by a constant if none of the high bits shifted in are
        // demanded.
        if (TLO.LegalTypes() &&
            !isTypeDesirableForOp(ISD::SRL, Op.getValueType()))
          // Do not turn (vt1 truncate (vt2 srl)) into (vt1 srl) if vt1 is
          // undesirable.
          break;
        ConstantSDNode *ShAmt = dyn_cast<ConstantSDNode>(In.getOperand(1));
        if (!ShAmt)
          break;
        SDValue Shift = In.getOperand(1);
        if (TLO.LegalTypes()) {
          uint64_t ShVal = ShAmt->getZExtValue();
          Shift =
            TLO.DAG.getConstant(ShVal, getShiftAmountTy(Op.getValueType()));
        }

        APInt HighBits = APInt::getHighBitsSet(OperandBitWidth,
                                               OperandBitWidth - BitWidth);
        HighBits = HighBits.lshr(ShAmt->getZExtValue()).trunc(BitWidth);

        if (ShAmt->getZExtValue() < BitWidth && !(HighBits & NewMask)) {
          // None of the shifted in bits are needed.  Add a truncate of the
          // shift input, then shift it.
          SDValue NewTrunc = TLO.DAG.getNode(ISD::TRUNCATE, dl,
                                             Op.getValueType(),
                                             In.getOperand(0));
          return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SRL, dl,
                                                   Op.getValueType(),
                                                   NewTrunc,
                                                   Shift));
        }
        break;
      }
    }

    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");
    break;
  }
  case ISD::AssertZext: {
    // AssertZext demands all of the high bits, plus any of the low bits
    // demanded by its users.
    EVT VT = cast<VTSDNode>(Op.getOperand(1))->getVT();
    APInt InMask = APInt::getLowBitsSet(BitWidth,
                                        VT.getSizeInBits());
    if (SimplifyDemandedBits(Op.getOperand(0), ~InMask | NewMask,
                             KnownZero, KnownOne, TLO, Depth+1))
      return true;
    assert((KnownZero & KnownOne) == 0 && "Bits known to be one AND zero?");

    KnownZero |= ~InMask & NewMask;
    break;
  }
  case ISD::BITCAST:
    // If this is an FP->Int bitcast and if the sign bit is the only
    // thing demanded, turn this into a FGETSIGN.
    if (!TLO.LegalOperations() &&
        !Op.getValueType().isVector() &&
        !Op.getOperand(0).getValueType().isVector() &&
        NewMask == APInt::getSignBit(Op.getValueType().getSizeInBits()) &&
        Op.getOperand(0).getValueType().isFloatingPoint()) {
      bool OpVTLegal = isOperationLegalOrCustom(ISD::FGETSIGN, Op.getValueType());
      bool i32Legal  = isOperationLegalOrCustom(ISD::FGETSIGN, MVT::i32);
      if ((OpVTLegal || i32Legal) && Op.getValueType().isSimple()) {
        EVT Ty = OpVTLegal ? Op.getValueType() : MVT::i32;
        // Make a FGETSIGN + SHL to move the sign bit into the appropriate
        // place.  We expect the SHL to be eliminated by other optimizations.
        SDValue Sign = TLO.DAG.getNode(ISD::FGETSIGN, dl, Ty, Op.getOperand(0));
        unsigned OpVTSizeInBits = Op.getValueType().getSizeInBits();
        if (!OpVTLegal && OpVTSizeInBits > 32)
          Sign = TLO.DAG.getNode(ISD::ZERO_EXTEND, dl, Op.getValueType(), Sign);
        unsigned ShVal = Op.getValueType().getSizeInBits()-1;
        SDValue ShAmt = TLO.DAG.getConstant(ShVal, Op.getValueType());
        return TLO.CombineTo(Op, TLO.DAG.getNode(ISD::SHL, dl,
                                                 Op.getValueType(),
                                                 Sign, ShAmt));
      }
    }
    break;
  case ISD::ADD:
  case ISD::MUL:
  case ISD::SUB: {
    // Add, Sub, and Mul don't demand any bits in positions beyond that
    // of the highest bit demanded of them.
    APInt LoMask = APInt::getLowBitsSet(BitWidth,
                                        BitWidth - NewMask.countLeadingZeros());
    if (SimplifyDemandedBits(Op.getOperand(0), LoMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    if (SimplifyDemandedBits(Op.getOperand(1), LoMask, KnownZero2,
                             KnownOne2, TLO, Depth+1))
      return true;
    // See if the operation should be performed at a smaller bit width.
    if (TLO.ShrinkDemandedOp(Op, BitWidth, NewMask, dl))
      return true;
  }
  // FALL THROUGH
  default:
    // Just use ComputeMaskedBits to compute output bits.
    TLO.DAG.ComputeMaskedBits(Op, KnownZero, KnownOne, Depth);
    break;
  }

  // If we know the value of all of the demanded bits, return this as a
  // constant.
  if ((NewMask & (KnownZero|KnownOne)) == NewMask)
    return TLO.CombineTo(Op, TLO.DAG.getConstant(KnownOne, Op.getValueType()));

  return false;
}

/// computeMaskedBitsForTargetNode - Determine which of the bits specified
/// in Mask are known to be either zero or one and return them in the
/// KnownZero/KnownOne bitsets.
void TargetLowering::computeMaskedBitsForTargetNode(const SDValue Op,
                                                    APInt &KnownZero,
                                                    APInt &KnownOne,
                                                    const SelectionDAG &DAG,
                                                    unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use MaskedValueIsZero if you don't know whether Op"
         " is a target node!");
  KnownZero = KnownOne = APInt(KnownOne.getBitWidth(), 0);
}

/// ComputeNumSignBitsForTargetNode - This method can be implemented by
/// targets that want to expose additional information about sign bits to the
/// DAG Combiner.
unsigned TargetLowering::ComputeNumSignBitsForTargetNode(SDValue Op,
                                                         unsigned Depth) const {
  assert((Op.getOpcode() >= ISD::BUILTIN_OP_END ||
          Op.getOpcode() == ISD::INTRINSIC_WO_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_W_CHAIN ||
          Op.getOpcode() == ISD::INTRINSIC_VOID) &&
         "Should use ComputeNumSignBits if you don't know whether Op"
         " is a target node!");
  return 1;
}

/// ValueHasExactlyOneBitSet - Test if the given value is known to have exactly
/// one bit set. This differs from ComputeMaskedBits in that it doesn't need to
/// determine which bit is set.
///
static bool ValueHasExactlyOneBitSet(SDValue Val, const SelectionDAG &DAG) {
  // A left-shift of a constant one will have exactly one bit set, because
  // shifting the bit off the end is undefined.
  if (Val.getOpcode() == ISD::SHL)
    if (ConstantSDNode *C =
         dyn_cast<ConstantSDNode>(Val.getNode()->getOperand(0)))
      if (C->getAPIntValue() == 1)
        return true;

  // Similarly, a right-shift of a constant sign-bit will have exactly
  // one bit set.
  if (Val.getOpcode() == ISD::SRL)
    if (ConstantSDNode *C =
         dyn_cast<ConstantSDNode>(Val.getNode()->getOperand(0)))
      if (C->getAPIntValue().isSignBit())
        return true;

  // More could be done here, though the above checks are enough
  // to handle some common cases.

  // Fall back to ComputeMaskedBits to catch other known cases.
  EVT OpVT = Val.getValueType();
  unsigned BitWidth = OpVT.getScalarType().getSizeInBits();
  APInt KnownZero, KnownOne;
  DAG.ComputeMaskedBits(Val, KnownZero, KnownOne);
  return (KnownZero.countPopulation() == BitWidth - 1) &&
         (KnownOne.countPopulation() == 1);
}

/// SimplifySetCC - Try to simplify a setcc built with the specified operands
/// and cc. If it is unable to simplify it, return a null SDValue.
SDValue
TargetLowering::SimplifySetCC(EVT VT, SDValue N0, SDValue N1,
                              ISD::CondCode Cond, bool foldBooleans,
                              DAGCombinerInfo &DCI, DebugLoc dl) const {
  SelectionDAG &DAG = DCI.DAG;

  // These setcc operations always fold.
  switch (Cond) {
  default: break;
  case ISD::SETFALSE:
  case ISD::SETFALSE2: return DAG.getConstant(0, VT);
  case ISD::SETTRUE:
  case ISD::SETTRUE2:  return DAG.getConstant(1, VT);
  }

  // Ensure that the constant occurs on the RHS, and fold constant
  // comparisons.
  if (isa<ConstantSDNode>(N0.getNode()))
    return DAG.getSetCC(dl, VT, N1, N0, ISD::getSetCCSwappedOperands(Cond));

  if (ConstantSDNode *N1C = dyn_cast<ConstantSDNode>(N1.getNode())) {
    const APInt &C1 = N1C->getAPIntValue();

    // If the LHS is '(srl (ctlz x), 5)', the RHS is 0/1, and this is an
    // equality comparison, then we're just comparing whether X itself is
    // zero.
    if (N0.getOpcode() == ISD::SRL && (C1 == 0 || C1 == 1) &&
        N0.getOperand(0).getOpcode() == ISD::CTLZ &&
        N0.getOperand(1).getOpcode() == ISD::Constant) {
      const APInt &ShAmt
        = cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          ShAmt == Log2_32(N0.getValueType().getSizeInBits())) {
        if ((C1 == 0) == (Cond == ISD::SETEQ)) {
          // (srl (ctlz x), 5) == 0  -> X != 0
          // (srl (ctlz x), 5) != 1  -> X != 0
          Cond = ISD::SETNE;
        } else {
          // (srl (ctlz x), 5) != 0  -> X == 0
          // (srl (ctlz x), 5) == 1  -> X == 0
          Cond = ISD::SETEQ;
        }
        SDValue Zero = DAG.getConstant(0, N0.getValueType());
        return DAG.getSetCC(dl, VT, N0.getOperand(0).getOperand(0),
                            Zero, Cond);
      }
    }

    SDValue CTPOP = N0;
    // Look through truncs that don't change the value of a ctpop.
    if (N0.hasOneUse() && N0.getOpcode() == ISD::TRUNCATE)
      CTPOP = N0.getOperand(0);

    if (CTPOP.hasOneUse() && CTPOP.getOpcode() == ISD::CTPOP &&
        (N0 == CTPOP || N0.getValueType().getSizeInBits() >
                        Log2_32_Ceil(CTPOP.getValueType().getSizeInBits()))) {
      EVT CTVT = CTPOP.getValueType();
      SDValue CTOp = CTPOP.getOperand(0);

      // (ctpop x) u< 2 -> (x & x-1) == 0
      // (ctpop x) u> 1 -> (x & x-1) != 0
      if ((Cond == ISD::SETULT && C1 == 2) || (Cond == ISD::SETUGT && C1 == 1)){
        SDValue Sub = DAG.getNode(ISD::SUB, dl, CTVT, CTOp,
                                  DAG.getConstant(1, CTVT));
        SDValue And = DAG.getNode(ISD::AND, dl, CTVT, CTOp, Sub);
        ISD::CondCode CC = Cond == ISD::SETULT ? ISD::SETEQ : ISD::SETNE;
        return DAG.getSetCC(dl, VT, And, DAG.getConstant(0, CTVT), CC);
      }

      // TODO: (ctpop x) == 1 -> x && (x & x-1) == 0 iff ctpop is illegal.
    }

    // (zext x) == C --> x == (trunc C)
    if (DCI.isBeforeLegalize() && N0->hasOneUse() &&
        (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
      unsigned MinBits = N0.getValueSizeInBits();
      SDValue PreZExt;
      if (N0->getOpcode() == ISD::ZERO_EXTEND) {
        // ZExt
        MinBits = N0->getOperand(0).getValueSizeInBits();
        PreZExt = N0->getOperand(0);
      } else if (N0->getOpcode() == ISD::AND) {
        // DAGCombine turns costly ZExts into ANDs
        if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(N0->getOperand(1)))
          if ((C->getAPIntValue()+1).isPowerOf2()) {
            MinBits = C->getAPIntValue().countTrailingOnes();
            PreZExt = N0->getOperand(0);
          }
      } else if (LoadSDNode *LN0 = dyn_cast<LoadSDNode>(N0)) {
        // ZEXTLOAD
        if (LN0->getExtensionType() == ISD::ZEXTLOAD) {
          MinBits = LN0->getMemoryVT().getSizeInBits();
          PreZExt = N0;
        }
      }

      // Make sure we're not losing bits from the constant.
      if (MinBits < C1.getBitWidth() && MinBits > C1.getActiveBits()) {
        EVT MinVT = EVT::getIntegerVT(*DAG.getContext(), MinBits);
        if (isTypeDesirableForOp(ISD::SETCC, MinVT)) {
          // Will get folded away.
          SDValue Trunc = DAG.getNode(ISD::TRUNCATE, dl, MinVT, PreZExt);
          SDValue C = DAG.getConstant(C1.trunc(MinBits), MinVT);
          return DAG.getSetCC(dl, VT, Trunc, C, Cond);
        }
      }
    }

    // If the LHS is '(and load, const)', the RHS is 0,
    // the test is for equality or unsigned, and all 1 bits of the const are
    // in the same partial word, see if we can shorten the load.
    if (DCI.isBeforeLegalize() &&
        N0.getOpcode() == ISD::AND && C1 == 0 &&
        N0.getNode()->hasOneUse() &&
        isa<LoadSDNode>(N0.getOperand(0)) &&
        N0.getOperand(0).getNode()->hasOneUse() &&
        isa<ConstantSDNode>(N0.getOperand(1))) {
      LoadSDNode *Lod = cast<LoadSDNode>(N0.getOperand(0));
      APInt bestMask;
      unsigned bestWidth = 0, bestOffset = 0;
      if (!Lod->isVolatile() && Lod->isUnindexed()) {
        unsigned origWidth = N0.getValueType().getSizeInBits();
        unsigned maskWidth = origWidth;
        // We can narrow (e.g.) 16-bit extending loads on 32-bit target to
        // 8 bits, but have to be careful...
        if (Lod->getExtensionType() != ISD::NON_EXTLOAD)
          origWidth = Lod->getMemoryVT().getSizeInBits();
        const APInt &Mask =
          cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue();
        for (unsigned width = origWidth / 2; width>=8; width /= 2) {
          APInt newMask = APInt::getLowBitsSet(maskWidth, width);
          for (unsigned offset=0; offset<origWidth/width; offset++) {
            if ((newMask & Mask) == Mask) {
              if (!TD->isLittleEndian())
                bestOffset = (origWidth/width - offset - 1) * (width/8);
              else
                bestOffset = (uint64_t)offset * (width/8);
              bestMask = Mask.lshr(offset * (width/8) * 8);
              bestWidth = width;
              break;
            }
            newMask = newMask << width;
          }
        }
      }
      if (bestWidth) {
        EVT newVT = EVT::getIntegerVT(*DAG.getContext(), bestWidth);
        if (newVT.isRound()) {
          EVT PtrType = Lod->getOperand(1).getValueType();
          SDValue Ptr = Lod->getBasePtr();
          if (bestOffset != 0)
            Ptr = DAG.getNode(ISD::ADD, dl, PtrType, Lod->getBasePtr(),
                              DAG.getConstant(bestOffset, PtrType));
          unsigned NewAlign = MinAlign(Lod->getAlignment(), bestOffset);
          SDValue NewLoad = DAG.getLoad(newVT, dl, Lod->getChain(), Ptr,
                                Lod->getPointerInfo().getWithOffset(bestOffset),
                                        false, false, false, NewAlign);
          return DAG.getSetCC(dl, VT,
                              DAG.getNode(ISD::AND, dl, newVT, NewLoad,
                                      DAG.getConstant(bestMask.trunc(bestWidth),
                                                      newVT)),
                              DAG.getConstant(0LL, newVT), Cond);
        }
      }
    }

    // If the LHS is a ZERO_EXTEND, perform the comparison on the input.
    if (N0.getOpcode() == ISD::ZERO_EXTEND) {
      unsigned InSize = N0.getOperand(0).getValueType().getSizeInBits();

      // If the comparison constant has bits in the upper part, the
      // zero-extended value could never match.
      if (C1.intersects(APInt::getHighBitsSet(C1.getBitWidth(),
                                              C1.getBitWidth() - InSize))) {
        switch (Cond) {
        case ISD::SETUGT:
        case ISD::SETUGE:
        case ISD::SETEQ: return DAG.getConstant(0, VT);
        case ISD::SETULT:
        case ISD::SETULE:
        case ISD::SETNE: return DAG.getConstant(1, VT);
        case ISD::SETGT:
        case ISD::SETGE:
          // True if the sign bit of C1 is set.
          return DAG.getConstant(C1.isNegative(), VT);
        case ISD::SETLT:
        case ISD::SETLE:
          // True if the sign bit of C1 isn't set.
          return DAG.getConstant(C1.isNonNegative(), VT);
        default:
          break;
        }
      }

      // Otherwise, we can perform the comparison with the low bits.
      switch (Cond) {
      case ISD::SETEQ:
      case ISD::SETNE:
      case ISD::SETUGT:
      case ISD::SETUGE:
      case ISD::SETULT:
      case ISD::SETULE: {
        EVT newVT = N0.getOperand(0).getValueType();
        if (DCI.isBeforeLegalizeOps() ||
            (isOperationLegal(ISD::SETCC, newVT) &&
              getCondCodeAction(Cond, newVT)==Legal))
          return DAG.getSetCC(dl, VT, N0.getOperand(0),
                              DAG.getConstant(C1.trunc(InSize), newVT),
                              Cond);
        break;
      }
      default:
        break;   // todo, be more careful with signed comparisons
      }
    } else if (N0.getOpcode() == ISD::SIGN_EXTEND_INREG &&
               (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
      EVT ExtSrcTy = cast<VTSDNode>(N0.getOperand(1))->getVT();
      unsigned ExtSrcTyBits = ExtSrcTy.getSizeInBits();
      EVT ExtDstTy = N0.getValueType();
      unsigned ExtDstTyBits = ExtDstTy.getSizeInBits();

      // If the constant doesn't fit into the number of bits for the source of
      // the sign extension, it is impossible for both sides to be equal.
      if (C1.getMinSignedBits() > ExtSrcTyBits)
        return DAG.getConstant(Cond == ISD::SETNE, VT);

      SDValue ZextOp;
      EVT Op0Ty = N0.getOperand(0).getValueType();
      if (Op0Ty == ExtSrcTy) {
        ZextOp = N0.getOperand(0);
      } else {
        APInt Imm = APInt::getLowBitsSet(ExtDstTyBits, ExtSrcTyBits);
        ZextOp = DAG.getNode(ISD::AND, dl, Op0Ty, N0.getOperand(0),
                              DAG.getConstant(Imm, Op0Ty));
      }
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(ZextOp.getNode());
      // Otherwise, make this a use of a zext.
      return DAG.getSetCC(dl, VT, ZextOp,
                          DAG.getConstant(C1 & APInt::getLowBitsSet(
                                                              ExtDstTyBits,
                                                              ExtSrcTyBits),
                                          ExtDstTy),
                          Cond);
    } else if ((N1C->isNullValue() || N1C->getAPIntValue() == 1) &&
                (Cond == ISD::SETEQ || Cond == ISD::SETNE)) {
      // SETCC (SETCC), [0|1], [EQ|NE]  -> SETCC
      if (N0.getOpcode() == ISD::SETCC &&
          isTypeLegal(VT) && VT.bitsLE(N0.getValueType())) {
        bool TrueWhenTrue = (Cond == ISD::SETEQ) ^ (N1C->getAPIntValue() != 1);
        if (TrueWhenTrue)
          return DAG.getNode(ISD::TRUNCATE, dl, VT, N0);
        // Invert the condition.
        ISD::CondCode CC = cast<CondCodeSDNode>(N0.getOperand(2))->get();
        CC = ISD::getSetCCInverse(CC,
                                  N0.getOperand(0).getValueType().isInteger());
        return DAG.getSetCC(dl, VT, N0.getOperand(0), N0.getOperand(1), CC);
      }

      if ((N0.getOpcode() == ISD::XOR ||
           (N0.getOpcode() == ISD::AND &&
            N0.getOperand(0).getOpcode() == ISD::XOR &&
            N0.getOperand(1) == N0.getOperand(0).getOperand(1))) &&
          isa<ConstantSDNode>(N0.getOperand(1)) &&
          cast<ConstantSDNode>(N0.getOperand(1))->getAPIntValue() == 1) {
        // If this is (X^1) == 0/1, swap the RHS and eliminate the xor.  We
        // can only do this if the top bits are known zero.
        unsigned BitWidth = N0.getValueSizeInBits();
        if (DAG.MaskedValueIsZero(N0,
                                  APInt::getHighBitsSet(BitWidth,
                                                        BitWidth-1))) {
          // Okay, get the un-inverted input value.
          SDValue Val;
          if (N0.getOpcode() == ISD::XOR)
            Val = N0.getOperand(0);
          else {
            assert(N0.getOpcode() == ISD::AND &&
                    N0.getOperand(0).getOpcode() == ISD::XOR);
            // ((X^1)&1)^1 -> X & 1
            Val = DAG.getNode(ISD::AND, dl, N0.getValueType(),
                              N0.getOperand(0).getOperand(0),
                              N0.getOperand(1));
          }

          return DAG.getSetCC(dl, VT, Val, N1,
                              Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
        }
      } else if (N1C->getAPIntValue() == 1 &&
                 (VT == MVT::i1 ||
                  getBooleanContents(false) == ZeroOrOneBooleanContent)) {
        SDValue Op0 = N0;
        if (Op0.getOpcode() == ISD::TRUNCATE)
          Op0 = Op0.getOperand(0);

        if ((Op0.getOpcode() == ISD::XOR) &&
            Op0.getOperand(0).getOpcode() == ISD::SETCC &&
            Op0.getOperand(1).getOpcode() == ISD::SETCC) {
          // (xor (setcc), (setcc)) == / != 1 -> (setcc) != / == (setcc)
          Cond = (Cond == ISD::SETEQ) ? ISD::SETNE : ISD::SETEQ;
          return DAG.getSetCC(dl, VT, Op0.getOperand(0), Op0.getOperand(1),
                              Cond);
        } else if (Op0.getOpcode() == ISD::AND &&
                isa<ConstantSDNode>(Op0.getOperand(1)) &&
                cast<ConstantSDNode>(Op0.getOperand(1))->getAPIntValue() == 1) {
          // If this is (X&1) == / != 1, normalize it to (X&1) != / == 0.
          if (Op0.getValueType().bitsGT(VT))
            Op0 = DAG.getNode(ISD::AND, dl, VT,
                          DAG.getNode(ISD::TRUNCATE, dl, VT, Op0.getOperand(0)),
                          DAG.getConstant(1, VT));
          else if (Op0.getValueType().bitsLT(VT))
            Op0 = DAG.getNode(ISD::AND, dl, VT,
                        DAG.getNode(ISD::ANY_EXTEND, dl, VT, Op0.getOperand(0)),
                        DAG.getConstant(1, VT));

          return DAG.getSetCC(dl, VT, Op0,
                              DAG.getConstant(0, Op0.getValueType()),
                              Cond == ISD::SETEQ ? ISD::SETNE : ISD::SETEQ);
        }
      }
    }

    APInt MinVal, MaxVal;
    unsigned OperandBitSize = N1C->getValueType(0).getSizeInBits();
    if (ISD::isSignedIntSetCC(Cond)) {
      MinVal = APInt::getSignedMinValue(OperandBitSize);
      MaxVal = APInt::getSignedMaxValue(OperandBitSize);
    } else {
      MinVal = APInt::getMinValue(OperandBitSize);
      MaxVal = APInt::getMaxValue(OperandBitSize);
    }

    // Canonicalize GE/LE comparisons to use GT/LT comparisons.
    if (Cond == ISD::SETGE || Cond == ISD::SETUGE) {
      if (C1 == MinVal) return DAG.getConstant(1, VT);   // X >= MIN --> true
      // X >= C0 --> X > (C0-1)
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(C1-1, N1.getValueType()),
                          (Cond == ISD::SETGE) ? ISD::SETGT : ISD::SETUGT);
    }

    if (Cond == ISD::SETLE || Cond == ISD::SETULE) {
      if (C1 == MaxVal) return DAG.getConstant(1, VT);   // X <= MAX --> true
      // X <= C0 --> X < (C0+1)
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(C1+1, N1.getValueType()),
                          (Cond == ISD::SETLE) ? ISD::SETLT : ISD::SETULT);
    }

    if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal)
      return DAG.getConstant(0, VT);      // X < MIN --> false
    if ((Cond == ISD::SETGE || Cond == ISD::SETUGE) && C1 == MinVal)
      return DAG.getConstant(1, VT);      // X >= MIN --> true
    if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal)
      return DAG.getConstant(0, VT);      // X > MAX --> false
    if ((Cond == ISD::SETLE || Cond == ISD::SETULE) && C1 == MaxVal)
      return DAG.getConstant(1, VT);      // X <= MAX --> true

    // Canonicalize setgt X, Min --> setne X, Min
    if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MinVal)
      return DAG.getSetCC(dl, VT, N0, N1, ISD::SETNE);
    // Canonicalize setlt X, Max --> setne X, Max
    if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MaxVal)
      return DAG.getSetCC(dl, VT, N0, N1, ISD::SETNE);

    // If we have setult X, 1, turn it into seteq X, 0
    if ((Cond == ISD::SETLT || Cond == ISD::SETULT) && C1 == MinVal+1)
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(MinVal, N0.getValueType()),
                          ISD::SETEQ);
    // If we have setugt X, Max-1, turn it into seteq X, Max
    else if ((Cond == ISD::SETGT || Cond == ISD::SETUGT) && C1 == MaxVal-1)
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(MaxVal, N0.getValueType()),
                          ISD::SETEQ);

    // If we have "setcc X, C0", check to see if we can shrink the immediate
    // by changing cc.

    // SETUGT X, SINTMAX  -> SETLT X, 0
    if (Cond == ISD::SETUGT &&
        C1 == APInt::getSignedMaxValue(OperandBitSize))
      return DAG.getSetCC(dl, VT, N0,
                          DAG.getConstant(0, N1.getValueType()),
                          ISD::SETLT);

    // SETULT X, SINTMIN  -> SETGT X, -1
    if (Cond == ISD::SETULT &&
        C1 == APInt::getSignedMinValue(OperandBitSize)) {
      SDValue ConstMinusOne =
          DAG.getConstant(APInt::getAllOnesValue(OperandBitSize),
                          N1.getValueType());
      return DAG.getSetCC(dl, VT, N0, ConstMinusOne, ISD::SETGT);
    }

    // Fold bit comparisons when we can.
    if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
        (VT == N0.getValueType() ||
         (isTypeLegal(VT) && VT.bitsLE(N0.getValueType()))) &&
        N0.getOpcode() == ISD::AND)
      if (ConstantSDNode *AndRHS =
                  dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
        EVT ShiftTy = DCI.isBeforeLegalizeOps() ?
          getPointerTy() : getShiftAmountTy(N0.getValueType());
        if (Cond == ISD::SETNE && C1 == 0) {// (X & 8) != 0  -->  (X & 8) >> 3
          // Perform the xform if the AND RHS is a single bit.
          if (AndRHS->getAPIntValue().isPowerOf2()) {
            return DAG.getNode(ISD::TRUNCATE, dl, VT,
                              DAG.getNode(ISD::SRL, dl, N0.getValueType(), N0,
                   DAG.getConstant(AndRHS->getAPIntValue().logBase2(), ShiftTy)));
          }
        } else if (Cond == ISD::SETEQ && C1 == AndRHS->getAPIntValue()) {
          // (X & 8) == 8  -->  (X & 8) >> 3
          // Perform the xform if C1 is a single bit.
          if (C1.isPowerOf2()) {
            return DAG.getNode(ISD::TRUNCATE, dl, VT,
                               DAG.getNode(ISD::SRL, dl, N0.getValueType(), N0,
                                      DAG.getConstant(C1.logBase2(), ShiftTy)));
          }
        }
      }

    if (C1.getMinSignedBits() <= 64 &&
        !isLegalICmpImmediate(C1.getSExtValue())) {
      // (X & -256) == 256 -> (X >> 8) == 1
      if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
          N0.getOpcode() == ISD::AND && N0.hasOneUse()) {
        if (ConstantSDNode *AndRHS =
            dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          const APInt &AndRHSC = AndRHS->getAPIntValue();
          if ((-AndRHSC).isPowerOf2() && (AndRHSC & C1) == C1) {
            unsigned ShiftBits = AndRHSC.countTrailingZeros();
            EVT ShiftTy = DCI.isBeforeLegalizeOps() ?
              getPointerTy() : getShiftAmountTy(N0.getValueType());
            EVT CmpTy = N0.getValueType();
            SDValue Shift = DAG.getNode(ISD::SRL, dl, CmpTy, N0.getOperand(0),
                                        DAG.getConstant(ShiftBits, ShiftTy));
            SDValue CmpRHS = DAG.getConstant(C1.lshr(ShiftBits), CmpTy);
            return DAG.getSetCC(dl, VT, Shift, CmpRHS, Cond);
          }
        }
      } else if (Cond == ISD::SETULT || Cond == ISD::SETUGE ||
                 Cond == ISD::SETULE || Cond == ISD::SETUGT) {
        bool AdjOne = (Cond == ISD::SETULE || Cond == ISD::SETUGT);
        // X <  0x100000000 -> (X >> 32) <  1
        // X >= 0x100000000 -> (X >> 32) >= 1
        // X <= 0x0ffffffff -> (X >> 32) <  1
        // X >  0x0ffffffff -> (X >> 32) >= 1
        unsigned ShiftBits;
        APInt NewC = C1;
        ISD::CondCode NewCond = Cond;
        if (AdjOne) {
          ShiftBits = C1.countTrailingOnes();
          NewC = NewC + 1;
          NewCond = (Cond == ISD::SETULE) ? ISD::SETULT : ISD::SETUGE;
        } else {
          ShiftBits = C1.countTrailingZeros();
        }
        NewC = NewC.lshr(ShiftBits);
        if (ShiftBits && isLegalICmpImmediate(NewC.getSExtValue())) {
          EVT ShiftTy = DCI.isBeforeLegalizeOps() ?
            getPointerTy() : getShiftAmountTy(N0.getValueType());
          EVT CmpTy = N0.getValueType();
          SDValue Shift = DAG.getNode(ISD::SRL, dl, CmpTy, N0,
                                      DAG.getConstant(ShiftBits, ShiftTy));
          SDValue CmpRHS = DAG.getConstant(NewC, CmpTy);
          return DAG.getSetCC(dl, VT, Shift, CmpRHS, NewCond);
        }
      }
    }
  }

  if (isa<ConstantFPSDNode>(N0.getNode())) {
    // Constant fold or commute setcc.
    SDValue O = DAG.FoldSetCC(VT, N0, N1, Cond, dl);
    if (O.getNode()) return O;
  } else if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(N1.getNode())) {
    // If the RHS of an FP comparison is a constant, simplify it away in
    // some cases.
    if (CFP->getValueAPF().isNaN()) {
      // If an operand is known to be a nan, we can fold it.
      switch (ISD::getUnorderedFlavor(Cond)) {
      default: llvm_unreachable("Unknown flavor!");
      case 0:  // Known false.
        return DAG.getConstant(0, VT);
      case 1:  // Known true.
        return DAG.getConstant(1, VT);
      case 2:  // Undefined.
        return DAG.getUNDEF(VT);
      }
    }

    // Otherwise, we know the RHS is not a NaN.  Simplify the node to drop the
    // constant if knowing that the operand is non-nan is enough.  We prefer to
    // have SETO(x,x) instead of SETO(x, 0.0) because this avoids having to
    // materialize 0.0.
    if (Cond == ISD::SETO || Cond == ISD::SETUO)
      return DAG.getSetCC(dl, VT, N0, N0, Cond);

    // If the condition is not legal, see if we can find an equivalent one
    // which is legal.
    if (!isCondCodeLegal(Cond, N0.getValueType())) {
      // If the comparison was an awkward floating-point == or != and one of
      // the comparison operands is infinity or negative infinity, convert the
      // condition to a less-awkward <= or >=.
      if (CFP->getValueAPF().isInfinity()) {
        if (CFP->getValueAPF().isNegative()) {
          if (Cond == ISD::SETOEQ &&
              isCondCodeLegal(ISD::SETOLE, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOLE);
          if (Cond == ISD::SETUEQ &&
              isCondCodeLegal(ISD::SETOLE, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETULE);
          if (Cond == ISD::SETUNE &&
              isCondCodeLegal(ISD::SETUGT, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETUGT);
          if (Cond == ISD::SETONE &&
              isCondCodeLegal(ISD::SETUGT, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOGT);
        } else {
          if (Cond == ISD::SETOEQ &&
              isCondCodeLegal(ISD::SETOGE, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOGE);
          if (Cond == ISD::SETUEQ &&
              isCondCodeLegal(ISD::SETOGE, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETUGE);
          if (Cond == ISD::SETUNE &&
              isCondCodeLegal(ISD::SETULT, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETULT);
          if (Cond == ISD::SETONE &&
              isCondCodeLegal(ISD::SETULT, N0.getValueType()))
            return DAG.getSetCC(dl, VT, N0, N1, ISD::SETOLT);
        }
      }
    }
  }

  if (N0 == N1) {
    // The sext(setcc()) => setcc() optimization relies on the appropriate
    // constant being emitted.
    uint64_t EqVal = 0;
    switch (getBooleanContents(N0.getValueType().isVector())) {
    case UndefinedBooleanContent:
    case ZeroOrOneBooleanContent:
      EqVal = ISD::isTrueWhenEqual(Cond);
      break;
    case ZeroOrNegativeOneBooleanContent:
      EqVal = ISD::isTrueWhenEqual(Cond) ? -1 : 0;
      break;
    }

    // We can always fold X == X for integer setcc's.
    if (N0.getValueType().isInteger()) {
      return DAG.getConstant(EqVal, VT);
    }
    unsigned UOF = ISD::getUnorderedFlavor(Cond);
    if (UOF == 2)   // FP operators that are undefined on NaNs.
      return DAG.getConstant(EqVal, VT);
    if (UOF == unsigned(ISD::isTrueWhenEqual(Cond)))
      return DAG.getConstant(EqVal, VT);
    // Otherwise, we can't fold it.  However, we can simplify it to SETUO/SETO
    // if it is not already.
    ISD::CondCode NewCond = UOF == 0 ? ISD::SETO : ISD::SETUO;
    if (NewCond != Cond && (DCI.isBeforeLegalizeOps() ||
          getCondCodeAction(NewCond, N0.getValueType()) == Legal))
      return DAG.getSetCC(dl, VT, N0, N1, NewCond);
  }

  if ((Cond == ISD::SETEQ || Cond == ISD::SETNE) &&
      N0.getValueType().isInteger()) {
    if (N0.getOpcode() == ISD::ADD || N0.getOpcode() == ISD::SUB ||
        N0.getOpcode() == ISD::XOR) {
      // Simplify (X+Y) == (X+Z) -->  Y == Z
      if (N0.getOpcode() == N1.getOpcode()) {
        if (N0.getOperand(0) == N1.getOperand(0))
          return DAG.getSetCC(dl, VT, N0.getOperand(1), N1.getOperand(1), Cond);
        if (N0.getOperand(1) == N1.getOperand(1))
          return DAG.getSetCC(dl, VT, N0.getOperand(0), N1.getOperand(0), Cond);
        if (DAG.isCommutativeBinOp(N0.getOpcode())) {
          // If X op Y == Y op X, try other combinations.
          if (N0.getOperand(0) == N1.getOperand(1))
            return DAG.getSetCC(dl, VT, N0.getOperand(1), N1.getOperand(0),
                                Cond);
          if (N0.getOperand(1) == N1.getOperand(0))
            return DAG.getSetCC(dl, VT, N0.getOperand(0), N1.getOperand(1),
                                Cond);
        }
      }

      // If RHS is a legal immediate value for a compare instruction, we need
      // to be careful about increasing register pressure needlessly.
      bool LegalRHSImm = false;

      if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(N1)) {
        if (ConstantSDNode *LHSR = dyn_cast<ConstantSDNode>(N0.getOperand(1))) {
          // Turn (X+C1) == C2 --> X == C2-C1
          if (N0.getOpcode() == ISD::ADD && N0.getNode()->hasOneUse()) {
            return DAG.getSetCC(dl, VT, N0.getOperand(0),
                                DAG.getConstant(RHSC->getAPIntValue()-
                                                LHSR->getAPIntValue(),
                                N0.getValueType()), Cond);
          }

          // Turn (X^C1) == C2 into X == C1^C2 iff X&~C1 = 0.
          if (N0.getOpcode() == ISD::XOR)
            // If we know that all of the inverted bits are zero, don't bother
            // performing the inversion.
            if (DAG.MaskedValueIsZero(N0.getOperand(0), ~LHSR->getAPIntValue()))
              return
                DAG.getSetCC(dl, VT, N0.getOperand(0),
                             DAG.getConstant(LHSR->getAPIntValue() ^
                                               RHSC->getAPIntValue(),
                                             N0.getValueType()),
                             Cond);
        }

        // Turn (C1-X) == C2 --> X == C1-C2
        if (ConstantSDNode *SUBC = dyn_cast<ConstantSDNode>(N0.getOperand(0))) {
          if (N0.getOpcode() == ISD::SUB && N0.getNode()->hasOneUse()) {
            return
              DAG.getSetCC(dl, VT, N0.getOperand(1),
                           DAG.getConstant(SUBC->getAPIntValue() -
                                             RHSC->getAPIntValue(),
                                           N0.getValueType()),
                           Cond);
          }
        }

        // Could RHSC fold directly into a compare?
        if (RHSC->getValueType(0).getSizeInBits() <= 64)
          LegalRHSImm = isLegalICmpImmediate(RHSC->getSExtValue());
      }

      // Simplify (X+Z) == X -->  Z == 0
      // Don't do this if X is an immediate that can fold into a cmp
      // instruction and X+Z has other uses. It could be an induction variable
      // chain, and the transform would increase register pressure.
      if (!LegalRHSImm || N0.getNode()->hasOneUse()) {
        if (N0.getOperand(0) == N1)
          return DAG.getSetCC(dl, VT, N0.getOperand(1),
                              DAG.getConstant(0, N0.getValueType()), Cond);
        if (N0.getOperand(1) == N1) {
          if (DAG.isCommutativeBinOp(N0.getOpcode()))
            return DAG.getSetCC(dl, VT, N0.getOperand(0),
                                DAG.getConstant(0, N0.getValueType()), Cond);
          else if (N0.getNode()->hasOneUse()) {
            assert(N0.getOpcode() == ISD::SUB && "Unexpected operation!");
            // (Z-X) == X  --> Z == X<<1
            SDValue SH = DAG.getNode(ISD::SHL, dl, N1.getValueType(), N1,
                       DAG.getConstant(1, getShiftAmountTy(N1.getValueType())));
            if (!DCI.isCalledByLegalizer())
              DCI.AddToWorklist(SH.getNode());
            return DAG.getSetCC(dl, VT, N0.getOperand(0), SH, Cond);
          }
        }
      }
    }

    if (N1.getOpcode() == ISD::ADD || N1.getOpcode() == ISD::SUB ||
        N1.getOpcode() == ISD::XOR) {
      // Simplify  X == (X+Z) -->  Z == 0
      if (N1.getOperand(0) == N0) {
        return DAG.getSetCC(dl, VT, N1.getOperand(1),
                        DAG.getConstant(0, N1.getValueType()), Cond);
      } else if (N1.getOperand(1) == N0) {
        if (DAG.isCommutativeBinOp(N1.getOpcode())) {
          return DAG.getSetCC(dl, VT, N1.getOperand(0),
                          DAG.getConstant(0, N1.getValueType()), Cond);
        } else if (N1.getNode()->hasOneUse()) {
          assert(N1.getOpcode() == ISD::SUB && "Unexpected operation!");
          // X == (Z-X)  --> X<<1 == Z
          SDValue SH = DAG.getNode(ISD::SHL, dl, N1.getValueType(), N0,
                       DAG.getConstant(1, getShiftAmountTy(N0.getValueType())));
          if (!DCI.isCalledByLegalizer())
            DCI.AddToWorklist(SH.getNode());
          return DAG.getSetCC(dl, VT, SH, N1.getOperand(0), Cond);
        }
      }
    }

    // Simplify x&y == y to x&y != 0 if y has exactly one bit set.
    // Note that where y is variable and is known to have at most
    // one bit set (for example, if it is z&1) we cannot do this;
    // the expressions are not equivalent when y==0.
    if (N0.getOpcode() == ISD::AND)
      if (N0.getOperand(0) == N1 || N0.getOperand(1) == N1) {
        if (ValueHasExactlyOneBitSet(N1, DAG)) {
          Cond = ISD::getSetCCInverse(Cond, /*isInteger=*/true);
          SDValue Zero = DAG.getConstant(0, N1.getValueType());
          return DAG.getSetCC(dl, VT, N0, Zero, Cond);
        }
      }
    if (N1.getOpcode() == ISD::AND)
      if (N1.getOperand(0) == N0 || N1.getOperand(1) == N0) {
        if (ValueHasExactlyOneBitSet(N0, DAG)) {
          Cond = ISD::getSetCCInverse(Cond, /*isInteger=*/true);
          SDValue Zero = DAG.getConstant(0, N0.getValueType());
          return DAG.getSetCC(dl, VT, N1, Zero, Cond);
        }
      }
  }

  // Fold away ALL boolean setcc's.
  SDValue Temp;
  if (N0.getValueType() == MVT::i1 && foldBooleans) {
    switch (Cond) {
    default: llvm_unreachable("Unknown integer setcc!");
    case ISD::SETEQ:  // X == Y  -> ~(X^Y)
      Temp = DAG.getNode(ISD::XOR, dl, MVT::i1, N0, N1);
      N0 = DAG.getNOT(dl, Temp, MVT::i1);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETNE:  // X != Y   -->  (X^Y)
      N0 = DAG.getNode(ISD::XOR, dl, MVT::i1, N0, N1);
      break;
    case ISD::SETGT:  // X >s Y   -->  X == 0 & Y == 1  -->  ~X & Y
    case ISD::SETULT: // X <u Y   -->  X == 0 & Y == 1  -->  ~X & Y
      Temp = DAG.getNOT(dl, N0, MVT::i1);
      N0 = DAG.getNode(ISD::AND, dl, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETLT:  // X <s Y   --> X == 1 & Y == 0  -->  ~Y & X
    case ISD::SETUGT: // X >u Y   --> X == 1 & Y == 0  -->  ~Y & X
      Temp = DAG.getNOT(dl, N1, MVT::i1);
      N0 = DAG.getNode(ISD::AND, dl, MVT::i1, N0, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETULE: // X <=u Y  --> X == 0 | Y == 1  -->  ~X | Y
    case ISD::SETGE:  // X >=s Y  --> X == 0 | Y == 1  -->  ~X | Y
      Temp = DAG.getNOT(dl, N0, MVT::i1);
      N0 = DAG.getNode(ISD::OR, dl, MVT::i1, N1, Temp);
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(Temp.getNode());
      break;
    case ISD::SETUGE: // X >=u Y  --> X == 1 | Y == 0  -->  ~Y | X
    case ISD::SETLE:  // X <=s Y  --> X == 1 | Y == 0  -->  ~Y | X
      Temp = DAG.getNOT(dl, N1, MVT::i1);
      N0 = DAG.getNode(ISD::OR, dl, MVT::i1, N0, Temp);
      break;
    }
    if (VT != MVT::i1) {
      if (!DCI.isCalledByLegalizer())
        DCI.AddToWorklist(N0.getNode());
      // FIXME: If running after legalize, we probably can't do this.
      N0 = DAG.getNode(ISD::ZERO_EXTEND, dl, VT, N0);
    }
    return N0;
  }

  // Could not fold it.
  return SDValue();
}

/// isGAPlusOffset - Returns true (and the GlobalValue and the offset) if the
/// node is a GlobalAddress + offset.
bool TargetLowering::isGAPlusOffset(SDNode *N, const GlobalValue *&GA,
                                    int64_t &Offset) const {
  if (isa<GlobalAddressSDNode>(N)) {
    GlobalAddressSDNode *GASD = cast<GlobalAddressSDNode>(N);
    GA = GASD->getGlobal();
    Offset += GASD->getOffset();
    return true;
  }

  if (N->getOpcode() == ISD::ADD) {
    SDValue N1 = N->getOperand(0);
    SDValue N2 = N->getOperand(1);
    if (isGAPlusOffset(N1.getNode(), GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N2);
      if (V) {
        Offset += V->getSExtValue();
        return true;
      }
    } else if (isGAPlusOffset(N2.getNode(), GA, Offset)) {
      ConstantSDNode *V = dyn_cast<ConstantSDNode>(N1);
      if (V) {
        Offset += V->getSExtValue();
        return true;
      }
    }
  }

  return false;
}


SDValue TargetLowering::
PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const {
  // Default implementation: no optimization.
  return SDValue();
}

//===----------------------------------------------------------------------===//
//  Inline Assembler Implementation Methods
//===----------------------------------------------------------------------===//


TargetLowering::ConstraintType
TargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'r': return C_RegisterClass;
    case 'm':    // memory
    case 'o':    // offsetable
    case 'V':    // not offsetable
      return C_Memory;
    case 'i':    // Simple Integer or Relocatable Constant
    case 'n':    // Simple Integer
    case 'E':    // Floating Point Constant
    case 'F':    // Floating Point Constant
    case 's':    // Relocatable Constant
    case 'p':    // Address.
    case 'X':    // Allow ANY value.
    case 'I':    // Target registers.
    case 'J':
    case 'K':
    case 'L':
    case 'M':
    case 'N':
    case 'O':
    case 'P':
    case '<':
    case '>':
      return C_Other;
    }
  }

  if (Constraint.size() > 1 && Constraint[0] == '{' &&
      Constraint[Constraint.size()-1] == '}')
    return C_Register;
  return C_Unknown;
}

/// LowerXConstraint - try to replace an X constraint, which matches anything,
/// with another that has more specific requirements based on the type of the
/// corresponding operand.
const char *TargetLowering::LowerXConstraint(EVT ConstraintVT) const{
  if (ConstraintVT.isInteger())
    return "r";
  if (ConstraintVT.isFloatingPoint())
    return "f";      // works for many targets
  return 0;
}

/// LowerAsmOperandForConstraint - Lower the specified operand into the Ops
/// vector.  If it is invalid, don't add anything to Ops.
void TargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                  std::string &Constraint,
                                                  std::vector<SDValue> &Ops,
                                                  SelectionDAG &DAG) const {

  if (Constraint.length() > 1) return;

  char ConstraintLetter = Constraint[0];
  switch (ConstraintLetter) {
  default: break;
  case 'X':     // Allows any operand; labels (basic block) use this.
    if (Op.getOpcode() == ISD::BasicBlock) {
      Ops.push_back(Op);
      return;
    }
    // fall through
  case 'i':    // Simple Integer or Relocatable Constant
  case 'n':    // Simple Integer
  case 's': {  // Relocatable Constant
    // These operands are interested in values of the form (GV+C), where C may
    // be folded in as an offset of GV, or it may be explicitly added.  Also, it
    // is possible and fine if either GV or C are missing.
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op);

    // If we have "(add GV, C)", pull out GV/C
    if (Op.getOpcode() == ISD::ADD) {
      C = dyn_cast<ConstantSDNode>(Op.getOperand(1));
      GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(0));
      if (C == 0 || GA == 0) {
        C = dyn_cast<ConstantSDNode>(Op.getOperand(0));
        GA = dyn_cast<GlobalAddressSDNode>(Op.getOperand(1));
      }
      if (C == 0 || GA == 0)
        C = 0, GA = 0;
    }

    // If we find a valid operand, map to the TargetXXX version so that the
    // value itself doesn't get selected.
    if (GA) {   // Either &GV   or   &GV+C
      if (ConstraintLetter != 'n') {
        int64_t Offs = GA->getOffset();
        if (C) Offs += C->getZExtValue();
        Ops.push_back(DAG.getTargetGlobalAddress(GA->getGlobal(),
                                                 C ? C->getDebugLoc() : DebugLoc(),
                                                 Op.getValueType(), Offs));
        return;
      }
    }
    if (C) {   // just C, no GV.
      // Simple constants are not allowed for 's'.
      if (ConstraintLetter != 's') {
        // gcc prints these as sign extended.  Sign extend value to 64 bits
        // now; without this it would get ZExt'd later in
        // ScheduleDAGSDNodes::EmitNode, which is very generic.
        Ops.push_back(DAG.getTargetConstant(C->getAPIntValue().getSExtValue(),
                                            MVT::i64));
        return;
      }
    }
    break;
  }
  }
}

std::pair<unsigned, const TargetRegisterClass*> TargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             EVT VT) const {
  if (Constraint[0] != '{')
    return std::make_pair(0u, static_cast<TargetRegisterClass*>(0));
  assert(*(Constraint.end()-1) == '}' && "Not a brace enclosed constraint?");

  // Remove the braces from around the name.
  StringRef RegName(Constraint.data()+1, Constraint.size()-2);

  // Figure out which register class contains this reg.
  const TargetRegisterInfo *RI = TM.getRegisterInfo();
  for (TargetRegisterInfo::regclass_iterator RCI = RI->regclass_begin(),
       E = RI->regclass_end(); RCI != E; ++RCI) {
    const TargetRegisterClass *RC = *RCI;

    // If none of the value types for this register class are valid, we
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    if (!isLegalRC(RC))
      continue;

    for (TargetRegisterClass::iterator I = RC->begin(), E = RC->end();
         I != E; ++I) {
      if (RegName.equals_lower(RI->getName(*I)))
        return std::make_pair(*I, RC);
    }
  }

  return std::make_pair(0u, static_cast<const TargetRegisterClass*>(0));
}

//===----------------------------------------------------------------------===//
// Constraint Selection.

/// isMatchingInputConstraint - Return true of this is an input operand that is
/// a matching constraint like "4".
bool TargetLowering::AsmOperandInfo::isMatchingInputConstraint() const {
  assert(!ConstraintCode.empty() && "No known constraint!");
  return isdigit(ConstraintCode[0]);
}

/// getMatchedOperand - If this is an input matching constraint, this method
/// returns the output operand it matches.
unsigned TargetLowering::AsmOperandInfo::getMatchedOperand() const {
  assert(!ConstraintCode.empty() && "No known constraint!");
  return atoi(ConstraintCode.c_str());
}


/// ParseConstraints - Split up the constraint string from the inline
/// assembly value into the specific constraints and their prefixes,
/// and also tie in the associated operand values.
/// If this returns an empty vector, and if the constraint string itself
/// isn't empty, there was an error parsing.
TargetLowering::AsmOperandInfoVector TargetLowering::ParseConstraints(
    ImmutableCallSite CS) const {
  /// ConstraintOperands - Information about all of the constraints.
  AsmOperandInfoVector ConstraintOperands;
  const InlineAsm *IA = cast<InlineAsm>(CS.getCalledValue());
  unsigned maCount = 0; // Largest number of multiple alternative constraints.

  // Do a prepass over the constraints, canonicalizing them, and building up the
  // ConstraintOperands list.
  InlineAsm::ConstraintInfoVector
    ConstraintInfos = IA->ParseConstraints();

  unsigned ArgNo = 0;   // ArgNo - The argument of the CallInst.
  unsigned ResNo = 0;   // ResNo - The result number of the next output.

  for (unsigned i = 0, e = ConstraintInfos.size(); i != e; ++i) {
    ConstraintOperands.push_back(AsmOperandInfo(ConstraintInfos[i]));
    AsmOperandInfo &OpInfo = ConstraintOperands.back();

    // Update multiple alternative constraint count.
    if (OpInfo.multipleAlternatives.size() > maCount)
      maCount = OpInfo.multipleAlternatives.size();

    OpInfo.ConstraintVT = MVT::Other;

    // Compute the value type for each operand.
    switch (OpInfo.Type) {
    case InlineAsm::isOutput:
      // Indirect outputs just consume an argument.
      if (OpInfo.isIndirect) {
        OpInfo.CallOperandVal = const_cast<Value *>(CS.getArgument(ArgNo++));
        break;
      }

      // The return value of the call is this value.  As such, there is no
      // corresponding argument.
      assert(!CS.getType()->isVoidTy() &&
             "Bad inline asm!");
      if (StructType *STy = dyn_cast<StructType>(CS.getType())) {
        OpInfo.ConstraintVT = getValueType(STy->getElementType(ResNo));
      } else {
        assert(ResNo == 0 && "Asm only has one result!");
        OpInfo.ConstraintVT = getValueType(CS.getType());
      }
      ++ResNo;
      break;
    case InlineAsm::isInput:
      OpInfo.CallOperandVal = const_cast<Value *>(CS.getArgument(ArgNo++));
      break;
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }

    if (OpInfo.CallOperandVal) {
      llvm::Type *OpTy = OpInfo.CallOperandVal->getType();
      if (OpInfo.isIndirect) {
        llvm::PointerType *PtrTy = dyn_cast<PointerType>(OpTy);
        if (!PtrTy)
          report_fatal_error("Indirect operand for inline asm not a pointer!");
        OpTy = PtrTy->getElementType();
      }

      // Look for vector wrapped in a struct. e.g. { <16 x i8> }.
      if (StructType *STy = dyn_cast<StructType>(OpTy))
        if (STy->getNumElements() == 1)
          OpTy = STy->getElementType(0);

      // If OpTy is not a single value, it may be a struct/union that we
      // can tile with integers.
      if (!OpTy->isSingleValueType() && OpTy->isSized()) {
        unsigned BitSize = TD->getTypeSizeInBits(OpTy);
        switch (BitSize) {
        default: break;
        case 1:
        case 8:
        case 16:
        case 32:
        case 64:
        case 128:
          OpInfo.ConstraintVT =
              EVT::getEVT(IntegerType::get(OpTy->getContext(), BitSize), true);
          break;
        }
      } else if (PointerType *PT = dyn_cast<PointerType>(OpTy)) {
        OpInfo.ConstraintVT = MVT::getIntegerVT(
            8*TD->getPointerSize(PT->getAddressSpace()));
      } else {
        OpInfo.ConstraintVT = EVT::getEVT(OpTy, true);
      }
    }
  }

  // If we have multiple alternative constraints, select the best alternative.
  if (ConstraintInfos.size()) {
    if (maCount) {
      unsigned bestMAIndex = 0;
      int bestWeight = -1;
      // weight:  -1 = invalid match, and 0 = so-so match to 5 = good match.
      int weight = -1;
      unsigned maIndex;
      // Compute the sums of the weights for each alternative, keeping track
      // of the best (highest weight) one so far.
      for (maIndex = 0; maIndex < maCount; ++maIndex) {
        int weightSum = 0;
        for (unsigned cIndex = 0, eIndex = ConstraintOperands.size();
            cIndex != eIndex; ++cIndex) {
          AsmOperandInfo& OpInfo = ConstraintOperands[cIndex];
          if (OpInfo.Type == InlineAsm::isClobber)
            continue;

          // If this is an output operand with a matching input operand,
          // look up the matching input. If their types mismatch, e.g. one
          // is an integer, the other is floating point, or their sizes are
          // different, flag it as an maCantMatch.
          if (OpInfo.hasMatchingInput()) {
            AsmOperandInfo &Input = ConstraintOperands[OpInfo.MatchingInput];
            if (OpInfo.ConstraintVT != Input.ConstraintVT) {
              if ((OpInfo.ConstraintVT.isInteger() !=
                   Input.ConstraintVT.isInteger()) ||
                  (OpInfo.ConstraintVT.getSizeInBits() !=
                   Input.ConstraintVT.getSizeInBits())) {
                weightSum = -1;  // Can't match.
                break;
              }
            }
          }
          weight = getMultipleConstraintMatchWeight(OpInfo, maIndex);
          if (weight == -1) {
            weightSum = -1;
            break;
          }
          weightSum += weight;
        }
        // Update best.
        if (weightSum > bestWeight) {
          bestWeight = weightSum;
          bestMAIndex = maIndex;
        }
      }

      // Now select chosen alternative in each constraint.
      for (unsigned cIndex = 0, eIndex = ConstraintOperands.size();
          cIndex != eIndex; ++cIndex) {
        AsmOperandInfo& cInfo = ConstraintOperands[cIndex];
        if (cInfo.Type == InlineAsm::isClobber)
          continue;
        cInfo.selectAlternative(bestMAIndex);
      }
    }
  }

  // Check and hook up tied operands, choose constraint code to use.
  for (unsigned cIndex = 0, eIndex = ConstraintOperands.size();
      cIndex != eIndex; ++cIndex) {
    AsmOperandInfo& OpInfo = ConstraintOperands[cIndex];

    // If this is an output operand with a matching input operand, look up the
    // matching input. If their types mismatch, e.g. one is an integer, the
    // other is floating point, or their sizes are different, flag it as an
    // error.
    if (OpInfo.hasMatchingInput()) {
      AsmOperandInfo &Input = ConstraintOperands[OpInfo.MatchingInput];

      if (OpInfo.ConstraintVT != Input.ConstraintVT) {
        std::pair<unsigned, const TargetRegisterClass*> MatchRC =
          getRegForInlineAsmConstraint(OpInfo.ConstraintCode,
                                       OpInfo.ConstraintVT);
        std::pair<unsigned, const TargetRegisterClass*> InputRC =
          getRegForInlineAsmConstraint(Input.ConstraintCode,
                                       Input.ConstraintVT);
        if ((OpInfo.ConstraintVT.isInteger() !=
             Input.ConstraintVT.isInteger()) ||
            (MatchRC.second != InputRC.second)) {
          report_fatal_error("Unsupported asm: input constraint"
                             " with a matching output constraint of"
                             " incompatible type!");
        }
      }

    }
  }

  return ConstraintOperands;
}


/// getConstraintGenerality - Return an integer indicating how general CT
/// is.
static unsigned getConstraintGenerality(TargetLowering::ConstraintType CT) {
  switch (CT) {
  case TargetLowering::C_Other:
  case TargetLowering::C_Unknown:
    return 0;
  case TargetLowering::C_Register:
    return 1;
  case TargetLowering::C_RegisterClass:
    return 2;
  case TargetLowering::C_Memory:
    return 3;
  }
  llvm_unreachable("Invalid constraint type");
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
  TargetLowering::getMultipleConstraintMatchWeight(
    AsmOperandInfo &info, int maIndex) const {
  InlineAsm::ConstraintCodeVector *rCodes;
  if (maIndex >= (int)info.multipleAlternatives.size())
    rCodes = &info.Codes;
  else
    rCodes = &info.multipleAlternatives[maIndex].Codes;
  ConstraintWeight BestWeight = CW_Invalid;

  // Loop over the options, keeping track of the most general one.
  for (unsigned i = 0, e = rCodes->size(); i != e; ++i) {
    ConstraintWeight weight =
      getSingleConstraintMatchWeight(info, (*rCodes)[i].c_str());
    if (weight > BestWeight)
      BestWeight = weight;
  }

  return BestWeight;
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
  TargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
    // If we don't have a value, we can't do a match,
    // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  // Look at the constraint type.
  switch (*constraint) {
    case 'i': // immediate integer.
    case 'n': // immediate integer with a known value.
      if (isa<ConstantInt>(CallOperandVal))
        weight = CW_Constant;
      break;
    case 's': // non-explicit intregal immediate.
      if (isa<GlobalValue>(CallOperandVal))
        weight = CW_Constant;
      break;
    case 'E': // immediate float if host format.
    case 'F': // immediate float.
      if (isa<ConstantFP>(CallOperandVal))
        weight = CW_Constant;
      break;
    case '<': // memory operand with autodecrement.
    case '>': // memory operand with autoincrement.
    case 'm': // memory operand.
    case 'o': // offsettable memory operand
    case 'V': // non-offsettable memory operand
      weight = CW_Memory;
      break;
    case 'r': // general register.
    case 'g': // general register, memory operand or immediate integer.
              // note: Clang converts "g" to "imr".
      if (CallOperandVal->getType()->isIntegerTy())
        weight = CW_Register;
      break;
    case 'X': // any operand.
    default:
      weight = CW_Default;
      break;
  }
  return weight;
}

/// ChooseConstraint - If there are multiple different constraints that we
/// could pick for this operand (e.g. "imr") try to pick the 'best' one.
/// This is somewhat tricky: constraints fall into four classes:
///    Other         -> immediates and magic values
///    Register      -> one specific register
///    RegisterClass -> a group of regs
///    Memory        -> memory
/// Ideally, we would pick the most specific constraint possible: if we have
/// something that fits into a register, we would pick it.  The problem here
/// is that if we have something that could either be in a register or in
/// memory that use of the register could cause selection of *other*
/// operands to fail: they might only succeed if we pick memory.  Because of
/// this the heuristic we use is:
///
///  1) If there is an 'other' constraint, and if the operand is valid for
///     that constraint, use it.  This makes us take advantage of 'i'
///     constraints when available.
///  2) Otherwise, pick the most general constraint present.  This prefers
///     'm' over 'r', for example.
///
static void ChooseConstraint(TargetLowering::AsmOperandInfo &OpInfo,
                             const TargetLowering &TLI,
                             SDValue Op, SelectionDAG *DAG) {
  assert(OpInfo.Codes.size() > 1 && "Doesn't have multiple constraint options");
  unsigned BestIdx = 0;
  TargetLowering::ConstraintType BestType = TargetLowering::C_Unknown;
  int BestGenerality = -1;

  // Loop over the options, keeping track of the most general one.
  for (unsigned i = 0, e = OpInfo.Codes.size(); i != e; ++i) {
    TargetLowering::ConstraintType CType =
      TLI.getConstraintType(OpInfo.Codes[i]);

    // If this is an 'other' constraint, see if the operand is valid for it.
    // For example, on X86 we might have an 'rI' constraint.  If the operand
    // is an integer in the range [0..31] we want to use I (saving a load
    // of a register), otherwise we must use 'r'.
    if (CType == TargetLowering::C_Other && Op.getNode()) {
      assert(OpInfo.Codes[i].size() == 1 &&
             "Unhandled multi-letter 'other' constraint");
      std::vector<SDValue> ResultOps;
      TLI.LowerAsmOperandForConstraint(Op, OpInfo.Codes[i],
                                       ResultOps, *DAG);
      if (!ResultOps.empty()) {
        BestType = CType;
        BestIdx = i;
        break;
      }
    }

    // Things with matching constraints can only be registers, per gcc
    // documentation.  This mainly affects "g" constraints.
    if (CType == TargetLowering::C_Memory && OpInfo.hasMatchingInput())
      continue;

    // This constraint letter is more general than the previous one, use it.
    int Generality = getConstraintGenerality(CType);
    if (Generality > BestGenerality) {
      BestType = CType;
      BestIdx = i;
      BestGenerality = Generality;
    }
  }

  OpInfo.ConstraintCode = OpInfo.Codes[BestIdx];
  OpInfo.ConstraintType = BestType;
}

/// ComputeConstraintToUse - Determines the constraint code and constraint
/// type to use for the specific AsmOperandInfo, setting
/// OpInfo.ConstraintCode and OpInfo.ConstraintType.
void TargetLowering::ComputeConstraintToUse(AsmOperandInfo &OpInfo,
                                            SDValue Op,
                                            SelectionDAG *DAG) const {
  assert(!OpInfo.Codes.empty() && "Must have at least one constraint");

  // Single-letter constraints ('r') are very common.
  if (OpInfo.Codes.size() == 1) {
    OpInfo.ConstraintCode = OpInfo.Codes[0];
    OpInfo.ConstraintType = getConstraintType(OpInfo.ConstraintCode);
  } else {
    ChooseConstraint(OpInfo, *this, Op, DAG);
  }

  // 'X' matches anything.
  if (OpInfo.ConstraintCode == "X" && OpInfo.CallOperandVal) {
    // Labels and constants are handled elsewhere ('X' is the only thing
    // that matches labels).  For Functions, the type here is the type of
    // the result, which is not what we want to look at; leave them alone.
    Value *v = OpInfo.CallOperandVal;
    if (isa<BasicBlock>(v) || isa<ConstantInt>(v) || isa<Function>(v)) {
      OpInfo.CallOperandVal = v;
      return;
    }

    // Otherwise, try to resolve it to something we know about by looking at
    // the actual operand type.
    if (const char *Repl = LowerXConstraint(OpInfo.ConstraintVT)) {
      OpInfo.ConstraintCode = Repl;
      OpInfo.ConstraintType = getConstraintType(OpInfo.ConstraintCode);
    }
  }
}

//===----------------------------------------------------------------------===//
//  Loop Strength Reduction hooks
//===----------------------------------------------------------------------===//

/// isLegalAddressingMode - Return true if the addressing mode represented
/// by AM is legal for this target, for a load/store of the specified type.
bool TargetLowering::isLegalAddressingMode(const AddrMode &AM,
                                           Type *Ty) const {
  // The default implementation of this implements a conservative RISCy, r+r and
  // r+i addr mode.

  // Allows a sign-extended 16-bit immediate field.
  if (AM.BaseOffs <= -(1LL << 16) || AM.BaseOffs >= (1LL << 16)-1)
    return false;

  // No global is ever allowed as a base.
  if (AM.BaseGV)
    return false;

  // Only support r+r,
  switch (AM.Scale) {
  case 0:  // "r+i" or just "i", depending on HasBaseReg.
    break;
  case 1:
    if (AM.HasBaseReg && AM.BaseOffs)  // "r+r+i" is not allowed.
      return false;
    // Otherwise we have r+r or r+i.
    break;
  case 2:
    if (AM.HasBaseReg || AM.BaseOffs)  // 2*r+r  or  2*r+i is not allowed.
      return false;
    // Allow 2*r as r+r.
    break;
  }

  return true;
}

/// BuildExactDiv - Given an exact SDIV by a constant, create a multiplication
/// with the multiplicative inverse of the constant.
SDValue TargetLowering::BuildExactSDIV(SDValue Op1, SDValue Op2, DebugLoc dl,
                                       SelectionDAG &DAG) const {
  ConstantSDNode *C = cast<ConstantSDNode>(Op2);
  APInt d = C->getAPIntValue();
  assert(d != 0 && "Division by zero!");

  // Shift the value upfront if it is even, so the LSB is one.
  unsigned ShAmt = d.countTrailingZeros();
  if (ShAmt) {
    // TODO: For UDIV use SRL instead of SRA.
    SDValue Amt = DAG.getConstant(ShAmt, getShiftAmountTy(Op1.getValueType()));
    Op1 = DAG.getNode(ISD::SRA, dl, Op1.getValueType(), Op1, Amt);
    d = d.ashr(ShAmt);
  }

  // Calculate the multiplicative inverse, using Newton's method.
  APInt t, xn = d;
  while ((t = d*xn) != 1)
    xn *= APInt(d.getBitWidth(), 2) - t;

  Op2 = DAG.getConstant(xn, Op1.getValueType());
  return DAG.getNode(ISD::MUL, dl, Op1.getValueType(), Op1, Op2);
}

/// BuildSDIVSequence - Given an ISD::SDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDValue TargetLowering::
BuildSDIV(SDNode *N, SelectionDAG &DAG, bool IsAfterLegalization,
          std::vector<SDNode*>* Created) const {
  EVT VT = N->getValueType(0);
  DebugLoc dl= N->getDebugLoc();

  // Check to see if we can do this.
  // FIXME: We should be more aggressive here.
  if (!isTypeLegal(VT))
    return SDValue();

  APInt d = cast<ConstantSDNode>(N->getOperand(1))->getAPIntValue();
  APInt::ms magics = d.magic();

  // Multiply the numerator (operand 0) by the magic value
  // FIXME: We should support doing a MUL in a wider type
  SDValue Q;
  if (IsAfterLegalization ? isOperationLegal(ISD::MULHS, VT) :
                            isOperationLegalOrCustom(ISD::MULHS, VT))
    Q = DAG.getNode(ISD::MULHS, dl, VT, N->getOperand(0),
                    DAG.getConstant(magics.m, VT));
  else if (IsAfterLegalization ? isOperationLegal(ISD::SMUL_LOHI, VT) :
                                 isOperationLegalOrCustom(ISD::SMUL_LOHI, VT))
    Q = SDValue(DAG.getNode(ISD::SMUL_LOHI, dl, DAG.getVTList(VT, VT),
                              N->getOperand(0),
                              DAG.getConstant(magics.m, VT)).getNode(), 1);
  else
    return SDValue();       // No mulhs or equvialent
  // If d > 0 and m < 0, add the numerator
  if (d.isStrictlyPositive() && magics.m.isNegative()) {
    Q = DAG.getNode(ISD::ADD, dl, VT, Q, N->getOperand(0));
    if (Created)
      Created->push_back(Q.getNode());
  }
  // If d < 0 and m > 0, subtract the numerator.
  if (d.isNegative() && magics.m.isStrictlyPositive()) {
    Q = DAG.getNode(ISD::SUB, dl, VT, Q, N->getOperand(0));
    if (Created)
      Created->push_back(Q.getNode());
  }
  // Shift right algebraic if shift value is nonzero
  if (magics.s > 0) {
    Q = DAG.getNode(ISD::SRA, dl, VT, Q,
                 DAG.getConstant(magics.s, getShiftAmountTy(Q.getValueType())));
    if (Created)
      Created->push_back(Q.getNode());
  }
  // Extract the sign bit and add it to the quotient
  SDValue T =
    DAG.getNode(ISD::SRL, dl, VT, Q, DAG.getConstant(VT.getSizeInBits()-1,
                                           getShiftAmountTy(Q.getValueType())));
  if (Created)
    Created->push_back(T.getNode());
  return DAG.getNode(ISD::ADD, dl, VT, Q, T);
}

/// BuildUDIVSequence - Given an ISD::UDIV node expressing a divide by constant,
/// return a DAG expression to select that will generate the same value by
/// multiplying by a magic number.  See:
/// <http://the.wall.riscom.net/books/proc/ppc/cwg/code2.html>
SDValue TargetLowering::
BuildUDIV(SDNode *N, SelectionDAG &DAG, bool IsAfterLegalization,
          std::vector<SDNode*>* Created) const {
  EVT VT = N->getValueType(0);
  DebugLoc dl = N->getDebugLoc();

  // Check to see if we can do this.
  // FIXME: We should be more aggressive here.
  if (!isTypeLegal(VT))
    return SDValue();

  // FIXME: We should use a narrower constant when the upper
  // bits are known to be zero.
  const APInt &N1C = cast<ConstantSDNode>(N->getOperand(1))->getAPIntValue();
  APInt::mu magics = N1C.magicu();

  SDValue Q = N->getOperand(0);

  // If the divisor is even, we can avoid using the expensive fixup by shifting
  // the divided value upfront.
  if (magics.a != 0 && !N1C[0]) {
    unsigned Shift = N1C.countTrailingZeros();
    Q = DAG.getNode(ISD::SRL, dl, VT, Q,
                    DAG.getConstant(Shift, getShiftAmountTy(Q.getValueType())));
    if (Created)
      Created->push_back(Q.getNode());

    // Get magic number for the shifted divisor.
    magics = N1C.lshr(Shift).magicu(Shift);
    assert(magics.a == 0 && "Should use cheap fixup now");
  }

  // Multiply the numerator (operand 0) by the magic value
  // FIXME: We should support doing a MUL in a wider type
  if (IsAfterLegalization ? isOperationLegal(ISD::MULHU, VT) :
                            isOperationLegalOrCustom(ISD::MULHU, VT))
    Q = DAG.getNode(ISD::MULHU, dl, VT, Q, DAG.getConstant(magics.m, VT));
  else if (IsAfterLegalization ? isOperationLegal(ISD::UMUL_LOHI, VT) :
                                 isOperationLegalOrCustom(ISD::UMUL_LOHI, VT))
    Q = SDValue(DAG.getNode(ISD::UMUL_LOHI, dl, DAG.getVTList(VT, VT), Q,
                            DAG.getConstant(magics.m, VT)).getNode(), 1);
  else
    return SDValue();       // No mulhu or equvialent
  if (Created)
    Created->push_back(Q.getNode());

  if (magics.a == 0) {
    assert(magics.s < N1C.getBitWidth() &&
           "We shouldn't generate an undefined shift!");
    return DAG.getNode(ISD::SRL, dl, VT, Q,
                 DAG.getConstant(magics.s, getShiftAmountTy(Q.getValueType())));
  } else {
    SDValue NPQ = DAG.getNode(ISD::SUB, dl, VT, N->getOperand(0), Q);
    if (Created)
      Created->push_back(NPQ.getNode());
    NPQ = DAG.getNode(ISD::SRL, dl, VT, NPQ,
                      DAG.getConstant(1, getShiftAmountTy(NPQ.getValueType())));
    if (Created)
      Created->push_back(NPQ.getNode());
    NPQ = DAG.getNode(ISD::ADD, dl, VT, NPQ, Q);
    if (Created)
      Created->push_back(NPQ.getNode());
    return DAG.getNode(ISD::SRL, dl, VT, NPQ,
             DAG.getConstant(magics.s-1, getShiftAmountTy(NPQ.getValueType())));
  }
}
