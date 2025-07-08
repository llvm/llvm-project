//===- RuntimeLibcalls.cpp - Interface for runtime libcalls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;
using namespace RTLIB;

#define GET_INIT_RUNTIME_LIBCALL_UTILS
#define GET_INIT_RUNTIME_LIBCALL_NAMES
#include "llvm/IR/RuntimeLibcalls.inc"
#undef GET_INIT_RUNTIME_LIBCALL_UTILS
#undef GET_INIT_RUNTIME_LIBCALL_NAMES

static cl::opt<bool>
    HexagonEnableFastMathRuntimeCalls("hexagon-fast-math", cl::Hidden,
                                      cl::desc("Enable Fast Math processing"));

static void setAArch64LibcallNames(RuntimeLibcallsInfo &Info,
                                   const Triple &TT) {
#define LCALLNAMES(A, B, N)                                                    \
  Info.setLibcallImpl(A##N##_RELAX, B##N##_relax);                             \
  Info.setLibcallImpl(A##N##_ACQ, B##N##_acq);                                 \
  Info.setLibcallImpl(A##N##_REL, B##N##_rel);                                 \
  Info.setLibcallImpl(A##N##_ACQ_REL, B##N##_acq_rel);
#define LCALLNAME4(A, B)                                                       \
  LCALLNAMES(A, B, 1)                                                          \
  LCALLNAMES(A, B, 2) LCALLNAMES(A, B, 4) LCALLNAMES(A, B, 8)
#define LCALLNAME5(A, B)                                                       \
  LCALLNAMES(A, B, 1)                                                          \
  LCALLNAMES(A, B, 2)                                                          \
  LCALLNAMES(A, B, 4) LCALLNAMES(A, B, 8) LCALLNAMES(A, B, 16)

  if (TT.isWindowsArm64EC()) {
    LCALLNAME5(RTLIB::OUTLINE_ATOMIC_CAS, RTLIB::arm64ec___aarch64_cas)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_SWP, RTLIB::arm64ec___aarch64_swp)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDADD, RTLIB::arm64ec___aarch64_ldadd)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDSET, RTLIB::arm64ec___aarch64_ldset)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDCLR, RTLIB::arm64ec___aarch64_ldclr)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDEOR, RTLIB::arm64ec___aarch64_ldeor)
  } else {
    LCALLNAME5(RTLIB::OUTLINE_ATOMIC_CAS, RTLIB::__aarch64_cas)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_SWP, RTLIB::__aarch64_swp)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDADD, RTLIB::__aarch64_ldadd)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDSET, RTLIB::__aarch64_ldset)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDCLR, RTLIB::__aarch64_ldclr)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDEOR, RTLIB::__aarch64_ldeor)
  }
#undef LCALLNAMES
#undef LCALLNAME4
#undef LCALLNAME5
}

static void setARMLibcallNames(RuntimeLibcallsInfo &Info, const Triple &TT,
                               FloatABI::ABIType FloatABIType,
                               EABI EABIVersion) {
  if (!TT.isOSDarwin() && !TT.isiOS() && !TT.isWatchOS() && !TT.isDriverKit()) {
    CallingConv::ID DefaultCC = FloatABIType == FloatABI::Hard
                                    ? CallingConv::ARM_AAPCS_VFP
                                    : CallingConv::ARM_AAPCS;
    for (RTLIB::Libcall LC : RTLIB::libcalls())
      Info.setLibcallCallingConv(LC, DefaultCC);
  }

  // Register based DivRem for AEABI (RTABI 4.2)
  if (TT.isTargetAEABI() || TT.isAndroid() || TT.isTargetGNUAEABI() ||
      TT.isTargetMuslAEABI() || TT.isOSWindows()) {
    if (TT.isOSWindows()) {
      const struct {
        const RTLIB::Libcall Op;
        const RTLIB::LibcallImpl Impl;
        const CallingConv::ID CC;
      } LibraryCalls[] = {
          {RTLIB::SDIVREM_I32, RTLIB::__rt_sdiv, CallingConv::ARM_AAPCS},
          {RTLIB::SDIVREM_I64, RTLIB::__rt_sdiv64, CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I32, RTLIB::__rt_udiv, CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I64, RTLIB::__rt_udiv64, CallingConv::ARM_AAPCS},
      };

      for (const auto &LC : LibraryCalls) {
        Info.setLibcallImpl(LC.Op, LC.Impl);
        Info.setLibcallCallingConv(LC.Op, LC.CC);
      }
    } else {
      const struct {
        const RTLIB::Libcall Op;
        const RTLIB::LibcallImpl Impl;
        const CallingConv::ID CC;
      } LibraryCalls[] = {
          {RTLIB::SDIVREM_I32, RTLIB::__aeabi_idivmod, CallingConv::ARM_AAPCS},
          {RTLIB::SDIVREM_I64, RTLIB::__aeabi_ldivmod, CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I32, RTLIB::__aeabi_uidivmod, CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I64, RTLIB::__aeabi_uldivmod, CallingConv::ARM_AAPCS},
      };

      for (const auto &LC : LibraryCalls) {
        Info.setLibcallImpl(LC.Op, LC.Impl);
        Info.setLibcallCallingConv(LC.Op, LC.CC);
      }
    }
  }

  if (TT.isOSWindows()) {
    static const struct {
      const RTLIB::Libcall Op;
      const RTLIB::LibcallImpl Impl;
      const CallingConv::ID CC;
    } LibraryCalls[] = {
        {RTLIB::FPTOSINT_F32_I64, RTLIB::__stoi64, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::FPTOSINT_F64_I64, RTLIB::__dtoi64, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::FPTOUINT_F32_I64, RTLIB::__stou64, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::FPTOUINT_F64_I64, RTLIB::__dtou64, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::SINTTOFP_I64_F32, RTLIB::__i64tos, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::SINTTOFP_I64_F64, RTLIB::__i64tod, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::UINTTOFP_I64_F32, RTLIB::__u64tos, CallingConv::ARM_AAPCS_VFP},
        {RTLIB::UINTTOFP_I64_F64, RTLIB::__u64tod, CallingConv::ARM_AAPCS_VFP},
    };

    for (const auto &LC : LibraryCalls) {
      Info.setLibcallImpl(LC.Op, LC.Impl);
      Info.setLibcallCallingConv(LC.Op, LC.CC);
    }
  }

  // Use divmod compiler-rt calls for iOS 5.0 and later.
  if (TT.isOSBinFormatMachO() && (!TT.isiOS() || !TT.isOSVersionLT(5, 0))) {
    Info.setLibcallImpl(RTLIB::SDIVREM_I32, RTLIB::__divmodsi4);
    Info.setLibcallImpl(RTLIB::UDIVREM_I32, RTLIB::__udivmodsi4);
  }
}

static void setMSP430Libcalls(RuntimeLibcallsInfo &Info, const Triple &TT) {
  // EABI Libcalls - EABI Section 6.2
  const struct {
    const RTLIB::Libcall Op;
    const RTLIB::LibcallImpl Impl;
  } LibraryCalls[] = {
      // Floating point conversions - EABI Table 6
      {RTLIB::FPROUND_F64_F32, RTLIB::__mspabi_cvtdf},
      {RTLIB::FPEXT_F32_F64, RTLIB::__mspabi_cvtfd},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOSINT_F64_I16,  RTLIB::__mspabi_fixdi },
      {RTLIB::FPTOSINT_F64_I32, RTLIB::__mspabi_fixdli},
      {RTLIB::FPTOSINT_F64_I64, RTLIB::__mspabi_fixdlli},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOUINT_F64_I16,  RTLIB::__mspabi_fixdu },
      {RTLIB::FPTOUINT_F64_I32, RTLIB::__mspabi_fixdul},
      {RTLIB::FPTOUINT_F64_I64, RTLIB::__mspabi_fixdull},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOSINT_F32_I16,  RTLIB::__mspabi_fixfi },
      {RTLIB::FPTOSINT_F32_I32, RTLIB::__mspabi_fixfli},
      {RTLIB::FPTOSINT_F32_I64, RTLIB::__mspabi_fixflli},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOUINT_F32_I16,  RTLIB::__mspabi_fixfu },
      {RTLIB::FPTOUINT_F32_I32, RTLIB::__mspabi_fixful},
      {RTLIB::FPTOUINT_F32_I64, RTLIB::__mspabi_fixfull},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::SINTTOFP_I16_F64,  RTLIB::__mspabi_fltid },
      {RTLIB::SINTTOFP_I32_F64, RTLIB::__mspabi_fltlid},
      // TODO The following IS implemented in libgcc but is not in the EABI
      {RTLIB::SINTTOFP_I64_F64, RTLIB::__mspabi_fltllid},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::UINTTOFP_I16_F64,  RTLIB::__mspabi_fltud },
      {RTLIB::UINTTOFP_I32_F64, RTLIB::__mspabi_fltuld},
      // The following IS implemented in libgcc but is not in the EABI
      {RTLIB::UINTTOFP_I64_F64, RTLIB::__mspabi_fltulld},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::SINTTOFP_I16_F32,  RTLIB::__mspabi_fltif },
      {RTLIB::SINTTOFP_I32_F32, RTLIB::__mspabi_fltlif},
      // TODO The following IS implemented in libgcc but is not in the EABI
      {RTLIB::SINTTOFP_I64_F32, RTLIB::__mspabi_fltllif},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::UINTTOFP_I16_F32,  RTLIB::__mspabi_fltuf },
      {RTLIB::UINTTOFP_I32_F32, RTLIB::__mspabi_fltulf},
      // The following IS implemented in libgcc but is not in the EABI
      {RTLIB::UINTTOFP_I64_F32, RTLIB::__mspabi_fltullf},

      // Floating point comparisons - EABI Table 7
      {RTLIB::OEQ_F64, RTLIB::__mspabi_cmpd__oeq},
      {RTLIB::UNE_F64, RTLIB::__mspabi_cmpd__une},
      {RTLIB::OGE_F64, RTLIB::__mspabi_cmpd__oge},
      {RTLIB::OLT_F64, RTLIB::__mspabi_cmpd__olt},
      {RTLIB::OLE_F64, RTLIB::__mspabi_cmpd__ole},
      {RTLIB::OGT_F64, RTLIB::__mspabi_cmpd__ogt},
      {RTLIB::OEQ_F32, RTLIB::__mspabi_cmpf__oeq},
      {RTLIB::UNE_F32, RTLIB::__mspabi_cmpf__une},
      {RTLIB::OGE_F32, RTLIB::__mspabi_cmpf__oge},
      {RTLIB::OLT_F32, RTLIB::__mspabi_cmpf__olt},
      {RTLIB::OLE_F32, RTLIB::__mspabi_cmpf__ole},
      {RTLIB::OGT_F32, RTLIB::__mspabi_cmpf__ogt},

      // Floating point arithmetic - EABI Table 8
      {RTLIB::ADD_F64, RTLIB::__mspabi_addd},
      {RTLIB::ADD_F32, RTLIB::__mspabi_addf},
      {RTLIB::DIV_F64, RTLIB::__mspabi_divd},
      {RTLIB::DIV_F32, RTLIB::__mspabi_divf},
      {RTLIB::MUL_F64, RTLIB::__mspabi_mpyd},
      {RTLIB::MUL_F32, RTLIB::__mspabi_mpyf},
      {RTLIB::SUB_F64, RTLIB::__mspabi_subd},
      {RTLIB::SUB_F32, RTLIB::__mspabi_subf},
      // The following are NOT implemented in libgcc
      // { RTLIB::NEG_F64,  RTLIB::__mspabi_negd },
      // { RTLIB::NEG_F32,  RTLIB::__mspabi_negf },

      // Universal Integer Operations - EABI Table 9
      {RTLIB::SDIV_I16, RTLIB::__mspabi_divi},
      {RTLIB::SDIV_I32, RTLIB::__mspabi_divli},
      {RTLIB::SDIV_I64, RTLIB::__mspabi_divlli},
      {RTLIB::UDIV_I16, RTLIB::__mspabi_divu},
      {RTLIB::UDIV_I32, RTLIB::__mspabi_divul},
      {RTLIB::UDIV_I64, RTLIB::__mspabi_divull},
      {RTLIB::SREM_I16, RTLIB::__mspabi_remi},
      {RTLIB::SREM_I32, RTLIB::__mspabi_remli},
      {RTLIB::SREM_I64, RTLIB::__mspabi_remlli},
      {RTLIB::UREM_I16, RTLIB::__mspabi_remu},
      {RTLIB::UREM_I32, RTLIB::__mspabi_remul},
      {RTLIB::UREM_I64, RTLIB::__mspabi_remull},

      // Bitwise Operations - EABI Table 10
      // TODO: __mspabi_[srli/srai/slli] ARE implemented in libgcc
      {RTLIB::SRL_I32, RTLIB::__mspabi_srll},
      {RTLIB::SRA_I32, RTLIB::__mspabi_sral},
      {RTLIB::SHL_I32, RTLIB::__mspabi_slll},
      // __mspabi_[srlll/srall/sllll/rlli/rlll] are NOT implemented in libgcc
  };

  for (const auto &LC : LibraryCalls)
    Info.setLibcallImpl(LC.Op, LC.Impl);

  // Several of the runtime library functions use a special calling conv
  Info.setLibcallCallingConv(RTLIB::UDIV_I64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::UREM_I64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::SDIV_I64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::SREM_I64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::ADD_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::SUB_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::MUL_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::DIV_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::OEQ_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::UNE_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::OGE_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::OLT_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::OLE_F64, CallingConv::MSP430_BUILTIN);
  Info.setLibcallCallingConv(RTLIB::OGT_F64, CallingConv::MSP430_BUILTIN);

  // TODO: __mspabi_srall, __mspabi_srlll, __mspabi_sllll
}

void RuntimeLibcallsInfo::initSoftFloatCmpLibcallPredicates() {
  SoftFloatCompareLibcallPredicates[RTLIB::OEQ_F32] = CmpInst::ICMP_EQ;
  SoftFloatCompareLibcallPredicates[RTLIB::OEQ_F64] = CmpInst::ICMP_EQ;
  SoftFloatCompareLibcallPredicates[RTLIB::OEQ_F128] = CmpInst::ICMP_EQ;
  SoftFloatCompareLibcallPredicates[RTLIB::OEQ_PPCF128] = CmpInst::ICMP_EQ;
  SoftFloatCompareLibcallPredicates[RTLIB::UNE_F32] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::UNE_F64] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::UNE_F128] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::UNE_PPCF128] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::OGE_F32] = CmpInst::ICMP_SGE;
  SoftFloatCompareLibcallPredicates[RTLIB::OGE_F64] = CmpInst::ICMP_SGE;
  SoftFloatCompareLibcallPredicates[RTLIB::OGE_F128] = CmpInst::ICMP_SGE;
  SoftFloatCompareLibcallPredicates[RTLIB::OGE_PPCF128] = CmpInst::ICMP_SGE;
  SoftFloatCompareLibcallPredicates[RTLIB::OLT_F32] = CmpInst::ICMP_SLT;
  SoftFloatCompareLibcallPredicates[RTLIB::OLT_F64] = CmpInst::ICMP_SLT;
  SoftFloatCompareLibcallPredicates[RTLIB::OLT_F128] = CmpInst::ICMP_SLT;
  SoftFloatCompareLibcallPredicates[RTLIB::OLT_PPCF128] = CmpInst::ICMP_SLT;
  SoftFloatCompareLibcallPredicates[RTLIB::OLE_F32] = CmpInst::ICMP_SLE;
  SoftFloatCompareLibcallPredicates[RTLIB::OLE_F64] = CmpInst::ICMP_SLE;
  SoftFloatCompareLibcallPredicates[RTLIB::OLE_F128] = CmpInst::ICMP_SLE;
  SoftFloatCompareLibcallPredicates[RTLIB::OLE_PPCF128] = CmpInst::ICMP_SLE;
  SoftFloatCompareLibcallPredicates[RTLIB::OGT_F32] = CmpInst::ICMP_SGT;
  SoftFloatCompareLibcallPredicates[RTLIB::OGT_F64] = CmpInst::ICMP_SGT;
  SoftFloatCompareLibcallPredicates[RTLIB::OGT_F128] = CmpInst::ICMP_SGT;
  SoftFloatCompareLibcallPredicates[RTLIB::OGT_PPCF128] = CmpInst::ICMP_SGT;
  SoftFloatCompareLibcallPredicates[RTLIB::UO_F32] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::UO_F64] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::UO_F128] = CmpInst::ICMP_NE;
  SoftFloatCompareLibcallPredicates[RTLIB::UO_PPCF128] = CmpInst::ICMP_NE;
}

static void setLongDoubleIsF128Libm(RuntimeLibcallsInfo &Info,
                                    bool FiniteOnlyFuncs = false) {
  Info.setLibcallImpl(RTLIB::REM_F128, RTLIB::fmodf128);
  Info.setLibcallImpl(RTLIB::FMA_F128, RTLIB::fmaf128);
  Info.setLibcallImpl(RTLIB::SQRT_F128, RTLIB::sqrtf128);
  Info.setLibcallImpl(RTLIB::CBRT_F128, RTLIB::cbrtf128);
  Info.setLibcallImpl(RTLIB::LOG_F128, RTLIB::logf128);
  Info.setLibcallImpl(RTLIB::LOG2_F128, RTLIB::log2f128);
  Info.setLibcallImpl(RTLIB::LOG10_F128, RTLIB::log10f128);
  Info.setLibcallImpl(RTLIB::EXP_F128, RTLIB::expf128);
  Info.setLibcallImpl(RTLIB::EXP2_F128, RTLIB::exp2f128);
  Info.setLibcallImpl(RTLIB::EXP10_F128, RTLIB::exp10f128);
  Info.setLibcallImpl(RTLIB::SIN_F128, RTLIB::sinf128);
  Info.setLibcallImpl(RTLIB::COS_F128, RTLIB::cosf128);
  Info.setLibcallImpl(RTLIB::TAN_F128, RTLIB::tanf128);
  Info.setLibcallImpl(RTLIB::SINCOS_F128, RTLIB::sincosf128);
  Info.setLibcallImpl(RTLIB::ASIN_F128, RTLIB::asinf128);
  Info.setLibcallImpl(RTLIB::ACOS_F128, RTLIB::acosf128);
  Info.setLibcallImpl(RTLIB::ATAN_F128, RTLIB::atanf128);
  Info.setLibcallImpl(RTLIB::ATAN2_F128, RTLIB::atan2f128);
  Info.setLibcallImpl(RTLIB::SINH_F128, RTLIB::sinhf128);
  Info.setLibcallImpl(RTLIB::COSH_F128, RTLIB::coshf128);
  Info.setLibcallImpl(RTLIB::TANH_F128, RTLIB::tanhf128);
  Info.setLibcallImpl(RTLIB::POW_F128, RTLIB::powf128);
  Info.setLibcallImpl(RTLIB::CEIL_F128, RTLIB::ceilf128);
  Info.setLibcallImpl(RTLIB::TRUNC_F128, RTLIB::truncf128);
  Info.setLibcallImpl(RTLIB::RINT_F128, RTLIB::rintf128);
  Info.setLibcallImpl(RTLIB::NEARBYINT_F128, RTLIB::nearbyintf128);
  Info.setLibcallImpl(RTLIB::ROUND_F128, RTLIB::roundf128);
  Info.setLibcallImpl(RTLIB::ROUNDEVEN_F128, RTLIB::roundevenf128);
  Info.setLibcallImpl(RTLIB::FLOOR_F128, RTLIB::floorf128);
  Info.setLibcallImpl(RTLIB::COPYSIGN_F128, RTLIB::copysignf128);
  Info.setLibcallImpl(RTLIB::FMIN_F128, RTLIB::fminf128);
  Info.setLibcallImpl(RTLIB::FMAX_F128, RTLIB::fmaxf128);
  Info.setLibcallImpl(RTLIB::FMINIMUM_F128, RTLIB::fminimumf128);
  Info.setLibcallImpl(RTLIB::FMAXIMUM_F128, RTLIB::fmaximumf128);
  Info.setLibcallImpl(RTLIB::FMINIMUM_NUM_F128, RTLIB::fminimum_numf128);
  Info.setLibcallImpl(RTLIB::FMAXIMUM_NUM_F128, RTLIB::fmaximum_numf128);
  Info.setLibcallImpl(RTLIB::LROUND_F128, RTLIB::lroundf128);
  Info.setLibcallImpl(RTLIB::LLROUND_F128, RTLIB::llroundf128);
  Info.setLibcallImpl(RTLIB::LRINT_F128, RTLIB::lrintf128);
  Info.setLibcallImpl(RTLIB::LLRINT_F128, RTLIB::llrintf128);
  Info.setLibcallImpl(RTLIB::LDEXP_F128, RTLIB::ldexpf128);
  Info.setLibcallImpl(RTLIB::FREXP_F128, RTLIB::frexpf128);
  Info.setLibcallImpl(RTLIB::MODF_F128, RTLIB::modff128);

  if (FiniteOnlyFuncs) {
    Info.setLibcallImpl(RTLIB::LOG_FINITE_F128, RTLIB::__logf128_finite);
    Info.setLibcallImpl(RTLIB::LOG2_FINITE_F128, RTLIB::__log2f128_finite);
    Info.setLibcallImpl(RTLIB::LOG10_FINITE_F128, RTLIB::__log10f128_finite);
    Info.setLibcallImpl(RTLIB::EXP_FINITE_F128, RTLIB::__expf128_finite);
    Info.setLibcallImpl(RTLIB::EXP2_FINITE_F128, RTLIB::__exp2f128_finite);
    Info.setLibcallImpl(RTLIB::POW_FINITE_F128, RTLIB::__powf128_finite);
  } else {
    Info.setLibcallImpl(RTLIB::LOG_FINITE_F128, RTLIB::Unsupported);
    Info.setLibcallImpl(RTLIB::LOG2_FINITE_F128, RTLIB::Unsupported);
    Info.setLibcallImpl(RTLIB::LOG10_FINITE_F128, RTLIB::Unsupported);
    Info.setLibcallImpl(RTLIB::EXP_FINITE_F128, RTLIB::Unsupported);
    Info.setLibcallImpl(RTLIB::EXP2_FINITE_F128, RTLIB::Unsupported);
    Info.setLibcallImpl(RTLIB::POW_FINITE_F128, RTLIB::Unsupported);
  }
}

void RTLIB::RuntimeLibcallsInfo::initDefaultLibCallImpls() {
  std::memcpy(LibcallImpls, DefaultLibcallImpls, sizeof(LibcallImpls));
  static_assert(sizeof(LibcallImpls) == sizeof(DefaultLibcallImpls),
                "libcall array size should match");
}

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       EABI EABIVersion, StringRef ABIName) {
  // Use the f128 variants of math functions on x86
  if (TT.isX86() && TT.isGNUEnvironment())
    setLongDoubleIsF128Libm(*this, /*FiniteOnlyFuncs=*/true);

  if (TT.isX86() || TT.isVE()) {
    if (ExceptionModel == ExceptionHandling::SjLj)
      setLibcallImpl(RTLIB::UNWIND_RESUME, RTLIB::_Unwind_SjLj_Resume);
  }

  if (TT.isPPC()) {
    setPPCLibCallNameOverrides();

    // TODO: Do the finite only functions exist?
    setLongDoubleIsF128Libm(*this, /*FiniteOnlyFuncs=*/false);

    // TODO: Tablegen predicate support
    if (TT.isOSAIX()) {
      if (TT.isPPC64()) {
        setLibcallImpl(RTLIB::MEMCPY, RTLIB::Unsupported);
        setLibcallImpl(RTLIB::MEMMOVE, RTLIB::___memmove64);
        setLibcallImpl(RTLIB::MEMSET, RTLIB::___memset64);
        setLibcallImpl(RTLIB::BZERO, RTLIB::___bzero64);
      } else {
        setLibcallImpl(RTLIB::MEMCPY, RTLIB::Unsupported);
        setLibcallImpl(RTLIB::MEMMOVE, RTLIB::___memmove);
        setLibcallImpl(RTLIB::MEMSET, RTLIB::___memset);
        setLibcallImpl(RTLIB::BZERO, RTLIB::___bzero);
      }
    }
  }

  // A few names are different on particular architectures or environments.
  if (TT.isOSDarwin()) {
    // For f16/f32 conversions, Darwin uses the standard naming scheme,
    // instead of the gnueabi-style __gnu_*_ieee.
    // FIXME: What about other targets?
    setLibcallImpl(RTLIB::FPEXT_F16_F32, RTLIB::__extendhfsf2);
    setLibcallImpl(RTLIB::FPROUND_F32_F16, RTLIB::__truncsfhf2);

    // Some darwins have an optimized __bzero/bzero function.
    if (TT.isX86()) {
      if (TT.isMacOSX() && !TT.isMacOSXVersionLT(10, 6))
        setLibcallImpl(RTLIB::BZERO, RTLIB::__bzero);
    } else if (TT.isAArch64())
      setLibcallImpl(RTLIB::BZERO, RTLIB::bzero);

    if (darwinHasSinCosStret(TT)) {
      setLibcallImpl(RTLIB::SINCOS_STRET_F32, RTLIB::__sincosf_stret);
      setLibcallImpl(RTLIB::SINCOS_STRET_F64, RTLIB::__sincos_stret);
      if (TT.isWatchABI()) {
        setLibcallCallingConv(RTLIB::SINCOS_STRET_F32,
                              CallingConv::ARM_AAPCS_VFP);
        setLibcallCallingConv(RTLIB::SINCOS_STRET_F64,
                              CallingConv::ARM_AAPCS_VFP);
      }
    }

    if (darwinHasExp10(TT)) {
      setLibcallImpl(RTLIB::EXP10_F32, RTLIB::__exp10f);
      setLibcallImpl(RTLIB::EXP10_F64, RTLIB::__exp10);
    } else {
      setLibcallImpl(RTLIB::EXP10_F32, RTLIB::Unsupported);
      setLibcallImpl(RTLIB::EXP10_F64, RTLIB::Unsupported);
    }
  }

  if (hasSinCos(TT)) {
    setLibcallImpl(RTLIB::SINCOS_F32, RTLIB::sincosf);
    setLibcallImpl(RTLIB::SINCOS_F64, RTLIB::sincos);
    setLibcallImpl(RTLIB::SINCOS_F80, RTLIB::sincos_f80);
    setLibcallImpl(RTLIB::SINCOS_F128, RTLIB::sincos_f128);
    setLibcallImpl(RTLIB::SINCOS_PPCF128, RTLIB::sincos_ppcf128);
  }

  if (TT.isPS()) {
    setLibcallImpl(RTLIB::SINCOS_F32, RTLIB::sincosf);
    setLibcallImpl(RTLIB::SINCOS_F64, RTLIB::sincos);
  }

  if (TT.isOSOpenBSD()) {
    setLibcallImpl(RTLIB::STACKPROTECTOR_CHECK_FAIL, RTLIB::Unsupported);
  }

  if (TT.isOSWindows() && !TT.isOSCygMing()) {
    setLibcallImpl(RTLIB::LDEXP_F32, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::LDEXP_F80, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::LDEXP_F128, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::LDEXP_PPCF128, RTLIB::Unsupported);

    setLibcallImpl(RTLIB::FREXP_F32, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::FREXP_F80, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::FREXP_F128, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::FREXP_PPCF128, RTLIB::Unsupported);
  }

  // Disable most libcalls on AMDGPU and NVPTX.
  if (TT.isAMDGPU() || TT.isNVPTX()) {
    for (RTLIB::Libcall LC : RTLIB::libcalls()) {
      if (!isAtomicLibCall(LC))
        setLibcallImpl(LC, RTLIB::Unsupported);
    }
  }

  if (TT.isOSMSVCRT()) {
    // MSVCRT doesn't have powi; fall back to pow
    setLibcallImpl(RTLIB::POWI_F32, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::POWI_F64, RTLIB::Unsupported);
  }

  // Setup Windows compiler runtime calls.
  if (TT.getArch() == Triple::x86 &&
      (TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment())) {
    static const struct {
      const RTLIB::Libcall Op;
      const RTLIB::LibcallImpl Impl;
      const CallingConv::ID CC;
    } LibraryCalls[] = {
        {RTLIB::SDIV_I64, RTLIB::_alldiv, CallingConv::X86_StdCall},
        {RTLIB::UDIV_I64, RTLIB::_aulldiv, CallingConv::X86_StdCall},
        {RTLIB::SREM_I64, RTLIB::_allrem, CallingConv::X86_StdCall},
        {RTLIB::UREM_I64, RTLIB::_aullrem, CallingConv::X86_StdCall},
        {RTLIB::MUL_I64, RTLIB::_allmul, CallingConv::X86_StdCall},
    };

    for (const auto &LC : LibraryCalls) {
      setLibcallImpl(LC.Op, LC.Impl);
      setLibcallCallingConv(LC.Op, LC.CC);
    }
  }

  if (TT.isAArch64()) {
    if (TT.isWindowsArm64EC()) {
      setWindowsArm64LibCallNameOverrides();
      setLibcallImpl(RTLIB::SC_MEMCPY, RTLIB::arm64ec___arm_sc_memcpy);
      setLibcallImpl(RTLIB::SC_MEMMOVE, RTLIB::arm64ec___arm_sc_memmove);
      setLibcallImpl(RTLIB::SC_MEMSET, RTLIB::arm64ec___arm_sc_memset);
    } else {
      setLibcallImpl(RTLIB::SC_MEMCPY, RTLIB::__arm_sc_memcpy);
      setLibcallImpl(RTLIB::SC_MEMMOVE, RTLIB::__arm_sc_memmove);
      setLibcallImpl(RTLIB::SC_MEMSET, RTLIB::__arm_sc_memset);
    }

    setAArch64LibcallNames(*this, TT);
  } else if (TT.isARM() || TT.isThumb()) {
    setARMLibcallNames(*this, TT, FloatABI, EABIVersion);
  } else if (TT.getArch() == Triple::ArchType::avr) {
    // Division rtlib functions (not supported), use divmod functions instead
    setLibcallImpl(RTLIB::SDIV_I8, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::SDIV_I16, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::SDIV_I32, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::UDIV_I8, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::UDIV_I16, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::UDIV_I32, RTLIB::Unsupported);

    // Modulus rtlib functions (not supported), use divmod functions instead
    setLibcallImpl(RTLIB::SREM_I8, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::SREM_I16, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::SREM_I32, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::UREM_I8, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::UREM_I16, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::UREM_I32, RTLIB::Unsupported);

    // Division and modulus rtlib functions
    setLibcallImpl(RTLIB::SDIVREM_I8, RTLIB::__divmodqi4);
    setLibcallImpl(RTLIB::SDIVREM_I16, RTLIB::__divmodhi4);
    setLibcallImpl(RTLIB::SDIVREM_I32, RTLIB::__divmodsi4);
    setLibcallImpl(RTLIB::UDIVREM_I8, RTLIB::__udivmodqi4);
    setLibcallImpl(RTLIB::UDIVREM_I16, RTLIB::__udivmodhi4);
    setLibcallImpl(RTLIB::UDIVREM_I32, RTLIB::__udivmodsi4);

    // Several of the runtime library functions use a special calling conv
    setLibcallCallingConv(RTLIB::SDIVREM_I8, CallingConv::AVR_BUILTIN);
    setLibcallCallingConv(RTLIB::SDIVREM_I16, CallingConv::AVR_BUILTIN);
    setLibcallCallingConv(RTLIB::UDIVREM_I8, CallingConv::AVR_BUILTIN);
    setLibcallCallingConv(RTLIB::UDIVREM_I16, CallingConv::AVR_BUILTIN);

    // Trigonometric rtlib functions
    setLibcallImpl(RTLIB::SIN_F32, RTLIB::avr_sin);
    setLibcallImpl(RTLIB::COS_F32, RTLIB::avr_cos);
  }

  if (!TT.isWasm()) {
    // These libcalls are only available in compiler-rt, not libgcc.
    if (TT.isArch32Bit()) {
      setLibcallImpl(RTLIB::SHL_I128, RTLIB::Unsupported);
      setLibcallImpl(RTLIB::SRL_I128, RTLIB::Unsupported);
      setLibcallImpl(RTLIB::SRA_I128, RTLIB::Unsupported);
      setLibcallImpl(RTLIB::MUL_I128, RTLIB::Unsupported);
      setLibcallImpl(RTLIB::MULO_I64, RTLIB::Unsupported);
    }

    setLibcallImpl(RTLIB::MULO_I128, RTLIB::Unsupported);
  } else {
    // Define the emscripten name for return address helper.
    // TODO: when implementing other Wasm backends, make this generic or only do
    // this on emscripten depending on what they end up doing.
    setLibcallImpl(RTLIB::RETURN_ADDRESS, RTLIB::emscripten_return_address);
  }

  if (TT.getArch() == Triple::ArchType::hexagon) {
    setLibcallImpl(RTLIB::SDIV_I32, RTLIB::__hexagon_divsi3);
    setLibcallImpl(RTLIB::SDIV_I64, RTLIB::__hexagon_divdi3);
    setLibcallImpl(RTLIB::UDIV_I32, RTLIB::__hexagon_udivsi3);
    setLibcallImpl(RTLIB::UDIV_I64, RTLIB::__hexagon_udivdi3);
    setLibcallImpl(RTLIB::SREM_I32, RTLIB::__hexagon_modsi3);
    setLibcallImpl(RTLIB::SREM_I64, RTLIB::__hexagon_moddi3);
    setLibcallImpl(RTLIB::UREM_I32, RTLIB::__hexagon_umodsi3);
    setLibcallImpl(RTLIB::UREM_I64, RTLIB::__hexagon_umoddi3);

    const bool FastMath = HexagonEnableFastMathRuntimeCalls;
    // This is the only fast library function for sqrtd.
    if (FastMath)
      setLibcallImpl(RTLIB::SQRT_F64, RTLIB::__hexagon_fast2_sqrtdf2);

    // Prefix is: nothing  for "slow-math",
    //            "fast2_" for V5+ fast-math double-precision
    // (actually, keep fast-math and fast-math2 separate for now)
    if (FastMath) {
      setLibcallImpl(RTLIB::ADD_F64, RTLIB::__hexagon_fast_adddf3);
      setLibcallImpl(RTLIB::SUB_F64, RTLIB::__hexagon_fast_subdf3);
      setLibcallImpl(RTLIB::MUL_F64, RTLIB::__hexagon_fast_muldf3);
      setLibcallImpl(RTLIB::DIV_F64, RTLIB::__hexagon_fast_divdf3);
      setLibcallImpl(RTLIB::DIV_F32, RTLIB::__hexagon_fast_divsf3);
    } else {
      setLibcallImpl(RTLIB::ADD_F64, RTLIB::__hexagon_adddf3);
      setLibcallImpl(RTLIB::SUB_F64, RTLIB::__hexagon_subdf3);
      setLibcallImpl(RTLIB::MUL_F64, RTLIB::__hexagon_muldf3);
      setLibcallImpl(RTLIB::DIV_F64, RTLIB::__hexagon_divdf3);
      setLibcallImpl(RTLIB::DIV_F32, RTLIB::__hexagon_divsf3);
    }

    if (FastMath)
      setLibcallImpl(RTLIB::SQRT_F32, RTLIB::__hexagon_fast2_sqrtf);
    else
      setLibcallImpl(RTLIB::SQRT_F32, RTLIB::__hexagon_sqrtf);

    setLibcallImpl(
        RTLIB::HEXAGON_MEMCPY_LIKELY_ALIGNED_MIN32BYTES_MULT8BYTES,
        RTLIB::__hexagon_memcpy_likely_aligned_min32bytes_mult8bytes);
  }

  if (TT.getArch() == Triple::ArchType::msp430)
    setMSP430Libcalls(*this, TT);

  if (TT.isSystemZ() && TT.isOSzOS())
    setZOSLibCallNameOverrides();

  if (TT.getArch() == Triple::ArchType::xcore)
    setLibcallImpl(RTLIB::MEMCPY_ALIGN_4, RTLIB::__memcpy_4);
}

bool RuntimeLibcallsInfo::darwinHasExp10(const Triple &TT) {
  assert(TT.isOSDarwin() && "should be called with darwin triple");

  switch (TT.getOS()) {
  case Triple::MacOSX:
    return !TT.isMacOSXVersionLT(10, 9);
  case Triple::IOS:
    return !TT.isOSVersionLT(7, 0);
  case Triple::DriverKit:
  case Triple::TvOS:
  case Triple::WatchOS:
  case Triple::XROS:
  case Triple::BridgeOS:
    return true;
  default:
    return false;
  }
}
