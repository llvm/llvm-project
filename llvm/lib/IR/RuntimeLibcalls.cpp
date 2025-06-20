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

static cl::opt<bool>
    HexagonEnableFastMathRuntimeCalls("hexagon-fast-math", cl::Hidden,
                                      cl::desc("Enable Fast Math processing"));

static void setAArch64LibcallNames(RuntimeLibcallsInfo &Info,
                                   const Triple &TT) {
#define LCALLNAMES(A, B, N)                                                    \
  Info.setLibcallName(A##N##_RELAX, #B #N "_relax");                           \
  Info.setLibcallName(A##N##_ACQ, #B #N "_acq");                               \
  Info.setLibcallName(A##N##_REL, #B #N "_rel");                               \
  Info.setLibcallName(A##N##_ACQ_REL, #B #N "_acq_rel");
#define LCALLNAME4(A, B)                                                       \
  LCALLNAMES(A, B, 1)                                                          \
  LCALLNAMES(A, B, 2) LCALLNAMES(A, B, 4) LCALLNAMES(A, B, 8)
#define LCALLNAME5(A, B)                                                       \
  LCALLNAMES(A, B, 1)                                                          \
  LCALLNAMES(A, B, 2)                                                          \
  LCALLNAMES(A, B, 4) LCALLNAMES(A, B, 8) LCALLNAMES(A, B, 16)
  LCALLNAME5(RTLIB::OUTLINE_ATOMIC_CAS, __aarch64_cas)
  LCALLNAME4(RTLIB::OUTLINE_ATOMIC_SWP, __aarch64_swp)
  LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDADD, __aarch64_ldadd)
  LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDSET, __aarch64_ldset)
  LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDCLR, __aarch64_ldclr)
  LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDEOR, __aarch64_ldeor)

  if (TT.isWindowsArm64EC()) {
    // FIXME: are there calls we need to exclude from this?
#define HANDLE_LIBCALL(code, name)                                             \
  if (sizeof(name) != 1) {                                                     \
    const char *libcallName = Info.getLibcallName(RTLIB::code);                \
    if (libcallName && libcallName[0] != '#') {                                \
      assert(strcmp(libcallName, name) == 0 && "Unexpected name");             \
      Info.setLibcallName(RTLIB::code, "#" name);                              \
    }                                                                          \
  }
#define LIBCALL_NO_NAME ""
#include "llvm/IR/RuntimeLibcalls.def"
#undef HANDLE_LIBCALL
#undef LIBCALL_NO_NAME

    LCALLNAME5(RTLIB::OUTLINE_ATOMIC_CAS, #__aarch64_cas)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_SWP, #__aarch64_swp)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDADD, #__aarch64_ldadd)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDSET, #__aarch64_ldset)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDCLR, #__aarch64_ldclr)
    LCALLNAME4(RTLIB::OUTLINE_ATOMIC_LDEOR, #__aarch64_ldeor)
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
        const char *const Name;
        const CallingConv::ID CC;
      } LibraryCalls[] = {
          {RTLIB::SDIVREM_I32, "__rt_sdiv", CallingConv::ARM_AAPCS},
          {RTLIB::SDIVREM_I64, "__rt_sdiv64", CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I32, "__rt_udiv", CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I64, "__rt_udiv64", CallingConv::ARM_AAPCS},
      };

      for (const auto &LC : LibraryCalls) {
        Info.setLibcallName(LC.Op, LC.Name);
        Info.setLibcallCallingConv(LC.Op, LC.CC);
      }
    } else {
      const struct {
        const RTLIB::Libcall Op;
        const char *const Name;
        const CallingConv::ID CC;
      } LibraryCalls[] = {
          {RTLIB::SDIVREM_I32, "__aeabi_idivmod", CallingConv::ARM_AAPCS},
          {RTLIB::SDIVREM_I64, "__aeabi_ldivmod", CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I32, "__aeabi_uidivmod", CallingConv::ARM_AAPCS},
          {RTLIB::UDIVREM_I64, "__aeabi_uldivmod", CallingConv::ARM_AAPCS},
      };

      for (const auto &LC : LibraryCalls) {
        Info.setLibcallName(LC.Op, LC.Name);
        Info.setLibcallCallingConv(LC.Op, LC.CC);
      }
    }
  }

  if (TT.isOSWindows()) {
    static const struct {
      const RTLIB::Libcall Op;
      const char *const Name;
      const CallingConv::ID CC;
    } LibraryCalls[] = {
        {RTLIB::FPTOSINT_F32_I64, "__stoi64", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::FPTOSINT_F64_I64, "__dtoi64", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::FPTOUINT_F32_I64, "__stou64", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::FPTOUINT_F64_I64, "__dtou64", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::SINTTOFP_I64_F32, "__i64tos", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::SINTTOFP_I64_F64, "__i64tod", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::UINTTOFP_I64_F32, "__u64tos", CallingConv::ARM_AAPCS_VFP},
        {RTLIB::UINTTOFP_I64_F64, "__u64tod", CallingConv::ARM_AAPCS_VFP},
    };

    for (const auto &LC : LibraryCalls) {
      Info.setLibcallName(LC.Op, LC.Name);
      Info.setLibcallCallingConv(LC.Op, LC.CC);
    }
  }

  // Use divmod compiler-rt calls for iOS 5.0 and later.
  if (TT.isOSBinFormatMachO() && (!TT.isiOS() || !TT.isOSVersionLT(5, 0))) {
    Info.setLibcallName(RTLIB::SDIVREM_I32, "__divmodsi4");
    Info.setLibcallName(RTLIB::UDIVREM_I32, "__udivmodsi4");
  }
}

static void setMSP430Libcalls(RuntimeLibcallsInfo &Info, const Triple &TT) {
  // EABI Libcalls - EABI Section 6.2
  const struct {
    const RTLIB::Libcall Op;
    const char *const Name;
  } LibraryCalls[] = {
      // Floating point conversions - EABI Table 6
      {RTLIB::FPROUND_F64_F32, "__mspabi_cvtdf"},
      {RTLIB::FPEXT_F32_F64, "__mspabi_cvtfd"},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOSINT_F64_I16,  "__mspabi_fixdi" },
      {RTLIB::FPTOSINT_F64_I32, "__mspabi_fixdli"},
      {RTLIB::FPTOSINT_F64_I64, "__mspabi_fixdlli"},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOUINT_F64_I16,  "__mspabi_fixdu" },
      {RTLIB::FPTOUINT_F64_I32, "__mspabi_fixdul"},
      {RTLIB::FPTOUINT_F64_I64, "__mspabi_fixdull"},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOSINT_F32_I16,  "__mspabi_fixfi" },
      {RTLIB::FPTOSINT_F32_I32, "__mspabi_fixfli"},
      {RTLIB::FPTOSINT_F32_I64, "__mspabi_fixflli"},
      // The following is NOT implemented in libgcc
      //{ RTLIB::FPTOUINT_F32_I16,  "__mspabi_fixfu" },
      {RTLIB::FPTOUINT_F32_I32, "__mspabi_fixful"},
      {RTLIB::FPTOUINT_F32_I64, "__mspabi_fixfull"},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::SINTTOFP_I16_F64,  "__mspabi_fltid" },
      {RTLIB::SINTTOFP_I32_F64, "__mspabi_fltlid"},
      // TODO The following IS implemented in libgcc but is not in the EABI
      {RTLIB::SINTTOFP_I64_F64, "__mspabi_fltllid"},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::UINTTOFP_I16_F64,  "__mspabi_fltud" },
      {RTLIB::UINTTOFP_I32_F64, "__mspabi_fltuld"},
      // The following IS implemented in libgcc but is not in the EABI
      {RTLIB::UINTTOFP_I64_F64, "__mspabi_fltulld"},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::SINTTOFP_I16_F32,  "__mspabi_fltif" },
      {RTLIB::SINTTOFP_I32_F32, "__mspabi_fltlif"},
      // TODO The following IS implemented in libgcc but is not in the EABI
      {RTLIB::SINTTOFP_I64_F32, "__mspabi_fltllif"},
      // TODO The following IS implemented in libgcc
      //{ RTLIB::UINTTOFP_I16_F32,  "__mspabi_fltuf" },
      {RTLIB::UINTTOFP_I32_F32, "__mspabi_fltulf"},
      // The following IS implemented in libgcc but is not in the EABI
      {RTLIB::UINTTOFP_I64_F32, "__mspabi_fltullf"},

      // Floating point comparisons - EABI Table 7
      {RTLIB::OEQ_F64, "__mspabi_cmpd"},
      {RTLIB::UNE_F64, "__mspabi_cmpd"},
      {RTLIB::OGE_F64, "__mspabi_cmpd"},
      {RTLIB::OLT_F64, "__mspabi_cmpd"},
      {RTLIB::OLE_F64, "__mspabi_cmpd"},
      {RTLIB::OGT_F64, "__mspabi_cmpd"},
      {RTLIB::OEQ_F32, "__mspabi_cmpf"},
      {RTLIB::UNE_F32, "__mspabi_cmpf"},
      {RTLIB::OGE_F32, "__mspabi_cmpf"},
      {RTLIB::OLT_F32, "__mspabi_cmpf"},
      {RTLIB::OLE_F32, "__mspabi_cmpf"},
      {RTLIB::OGT_F32, "__mspabi_cmpf"},

      // Floating point arithmetic - EABI Table 8
      {RTLIB::ADD_F64, "__mspabi_addd"},
      {RTLIB::ADD_F32, "__mspabi_addf"},
      {RTLIB::DIV_F64, "__mspabi_divd"},
      {RTLIB::DIV_F32, "__mspabi_divf"},
      {RTLIB::MUL_F64, "__mspabi_mpyd"},
      {RTLIB::MUL_F32, "__mspabi_mpyf"},
      {RTLIB::SUB_F64, "__mspabi_subd"},
      {RTLIB::SUB_F32, "__mspabi_subf"},
      // The following are NOT implemented in libgcc
      // { RTLIB::NEG_F64,  "__mspabi_negd" },
      // { RTLIB::NEG_F32,  "__mspabi_negf" },

      // Universal Integer Operations - EABI Table 9
      {RTLIB::SDIV_I16, "__mspabi_divi"},
      {RTLIB::SDIV_I32, "__mspabi_divli"},
      {RTLIB::SDIV_I64, "__mspabi_divlli"},
      {RTLIB::UDIV_I16, "__mspabi_divu"},
      {RTLIB::UDIV_I32, "__mspabi_divul"},
      {RTLIB::UDIV_I64, "__mspabi_divull"},
      {RTLIB::SREM_I16, "__mspabi_remi"},
      {RTLIB::SREM_I32, "__mspabi_remli"},
      {RTLIB::SREM_I64, "__mspabi_remlli"},
      {RTLIB::UREM_I16, "__mspabi_remu"},
      {RTLIB::UREM_I32, "__mspabi_remul"},
      {RTLIB::UREM_I64, "__mspabi_remull"},

      // Bitwise Operations - EABI Table 10
      // TODO: __mspabi_[srli/srai/slli] ARE implemented in libgcc
      {RTLIB::SRL_I32, "__mspabi_srll"},
      {RTLIB::SRA_I32, "__mspabi_sral"},
      {RTLIB::SHL_I32, "__mspabi_slll"},
      // __mspabi_[srlll/srall/sllll/rlli/rlll] are NOT implemented in libgcc
  };

  for (const auto &LC : LibraryCalls)
    Info.setLibcallName(LC.Op, LC.Name);

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
  Info.setLibcallName(RTLIB::REM_F128, "fmodf128");
  Info.setLibcallName(RTLIB::FMA_F128, "fmaf128");
  Info.setLibcallName(RTLIB::SQRT_F128, "sqrtf128");
  Info.setLibcallName(RTLIB::CBRT_F128, "cbrtf128");
  Info.setLibcallName(RTLIB::LOG_F128, "logf128");
  Info.setLibcallName(RTLIB::LOG2_F128, "log2f128");
  Info.setLibcallName(RTLIB::LOG10_F128, "log10f128");
  Info.setLibcallName(RTLIB::EXP_F128, "expf128");
  Info.setLibcallName(RTLIB::EXP2_F128, "exp2f128");
  Info.setLibcallName(RTLIB::EXP10_F128, "exp10f128");
  Info.setLibcallName(RTLIB::SIN_F128, "sinf128");
  Info.setLibcallName(RTLIB::COS_F128, "cosf128");
  Info.setLibcallName(RTLIB::TAN_F128, "tanf128");
  Info.setLibcallName(RTLIB::SINCOS_F128, "sincosf128");
  Info.setLibcallName(RTLIB::ASIN_F128, "asinf128");
  Info.setLibcallName(RTLIB::ACOS_F128, "acosf128");
  Info.setLibcallName(RTLIB::ATAN_F128, "atanf128");
  Info.setLibcallName(RTLIB::ATAN2_F128, "atan2f128");
  Info.setLibcallName(RTLIB::SINH_F128, "sinhf128");
  Info.setLibcallName(RTLIB::COSH_F128, "coshf128");
  Info.setLibcallName(RTLIB::TANH_F128, "tanhf128");
  Info.setLibcallName(RTLIB::POW_F128, "powf128");
  Info.setLibcallName(RTLIB::CEIL_F128, "ceilf128");
  Info.setLibcallName(RTLIB::TRUNC_F128, "truncf128");
  Info.setLibcallName(RTLIB::RINT_F128, "rintf128");
  Info.setLibcallName(RTLIB::NEARBYINT_F128, "nearbyintf128");
  Info.setLibcallName(RTLIB::ROUND_F128, "roundf128");
  Info.setLibcallName(RTLIB::ROUNDEVEN_F128, "roundevenf128");
  Info.setLibcallName(RTLIB::FLOOR_F128, "floorf128");
  Info.setLibcallName(RTLIB::COPYSIGN_F128, "copysignf128");
  Info.setLibcallName(RTLIB::FMIN_F128, "fminf128");
  Info.setLibcallName(RTLIB::FMAX_F128, "fmaxf128");
  Info.setLibcallName(RTLIB::FMINIMUM_F128, "fminimumf128");
  Info.setLibcallName(RTLIB::FMAXIMUM_F128, "fmaximumf128");
  Info.setLibcallName(RTLIB::FMINIMUM_NUM_F128, "fminimum_numf128");
  Info.setLibcallName(RTLIB::FMAXIMUM_NUM_F128, "fmaximum_numf128");
  Info.setLibcallName(RTLIB::LROUND_F128, "lroundf128");
  Info.setLibcallName(RTLIB::LLROUND_F128, "llroundf128");
  Info.setLibcallName(RTLIB::LRINT_F128, "lrintf128");
  Info.setLibcallName(RTLIB::LLRINT_F128, "llrintf128");
  Info.setLibcallName(RTLIB::LDEXP_F128, "ldexpf128");
  Info.setLibcallName(RTLIB::FREXP_F128, "frexpf128");
  Info.setLibcallName(RTLIB::MODF_F128, "modff128");

  if (FiniteOnlyFuncs) {
    Info.setLibcallName(RTLIB::LOG_FINITE_F128, "__logf128_finite");
    Info.setLibcallName(RTLIB::LOG2_FINITE_F128, "__log2f128_finite");
    Info.setLibcallName(RTLIB::LOG10_FINITE_F128, "__log10f128_finite");
    Info.setLibcallName(RTLIB::EXP_FINITE_F128, "__expf128_finite");
    Info.setLibcallName(RTLIB::EXP2_FINITE_F128, "__exp2f128_finite");
    Info.setLibcallName(RTLIB::POW_FINITE_F128, "__powf128_finite");
  } else {
    Info.setLibcallName(RTLIB::LOG_FINITE_F128, nullptr);
    Info.setLibcallName(RTLIB::LOG2_FINITE_F128, nullptr);
    Info.setLibcallName(RTLIB::LOG10_FINITE_F128, nullptr);
    Info.setLibcallName(RTLIB::EXP_FINITE_F128, nullptr);
    Info.setLibcallName(RTLIB::EXP2_FINITE_F128, nullptr);
    Info.setLibcallName(RTLIB::POW_FINITE_F128, nullptr);
  }
}

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       EABI EABIVersion) {
  initSoftFloatCmpLibcallPredicates();

  initSoftFloatCmpLibcallPredicates();

#define HANDLE_LIBCALL(code, name) setLibcallName(RTLIB::code, name);
#define LIBCALL_NO_NAME nullptr
#include "llvm/IR/RuntimeLibcalls.def"
#undef HANDLE_LIBCALL
#undef LIBCALL_NO_NAME

  // Use the f128 variants of math functions on x86
  if (TT.isX86() && TT.isGNUEnvironment())
    setLongDoubleIsF128Libm(*this, /*FiniteOnlyFuncs=*/true);

  if (TT.isX86() || TT.isVE()) {
    if (ExceptionModel == ExceptionHandling::SjLj)
      setLibcallName(RTLIB::UNWIND_RESUME, "_Unwind_SjLj_Resume");
  }

  // For IEEE quad-precision libcall names, PPC uses "kf" instead of "tf".
  if (TT.isPPC()) {
    setLibcallName(RTLIB::ADD_F128, "__addkf3");
    setLibcallName(RTLIB::SUB_F128, "__subkf3");
    setLibcallName(RTLIB::MUL_F128, "__mulkf3");
    setLibcallName(RTLIB::DIV_F128, "__divkf3");
    setLibcallName(RTLIB::POWI_F128, "__powikf2");
    setLibcallName(RTLIB::FPEXT_F32_F128, "__extendsfkf2");
    setLibcallName(RTLIB::FPEXT_F64_F128, "__extenddfkf2");
    setLibcallName(RTLIB::FPROUND_F128_F16, "__trunckfhf2");
    setLibcallName(RTLIB::FPROUND_F128_F32, "__trunckfsf2");
    setLibcallName(RTLIB::FPROUND_F128_F64, "__trunckfdf2");
    setLibcallName(RTLIB::FPTOSINT_F128_I32, "__fixkfsi");
    setLibcallName(RTLIB::FPTOSINT_F128_I64, "__fixkfdi");
    setLibcallName(RTLIB::FPTOSINT_F128_I128, "__fixkfti");
    setLibcallName(RTLIB::FPTOUINT_F128_I32, "__fixunskfsi");
    setLibcallName(RTLIB::FPTOUINT_F128_I64, "__fixunskfdi");
    setLibcallName(RTLIB::FPTOUINT_F128_I128, "__fixunskfti");
    setLibcallName(RTLIB::SINTTOFP_I32_F128, "__floatsikf");
    setLibcallName(RTLIB::SINTTOFP_I64_F128, "__floatdikf");
    setLibcallName(RTLIB::SINTTOFP_I128_F128, "__floattikf");
    setLibcallName(RTLIB::UINTTOFP_I32_F128, "__floatunsikf");
    setLibcallName(RTLIB::UINTTOFP_I64_F128, "__floatundikf");
    setLibcallName(RTLIB::UINTTOFP_I128_F128, "__floatuntikf");
    setLibcallName(RTLIB::OEQ_F128, "__eqkf2");
    setLibcallName(RTLIB::UNE_F128, "__nekf2");
    setLibcallName(RTLIB::OGE_F128, "__gekf2");
    setLibcallName(RTLIB::OLT_F128, "__ltkf2");
    setLibcallName(RTLIB::OLE_F128, "__lekf2");
    setLibcallName(RTLIB::OGT_F128, "__gtkf2");
    setLibcallName(RTLIB::UO_F128, "__unordkf2");

    // TODO: Do the finite only functions exist?
    setLongDoubleIsF128Libm(*this, /*FiniteOnlyFuncs=*/false);

    if (TT.isOSAIX()) {
      bool isPPC64 = TT.isPPC64();
      setLibcallName(RTLIB::MEMCPY, isPPC64 ? "___memmove64" : "___memmove");
      setLibcallName(RTLIB::MEMMOVE, isPPC64 ? "___memmove64" : "___memmove");
      setLibcallName(RTLIB::MEMSET, isPPC64 ? "___memset64" : "___memset");
      setLibcallName(RTLIB::BZERO, isPPC64 ? "___bzero64" : "___bzero");
    }
  }

  // A few names are different on particular architectures or environments.
  if (TT.isOSDarwin()) {
    // For f16/f32 conversions, Darwin uses the standard naming scheme,
    // instead of the gnueabi-style __gnu_*_ieee.
    // FIXME: What about other targets?
    setLibcallName(RTLIB::FPEXT_F16_F32, "__extendhfsf2");
    setLibcallName(RTLIB::FPROUND_F32_F16, "__truncsfhf2");

    // Some darwins have an optimized __bzero/bzero function.
    switch (TT.getArch()) {
    case Triple::x86:
    case Triple::x86_64:
      if (TT.isMacOSX() && !TT.isMacOSXVersionLT(10, 6))
        setLibcallName(RTLIB::BZERO, "__bzero");
      break;
    case Triple::aarch64:
    case Triple::aarch64_32:
      setLibcallName(RTLIB::BZERO, "bzero");
      break;
    default:
      break;
    }

    if (darwinHasSinCosStret(TT)) {
      setLibcallName(RTLIB::SINCOS_STRET_F32, "__sincosf_stret");
      setLibcallName(RTLIB::SINCOS_STRET_F64, "__sincos_stret");
      if (TT.isWatchABI()) {
        setLibcallCallingConv(RTLIB::SINCOS_STRET_F32,
                              CallingConv::ARM_AAPCS_VFP);
        setLibcallCallingConv(RTLIB::SINCOS_STRET_F64,
                              CallingConv::ARM_AAPCS_VFP);
      }
    }

    switch (TT.getOS()) {
    case Triple::MacOSX:
      if (TT.isMacOSXVersionLT(10, 9)) {
        setLibcallName(RTLIB::EXP10_F32, nullptr);
        setLibcallName(RTLIB::EXP10_F64, nullptr);
      } else {
        setLibcallName(RTLIB::EXP10_F32, "__exp10f");
        setLibcallName(RTLIB::EXP10_F64, "__exp10");
      }
      break;
    case Triple::IOS:
      if (TT.isOSVersionLT(7, 0)) {
        setLibcallName(RTLIB::EXP10_F32, nullptr);
        setLibcallName(RTLIB::EXP10_F64, nullptr);
        break;
      }
      [[fallthrough]];
    case Triple::DriverKit:
    case Triple::TvOS:
    case Triple::WatchOS:
    case Triple::XROS:
      setLibcallName(RTLIB::EXP10_F32, "__exp10f");
      setLibcallName(RTLIB::EXP10_F64, "__exp10");
      break;
    default:
      break;
    }
  } else if (TT.getOS() == Triple::BridgeOS) {
    // TODO: BridgeOS should be included in isOSDarwin.
    setLibcallName(RTLIB::EXP10_F32, "__exp10f");
    setLibcallName(RTLIB::EXP10_F64, "__exp10");
  }

  if (hasSinCos(TT)) {
    setLibcallName(RTLIB::SINCOS_F32, "sincosf");
    setLibcallName(RTLIB::SINCOS_F64, "sincos");
    setLibcallName(RTLIB::SINCOS_F80, "sincosl");
    setLibcallName(RTLIB::SINCOS_F128, "sincosl");
    setLibcallName(RTLIB::SINCOS_PPCF128, "sincosl");
  }

  if (TT.isPS()) {
    setLibcallName(RTLIB::SINCOS_F32, "sincosf");
    setLibcallName(RTLIB::SINCOS_F64, "sincos");
  }

  if (TT.isOSOpenBSD()) {
    setLibcallName(RTLIB::STACKPROTECTOR_CHECK_FAIL, nullptr);
  }

  if (TT.isOSWindows() && !TT.isOSCygMing()) {
    setLibcallName(RTLIB::LDEXP_F32, nullptr);
    setLibcallName(RTLIB::LDEXP_F80, nullptr);
    setLibcallName(RTLIB::LDEXP_F128, nullptr);
    setLibcallName(RTLIB::LDEXP_PPCF128, nullptr);

    setLibcallName(RTLIB::FREXP_F32, nullptr);
    setLibcallName(RTLIB::FREXP_F80, nullptr);
    setLibcallName(RTLIB::FREXP_F128, nullptr);
    setLibcallName(RTLIB::FREXP_PPCF128, nullptr);
  }

  // Disable most libcalls on AMDGPU and NVPTX.
  if (TT.isAMDGPU() || TT.isNVPTX()) {
    for (RTLIB::Libcall LC : RTLIB::libcalls()) {
      if (LC < RTLIB::ATOMIC_LOAD || LC > RTLIB::ATOMIC_FETCH_NAND_16)
        setLibcallName(LC, nullptr);
    }
  }

  if (TT.isOSMSVCRT()) {
    // MSVCRT doesn't have powi; fall back to pow
    setLibcallName(RTLIB::POWI_F32, nullptr);
    setLibcallName(RTLIB::POWI_F64, nullptr);
  }

  // Setup Windows compiler runtime calls.
  if (TT.getArch() == Triple::x86 &&
      (TT.isWindowsMSVCEnvironment() || TT.isWindowsItaniumEnvironment())) {
    static const struct {
      const RTLIB::Libcall Op;
      const char *const Name;
      const CallingConv::ID CC;
    } LibraryCalls[] = {
        {RTLIB::SDIV_I64, "_alldiv", CallingConv::X86_StdCall},
        {RTLIB::UDIV_I64, "_aulldiv", CallingConv::X86_StdCall},
        {RTLIB::SREM_I64, "_allrem", CallingConv::X86_StdCall},
        {RTLIB::UREM_I64, "_aullrem", CallingConv::X86_StdCall},
        {RTLIB::MUL_I64, "_allmul", CallingConv::X86_StdCall},
    };

    for (const auto &LC : LibraryCalls) {
      setLibcallName(LC.Op, LC.Name);
      setLibcallCallingConv(LC.Op, LC.CC);
    }
  }

  if (TT.isAArch64())
    setAArch64LibcallNames(*this, TT);
  else if (TT.isARM() || TT.isThumb())
    setARMLibcallNames(*this, TT, FloatABI, EABIVersion);
  else if (TT.getArch() == Triple::ArchType::avr) {
    // Division rtlib functions (not supported), use divmod functions instead
    setLibcallName(RTLIB::SDIV_I8, nullptr);
    setLibcallName(RTLIB::SDIV_I16, nullptr);
    setLibcallName(RTLIB::SDIV_I32, nullptr);
    setLibcallName(RTLIB::UDIV_I8, nullptr);
    setLibcallName(RTLIB::UDIV_I16, nullptr);
    setLibcallName(RTLIB::UDIV_I32, nullptr);

    // Modulus rtlib functions (not supported), use divmod functions instead
    setLibcallName(RTLIB::SREM_I8, nullptr);
    setLibcallName(RTLIB::SREM_I16, nullptr);
    setLibcallName(RTLIB::SREM_I32, nullptr);
    setLibcallName(RTLIB::UREM_I8, nullptr);
    setLibcallName(RTLIB::UREM_I16, nullptr);
    setLibcallName(RTLIB::UREM_I32, nullptr);

    // Division and modulus rtlib functions
    setLibcallName(RTLIB::SDIVREM_I8, "__divmodqi4");
    setLibcallName(RTLIB::SDIVREM_I16, "__divmodhi4");
    setLibcallName(RTLIB::SDIVREM_I32, "__divmodsi4");
    setLibcallName(RTLIB::UDIVREM_I8, "__udivmodqi4");
    setLibcallName(RTLIB::UDIVREM_I16, "__udivmodhi4");
    setLibcallName(RTLIB::UDIVREM_I32, "__udivmodsi4");

    // Several of the runtime library functions use a special calling conv
    setLibcallCallingConv(RTLIB::SDIVREM_I8, CallingConv::AVR_BUILTIN);
    setLibcallCallingConv(RTLIB::SDIVREM_I16, CallingConv::AVR_BUILTIN);
    setLibcallCallingConv(RTLIB::UDIVREM_I8, CallingConv::AVR_BUILTIN);
    setLibcallCallingConv(RTLIB::UDIVREM_I16, CallingConv::AVR_BUILTIN);

    // Trigonometric rtlib functions
    setLibcallName(RTLIB::SIN_F32, "sin");
    setLibcallName(RTLIB::COS_F32, "cos");
  }

  if (!TT.isWasm()) {
    // These libcalls are only available in compiler-rt, not libgcc.
    if (TT.isArch32Bit()) {
      setLibcallName(RTLIB::SHL_I128, nullptr);
      setLibcallName(RTLIB::SRL_I128, nullptr);
      setLibcallName(RTLIB::SRA_I128, nullptr);
      setLibcallName(RTLIB::MUL_I128, nullptr);
      setLibcallName(RTLIB::MULO_I64, nullptr);
    }
    setLibcallName(RTLIB::MULO_I128, nullptr);
  } else {
    // Define the emscripten name for return address helper.
    // TODO: when implementing other Wasm backends, make this generic or only do
    // this on emscripten depending on what they end up doing.
    setLibcallName(RTLIB::RETURN_ADDRESS, "emscripten_return_address");
  }

  if (TT.isSystemZ() && TT.isOSzOS()) {
    struct RTLibCallMapping {
      RTLIB::Libcall Code;
      const char *Name;
    };
    static RTLibCallMapping RTLibCallCommon[] = {
#define HANDLE_LIBCALL(code, name) {RTLIB::code, name},
#include "ZOSLibcallNames.def"
    };
    for (auto &E : RTLibCallCommon)
      setLibcallName(E.Code, E.Name);
  }

  if (TT.getArch() == Triple::ArchType::hexagon) {
    setLibcallName(RTLIB::SDIV_I32, "__hexagon_divsi3");
    setLibcallName(RTLIB::SDIV_I64, "__hexagon_divdi3");
    setLibcallName(RTLIB::UDIV_I32, "__hexagon_udivsi3");
    setLibcallName(RTLIB::UDIV_I64, "__hexagon_udivdi3");
    setLibcallName(RTLIB::SREM_I32, "__hexagon_modsi3");
    setLibcallName(RTLIB::SREM_I64, "__hexagon_moddi3");
    setLibcallName(RTLIB::UREM_I32, "__hexagon_umodsi3");
    setLibcallName(RTLIB::UREM_I64, "__hexagon_umoddi3");

    const bool FastMath = HexagonEnableFastMathRuntimeCalls;
    // This is the only fast library function for sqrtd.
    if (FastMath)
      setLibcallName(RTLIB::SQRT_F64, "__hexagon_fast2_sqrtdf2");

    // Prefix is: nothing  for "slow-math",
    //            "fast2_" for V5+ fast-math double-precision
    // (actually, keep fast-math and fast-math2 separate for now)
    if (FastMath) {
      setLibcallName(RTLIB::ADD_F64, "__hexagon_fast_adddf3");
      setLibcallName(RTLIB::SUB_F64, "__hexagon_fast_subdf3");
      setLibcallName(RTLIB::MUL_F64, "__hexagon_fast_muldf3");
      setLibcallName(RTLIB::DIV_F64, "__hexagon_fast_divdf3");
      setLibcallName(RTLIB::DIV_F32, "__hexagon_fast_divsf3");
    } else {
      setLibcallName(RTLIB::ADD_F64, "__hexagon_adddf3");
      setLibcallName(RTLIB::SUB_F64, "__hexagon_subdf3");
      setLibcallName(RTLIB::MUL_F64, "__hexagon_muldf3");
      setLibcallName(RTLIB::DIV_F64, "__hexagon_divdf3");
      setLibcallName(RTLIB::DIV_F32, "__hexagon_divsf3");
    }

    if (FastMath)
      setLibcallName(RTLIB::SQRT_F32, "__hexagon_fast2_sqrtf");
    else
      setLibcallName(RTLIB::SQRT_F32, "__hexagon_sqrtf");
  }

  if (TT.getArch() == Triple::ArchType::msp430)
    setMSP430Libcalls(*this, TT);
}
