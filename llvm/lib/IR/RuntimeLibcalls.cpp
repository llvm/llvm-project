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

#define GET_INIT_RUNTIME_LIBCALL_NAMES
#define GET_SET_TARGET_RUNTIME_LIBCALL_SETS
#include "llvm/IR/RuntimeLibcalls.inc"
#undef GET_INIT_RUNTIME_LIBCALL_NAMES
#undef GET_SET_TARGET_RUNTIME_LIBCALL_SETS

static cl::opt<bool>
    HexagonEnableFastMathRuntimeCalls("hexagon-fast-math", cl::Hidden,
                                      cl::desc("Enable Fast Math processing"));

static void setARMLibcallNames(RuntimeLibcallsInfo &Info, const Triple &TT,
                               FloatABI::ABIType FloatABIType,
                               EABI EABIVersion) {
  if (!TT.isOSDarwin() && !TT.isiOS() && !TT.isWatchOS() && !TT.isDriverKit()) {
    CallingConv::ID DefaultCC = FloatABIType == FloatABI::Hard
                                    ? CallingConv::ARM_AAPCS_VFP
                                    : CallingConv::ARM_AAPCS;
    for (RTLIB::LibcallImpl LC : RTLIB::libcall_impls())
      Info.setLibcallImplCallingConv(LC, DefaultCC);
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
        Info.setLibcallImplCallingConv(LC.Impl, LC.CC);
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
        Info.setLibcallImplCallingConv(LC.Impl, LC.CC);
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
      Info.setLibcallImplCallingConv(LC.Impl, LC.CC);
    }
  }

  // Use divmod compiler-rt calls for iOS 5.0 and later.
  if (TT.isOSBinFormatMachO() && (!TT.isiOS() || !TT.isOSVersionLT(5, 0))) {
    Info.setLibcallImpl(RTLIB::SDIVREM_I32, RTLIB::__divmodsi4);
    Info.setLibcallImpl(RTLIB::UDIVREM_I32, RTLIB::__udivmodsi4);
  }
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
  setTargetRuntimeLibcallSets(TT);

  // Use the f128 variants of math functions on x86
  if (TT.isX86() && TT.isGNUEnvironment())
    setLongDoubleIsF128Libm(*this, /*FiniteOnlyFuncs=*/true);

  if (TT.isX86() || TT.isVE() || TT.isARM() || TT.isThumb()) {
    if (ExceptionModel == ExceptionHandling::SjLj)
      setLibcallImpl(RTLIB::UNWIND_RESUME, RTLIB::_Unwind_SjLj_Resume);
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
    }

    if (darwinHasSinCosStret(TT)) {
      setLibcallImpl(RTLIB::SINCOS_STRET_F32, RTLIB::__sincosf_stret);
      setLibcallImpl(RTLIB::SINCOS_STRET_F64, RTLIB::__sincos_stret);
      if (TT.isWatchABI()) {
        setLibcallImplCallingConv(RTLIB::__sincosf_stret,
                                  CallingConv::ARM_AAPCS_VFP);
        setLibcallImplCallingConv(RTLIB::__sincos_stret,
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
      setLibcallImplCallingConv(LC.Impl, LC.CC);
    }
  }

  if (TT.isARM() || TT.isThumb())
    setARMLibcallNames(*this, TT, FloatABI, EABIVersion);

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

  if (TT.getArch() == Triple::ArchType::msp430) {
    setLibcallImplCallingConv(RTLIB::__mspabi_mpyll,
                              CallingConv::MSP430_BUILTIN);
  }
}

bool RuntimeLibcallsInfo::darwinHasExp10(const Triple &TT) {
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
