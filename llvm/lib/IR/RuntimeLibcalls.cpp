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

static void setARMLibcallNames(RuntimeLibcallsInfo &Info, const Triple &TT,
                               FloatABI::ABIType FloatABIType,
                               EABI EABIVersion) {
  static const RTLIB::LibcallImpl AAPCS_Libcalls[] = {
      RTLIB::__aeabi_dadd,        RTLIB::__aeabi_ddiv,
      RTLIB::__aeabi_dmul,        RTLIB::__aeabi_dsub,
      RTLIB::__aeabi_dcmpeq__oeq, RTLIB::__aeabi_dcmpeq__une,
      RTLIB::__aeabi_dcmplt,      RTLIB::__aeabi_dcmple,
      RTLIB::__aeabi_dcmpge,      RTLIB::__aeabi_dcmpgt,
      RTLIB::__aeabi_dcmpun,      RTLIB::__aeabi_fadd,
      RTLIB::__aeabi_fdiv,        RTLIB::__aeabi_fmul,
      RTLIB::__aeabi_fsub,        RTLIB::__aeabi_fcmpeq__oeq,
      RTLIB::__aeabi_fcmpeq__une, RTLIB::__aeabi_fcmplt,
      RTLIB::__aeabi_fcmple,      RTLIB::__aeabi_fcmpge,
      RTLIB::__aeabi_fcmpgt,      RTLIB::__aeabi_fcmpun,
      RTLIB::__aeabi_d2iz,        RTLIB::__aeabi_d2uiz,
      RTLIB::__aeabi_d2lz,        RTLIB::__aeabi_d2ulz,
      RTLIB::__aeabi_f2iz,        RTLIB::__aeabi_f2uiz,
      RTLIB::__aeabi_f2lz,        RTLIB::__aeabi_f2ulz,
      RTLIB::__aeabi_d2f,         RTLIB::__aeabi_d2h,
      RTLIB::__aeabi_f2d,         RTLIB::__aeabi_i2d,
      RTLIB::__aeabi_ui2d,        RTLIB::__aeabi_l2d,
      RTLIB::__aeabi_ul2d,        RTLIB::__aeabi_i2f,
      RTLIB::__aeabi_ui2f,        RTLIB::__aeabi_l2f,
      RTLIB::__aeabi_ul2f,        RTLIB::__aeabi_lmul,
      RTLIB::__aeabi_llsl,        RTLIB::__aeabi_llsr,
      RTLIB::__aeabi_lasr,        RTLIB::__aeabi_idiv__i8,
      RTLIB::__aeabi_idiv__i16,   RTLIB::__aeabi_idiv__i32,
      RTLIB::__aeabi_idivmod,     RTLIB::__aeabi_uidivmod,
      RTLIB::__aeabi_ldivmod,     RTLIB::__aeabi_uidiv__i8,
      RTLIB::__aeabi_uidiv__i16,  RTLIB::__aeabi_uidiv__i32,
      RTLIB::__aeabi_uldivmod,    RTLIB::__aeabi_f2h,
      RTLIB::__aeabi_d2h,         RTLIB::__aeabi_h2f,
      RTLIB::__aeabi_memcpy,      RTLIB::__aeabi_memmove,
      RTLIB::__aeabi_memset,      RTLIB::__aeabi_memcpy4,
      RTLIB::__aeabi_memcpy8,     RTLIB::__aeabi_memmove4,
      RTLIB::__aeabi_memmove8,    RTLIB::__aeabi_memset4,
      RTLIB::__aeabi_memset8,     RTLIB::__aeabi_memclr,
      RTLIB::__aeabi_memclr4,     RTLIB::__aeabi_memclr8};

  for (RTLIB::LibcallImpl Impl : AAPCS_Libcalls)
    Info.setLibcallImplCallingConv(Impl, CallingConv::ARM_AAPCS);
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
  setTargetRuntimeLibcallSets(TT, FloatABI);

  // Early exit for targets that have fully ported to tablegen.
  if (TT.isAMDGPU() || TT.isNVPTX() || TT.isWasm())
    return;

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

  // These libcalls are only available in compiler-rt, not libgcc.
  if (TT.isArch64Bit()) {
    setLibcallImpl(RTLIB::SHL_I128, RTLIB::__ashlti3);
    setLibcallImpl(RTLIB::SRL_I128, RTLIB::__lshrti3);
    setLibcallImpl(RTLIB::SRA_I128, RTLIB::__ashrti3);
    setLibcallImpl(RTLIB::MUL_I128, RTLIB::__multi3);
    setLibcallImpl(RTLIB::MULO_I64, RTLIB::__mulodi4);
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
