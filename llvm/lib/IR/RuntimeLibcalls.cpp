//===- RuntimeLibcalls.cpp - Interface for runtime libcalls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"

using namespace llvm;
using namespace RTLIB;

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT) {
  std::fill(std::begin(LibcallRoutineNames), std::end(LibcallRoutineNames),
            nullptr);

#define HANDLE_LIBCALL(code, name) setLibcallName(RTLIB::code, name);
#include "llvm/IR/RuntimeLibcalls.def"
#undef HANDLE_LIBCALL

  // Initialize calling conventions to their default.
  for (int LC = 0; LC < RTLIB::UNKNOWN_LIBCALL; ++LC)
    setLibcallCallingConv((RTLIB::Libcall)LC, CallingConv::C);

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

    if (darwinHasSinCos(TT)) {
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

  if (TT.isGNUEnvironment() || TT.isOSFuchsia() ||
      (TT.isAndroid() && !TT.isAndroidVersionLT(9))) {
    setLibcallName(RTLIB::SINCOS_F32, "sincosf");
    setLibcallName(RTLIB::SINCOS_F64, "sincos");
    setLibcallName(RTLIB::SINCOS_F80, "sincosl");
    setLibcallName(RTLIB::SINCOS_F128, "sincosf128");
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

  // Disable most libcalls on AMDGPU.
  if (TT.isAMDGPU()) {
    for (int I = 0; I < RTLIB::UNKNOWN_LIBCALL; ++I) {
      if (I < RTLIB::ATOMIC_LOAD || I > RTLIB::ATOMIC_FETCH_NAND_16)
        setLibcallName(static_cast<RTLIB::Libcall>(I), nullptr);
    }
  }

  // Disable most libcalls on NVPTX.
  if (TT.isNVPTX()) {
    for (int I = 0; I < RTLIB::UNKNOWN_LIBCALL; ++I)
      if (I < RTLIB::ATOMIC_LOAD || I > RTLIB::ATOMIC_FETCH_NAND_16)
        setLibcallName(static_cast<RTLIB::Libcall>(I), nullptr);
  }

  if (TT.isOSMSVCRT()) {
    // MSVCRT doesn't have powi; fall back to pow
    setLibcallName(RTLIB::POWI_F32, nullptr);
    setLibcallName(RTLIB::POWI_F64, nullptr);
  }

  if (TT.getArch() == Triple::ArchType::avr) {
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
  }

  // By default fp128 libcalls get lowered to `*f128` symbols, which is
  // safest because the symbols are only ever for binary128 on all platforms.
  // Unfortunately many platforms only have the `*l` (`long double`) symbols,
  // which vary by architecture and compilation flags, so we have to use them
  // sometimes.
  if (TT.shouldLowerf128AsLongDouble())
    setF128LibcallFormat(F128LibcallFormat::LongDouble);
}

void RuntimeLibcallsInfo::setF128LibcallFormat(F128LibcallFormat Format) {
  bool UseLD = Format == F128LibcallFormat::LongDouble;

  setLibcallName(RTLIB::ACOS_F128, UseLD ? "acosl" : "acosf128");
  setLibcallName(RTLIB::ASIN_F128, UseLD ? "asinl" : "asinf128");
  setLibcallName(RTLIB::ATAN2_F128, UseLD ? "atan2l" : "atan2f128");
  setLibcallName(RTLIB::ATAN_F128, UseLD ? "atanl" : "atanf128");
  setLibcallName(RTLIB::CBRT_F128, UseLD ? "cbrtl" : "cbrtf128");
  setLibcallName(RTLIB::CEIL_F128, UseLD ? "ceill" : "ceilf128");
  setLibcallName(RTLIB::COPYSIGN_F128, UseLD ? "copysignl" : "copysignf128");
  setLibcallName(RTLIB::COSH_F128, UseLD ? "coshl" : "coshf128");
  setLibcallName(RTLIB::COS_F128, UseLD ? "cosl" : "cosf128");
  setLibcallName(RTLIB::EXP10_F128, UseLD ? "exp10l" : "exp10f128");
  setLibcallName(RTLIB::EXP2_F128, UseLD ? "exp2l" : "exp2f128");
  setLibcallName(RTLIB::EXP_F128, UseLD ? "expl" : "expf128");
  setLibcallName(RTLIB::FLOOR_F128, UseLD ? "floorl" : "floorf128");
  setLibcallName(RTLIB::FMAXIMUMNUM_F128,
                 UseLD ? "fmaximum_numl" : "fmaximum_numf128");
  setLibcallName(RTLIB::FMAXIMUM_F128, UseLD ? "fmaximuml" : "fmaximumf128");
  setLibcallName(RTLIB::FMAX_F128, UseLD ? "fmaxl" : "fmaxf128");
  setLibcallName(RTLIB::FMA_F128, UseLD ? "fmal" : "fmaf128");
  setLibcallName(RTLIB::FMINIMUMNUM_F128,
                 UseLD ? "fminimum_numl" : "fminimum_numf128");
  setLibcallName(RTLIB::FMINIMUM_F128, UseLD ? "fminimuml" : "fminimumf128");
  setLibcallName(RTLIB::FMIN_F128, UseLD ? "fminl" : "fminf128");
  setLibcallName(RTLIB::FREXP_F128, UseLD ? "frexpl" : "frexpf128");
  setLibcallName(RTLIB::LDEXP_F128, UseLD ? "ldexpl" : "ldexpf128");
  setLibcallName(RTLIB::LLRINT_F128, UseLD ? "llrintl" : "llrintf128");
  setLibcallName(RTLIB::LLROUND_F128, UseLD ? "llroundl" : "llroundf128");
  setLibcallName(RTLIB::LOG10_F128, UseLD ? "log10l" : "log10f128");
  setLibcallName(RTLIB::LOG2_F128, UseLD ? "log2l" : "log2f128");
  setLibcallName(RTLIB::LOG_F128, UseLD ? "logl" : "logf128");
  setLibcallName(RTLIB::LRINT_F128, UseLD ? "lrintl" : "lrintf128");
  setLibcallName(RTLIB::LROUND_F128, UseLD ? "lroundl" : "lroundf128");
  setLibcallName(RTLIB::MODF_F128, UseLD ? "modfl" : "modff128");
  setLibcallName(RTLIB::NEARBYINT_F128, UseLD ? "nearbyintl" : "nearbyintf128");
  setLibcallName(RTLIB::POW_F128, UseLD ? "powl" : "powf128");
  setLibcallName(RTLIB::REM_F128, UseLD ? "fmodl" : "fmodf128");
  setLibcallName(RTLIB::RINT_F128, UseLD ? "rintl" : "rintf128");
  setLibcallName(RTLIB::ROUNDEVEN_F128, UseLD ? "roundevenl" : "roundevenf128");
  setLibcallName(RTLIB::ROUND_F128, UseLD ? "roundl" : "roundf128");
  setLibcallName(RTLIB::SINCOSPI_F128, UseLD ? "sincospil" : "sincospif128");
  setLibcallName(RTLIB::SINH_F128, UseLD ? "sinhl" : "sinhf128");
  setLibcallName(RTLIB::SIN_F128, UseLD ? "sinl" : "sinf128");
  setLibcallName(RTLIB::SQRT_F128, UseLD ? "sqrtl" : "sqrtf128");
  setLibcallName(RTLIB::TANH_F128, UseLD ? "tanhl" : "tanhf128");
  setLibcallName(RTLIB::TAN_F128, UseLD ? "tanl" : "tanf128");
  setLibcallName(RTLIB::TRUNC_F128, UseLD ? "truncl" : "truncf128");

  if (nullptr != getLibcallName(RTLIB::SINCOS_F128)) {
    // Upsate sincos only if already set (sincos is allowed to be null to use
    // sin+cos instead).
    setLibcallName(RTLIB::SINCOS_F128, UseLD ? "sincosl" : "sincosf128");
  }
}
