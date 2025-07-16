//===- RuntimeLibcalls.cpp - Interface for runtime libcalls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/StringTable.h"

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

    if (!darwinHasExp10(TT)) {
      setLibcallImpl(RTLIB::EXP10_F32, RTLIB::Unsupported);
      setLibcallImpl(RTLIB::EXP10_F64, RTLIB::Unsupported);
    }
  }

  if (TT.isOSOpenBSD()) {
    setLibcallImpl(RTLIB::STACKPROTECTOR_CHECK_FAIL, RTLIB::Unsupported);
    setLibcallImpl(RTLIB::STACK_SMASH_HANDLER, RTLIB::__stack_smash_handler);
  }

  // Skip default manual processing for targets that have been fully ported to
  // tablegen for now. Eventually the rest of this should be deleted.
  if (TT.isX86() || TT.isAArch64() || TT.isWasm())
    return;

  if (TT.isARM() || TT.isThumb()) {
    setARMLibcallNames(*this, TT, FloatABI, EABIVersion);
    return;
  }

  if (hasSinCos(TT)) {
    setLibcallImpl(RTLIB::SINCOS_F32, RTLIB::sincosf);
    setLibcallImpl(RTLIB::SINCOS_F64, RTLIB::sincos);
    setLibcallImpl(RTLIB::SINCOS_F128, RTLIB::sincos_f128);
  }

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
