//===- RuntimeLibcalls.cpp - Interface for runtime libcalls -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/xxhash.h"

#define DEBUG_TYPE "runtime-libcalls-info"

using namespace llvm;
using namespace RTLIB;

#define GET_INIT_RUNTIME_LIBCALL_NAMES
#define GET_SET_TARGET_RUNTIME_LIBCALL_SETS
#define DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME
#include "llvm/IR/RuntimeLibcalls.inc"
#undef GET_INIT_RUNTIME_LIBCALL_NAMES
#undef GET_SET_TARGET_RUNTIME_LIBCALL_SETS
#undef DEFINE_GET_LOOKUP_LIBCALL_IMPL_NAME

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
      RTLIB::__aeabi_lasr,        RTLIB::__aeabi_idiv,
      RTLIB::__aeabi_idivmod,     RTLIB::__aeabi_uidivmod,
      RTLIB::__aeabi_ldivmod,     RTLIB::__aeabi_uidiv,
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

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       EABI EABIVersion, StringRef ABIName) {
  setTargetRuntimeLibcallSets(TT, FloatABI);

  if (ExceptionModel == ExceptionHandling::SjLj)
    setLibcallImpl(RTLIB::UNWIND_RESUME, RTLIB::_Unwind_SjLj_Resume);

  if (TT.isARM() || TT.isThumb()) {
    setARMLibcallNames(*this, TT, FloatABI, EABIVersion);
    return;
  }

  if (TT.getArch() == Triple::ArchType::msp430) {
    setLibcallImplCallingConv(RTLIB::__mspabi_mpyll,
                              CallingConv::MSP430_BUILTIN);
  }
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
iota_range<RTLIB::LibcallImpl>
RuntimeLibcallsInfo::libcallImplNameHit(uint16_t NameOffsetEntry,
                                        uint16_t StrOffset) {
  int NumAliases = 1;
  for (int E = std::size(RuntimeLibcallNameOffsetTable);
       NameOffsetEntry + NumAliases != E &&
       RuntimeLibcallNameOffsetTable[NameOffsetEntry + NumAliases] == StrOffset;
       ++NumAliases) {
  }

  RTLIB::LibcallImpl ImplStart = static_cast<RTLIB::LibcallImpl>(
      &RuntimeLibcallNameOffsetTable[NameOffsetEntry] -
      &RuntimeLibcallNameOffsetTable[0]);
  return enum_seq(ImplStart,
                  static_cast<RTLIB::LibcallImpl>(ImplStart + NumAliases));
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
