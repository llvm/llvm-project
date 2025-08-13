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
#include "llvm/TargetParser/ARMTargetParser.h"

#define DEBUG_TYPE "runtime-libcalls-info"

using namespace llvm;
using namespace RTLIB;

#define GET_INIT_RUNTIME_LIBCALL_NAMES
#define GET_SET_TARGET_RUNTIME_LIBCALL_SETS
#include "llvm/IR/RuntimeLibcalls.inc"
#undef GET_INIT_RUNTIME_LIBCALL_NAMES
#undef GET_SET_TARGET_RUNTIME_LIBCALL_SETS

static void setARMLibcallNames(RuntimeLibcallsInfo &Info, const Triple &TT,
                               FloatABI::ABIType FloatABIType, EABI EABIVersion,
                               StringRef ABIName) {
  // The half <-> float conversion functions are always soft-float on
  // non-watchos platforms, but are needed for some targets which use a
  // hard-float calling convention by default.
  if (!TT.isWatchABI()) {
    ARM::ARMABI TargetABI = ARM::computeTargetABI(TT, ABIName);

    if (TargetABI == ARM::ARM_ABI_AAPCS || TargetABI == ARM::ARM_ABI_AAPCS16) {
      Info.setLibcallImplCallingConv(RTLIB::__truncsfhf2,
                                     CallingConv::ARM_AAPCS);
      Info.setLibcallImplCallingConv(RTLIB::__truncdfhf2,
                                     CallingConv::ARM_AAPCS);
      Info.setLibcallImplCallingConv(RTLIB::__extendhfsf2,
                                     CallingConv::ARM_AAPCS);
      Info.setLibcallImplCallingConv(RTLIB::__gnu_h2f_ieee,
                                     CallingConv::ARM_AAPCS);
      Info.setLibcallImplCallingConv(RTLIB::__gnu_f2h_ieee,
                                     CallingConv::ARM_AAPCS);
    } else {
      Info.setLibcallImplCallingConv(RTLIB::__truncsfhf2,
                                     CallingConv::ARM_APCS);
      Info.setLibcallImplCallingConv(RTLIB::__truncdfhf2,
                                     CallingConv::ARM_APCS);
      Info.setLibcallImplCallingConv(RTLIB::__extendhfsf2,
                                     CallingConv::ARM_APCS);
      Info.setLibcallImplCallingConv(RTLIB::__gnu_h2f_ieee,
                                     CallingConv::ARM_APCS);
      Info.setLibcallImplCallingConv(RTLIB::__gnu_f2h_ieee,
                                     CallingConv::ARM_APCS);
    }
  }

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
    setARMLibcallNames(*this, TT, FloatABI, EABIVersion, ABIName);
    return;
  }

  if (TT.getArch() == Triple::ArchType::msp430) {
    setLibcallImplCallingConv(RTLIB::__mspabi_mpyll,
                              CallingConv::MSP430_BUILTIN);
  }
}

RTLIB::LibcallImpl
RuntimeLibcallsInfo::getSupportedLibcallImpl(StringRef FuncName) const {
  const ArrayRef<uint16_t> RuntimeLibcallNameOffsets(
      RuntimeLibcallNameOffsetTable);

  iterator_range<ArrayRef<uint16_t>::const_iterator> Range =
      getRecognizedLibcallImpls(FuncName);

  for (auto I = Range.begin(); I != Range.end(); ++I) {
    RTLIB::LibcallImpl Impl =
        static_cast<RTLIB::LibcallImpl>(I - RuntimeLibcallNameOffsets.begin());

    // FIXME: This should not depend on looking up ImplToLibcall, only the list
    // of libcalls for the module.
    RTLIB::LibcallImpl Recognized = LibcallImpls[ImplToLibcall[Impl]];
    if (Recognized != RTLIB::Unsupported)
      return Recognized;
  }

  return RTLIB::Unsupported;
}

iterator_range<ArrayRef<uint16_t>::const_iterator>
RuntimeLibcallsInfo::getRecognizedLibcallImpls(StringRef FuncName) {
  StringTable::Iterator It = lower_bound(RuntimeLibcallImplNameTable, FuncName);
  if (It == RuntimeLibcallImplNameTable.end() || *It != FuncName)
    return iterator_range(ArrayRef<uint16_t>());

  uint16_t IndexVal = It.offset().value();
  const ArrayRef<uint16_t> TableRef(RuntimeLibcallNameOffsetTable);

  ArrayRef<uint16_t>::const_iterator E = TableRef.end();
  ArrayRef<uint16_t>::const_iterator EntriesBegin =
      std::lower_bound(TableRef.begin(), E, IndexVal);
  ArrayRef<uint16_t>::const_iterator EntriesEnd = EntriesBegin;

  while (EntriesEnd != E && *EntriesEnd == IndexVal)
    ++EntriesEnd;

  assert(EntriesBegin != E &&
         "libcall found in name table but not offset table");

  return make_range(EntriesBegin, EntriesEnd);
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
