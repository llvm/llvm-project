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

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       EABI EABIVersion, StringRef ABIName) {
  setTargetRuntimeLibcallSets(TT, FloatABI, EABIVersion, ABIName);

  if (ExceptionModel == ExceptionHandling::SjLj)
    setLibcallImpl(RTLIB::UNWIND_RESUME, RTLIB::_Unwind_SjLj_Resume);

  if (TT.isARM() || TT.isThumb()) {
    // The half <-> float conversion functions are always soft-float on
    // non-watchos platforms, but are needed for some targets which use a
    // hard-float calling convention by default.
    if (!TT.isWatchABI()) {
      if (isAAPCS_ABI(TT, ABIName)) {
        setLibcallImplCallingConv(RTLIB::__truncsfhf2, CallingConv::ARM_AAPCS);
        setLibcallImplCallingConv(RTLIB::__truncdfhf2, CallingConv::ARM_AAPCS);
        setLibcallImplCallingConv(RTLIB::__extendhfsf2, CallingConv::ARM_AAPCS);
        setLibcallImplCallingConv(RTLIB::__gnu_h2f_ieee,
                                  CallingConv::ARM_AAPCS);
        setLibcallImplCallingConv(RTLIB::__gnu_f2h_ieee,
                                  CallingConv::ARM_AAPCS);
      } else {
        setLibcallImplCallingConv(RTLIB::__truncsfhf2, CallingConv::ARM_APCS);
        setLibcallImplCallingConv(RTLIB::__truncdfhf2, CallingConv::ARM_APCS);
        setLibcallImplCallingConv(RTLIB::__extendhfsf2, CallingConv::ARM_APCS);
        setLibcallImplCallingConv(RTLIB::__gnu_h2f_ieee, CallingConv::ARM_APCS);
        setLibcallImplCallingConv(RTLIB::__gnu_f2h_ieee, CallingConv::ARM_APCS);
      }
    }

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

bool RuntimeLibcallsInfo::isAAPCS_ABI(const Triple &TT, StringRef ABIName) {
  const ARM::ARMABI TargetABI = ARM::computeTargetABI(TT, ABIName);
  return TargetABI == ARM::ARM_ABI_AAPCS || TargetABI == ARM::ARM_ABI_AAPCS16;
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
