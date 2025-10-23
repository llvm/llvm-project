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
#include "llvm/TargetParser/ARMTargetParser.h"

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

/// Set default libcall names. If a target wants to opt-out of a libcall it
/// should be placed here.
void RuntimeLibcallsInfo::initLibcalls(const Triple &TT,
                                       ExceptionHandling ExceptionModel,
                                       FloatABI::ABIType FloatABI,
                                       EABI EABIVersion, StringRef ABIName) {
  setTargetRuntimeLibcallSets(TT, ExceptionModel, FloatABI, EABIVersion,
                              ABIName);

  if (TT.isARM() || TT.isThumb()) {
    // The half <-> float conversion functions are always soft-float on
    // non-watchos platforms, but are needed for some targets which use a
    // hard-float calling convention by default.
    if (!TT.isWatchABI()) {
      if (isAAPCS_ABI(TT, ABIName)) {
        setLibcallImplCallingConv(RTLIB::impl___truncsfhf2,
                                  CallingConv::ARM_AAPCS);
        setLibcallImplCallingConv(RTLIB::impl___truncdfhf2,
                                  CallingConv::ARM_AAPCS);
        setLibcallImplCallingConv(RTLIB::impl___extendhfsf2,
                                  CallingConv::ARM_AAPCS);
      } else {
        setLibcallImplCallingConv(RTLIB::impl___truncsfhf2,
                                  CallingConv::ARM_APCS);
        setLibcallImplCallingConv(RTLIB::impl___truncdfhf2,
                                  CallingConv::ARM_APCS);
        setLibcallImplCallingConv(RTLIB::impl___extendhfsf2,
                                  CallingConv::ARM_APCS);
      }
    }

    return;
  }
}

LLVM_ATTRIBUTE_ALWAYS_INLINE
iota_range<RTLIB::LibcallImpl>
RuntimeLibcallsInfo::libcallImplNameHit(uint16_t NameOffsetEntry,
                                        uint16_t StrOffset) {
  int NumAliases = 1;
  for (uint16_t Entry : ArrayRef(RuntimeLibcallNameOffsetTable)
                            .drop_front(NameOffsetEntry + 1)) {
    if (Entry != StrOffset)
      break;
    ++NumAliases;
  }

  RTLIB::LibcallImpl ImplStart = static_cast<RTLIB::LibcallImpl>(
      &RuntimeLibcallNameOffsetTable[NameOffsetEntry] -
      &RuntimeLibcallNameOffsetTable[0]);
  return enum_seq(ImplStart,
                  static_cast<RTLIB::LibcallImpl>(ImplStart + NumAliases));
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
