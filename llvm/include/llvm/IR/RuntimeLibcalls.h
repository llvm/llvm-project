//===- RuntimeLibcalls.h - Interface for runtime libcalls -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a common interface to work with library calls into a
// runtime that may be emitted by a given backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_RUNTIME_LIBCALLS_H
#define LLVM_IR_RUNTIME_LIBCALLS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {
namespace RTLIB {

/// RTLIB::Libcall enum - This enum defines all of the runtime library calls
/// the backend can emit.  The various long double types cannot be merged,
/// because 80-bit library functions use "xf" and 128-bit use "tf".
///
/// When adding PPCF128 functions here, note that their names generally need
/// to be overridden for Darwin with the xxx$LDBL128 form.  See
/// PPCISelLowering.cpp.
///
enum Libcall {
#define HANDLE_LIBCALL(code, name) code,
#include "llvm/IR/RuntimeLibcalls.def"
#undef HANDLE_LIBCALL
};
} // namespace RTLIB

template <> struct enum_iteration_traits<RTLIB::Libcall> {
  static constexpr bool is_iterable = true;
};

namespace RTLIB {

// Return an iterator over all Libcall values.
static inline auto libcalls() {
  return enum_seq(static_cast<RTLIB::Libcall>(0), RTLIB::UNKNOWN_LIBCALL);
}

/// A simple container for information about the supported runtime calls.
struct RuntimeLibcallsInfo {
  explicit RuntimeLibcallsInfo(const Triple &TT) {
    initLibcalls(TT);
  }

  /// Rename the default libcall routine name for the specified libcall.
  void setLibcallName(RTLIB::Libcall Call, const char *Name) {
    LibcallRoutineNames[Call] = Name;
  }

  void setLibcallName(ArrayRef<RTLIB::Libcall> Calls, const char *Name) {
    for (auto Call : Calls)
      setLibcallName(Call, Name);
  }

  /// Get the libcall routine name for the specified libcall.
  const char *getLibcallName(RTLIB::Libcall Call) const {
    return LibcallRoutineNames[Call];
  }

  /// Set the CallingConv that should be used for the specified libcall.
  void setLibcallCallingConv(RTLIB::Libcall Call, CallingConv::ID CC) {
    LibcallCallingConvs[Call] = CC;
  }

  /// Get the CallingConv that should be used for the specified libcall.
  CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return LibcallCallingConvs[Call];
  }

  ArrayRef<const char *> getLibcallNames() const {
    // Trim UNKNOWN_LIBCALL from the end
    return ArrayRef(LibcallRoutineNames).drop_back();
  }

  /// Get the comparison predicate that's to be used to test the result of the
  /// comparison libcall against zero. This should only be used with
  /// floating-point compare libcalls.
  CmpInst::Predicate
  getSoftFloatCmpLibcallPredicate(RTLIB::Libcall Call) const {
    return SoftFloatCompareLibcallPredicates[Call];
  }

  // FIXME: This should be removed. This should be private constant.
  void setSoftFloatCmpLibcallPredicate(RTLIB::Libcall Call,
                                       CmpInst::Predicate Pred) {
    SoftFloatCompareLibcallPredicates[Call] = Pred;
  }

private:
  /// Stores the name each libcall.
  const char *LibcallRoutineNames[RTLIB::UNKNOWN_LIBCALL + 1];

  /// Stores the CallingConv that should be used for each libcall.
  CallingConv::ID LibcallCallingConvs[RTLIB::UNKNOWN_LIBCALL];

  /// The condition type that should be used to test the result of each of the
  /// soft floating-point comparison libcall against integer zero.
  ///
  // FIXME: This is only relevant for the handful of floating-point comparison
  // runtime calls; it's excessive to have a table entry for every single
  // opcode.
  CmpInst::Predicate SoftFloatCompareLibcallPredicates[RTLIB::UNKNOWN_LIBCALL];

  static bool darwinHasSinCos(const Triple &TT) {
    assert(TT.isOSDarwin() && "should be called with darwin triple");
    // Don't bother with 32 bit x86.
    if (TT.getArch() == Triple::x86)
      return false;
    // Macos < 10.9 has no sincos_stret.
    if (TT.isMacOSX())
      return !TT.isMacOSXVersionLT(10, 9) && TT.isArch64Bit();
    // iOS < 7.0 has no sincos_stret.
    if (TT.isiOS())
      return !TT.isOSVersionLT(7, 0);
    // Any other darwin such as WatchOS/TvOS is new enough.
    return true;
  }

  void initSoftFloatCmpLibcallPredicates();

  /// Set default libcall names. If a target wants to opt-out of a libcall it
  /// should be placed here.
  LLVM_ABI void initLibcalls(const Triple &TT);
};

} // namespace RTLIB
} // namespace llvm

#endif // LLVM_IR_RUNTIME_LIBCALLS_H
