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
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

/// TableGen will produce 2 enums, RTLIB::Libcall and
/// RTLIB::LibcallImpl. RTLIB::Libcall describes abstract functionality the
/// compiler may choose to access, RTLIB::LibcallImpl describes a particular ABI
/// implementation, which includes a name and type signature.
#define GET_RUNTIME_LIBCALL_ENUM
#include "llvm/IR/RuntimeLibcalls.inc"
#undef GET_RUNTIME_LIBCALL_ENUM

namespace llvm {

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
  explicit RuntimeLibcallsInfo(
      const Triple &TT,
      ExceptionHandling ExceptionModel = ExceptionHandling::None,
      FloatABI::ABIType FloatABI = FloatABI::Default,
      EABI EABIVersion = EABI::Default, StringRef ABIName = "") {
    initSoftFloatCmpLibcallPredicates();
    initDefaultLibCallImpls();
    initLibcalls(TT, ExceptionModel, FloatABI, EABIVersion, ABIName);
  }

  /// Rename the default libcall routine name for the specified libcall.
  void setLibcallImpl(RTLIB::Libcall Call, RTLIB::LibcallImpl Impl) {
    LibcallImpls[Call] = Impl;
  }

  /// Get the libcall routine name for the specified libcall.
  // FIXME: This should be removed. Only LibcallImpl should have a name.
  const char *getLibcallName(RTLIB::Libcall Call) const {
    return LibCallImplNames[LibcallImpls[Call]];
  }

  /// Get the libcall routine name for the specified libcall implementation.
  const char *getLibcallImplName(RTLIB::LibcallImpl CallImpl) const {
    return LibCallImplNames[CallImpl];
  }

  /// Return the lowering's selection of implementation call for \p Call
  RTLIB::LibcallImpl getLibcallImpl(RTLIB::Libcall Call) const {
    return LibcallImpls[Call];
  }

  /// Set the CallingConv that should be used for the specified libcall.
  // FIXME: This should be a function of RTLIB::LibcallImpl
  void setLibcallCallingConv(RTLIB::Libcall Call, CallingConv::ID CC) {
    LibcallCallingConvs[Call] = CC;
  }

  /// Get the CallingConv that should be used for the specified libcall.
  // FIXME: This should be a function of RTLIB::LibcallImpl
  CallingConv::ID getLibcallCallingConv(RTLIB::Libcall Call) const {
    return LibcallCallingConvs[Call];
  }

  ArrayRef<RTLIB::LibcallImpl> getLibcallImpls() const {
    // Trim Unsupported from the start
    return ArrayRef(LibcallImpls).drop_front();
  }

  /// Get the comparison predicate that's to be used to test the result of the
  /// comparison libcall against zero. This should only be used with
  /// floating-point compare libcalls.
  CmpInst::Predicate
  getSoftFloatCmpLibcallPredicate(RTLIB::Libcall Call) const {
    return SoftFloatCompareLibcallPredicates[Call];
  }

  // FIXME: This should be removed. This should be private constant.
  // FIXME: This should be a function of RTLIB::LibcallImpl
  void setSoftFloatCmpLibcallPredicate(RTLIB::Libcall Call,
                                       CmpInst::Predicate Pred) {
    SoftFloatCompareLibcallPredicates[Call] = Pred;
  }

  /// Return a function name compatible with RTLIB::MEMCPY, or nullptr if fully
  /// unsupported.
  const char *getMemcpyName() const {
    if (const char *Memcpy = getLibcallName(RTLIB::MEMCPY))
      return Memcpy;

    // Fallback to memmove if memcpy isn't available.
    return getLibcallName(RTLIB::MEMMOVE);
  }

private:
  static const RTLIB::LibcallImpl
      DefaultLibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1];

  /// Stores the implementation choice for each each libcall.
  RTLIB::LibcallImpl LibcallImpls[RTLIB::UNKNOWN_LIBCALL + 1] = {
      RTLIB::Unsupported};

  static_assert(static_cast<int>(CallingConv::C) == 0,
                "default calling conv should be encoded as 0");

  /// Stores the CallingConv that should be used for each libcall.
  CallingConv::ID LibcallCallingConvs[RTLIB::UNKNOWN_LIBCALL] = {};

  /// The condition type that should be used to test the result of each of the
  /// soft floating-point comparison libcall against integer zero.
  ///
  // FIXME: This is only relevant for the handful of floating-point comparison
  // runtime calls; it's excessive to have a table entry for every single
  // opcode.
  CmpInst::Predicate SoftFloatCompareLibcallPredicates[RTLIB::UNKNOWN_LIBCALL];

  /// Names of concrete implementations of runtime calls. e.g. __ashlsi3 for
  /// SHL_I32
  static const char *const LibCallImplNames[RTLIB::NumLibcallImpls];

  /// Map from a concrete LibcallImpl implementation to its RTLIB::Libcall kind.
  static const RTLIB::Libcall ImplToLibcall[RTLIB::NumLibcallImpls];

  static bool darwinHasSinCosStret(const Triple &TT) {
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

  static bool darwinHasExp10(const Triple &TT);

  /// Return true if the target has sincosf/sincos/sincosl functions
  static bool hasSinCos(const Triple &TT) {
    return TT.isGNUEnvironment() || TT.isOSFuchsia() ||
           (TT.isAndroid() && !TT.isAndroidVersionLT(9));
  }

  void initDefaultLibCallImpls();

  /// Generated by tablegen.
  void setPPCLibCallNameOverrides();

  /// Generated by tablegen.
  void setZOSLibCallNameOverrides();

  /// Generated by tablegen.
  void setWindowsArm64LibCallNameOverrides();

  void initSoftFloatCmpLibcallPredicates();

  /// Set default libcall names. If a target wants to opt-out of a libcall it
  /// should be placed here.
  LLVM_ABI void initLibcalls(const Triple &TT, ExceptionHandling ExceptionModel,
                             FloatABI::ABIType FloatABI, EABI ABIType,
                             StringRef ABIName);
};

} // namespace RTLIB
} // namespace llvm

#endif // LLVM_IR_RUNTIME_LIBCALLS_H
