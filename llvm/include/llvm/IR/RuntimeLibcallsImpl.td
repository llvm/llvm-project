//===-- llvm/RuntimeLibcalls.td - File that describes libcalls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SetTheory.td"

// Predicate for whether a libcall exists for the target ABI. This is
// a module level property that should only be computed based on the
// triple.
class RuntimeLibcallPredicate<code cond> {
  // Expression of an llvm::Triple named TT for whether a libcall
  // should exist.
  code Cond = cond;
}

// Predicate for whether a libcall should be used for the current
// function/subtarget.
class LibcallLoweringPredicate<code cond> { code Cond = cond; }

def AlwaysAvailable : RuntimeLibcallPredicate<[{}]>;

class LibcallCallingConv<code CC = [{}]> {
  // Enum value for the calling convention. Empty string defaults to
  // CallingConv::C.
  code CallingConv = CC;
}

def DefaultCC : LibcallCallingConv;
def FASTCC : LibcallCallingConv<[{CallingConv::Fast}]>;
def MSP430_BUILTIN : LibcallCallingConv<[{CallingConv::MSP430_BUILTIN}]>;
def ARM_AAPCS : LibcallCallingConv<[{CallingConv::ARM_AAPCS}]>;
def ARM_AAPCS_VFP : LibcallCallingConv<[{CallingConv::ARM_AAPCS_VFP}]>;
def X86_STDCALL : LibcallCallingConv<[{CallingConv::X86_StdCall}]>;
def AVR_BUILTIN : LibcallCallingConv<[{CallingConv::AVR_BUILTIN}]>;

/// Abstract definition for functionality the compiler may need to
/// emit a call to. Emits the RTLIB::Libcall enum - This enum defines
/// all of the runtime library calls the backend can emit. The various
/// long double types cannot be merged, because 80-bit library
/// functions use "xf" and 128-bit use "tf".
///
/// When adding PPCF128 functions here, note that their names generally need
/// to be overridden for Darwin with the xxx$LDBL128 form.  See
/// PPCISelLowering.cpp.
class RuntimeLibcall {
  string Name = NAME;

  // TODO: Record type signature and attributes
}

// Concrete implementation of a libcall, which may have a different
// name and only be valid on some subset of targets.

class RuntimeLibcallImpl<RuntimeLibcall P, string Name = NAME> {
  RuntimeLibcall Provides = P;
  string LibCallFuncName = Name;
  list<LibcallLoweringPredicate> LoweringPredicates;
  bit IsDefault = false;
}

class LibcallImpls<dag funcList,
                   RuntimeLibcallPredicate Pred = AlwaysAvailable> {
  // Function of the triple where this applies
  RuntimeLibcallPredicate AvailabilityPredicate = Pred;
  dag MemberList = funcList;

  // If set, applies calling convention to every entry instead of
  // taking the system default.
  LibcallCallingConv CallingConv = ?;
}

/// Convenience wrapper around LibcallImplSet to make a single libcall
/// implementation conditionally conditionally available.
class AvailableIf<RuntimeLibcallImpl Impl, RuntimeLibcallPredicate Pred>
    : LibcallImpls<(add Impl), Pred>;

class LibcallsWithCC<dag funcList, LibcallCallingConv CC,
                     RuntimeLibcallPredicate Pred = AlwaysAvailable>
    : LibcallImpls<funcList, Pred> {
  LibcallCallingConv CallingConv = CC;
}

/// Define a complete top level set of runtime libcalls for a target.
class SystemRuntimeLibrary<RuntimeLibcallPredicate Pred, dag funcList> {
  /// Set the default calling convention assumed for RuntimeLibcallImpl members.
  LibcallCallingConv DefaultLibcallCallingConv = DefaultCC;
  RuntimeLibcallPredicate TriplePred = Pred;
  LibcallImpls MemberList = LibcallImpls<funcList>;
}
