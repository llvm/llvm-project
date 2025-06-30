//===-- llvm/RuntimeLibcalls.td - File that describes libcalls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

// TODO: Do we need a way to conditionally select calling convention?
class RuntimeLibcallImpl<RuntimeLibcall P, string Name = NAME> {
  RuntimeLibcall Provides = P;
  string LibCallFuncName = Name;
  bit IsDefault = false;
}
