//===- DWARFEHFrameRegistrar.h - DWARF EH frame registration ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DWARF EH frame registration with the process's unwinder.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_INTERNAL_BEDROCK_SYS_POSIX_DWARFEHFRAMEREGISTRAR_H
#define ORC_RT_INTERNAL_BEDROCK_SYS_POSIX_DWARFEHFRAMEREGISTRAR_H

#include "orc-rt/support/Error.h"
#include "orc-rt/support/span.h"

namespace orc_rt::sys::posix {

/// Registers and deregisters JIT'd .eh_frame sections with the process's
/// unwinder (libgcc or libunwind), so that exceptions can be thrown through
/// JIT'd code.
class DWARFEHFrameRegistrar {
public:
  /// Register the .eh_frame section in EHFrame. Returns an error if the
  /// section is malformed, in which case nothing is registered.
  ///
  /// The section must remain mapped and unmodified until it is deregistered.
  static Error registerSection(span<const char> EHFrame) noexcept;

  /// Deregister an .eh_frame section previously registered with
  /// registerSection.
  static Error deregisterSection(span<const char> EHFrame) noexcept;
};

} // namespace orc_rt::sys::posix

#endif // ORC_RT_INTERNAL_BEDROCK_SYS_POSIX_DWARFEHFRAMEREGISTRAR_H
