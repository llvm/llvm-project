//===-- CodeViewRegisterMapping.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_CODEVIEWREGISTERMAPPING_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_NATIVEPDB_CODEVIEWREGISTERMAPPING_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/TargetParser/Triple.h"

namespace lldb_private {
namespace npdb {

uint32_t GetLLDBRegisterNumber(llvm::Triple::ArchType arch_type,
                               llvm::codeview::RegisterId register_id);

/// Maps a CodeView register id to the architecture-standard DWARF register
/// number (eRegisterKindDWARF), when one is defined for that register.
/// Unlike GetLLDBRegisterNumber(), a DWARF register number is independent of
/// any particular RegisterContext's own internal numbering, so it can be
/// correctly translated at evaluation time regardless of how the live
/// process's register list happens to be ordered (e.g. a raw gdb-remote
/// connection to a stub with no target description, which assigns its own
/// eRegisterKindLLDB numbering that need not match lldb-server's). Returns
/// LLDB_INVALID_REGNUM if no DWARF register number is known for this
/// register/architecture; callers should fall back to
/// GetLLDBRegisterNumber() in that case.
uint32_t GetDWARFRegisterNumber(llvm::Triple::ArchType arch_type,
                                llvm::codeview::RegisterId register_id);

uint32_t GetRegisterSize(llvm::codeview::RegisterId register_id);

} // namespace npdb
} // namespace lldb_private

#endif
