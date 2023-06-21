//=== OutputSections.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "llvm/ADT/StringSwitch.h"

namespace llvm {
namespace dwarflinker_parallel {

std::optional<OutputSections::DebugSectionKind>
OutputSections::parseDebugSectionName(llvm::StringRef SecName) {
  return llvm::StringSwitch<std::optional<OutputSections::DebugSectionKind>>(
             SecName)
      .Case("debug_info", DebugSectionKind::DebugInfo)
      .Case("debug_line", DebugSectionKind::DebugLine)
      .Case("debug_frame", DebugSectionKind::DebugFrame)
      .Case("debug_ranges", DebugSectionKind::DebugRange)
      .Case("debug_rnglists", DebugSectionKind::DebugRngLists)
      .Case("debug_loc", DebugSectionKind::DebugLoc)
      .Case("debug_loclists", DebugSectionKind::DebugLocLists)
      .Case("debug_aranges", DebugSectionKind::DebugARanges)
      .Case("debug_abbrev", DebugSectionKind::DebugAbbrev)
      .Case("debug_macinfo", DebugSectionKind::DebugMacinfo)
      .Case("debug_macro", DebugSectionKind::DebugMacro)
      .Default(std::nullopt);

  return std::nullopt;
}

} // end of namespace dwarflinker_parallel
} // end of namespace llvm
