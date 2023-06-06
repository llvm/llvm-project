//===- OutputSections.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_OUTPUTSECTIONS_H
#define LLVM_LIB_DWARFLINKERPARALLEL_OUTPUTSECTIONS_H

#include "llvm/ADT/StringRef.h"
#include <array>
#include <cstdint>

namespace llvm {
namespace dwarflinker_parallel {

/// This class keeps offsets to the debug sections. Any object which is
/// supposed to be emitted into the debug section should use this class to
/// track debug sections offsets.
class OutputSections {
public:
  /// List of tracked debug sections.
  enum class DebugSectionKind : uint8_t {
    DebugInfo = 0,
    DebugLine,
    DebugFrame,
    DebugRange,
    DebugRngLists,
    DebugLoc,
    DebugLocLists,
    DebugARanges,
    DebugAbbrev,
    DebugMacinfo,
    DebugMacro,
  };
  constexpr static size_t SectionKindsNum = 11;

  /// Recognise the section name and match it with the DebugSectionKind.
  static std::optional<DebugSectionKind> parseDebugSectionName(StringRef Name);

  /// When objects(f.e. compile units) are glued into the single file,
  /// the debug sections corresponding to the concrete object are assigned
  /// with offsets inside the whole file. This method returns offset
  /// to the \p SectionKind debug section, corresponding to this object.
  uint64_t getStartOffset(DebugSectionKind SectionKind) const {
    return Offsets[static_cast<
        typename std::underlying_type<DebugSectionKind>::type>(SectionKind)];
  }

  /// Set offset to the start of specified \p SectionKind debug section,
  /// corresponding to this object.
  void setStartOffset(DebugSectionKind SectionKind, uint64_t Offset) {
    Offsets[static_cast<typename std::underlying_type<DebugSectionKind>::type>(
        SectionKind)] = Offset;
  }

protected:
  /// Offsets to the debug sections composing this object.
  std::array<uint64_t, SectionKindsNum> Offsets = {0};
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_OUTPUTSECTIONS_H
