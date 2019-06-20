//===-- DIERef.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DIERef_h_
#define SymbolFileDWARF_DIERef_h_

#include "lldb/Core/dwarf.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FormatProviders.h"
#include <vector>

/// Identifies a DWARF debug info entry within a given Module. It contains three
/// "coordinates":
/// - section: identifies the section of the debug info entry: debug_info or
///   debug_types
/// - unit_offset: the offset of the unit containing the debug info entry. For
///   regular (unsplit) units, this field is optional, as the die_offset is
///   enough to uniquely identify the containing unit. For split units, this
///   field must contain the offset of the skeleton unit in the main object
///   file.
/// - die_offset: The offset of te debug info entry as an absolute offset from
///   the beginning of the section specified in the section field.
class DIERef {
public:
  enum Section : uint8_t { DebugInfo, DebugTypes };

  DIERef(Section s, llvm::Optional<dw_offset_t> u, dw_offset_t d)
      : m_section(s), m_unit_offset(u.getValueOr(DW_INVALID_OFFSET)),
        m_die_offset(d) {}

  Section section() const { return static_cast<Section>(m_section); }

  llvm::Optional<dw_offset_t> unit_offset() const {
    if (m_unit_offset != DW_INVALID_OFFSET)
      return m_unit_offset;
    return llvm::None;
  }

  dw_offset_t die_offset() const { return m_die_offset; }

private:
  unsigned m_section : 1;
  dw_offset_t m_unit_offset;
  dw_offset_t m_die_offset;
};

typedef std::vector<DIERef> DIEArray;

namespace llvm {
template<> struct format_provider<DIERef> {
  static void format(const DIERef &ref, raw_ostream &OS, StringRef Style);
};
} // namespace llvm

#endif // SymbolFileDWARF_DIERef_h_
