//===-- DIERef.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DIEREF_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DIEREF_H

#include "lldb/Core/dwarf.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/FormatProviders.h"
#include <cassert>
#include <vector>

/// Identifies a DWARF debug info entry within a given Module. It contains three
/// "coordinates":
/// - dwo_num: identifies the dwo file in the Module. If this field is not set,
///   the DIERef references the main file.
/// - section: identifies the section of the debug info entry in the given file:
///   debug_info or debug_types.
/// - die_offset: The offset of the debug info entry as an absolute offset from
///   the beginning of the section specified in the section field.
class DIERef {
public:
  enum Section : uint8_t { DebugInfo, DebugTypes };

  DIERef(llvm::Optional<uint32_t> dwo_num, Section section,
         dw_offset_t die_offset)
      : m_dwo_num(dwo_num.value_or(0)), m_dwo_num_valid(bool(dwo_num)),
        m_section(section), m_die_offset(die_offset) {
    assert(this->dwo_num() == dwo_num && "Dwo number out of range?");
  }

  llvm::Optional<uint32_t> dwo_num() const {
    if (m_dwo_num_valid)
      return m_dwo_num;
    return llvm::None;
  }

  Section section() const { return static_cast<Section>(m_section); }

  dw_offset_t die_offset() const { return m_die_offset; }

  bool operator<(DIERef other) const {
    if (m_dwo_num_valid != other.m_dwo_num_valid)
      return m_dwo_num_valid < other.m_dwo_num_valid;
    if (m_dwo_num_valid && (m_dwo_num != other.m_dwo_num))
      return m_dwo_num < other.m_dwo_num;
    if (m_section != other.m_section)
      return m_section < other.m_section;
    return m_die_offset < other.m_die_offset;
  }

  bool operator==(const DIERef &rhs) const {
    return dwo_num() == rhs.dwo_num() && m_section == rhs.m_section &&
           m_die_offset == rhs.m_die_offset;
  }

  bool operator!=(const DIERef &rhs) const { return !(*this == rhs); }

  /// Decode a serialized version of this object from data.
  ///
  /// \param data
  ///   The decoder object that references the serialized data.
  ///
  /// \param offset_ptr
  ///   A pointer that contains the offset from which the data will be decoded
  ///   from that gets updated as data gets decoded.
  ///
  /// \return
  ///   Returns a valid DIERef if decoding succeeded, llvm::None if there was
  ///   unsufficient or invalid values that were decoded.
  static llvm::Optional<DIERef> Decode(const lldb_private::DataExtractor &data,
                                       lldb::offset_t *offset_ptr);

  /// Encode this object into a data encoder object.
  ///
  /// This allows this object to be serialized to disk.
  ///
  /// \param encoder
  ///   A data encoder object that serialized bytes will be encoded into.
  ///
  void Encode(lldb_private::DataEncoder &encoder) const;

private:
  uint32_t m_dwo_num : 30;
  uint32_t m_dwo_num_valid : 1;
  uint32_t m_section : 1;
  dw_offset_t m_die_offset;
};
static_assert(sizeof(DIERef) == 8);

typedef std::vector<DIERef> DIEArray;

namespace llvm {
template<> struct format_provider<DIERef> {
  static void format(const DIERef &ref, raw_ostream &OS, StringRef Style);
};
} // namespace llvm

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DIEREF_H
