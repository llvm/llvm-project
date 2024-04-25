//===-- DWARFFormValue.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFFORMVALUE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFFORMVALUE_H

#include "DWARFDataExtractor.h"
#include <cstddef>
#include <optional>

namespace lldb_private::plugin {
namespace dwarf {
class DWARFUnit;
class SymbolFileDWARF;
class DWARFDIE;

class DWARFFormValue {
public:
  typedef struct ValueTypeTag {
    ValueTypeTag() : value() { value.uval = 0; }

    union {
      uint64_t uval;
      int64_t sval;
      const char *cstr;
    } value;
    const uint8_t *data = nullptr;
  } ValueType;

  enum {
    eValueTypeInvalid = 0,
    eValueTypeUnsigned,
    eValueTypeSigned,
    eValueTypeCStr,
    eValueTypeBlock
  };

  DWARFFormValue() = default;
  DWARFFormValue(const DWARFUnit *unit) : m_unit(unit) {}
  DWARFFormValue(const DWARFUnit *unit, dw_form_t form)
      : m_unit(unit), m_form(form) {}
  const DWARFUnit *GetUnit() const { return m_unit; }
  void SetUnit(const DWARFUnit *unit) { m_unit = unit; }
  dw_form_t Form() const { return m_form; }
  dw_form_t &FormRef() { return m_form; }
  void SetForm(dw_form_t form) { m_form = form; }
  const ValueType &Value() const { return m_value; }
  ValueType &ValueRef() { return m_value; }
  void SetValue(const ValueType &val) { m_value = val; }

  void Dump(Stream &s) const;
  bool ExtractValue(const DWARFDataExtractor &data, lldb::offset_t *offset_ptr);
  const uint8_t *BlockData() const;
  static std::optional<uint8_t> GetFixedSize(dw_form_t form,
                                             const DWARFUnit *u);
  std::optional<uint8_t> GetFixedSize() const;
  DWARFDIE Reference() const;

  /// If this is a reference to another DIE, return the corresponding DWARFUnit
  /// and DIE offset such that Unit->GetDIE(offset) produces the desired DIE.
  /// Otherwise, a nullptr and unspecified offset are returned.
  std::pair<DWARFUnit *, uint64_t> ReferencedUnitAndOffset() const;

  uint64_t Reference(dw_offset_t offset) const;
  bool Boolean() const { return m_value.value.uval != 0; }
  uint64_t Unsigned() const { return m_value.value.uval; }
  std::optional<uint64_t> getAsUnsignedConstant() const {
    if ((!IsDataForm(m_form)) || m_form == lldb_private::dwarf::DW_FORM_sdata)
      return std::nullopt;
    return m_value.value.uval;
  }
  std::optional<int64_t> getAsSignedConstant() const {
    if ((!IsDataForm(m_form)) ||
        (m_form == lldb_private::dwarf::DW_FORM_udata &&
         uint64_t(std::numeric_limits<int64_t>::max()) < m_value.value.uval))
      return std::nullopt;
    switch (m_form) {
    case lldb_private::dwarf::DW_FORM_data4:
      return int32_t(m_value.value.uval);
    case lldb_private::dwarf::DW_FORM_data2:
      return int16_t(m_value.value.uval);
    case lldb_private::dwarf::DW_FORM_data1:
      return int8_t(m_value.value.uval);
    case lldb_private::dwarf::DW_FORM_sdata:
    case lldb_private::dwarf::DW_FORM_data8:
    default:
      return m_value.value.sval;
    }
  }

  void SetUnsigned(uint64_t uval) { m_value.value.uval = uval; }
  int64_t Signed() const { return m_value.value.sval; }
  void SetSigned(int64_t sval) { m_value.value.sval = sval; }
  const char *AsCString() const;
  dw_addr_t Address() const;
  bool IsValid() const { return m_form != 0; }
  bool SkipValue(const DWARFDataExtractor &debug_info_data,
                 lldb::offset_t *offset_ptr) const;
  static bool SkipValue(const dw_form_t form,
                        const DWARFDataExtractor &debug_info_data,
                        lldb::offset_t *offset_ptr, const DWARFUnit *unit);
  static bool IsBlockForm(const dw_form_t form);
  static bool IsDataForm(const dw_form_t form);
  static int Compare(const DWARFFormValue &a, const DWARFFormValue &b);
  void Clear();
  static bool FormIsSupported(dw_form_t form);

protected:
  // Compile unit where m_value was located.
  // It may be different from compile unit where m_value refers to.
  const DWARFUnit *m_unit = nullptr; // Unit for this form
  dw_form_t m_form = dw_form_t(0);   // Form for this value
  ValueType m_value;                 // Contains all data for the form
};

inline const char* toString(DWARFFormValue Value, const char* Default) {
  if (const char* R = Value.AsCString())
    return R;
  return Default;
}
inline const char* toString(std::optional<DWARFFormValue> Value, const char* Default) {
  if (!Value)
    return Default;
  if (const char* R = Value->AsCString())
    return R;
  return Default;
}
} // namespace dwarf
} // namespace lldb_private::plugin

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFFORMVALUE_H
