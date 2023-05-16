//===-- DWARFAbbreviationDeclaration.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFABBREVIATIONDECLARATION_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFABBREVIATIONDECLARATION_H

#include "DWARFAttribute.h"
#include "DWARFDefines.h"
#include "SymbolFileDWARF.h"
#include "llvm/Support/Error.h"

class DWARFAbbreviationDeclaration {
public:
  struct AttributeSpec {
    AttributeSpec(dw_attr_t attr, dw_form_t form, int64_t value)
        : m_attr(attr), m_form(form), m_value(value) {}

    AttributeSpec(dw_attr_t attr, dw_form_t form)
        : m_attr(attr), m_form(form), m_value(0) {}

    bool IsImplicitConst() const {
      return m_form == lldb_private::dwarf::DW_FORM_implicit_const;
    }

    int64_t GetImplicitConstValue() const { return m_value; }

    dw_attr_t GetAttribute() const { return m_attr; }

    dw_form_t GetForm() const { return m_form; }

  private:
    dw_attr_t m_attr;
    dw_form_t m_form;
    int64_t m_value;
  };

  enum { InvalidCode = 0 };
  DWARFAbbreviationDeclaration();

  // For hand crafting an abbreviation declaration
  DWARFAbbreviationDeclaration(dw_tag_t tag, uint8_t has_children);

  uint32_t Code() const { return m_code; }
  void SetCode(uint32_t code) { m_code = code; }
  dw_tag_t Tag() const { return m_tag; }
  bool HasChildren() const { return m_has_children; }
  size_t NumAttributes() const { return m_attributes.size(); }
  dw_form_t GetFormByIndex(uint32_t idx) const {
    return m_attributes.size() > idx ? m_attributes[idx].GetForm()
                                     : dw_form_t(0);
  }

  // idx is assumed to be valid when calling GetAttrAndFormByIndex()
  void GetAttrAndFormValueByIndex(uint32_t idx, dw_attr_t &attr,
                                  DWARFFormValue &form_value) const {
    const AttributeSpec &spec = m_attributes[idx];
    attr = spec.GetAttribute();
    form_value.FormRef() = spec.GetForm();
    if (spec.IsImplicitConst())
      form_value.SetSigned(spec.GetImplicitConstValue());
  }
  dw_form_t GetFormByIndexUnchecked(uint32_t idx) const {
    return m_attributes[idx].GetForm();
  }
  uint32_t FindAttributeIndex(dw_attr_t attr) const;

  /// Extract one abbreviation declaration and all of its associated attributes.
  /// Possible return values:
  ///   DWARFEnumState::Complete - the extraction completed successfully.  This
  ///       was the last abbrev decl in a sequence, and the user should not call
  ///       this function again.
  ///   DWARFEnumState::MoreItems - the extraction completed successfully.  The
  ///       user should call this function again to retrieve the next decl.
  ///   llvm::Error - A parsing error occurred.  The debug info is malformed.
  llvm::Expected<lldb_private::DWARFEnumState>
  extract(const lldb_private::DWARFDataExtractor &data,
          lldb::offset_t *offset_ptr);
  bool IsValid();

protected:
  uint32_t m_code = InvalidCode;
  dw_tag_t m_tag = llvm::dwarf::DW_TAG_null;
  uint8_t m_has_children = 0;
  llvm::SmallVector<AttributeSpec, 4> m_attributes;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFABBREVIATIONDECLARATION_H
