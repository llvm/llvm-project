//===-- DWARFAbbreviationDeclaration.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFAbbreviationDeclaration.h"

#include "lldb/Core/dwarf.h"
#include "lldb/Utility/Stream.h"

#include "llvm/Object/Error.h"

#include "DWARFFormValue.h"

using namespace lldb_private;

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration()
    : m_code(InvalidCode), m_tag(0), m_has_children(0), m_attributes() {}

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration(dw_tag_t tag,
                                                           uint8_t has_children)
    : m_code(InvalidCode), m_tag(tag), m_has_children(has_children),
      m_attributes() {}

llvm::Expected<DWARFEnumState>
DWARFAbbreviationDeclaration::extract(const DWARFDataExtractor &data,
                                      lldb::offset_t *offset_ptr) {
  m_code = data.GetULEB128(offset_ptr);
  if (m_code == 0)
    return DWARFEnumState::Complete;

  m_attributes.clear();
  m_tag = data.GetULEB128(offset_ptr);
  if (m_tag == DW_TAG_null)
    return llvm::make_error<llvm::object::GenericBinaryError>(
        "abbrev decl requires non-null tag.");

  m_has_children = data.GetU8(offset_ptr);

  while (data.ValidOffset(*offset_ptr)) {
    dw_attr_t attr = data.GetULEB128(offset_ptr);
    dw_form_t form = data.GetULEB128(offset_ptr);

    // This is the last attribute for this abbrev decl, but there may still be
    // more abbrev decls, so return MoreItems to indicate to the caller that
    // they should call this function again.
    if (!attr && !form)
      return DWARFEnumState::MoreItems;

    if (!attr || !form)
      return llvm::make_error<llvm::object::GenericBinaryError>(
          "malformed abbreviation declaration attribute");

    DWARFFormValue::ValueType val;

    if (form == DW_FORM_implicit_const)
      val.value.sval = data.GetULEB128(offset_ptr);

    m_attributes.push_back(DWARFAttribute(attr, form, val));
  }

  return llvm::make_error<llvm::object::GenericBinaryError>(
      "abbreviation declaration attribute list not terminated with a null "
      "entry");
}

bool DWARFAbbreviationDeclaration::IsValid() {
  return m_code != 0 && m_tag != 0;
}

uint32_t
DWARFAbbreviationDeclaration::FindAttributeIndex(dw_attr_t attr) const {
  uint32_t i;
  const uint32_t kNumAttributes = m_attributes.size();
  for (i = 0; i < kNumAttributes; ++i) {
    if (m_attributes[i].get_attr() == attr)
      return i;
  }
  return DW_INVALID_INDEX;
}

bool DWARFAbbreviationDeclaration::
operator==(const DWARFAbbreviationDeclaration &rhs) const {
  return Tag() == rhs.Tag() && HasChildren() == rhs.HasChildren() &&
         m_attributes == rhs.m_attributes;
}
