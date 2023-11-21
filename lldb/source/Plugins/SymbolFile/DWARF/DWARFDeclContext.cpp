//===-- DWARFDeclContext.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDeclContext.h"

using namespace lldb_private::dwarf;
using namespace lldb_private::plugin::dwarf;

const char *DWARFDeclContext::GetQualifiedName() const {
  if (m_qualified_name.empty()) {
    // The declaration context array for a class named "foo" in namespace
    // "a::b::c" will be something like:
    //  [0] DW_TAG_class_type "foo"
    //  [1] DW_TAG_namespace "c"
    //  [2] DW_TAG_namespace "b"
    //  [3] DW_TAG_namespace "a"
    if (!m_entries.empty()) {
      if (m_entries.size() == 1) {
        if (m_entries[0].name) {
          m_qualified_name.append("::");
          m_qualified_name.append(m_entries[0].name);
        }
      } else {
        collection::const_reverse_iterator pos;
        collection::const_reverse_iterator begin = m_entries.rbegin();
        collection::const_reverse_iterator end = m_entries.rend();
        for (pos = begin; pos != end; ++pos) {
          if (pos != begin)
            m_qualified_name.append("::");
          if (pos->name == nullptr) {
            if (pos->tag == DW_TAG_namespace)
              m_qualified_name.append("(anonymous namespace)");
            else if (pos->tag == DW_TAG_class_type)
              m_qualified_name.append("(anonymous class)");
            else if (pos->tag == DW_TAG_structure_type)
              m_qualified_name.append("(anonymous struct)");
            else if (pos->tag == DW_TAG_union_type)
              m_qualified_name.append("(anonymous union)");
            else
              m_qualified_name.append("(anonymous)");
          } else
            m_qualified_name.append(pos->name);
        }
      }
    }
  }
  if (m_qualified_name.empty())
    return nullptr;
  return m_qualified_name.c_str();
}

bool DWARFDeclContext::operator==(const DWARFDeclContext &rhs) const {
  if (m_entries.size() != rhs.m_entries.size())
    return false;

  collection::const_iterator pos;
  collection::const_iterator begin = m_entries.begin();
  collection::const_iterator end = m_entries.end();

  collection::const_iterator rhs_pos;
  collection::const_iterator rhs_begin = rhs.m_entries.begin();
  // The two entry arrays have the same size

  // First compare the tags before we do expensive name compares
  for (pos = begin, rhs_pos = rhs_begin; pos != end; ++pos, ++rhs_pos) {
    if (pos->tag != rhs_pos->tag) {
      // Check for DW_TAG_structure_type and DW_TAG_class_type as they are
      // often used interchangeably in GCC
      if (pos->tag == DW_TAG_structure_type &&
          rhs_pos->tag == DW_TAG_class_type)
        continue;
      if (pos->tag == DW_TAG_class_type &&
          rhs_pos->tag == DW_TAG_structure_type)
        continue;
      return false;
    }
  }
  // The tags all match, now compare the names
  for (pos = begin, rhs_pos = rhs_begin; pos != end; ++pos, ++rhs_pos) {
    if (!pos->NameMatches(*rhs_pos))
      return false;
  }
  // All tags and names match
  return true;
}
