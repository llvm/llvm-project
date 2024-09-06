//===-- RegisterTypeUnion.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterTypeUnion.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

using namespace lldb_private;

RegisterTypeUnion::RegisterTypeUnion(std::string id,
                                     const RegisterTypeUnion::Fields &fields)
    : RegisterType(eRegisterTypeKindUnion, id), m_fields(fields) {
  // We assume the XML processor also checked this, so this assert is only for
  // unions created directly from C++ (or in other words, for lldb's built in
  // register types).
  assert(ValidateFields(m_fields) &&
         "All fields of a union must have the same size, "
         "and their size must be non-zero.");

  std::vector<const RegisterType *> dependencies;
  for (const auto &field : m_fields)
    dependencies.push_back(field.second);

  SetDependencies(dependencies);
}

bool RegisterTypeUnion::ValidateFields(
    const RegisterTypeUnion::Fields &fields) {
  std::optional<unsigned> size;
  for (const auto &field : fields) {
    // All fields of the union must have the same size, and no field can have 0
    // size.
    if (size) {
      if (!size || field.second->GetSize() != *size)
        return false;
    } else
      size = field.second->GetSize();
  }

  return true;
}

void RegisterTypeUnion::ToXMLElement(Stream &strm,
                                     const RegisterType *user) const {
  (void)user;
  // Example XML:
  // <union id="foo">
  //  <field name="some name" type="some type"/>
  // </union>
  strm.Indent();
  strm << "<union id=\"" << GetID() << "\"";

  if (m_fields.empty()) {
    strm << "/>\n";
    return;
  } else
    strm << ">\n";

  strm.IndentMore();
  for (const auto &field : m_fields) {
    strm.Indent("<field name=\"");
    strm << field.first << "\" type=\"" << field.second->GetID() << "\"/>\n";
  }
  strm.IndentLess();
  strm.Indent("</union>\n");
}

void RegisterTypeUnion::DumpToLog(Log *log) const {
  LLDB_LOG(log, "union ID: \"{0}\"", GetID().c_str());
  for (const auto &field : m_fields)
    LLDB_LOG(log, "  Name: \"{0}\" Type: \"{1}\"", field.first.c_str(),
             field.second->GetID());
}

unsigned RegisterTypeUnion::GetSize() const {
  // We assume that the XML parser and/or class constructor checked that all
  // fields have the same size. A union with no fields is valid, but you'll
  // never be able to attach it to a register, which is what size of 0 means.
  return m_fields.size() ? m_fields[0].second->GetSize() : 0;
}