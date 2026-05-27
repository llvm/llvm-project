//===-- RegisterTypeVector.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/RegisterTypeVector.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"

#include "llvm/ADT/StringSwitch.h"

using namespace lldb_private;

// TODO: this code may already exist in the XML processor.
static RegisterTypeVector::ElementTypeInfo
LookupElementType(const std::string &type) {
  // See
  // https://sourceware.org/gdb/current/onlinedocs/gdb.html/Predefined-Target-Types.html#Predefined-Target-Types.
  // Currently this is just the ones qemu sends for SVE vectors.
  return llvm::StringSwitch<RegisterTypeVector::ElementTypeInfo>(type)
      .Case("int8", {lldb::eEncodingSint, 1})
      .Case("int16", {lldb::eEncodingSint, 2})
      .Case("int32", {lldb::eEncodingSint, 4})
      .Case("int64", {lldb::eEncodingSint, 8})
      .Case("int128", {lldb::eEncodingSint, 16})
      .Case("uint8", {lldb::eEncodingUint, 1})
      .Case("uint16", {lldb::eEncodingUint, 2})
      .Case("uint32", {lldb::eEncodingUint, 4})
      .Case("uint64", {lldb::eEncodingUint, 8})
      .Case("uint128", {lldb::eEncodingUint, 16})
      .Case("ieee_half", {lldb::eEncodingIEEE754, 2})
      .Case("ieee_single", {lldb::eEncodingIEEE754, 4})
      .Case("ieee_double", {lldb::eEncodingIEEE754, 8})
      .Default({});
}

RegisterTypeVector::RegisterTypeVector(std::string id, std::string type,
                                       unsigned count)
    : RegisterType(eRegisterTypeKindVector, id), m_type(type), m_count(count),
      m_element_type_info(LookupElementType(type)) {}

// TODO: test me!
void RegisterTypeVector::ToXMLElement(Stream &strm,
                                      const RegisterType *user) const {
  (void)user;
  // Example XML:
  // <vector id="foo" type="some type" count="4"/>
  strm.Indent();
  strm << "<vector id=\"" << GetID() << "\" type=\"" << GetType()
       << "\" count=\"";
  strm.Printf("%d\"/>\n", GetCount());
}

void RegisterTypeVector::DumpToLog(Log *log) const {
  LLDB_LOG(log, "vector ID: \"{0}\", type: \"{1}\", count: {2}",
           GetID().c_str(), GetType().c_str(), GetCount());
}

unsigned RegisterTypeVector::GetSize() const {
  return GetCount() * m_element_type_info.size;
}