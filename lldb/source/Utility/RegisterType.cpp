//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegisterType.h"

#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cassert>
#include <cinttypes>
#include <limits>

using namespace lldb_private;

namespace {

std::atomic<uint64_t> g_next_register_type_uid{1};

void PrintXMLAttributeValue(Stream &strm, llvm::StringRef value) {
  std::string escaped;
  llvm::raw_string_ostream escape_strm(escaped);
  llvm::printHTMLEscaped(value, escape_strm);
  strm << escaped;
}

} // namespace

RegisterType::RegisterType(RegisterTypeKind kind, std::string id)
    : m_kind(kind), m_id(std::move(id)),
      m_uid(g_next_register_type_uid.fetch_add(1, std::memory_order_relaxed)) {}

void RegisterType::ToXML(Stream &strm,
                         std::unordered_set<std::string> &previously_emitted,
                         const RegisterType *user) const {
  if (getKind() == eRegisterTypeKindBuiltin)
    return;

  if (!previously_emitted.insert(GetID()).second)
    return;

  for (auto dep : m_dependencies)
    dep->ToXML(strm, previously_emitted, this);

  ToXMLElement(strm, user);
}

RegisterTypeBuiltin::RegisterTypeBuiltin(std::string id,
                                         lldb::Encoding encoding,
                                         lldb::Format format,
                                         uint32_t byte_size)
    : RegisterType(eRegisterTypeKindBuiltin, std::move(id)),
      m_encoding(encoding), m_format(format), m_byte_size(byte_size) {}

void RegisterTypeBuiltin::ToXMLElement(Stream &, const RegisterType *) const {}

RegisterTypeVector::RegisterTypeVector(std::string id,
                                       const RegisterType *element_type,
                                       uint32_t count)
    : RegisterType(eRegisterTypeKindVector, std::move(id)),
      m_element_type(element_type), m_count(count) {
  assert(m_element_type && "Vector element type cannot be null");
  assert(m_count && "Vector element count cannot be zero");
  SetDependencies({m_element_type});
}

std::optional<uint64_t> RegisterTypeVector::GetByteSize() const {
  std::optional<uint64_t> element_size = m_element_type->GetByteSize();
  if (!element_size ||
      *element_size > std::numeric_limits<uint64_t>::max() / m_count)
    return std::nullopt;
  return *element_size * m_count;
}

void RegisterTypeVector::ToXMLElement(Stream &strm,
                                      const RegisterType *) const {
  strm.Indent();
  strm << "<vector id=\"";
  PrintXMLAttributeValue(strm, GetID());
  strm << "\" type=\"";
  PrintXMLAttributeValue(strm, m_element_type->GetID());
  strm.Printf("\" count=\"%" PRIu32 "\"/>\n", m_count);
}
