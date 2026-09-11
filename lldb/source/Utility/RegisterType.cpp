//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegisterType.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <cassert>
#include <cinttypes>
#include <limits>

using namespace lldb_private;

static std::atomic<uint64_t> g_next_register_type_uid{1};

RegisterType::RegisterType(RegisterTypeKind kind, std::string id)
    : m_kind(kind), m_id(std::move(id)),
      m_uid(g_next_register_type_uid.fetch_add(1, std::memory_order_relaxed)) {}

void RegisterType::ToXML(Stream &strm,
                         std::unordered_set<std::string> &previously_emitted,
                         const RegisterType *user) const {
  // XML type references use IDs, so definitions must also be unique by ID.
  if (!previously_emitted.insert(GetID()).second)
    return;

  // Emit this type's dependencies first.
  for (auto dep : m_dependencies)
    dep->ToXML(strm, previously_emitted, this);

  // Finally emit this type.
  ToXMLElement(strm, user);
}

void RegisterType::PrintXMLAttributeValue(Stream &strm, llvm::StringRef value) {
  std::string escaped;
  llvm::raw_string_ostream escape_strm(escaped);
  llvm::printHTMLEscaped(value, escape_strm);
  strm << escaped;
}

RegisterTypeBuiltin::RegisterTypeBuiltin(std::string id,
                                         lldb::Encoding encoding,
                                         lldb::Format format,
                                         std::optional<uint64_t> byte_size)
    : RegisterType(eRegisterTypeKindBuiltin, std::move(id)),
      m_encoding(encoding), m_format(format), m_byte_size(byte_size) {}

void RegisterTypeBuiltin::ToXML(Stream &, std::unordered_set<std::string> &,
                                const RegisterType *) const {}

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

bool RegisterTypeVector::IsByteSizeCompatible(uint64_t byte_size) const {
  if (!byte_size || byte_size % m_count)
    return false;

  uint64_t element_byte_size = byte_size / m_count;
  if (std::optional<uint64_t> fixed_size = m_element_type->GetByteSize())
    return *fixed_size == element_byte_size;
  if (const auto *vector = llvm::dyn_cast<RegisterTypeVector>(m_element_type))
    return vector->IsByteSizeCompatible(element_byte_size);
  return llvm::isa<RegisterTypeBuiltin>(m_element_type);
}

void RegisterTypeVector::DumpToLog(Log *log) const {
  LLDB_LOG(log, "ID: \"{0}\" Element type: \"{1}\" Count: {2}", GetID().c_str(),
           m_element_type->GetID().c_str(), m_count);
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
