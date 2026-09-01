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
