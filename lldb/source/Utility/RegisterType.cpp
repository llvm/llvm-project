//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegisterType.h"

#include <atomic>

using namespace lldb_private;

static std::atomic<uint64_t> g_next_register_type_uid{1};

RegisterType::RegisterType(RegisterTypeKind kind, std::string id)
    : m_kind(kind), m_id(std::move(id)),
      m_uid(g_next_register_type_uid.fetch_add(1, std::memory_order_relaxed)) {}

void RegisterType::ToXML(
    Stream &strm, std::unordered_set<const RegisterType *> &previously_emitted,
    const RegisterType *user) const {
  // If we already emitted this, don't emit it again.
  if (!previously_emitted.insert(this).second)
    return;

  // Emit this type's dependencies first.
  for (auto dep : m_dependencies)
    dep->ToXML(strm, previously_emitted, this);

  // Finally emit this type.
  ToXMLElement(strm, user);
}
