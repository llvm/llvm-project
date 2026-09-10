//===-- Identifier.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Identifier.h"

#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstring>

using namespace lldb_private;
using namespace lldb_private::clike_typesystem;

Identifier IdentifierMap::get(llvm::StringRef name) {
  auto it = m_names.find(name);
  if (it != m_names.end())
    return Identifier(*it);

  // Copy the string into our own storage.
  char *storage = m_string_storage.Allocate<char>(name.size());
  std::memcpy(storage, name.data(), name.size());
  llvm::StringRef owned(storage, name.size());
  m_names.insert(owned);
  return Identifier(owned);
}

Identifier IdentifierMap::getWithStaticStorageStr(llvm::StringRef name) {
  // The caller promises the backing storage outlives this map.
  m_names.insert(name);
  return Identifier(name);
}

IdentifierMap::~IdentifierMap() {
#if LLVM_ADDRESS_SANITIZER_BUILD
  // Verify every identifier still points at live memory.
  for (llvm::StringRef name : m_names) {
    if (!name.empty() &&
        __asan_region_is_poisoned(const_cast<char *>(name.data()), name.size()))
      llvm::report_fatal_error("IdentifierMap holds a freed string "
                               "(misuse of getWithStaticStorageStr?)");
  }
#endif
}
