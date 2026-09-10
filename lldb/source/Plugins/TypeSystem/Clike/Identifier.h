//===-- Identifier.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_IDENTIFIER_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_IDENTIFIER_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"

#include <vector>

namespace lldb_private {
namespace clike_typesystem {

class IdentifierMap;

/// Represents a name in TypeSystemClike.
class Identifier {
public:
  Identifier() = default;

  llvm::StringRef GetName() const { return m_name; }

private:
  // Only IdentifierMap may build a non-empty Identifier.
  friend class IdentifierMap;
  explicit Identifier(llvm::StringRef name) : m_name(name) {}

  llvm::StringRef m_name;
};

/// Turns strings into unique Identifier objects.
class IdentifierMap {
public:
  ~IdentifierMap();

  /// Returns an Identifier for \p name.
  ///
  /// This copies the string into storage owned by this map.
  Identifier get(llvm::StringRef name);

  /// Returns an Identifier for \p name.
  ///
  /// This does not make a copy of the passed string and the string storage
  /// needs to outlive this IdentifierMap. This is used if `name` is a string
  /// literal or backed by ConstString.
  Identifier getWithStaticStorageStr(llvm::StringRef name);

private:
  /// Owns the copies made by get().
  llvm::BumpPtrAllocator m_string_storage;
  /// Set of all created Identifiers strings.
  llvm::DenseSet<llvm::StringRef> m_names;
};

} // namespace clike_typesystem
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_IDENTIFIER_H
