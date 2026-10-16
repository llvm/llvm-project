//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_NAMEDTYPE_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_NAMEDTYPE_H

#include <utility>

#include "Identifier.h"
#include "Type.h"

namespace lldb_private {
namespace clike {

/// Represents a type with a name.
///
/// \see ScopedType if you need a type with a name within a namespace/scope.
template <typename Base> class NamedType : public Base {
public:
  /// \p name The name of the type.
  ///
  /// Any remaining arguments are forwarded to \p Base.
  template <typename... Args>
  NamedType(Identifier name, Args &&...args)
      : Base(std::forward<Args>(args)...), m_name(name) {}

  Identifier GetName() const override { return m_name; }
  Identifier GetUnqualifiedName() const override { return m_name; }

private:
  Identifier m_name;
};

} // namespace clike
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_NAMEDTYPE_H
