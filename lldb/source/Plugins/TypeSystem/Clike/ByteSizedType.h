//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_BYTESIZEDTYPE_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_BYTESIZEDTYPE_H

#include <cstdint>
#include <optional>
#include <utility>

namespace lldb_private {
namespace clike {

/// Represents type with a fixed byte size.
///
/// This is only used for types that do not derive their size from another type.
template <typename Base> class ByteSizedType : public Base {
public:
  /// \p byte_size is std::nullopt if unknown.
  ///
  /// Any remaining arguments are forwarded to \p Base.
  template <typename... Args>
  explicit ByteSizedType(std::optional<uint64_t> byte_size, Args &&...args)
      : Base(std::forward<Args>(args)...), m_byte_size(byte_size) {}

  std::optional<uint64_t> GetByteSize() const override { return m_byte_size; }

protected:
  /// Used by records, whose size might only be known after completion.
  void SetByteSize(std::optional<uint64_t> byte_size) {
    m_byte_size = byte_size;
  }

private:
  // TODO: We could maybe save some memory here by using a sentinel value.
  std::optional<uint64_t> m_byte_size;
};

} // namespace clike
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_CLIKE_BYTESIZEDTYPE_H
