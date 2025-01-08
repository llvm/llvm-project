//===-- NameToDIE.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H

#include "DIERef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DJB.h"
#include <cstddef>
#include <tuple>
#include <vector>

namespace lldb_private::plugin {
namespace dwarf {
class DWARFUnit;

class NameToDIEX {
public:
  using Tuple = std::tuple<uint32_t, uint32_t, const char *, lldb::user_id_t>;

  void Insert(const char *name, DIERef die_ref) {
    m_map.emplace_back(0, 0, name, die_ref.get_id());
  }

  void Finalize() {
    llvm::sort(m_map, [](const auto &lhs, const auto &rhs) {
      return std::get<2>(lhs) < std::get<2>(rhs);
    });
    const char *prev_name = nullptr;
    uint32_t prev_len = 0;
    uint32_t prev_hash = 0;
    for (auto &[hash, len, name, ref] : m_map) {
      if (name != prev_name) {
        prev_name = name;
        prev_len = strlen(prev_name);
        prev_hash = llvm::djbHash(llvm::StringRef(prev_name, prev_len));
      }
      hash = prev_hash;
      len = prev_len;
    }
    llvm::sort(m_map, llvm::less_first());
    for (uint32_t i = 0; i < m_starts.size(); ++i) {
      m_starts[i] =
          llvm::lower_bound(m_map,
                            std::make_tuple(i << 30u, UINT32_C(0),
                                            (const char *)nullptr, UINT64_C(0)),
                            llvm::less_first());
    }
  }

  std::vector<Tuple> m_map;
  std::array<std::vector<Tuple>::const_iterator, 4> m_starts;
};

class NameToDIE {
public:
  NameToDIE() : m_map() {}

  ~NameToDIE() = default;

  void Dump(Stream *s);

  void Append(const NameToDIEX &other, unsigned slice);
  void Reserve(size_t count, unsigned slice) { m_map[slice].reserve(count); }

  bool Find(llvm::StringRef name,
            llvm::function_ref<bool(DIERef ref)> callback) const;

  bool Find(const RegularExpression &regex,
            llvm::function_ref<bool(DIERef ref)> callback) const;

  /// \a unit must be the skeleton unit if possible, not GetNonSkeletonUnit().
  void
  FindAllEntriesForUnit(DWARFUnit &unit,
                        llvm::function_ref<bool(DIERef ref)> callback) const;

  void
  ForEach(llvm::function_ref<bool(llvm::StringRef name, const DIERef &die_ref)>
              callback) const;

  /// Decode a serialized version of this object from data.
  ///
  /// \param data
  ///   The decoder object that references the serialized data.
  ///
  /// \param offset_ptr
  ///   A pointer that contains the offset from which the data will be decoded
  ///   from that gets updated as data gets decoded.
  ///
  /// \param strtab
  ///   All strings in cache files are put into string tables for efficiency
  ///   and cache file size reduction. Strings are stored as uint32_t string
  ///   table offsets in the cache data.
  bool Decode(const DataExtractor &data, lldb::offset_t *offset_ptr,
              const StringTableReader &strtab);

  /// Encode this object into a data encoder object.
  ///
  /// This allows this object to be serialized to disk.
  ///
  /// \param encoder
  ///   A data encoder object that serialized bytes will be encoded into.
  ///
  /// \param strtab
  ///   All strings in cache files are put into string tables for efficiency
  ///   and cache file size reduction. Strings are stored as uint32_t string
  ///   table offsets in the cache data.
  void Encode(DataEncoder &encoder, ConstStringTable &strtab) const;

  /// Used for unit testing the encoding and decoding.
  bool operator==(const NameToDIE &rhs) const;

  bool IsEmpty() const {
    return llvm::all_of(m_map, [](const auto &map) { return map.empty(); });
  }

  void Clear() {
    for (auto &map : m_map)
      map.clear();
  }

  void Finalize(unsigned slice) {
    auto &map = m_map[slice];
    llvm::sort(map, [](const auto &lhs, const auto &rhs) {
      return std::tie(std::get<0>(lhs), std::get<3>(lhs)) <
             std::tie(std::get<0>(rhs), std::get<3>(rhs));
    });
  }

protected:
  std::array<std::vector<NameToDIEX::Tuple>, 4> m_map;
};
} // namespace dwarf
} // namespace lldb_private::plugin

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H
