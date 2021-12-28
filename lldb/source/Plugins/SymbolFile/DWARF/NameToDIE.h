//===-- NameToDIE.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H

#include <functional>

#include "DIERef.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Core/dwarf.h"
#include "lldb/lldb-defines.h"

class DWARFUnit;

class NameToDIE {
public:
  NameToDIE() : m_map() {}

  ~NameToDIE() = default;

  void Dump(lldb_private::Stream *s);

  void Insert(lldb_private::ConstString name, const DIERef &die_ref);

  void Append(const NameToDIE &other);

  void Finalize();

  bool Find(lldb_private::ConstString name,
            llvm::function_ref<bool(DIERef ref)> callback) const;

  bool Find(const lldb_private::RegularExpression &regex,
            llvm::function_ref<bool(DIERef ref)> callback) const;

  /// \a unit must be the skeleton unit if possible, not GetNonSkeletonUnit().
  void
  FindAllEntriesForUnit(DWARFUnit &unit,
                        llvm::function_ref<bool(DIERef ref)> callback) const;

  void
  ForEach(std::function<bool(lldb_private::ConstString name,
                             const DIERef &die_ref)> const
              &callback) const;

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
  bool Decode(const lldb_private::DataExtractor &data,
              lldb::offset_t *offset_ptr,
              const lldb_private::StringTableReader &strtab);

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
  void Encode(lldb_private::DataEncoder &encoder,
              lldb_private::ConstStringTable &strtab) const;

  /// Used for unit testing the encoding and decoding.
  bool operator==(const NameToDIE &rhs) const;

  bool IsEmpty() const { return m_map.IsEmpty(); }

  void Clear() { m_map.Clear(); }

protected:
  lldb_private::UniqueCStringMap<DIERef> m_map;
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_NAMETODIE_H
