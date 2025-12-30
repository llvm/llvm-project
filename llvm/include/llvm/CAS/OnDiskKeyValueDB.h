//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This declares OnDiskKeyValueDB, a key value storage database of fixed size
/// key and value.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_ONDISKKEYVALUEDB_H
#define LLVM_CAS_ONDISKKEYVALUEDB_H

#include "llvm/CAS/OnDiskTrieRawHashMap.h"

namespace llvm::cas::ondisk {

class UnifiedOnDiskCache;

/// An on-disk key-value data store with the following properties:
/// * Keys are fixed length binary hashes with expected normal distribution.
/// * Values are buffers of the same size, specified at creation time.
/// * The value of a key cannot be changed once it is set.
/// * The value buffers returned from a key lookup have 8-byte alignment.
class OnDiskKeyValueDB {
public:
  /// Associate a value with a key.
  ///
  /// \param Key the hash bytes for the key
  /// \param Value the value bytes, same size as \p ValueSize parameter of
  /// \p open call.
  ///
  /// \returns the value associated with the \p Key. It may be different than
  /// \p Value if another value is already associated with this key.
  LLVM_ABI_FOR_TEST Expected<ArrayRef<char>> put(ArrayRef<uint8_t> Key,
                                                 ArrayRef<char> Value);

  /// \returns the value associated with the \p Key, or \p std::nullopt if the
  /// key does not exist.
  LLVM_ABI_FOR_TEST Expected<std::optional<ArrayRef<char>>>
  get(ArrayRef<uint8_t> Key);

  /// \returns Total size of stored data.
  size_t getStorageSize() const { return Cache.size(); }

  /// \returns The precentage of space utilization of hard space limits.
  ///
  /// Return value is an integer between 0 and 100 for percentage.
  unsigned getHardStorageLimitUtilization() const {
    return Cache.size() * 100ULL / Cache.capacity();
  }

  /// Open the on-disk store from a directory.
  ///
  /// \param Path directory for the on-disk store. The directory will be created
  /// if it doesn't exist.
  /// \param HashName Identifier name for the hashing algorithm that is going to
  /// be used.
  /// \param KeySize Size for the key hash bytes.
  /// \param ValueName Identifier name for the values.
  /// \param ValueSize Size for the value bytes.
  /// \param UnifiedCache An optional UnifiedOnDiskCache that manages the size
  /// and lifetime of the CAS instance and it must owns current initializing
  /// KeyValueDB after initialized.
  LLVM_ABI_FOR_TEST static Expected<std::unique_ptr<OnDiskKeyValueDB>>
  open(StringRef Path, StringRef HashName, unsigned KeySize,
       StringRef ValueName, size_t ValueSize,
       UnifiedOnDiskCache *UnifiedCache = nullptr);

  using CheckValueT =
      function_ref<Error(FileOffset Offset, ArrayRef<char> Data)>;
  /// Validate the storage with a callback \p CheckValue to check the stored
  /// value.
  LLVM_ABI_FOR_TEST Error validate(CheckValueT CheckValue) const;

private:
  OnDiskKeyValueDB(size_t ValueSize, OnDiskTrieRawHashMap Cache,
                   UnifiedOnDiskCache *UnifiedCache)
      : ValueSize(ValueSize), Cache(std::move(Cache)),
        UnifiedCache(UnifiedCache) {}

  const size_t ValueSize;
  OnDiskTrieRawHashMap Cache;
  UnifiedOnDiskCache *UnifiedCache = nullptr;
};

} // namespace llvm::cas::ondisk

#endif // LLVM_CAS_ONDISKKEYVALUEDB_H
