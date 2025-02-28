//===-- SwiftMetadataCache.h ------------------------------------*- C++ -*-===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_TypeRefCacher_h_
#define liblldb_TypeRefCacher_h_

#include <mutex>

#include "lldb/Core/DataFileCache.h"
#include "lldb/Core/Module.h"
#include "lldb/Host/SafeMachO.h"

#include "llvm/Support/DJB.h"
#include "llvm/Support/OnDiskHashTable.h"

#include "swift/Remote/ExternalTypeRefCache.h"

namespace lldb_private {

/// An info object to support serializing/deserializing on disk hash tables.
/// Check the OnDiskChainedHashTableGenerator and OnDiskChainedHashTable
/// comments for more info about the interface.
class TypeRefInfo {
public:
  /// The key is the mangled name.
  using key_type = llvm::StringRef;
  using key_type_ref = key_type;
  using internal_key_type = key_type;
  using external_key_type = key_type;
  /// The data is the TypeRefInfoLocator's offset.
  using data_type = uint64_t;
  using data_type_ref = data_type;

  using hash_value_type = uint32_t;
  using offset_type = uint32_t;

  explicit TypeRefInfo() = default;

  // Common encoder decoder functions.

  hash_value_type ComputeHash(key_type_ref key) {
    assert(!key.empty());
    return llvm::djbHash(key);
  }

  static bool EqualKey(internal_key_type lhs, internal_key_type rhs) {
    return lhs == rhs;
  }

  // Encoder functions.

  std::pair<uint32_t, uint32_t> EmitKeyDataLength(llvm::raw_ostream &out,
                                                  key_type_ref key,
                                                  data_type_ref data) {
    assert(key.size() < std::numeric_limits<offset_type>::max() &&
           "Key size is too long!");
    offset_type key_len = key.size();
    // Write the key length so we don't have to traverse it later.
    llvm::support::endian::write<offset_type>(out, key_len,
                                              llvm::endianness::little);
    // Since the data type is always a constant size there's no need to write
    // it.
    offset_type data_len = sizeof(data_type);
    return std::make_pair(key_len, data_len);
  }

  void EmitKey(llvm::raw_ostream &out, key_type_ref key, unsigned len) {
    out << key;
  }

  void EmitData(llvm::raw_ostream &out, key_type_ref key, data_type_ref data,
                unsigned len) {
    llvm::support::endian::write<data_type>(out, data, llvm::endianness::little);
  }

  // Decoder functions.

  internal_key_type GetInternalKey(external_key_type ID) { return ID; }

  external_key_type GetExternalKey(internal_key_type ID) { return ID; }

  static std::pair<offset_type, offset_type>
  ReadKeyDataLength(const unsigned char *&data) {
    offset_type key_len =
        llvm::support::endian::readNext<offset_type, llvm::endianness::little,
                                        llvm::support::unaligned>(data);
    offset_type data_len = sizeof(data_type);
    return std::make_pair(key_len, data_len);
  }

  internal_key_type ReadKey(const unsigned char *data, unsigned length) {
    return llvm::StringRef((const char *)data, length);
  }

  static data_type ReadData(internal_key_type key, const uint8_t *data,
                            unsigned length) {
    data_type result =
        llvm::support::endian::readNext<uint32_t, llvm::endianness::little,
                                        llvm::support::unaligned>(data);
    return result;
  }
};

/// A cache for data used to speed up retrieving swift metadata.
/// Currently we only cache typeref information.
/// This caches files in the directory specified by the SwiftMetadataCachePath
/// setting. The file format is the following:
/// A header consisting of:
/// - The size of the module's UUID.
/// - The module's UUID.
/// - The on hash table's control structure offset.
/// A body consisting of the on disk hash table.
struct SwiftMetadataCache : swift::remote::ExternalTypeRefCache {
public:
  SwiftMetadataCache();

  /// Whether the cache is enabled or disabled. This is controlled by the
  /// enable-swift-metadata-cache setting.
  bool is_enabled();

  /// Relate the lldb module with the reflection info id.
  void registerModuleWithReflectionInfoID(lldb::ModuleSP module,
                                          uint64_t info_id);

  /// Cache the field descriptors in the info with the mangled names
  /// passed in. The number of field descriptors and mangled names passed should
  /// be the same, otherwise caching is aborted.
  void cacheFieldDescriptors(
      uint64_t info_id,
      const swift::reflection::FieldSection &field_descriptors,
      llvm::ArrayRef<std::string> mangled_names) override;

  std::optional<swift::remote::FieldDescriptorLocator>
  getFieldDescriptorLocator(const std::string &mangled_name) override;

  bool isReflectionInfoCached(uint64_t info_id) override;

private:
  /// Generate the on disk hash table data structure into a blob. Returns
  /// the start on the hash table's control structure and the blob itself.
  std::optional<std::pair<uint32_t, llvm::SmallString<32>>>
  generateHashTableBlob(
      uint64_t info_id,
      const swift::reflection::FieldSection &field_descriptors,
      const std::vector<std::string> &mangled_names);

  /// Gets the file name that the cache file should use for a given module.
  std::string getTyperefCacheFileNameForModule(const lldb::ModuleSP &module);

  /// The memory buffers that own the data of the hash tables.
  std::vector<std::unique_ptr<llvm::MemoryBuffer>> m_hash_table_buffers;

  /// A single info object to query all the hash tables with.
  TypeRefInfo m_info;

  struct ModuleCacheInfo {
    lldb::ModuleSP module;
    /// The on disk hash table for this module. The hash tables map mangled
    /// names to field descriptor offsets. A null pointer means that we have
    // no cache info for this module yet.
    std::unique_ptr<llvm::OnDiskChainedHashTable<TypeRefInfo>> cache_hash_table;

    ModuleCacheInfo(lldb::ModuleSP module)
        : module(module), cache_hash_table() {}
  };

  /// A map from reflection infos ids to a pair constituting of its
  /// corresponding module and whether or not we've inserted the cached metadata
  /// for that reflection info already.
  llvm::DenseMap<uint32_t, ModuleCacheInfo> m_reflection_info_to_module;

  std::recursive_mutex m_mutex;

  std::optional<DataFileCache> m_data_file_cache;
};
} // namespace lldb_private
#endif
