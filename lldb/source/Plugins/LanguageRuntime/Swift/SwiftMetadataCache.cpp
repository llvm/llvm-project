//===-- SwiftMetadataCache.cpp --------------------------------------------===//
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

#include "SwiftMetadataCache.h"

#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Version/Version.h"
#include "swift/RemoteInspection/ReflectionContext.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CachePruning.h"

using namespace lldb;
using namespace lldb_private;
using namespace swift::reflection;
using namespace swift::remote;

SwiftMetadataCache::SwiftMetadataCache() {
  if (ModuleList::GetGlobalModuleListProperties()
          .GetEnableSwiftMetadataCache()) {
    llvm::CachePruningPolicy policy;
    ModuleListProperties &properties =
        ModuleList::GetGlobalModuleListProperties();
    policy.Interval = std::chrono::hours(1);
    policy.MaxSizeBytes = properties.GetSwiftMetadataCacheMaxByteSize();
    policy.Expiration = std::chrono::hours(
        properties.GetSwiftMetadataCacheExpirationDays() * 24);
    m_data_file_cache.emplace(ModuleList::GetGlobalModuleListProperties()
                                  .GetSwiftMetadataCachePath()
                                  .GetPath(),
                              policy);
  }
}

bool SwiftMetadataCache::is_enabled() { return m_data_file_cache.has_value(); }

void SwiftMetadataCache::registerModuleWithReflectionInfoID(ModuleSP module,
                                                            uint64_t info_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);

  if (!is_enabled())
    return;

  /// Insert the module cache info as not processed.
  m_reflection_info_to_module.insert({info_id, module});

  // Attempt to load the cached file.
  auto module_name = getTyperefCacheFileNameForModule(module);
  auto mem_buffer_up = m_data_file_cache->GetCachedData(module_name);

  // Nothing cached.
  if (!mem_buffer_up) {
    LLDB_LOGV(log, "[SwiftMetadataCache] No cached file found for module {0}.",
              module->GetFileSpec().GetFilename());
    return;
  }

  // Move it to the instance variable so references to this data don't go
  // out of scope.
  m_hash_table_buffers.emplace_back(std::move(mem_buffer_up));
  auto &mem_buffer = m_hash_table_buffers.back();

  // Extractor used to extract the header information (see the .h file for
  // details on the format).
  DataExtractor header_extractor(mem_buffer->getBufferStart(),
                                 mem_buffer->getBufferSize(),
                                 module->GetObjectFile()->GetByteOrder(),
                                 module->GetObjectFile()->GetAddressByteSize());

  lldb::offset_t read_offset = 0;
  uint16_t cached_UUID_size = 0;
  if (!header_extractor.GetU16(&read_offset, &cached_UUID_size, 1)) {
    LLDB_LOG(log,
             "[SwiftMetadataCache] Failed to read cached UUID size for module {0}.",
             module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  const auto *cached_UUID_data = reinterpret_cast<const uint8_t *>(
      header_extractor.GetData(&read_offset, cached_UUID_size));

  llvm::ArrayRef<uint8_t> cached_UUID(cached_UUID_data, cached_UUID_size);
  // If no uuid in the file something is wrong with the cache.
  if (cached_UUID.empty()) {
    LLDB_LOG(log,
             "[SwiftMetadataCache] Failed to read cached UUID for module {0}.",
             module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  auto UUID = module->GetUUID().GetBytes();
  // If the UUIDs don't match this is most likely a stale cache.
  if (cached_UUID != UUID) {
    LLDB_LOGV(log, "[SwiftMetadataCache] Module UUID mismatch for {0}.",
              module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  // The on disk hash table must have a 4-byte alignment, skip
  // the padding when reading.
  read_offset = llvm::alignTo(read_offset, 4);

  // The offset of the hash table control structure, which follows the payload.
  uint32_t table_control_offset = 0;
  if (!header_extractor.GetU32(&read_offset, &table_control_offset, 1)) {
    LLDB_LOGV(log,
              "[SwiftMetadataCache] Failed to read table offset for "
              "module {0}.",
              module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  const auto *table_contents = reinterpret_cast<const uint8_t *>(
      header_extractor.GetData(&read_offset, 0));

  const auto *table_control = table_contents + table_control_offset;

  // Store the hash table.
  m_reflection_info_to_module.find(info_id)->second.cache_hash_table.reset(
      llvm::OnDiskChainedHashTable<TypeRefInfo>::Create(
          table_control, table_contents, m_info));

  LLDB_LOGV(log, "[SwiftMetadataCache] Loaded cache for module {0}.",
            module->GetFileSpec().GetFilename());
}

static bool areMangledNamesAndFieldSectionSameSize(
    const swift::reflection::FieldSection &field_descriptors,
    const std::vector<std::string> &mangled_names) {
  // FieldSection is not random access, so we have to iterate over it in it's
  // entirety to find out it's true size
  uint64_t field_descriptors_size =
      std::distance(field_descriptors.begin(), field_descriptors.end());

  return field_descriptors_size == mangled_names.size();
}

std::optional<std::pair<uint32_t, llvm::SmallString<32>>>
SwiftMetadataCache::generateHashTableBlob(
    uint64_t info_id, const swift::reflection::FieldSection &field_descriptors,
    const std::vector<std::string> &mangled_names) {
  Log *log = GetLog(LLDBLog::Types);
  llvm::SmallString<32> hash_table_blob;
  llvm::raw_svector_ostream blobStream(hash_table_blob);

  // If the amount of mangled names and field descriptors don't match something
  // unexpected happened.
  if (!areMangledNamesAndFieldSectionSameSize(field_descriptors,
                                              mangled_names)) {
    LLDB_LOG(log, "[SwiftMetadataCache] Mismatch between number of mangled "
                  "names and field descriptors passed in.");
    return {};
  }

  llvm::OnDiskChainedHashTableGenerator<TypeRefInfo> table_generator;
  for (auto pair : llvm::zip(field_descriptors, mangled_names)) {
    auto field_descriptor = std::get<0>(pair);
    auto &mangled_name = std::get<1>(pair);
    if (mangled_name.empty())
      continue;
    auto offset = field_descriptor.getAddressData() -
                  field_descriptors.startAddress().getAddressData();
    table_generator.insert(mangled_name, offset, m_info);
  }

  // Make sure that no bucket is at offset 0.
  llvm::support::endian::write<uint32_t>(blobStream, 0, llvm::endianness::little);
  uint32_t table_control_offset = table_generator.Emit(blobStream, m_info);
  return {{std::move(table_control_offset), std::move(hash_table_blob)}};
}

void SwiftMetadataCache::cacheFieldDescriptors(
    uint64_t info_id, const swift::reflection::FieldSection &field_descriptors,
    llvm::ArrayRef<std::string> mangled_names) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);

  if (!is_enabled())
    return;

  auto it = m_reflection_info_to_module.find(info_id);
  if (it == m_reflection_info_to_module.end()) {
    LLDB_LOGV(log, "[SwiftMetadataCache] No module found with module id {0}.",
              info_id);
    return;
  }

  auto &module = it->second.module;

  auto maybe_pair =
      generateHashTableBlob(info_id, field_descriptors, mangled_names);
  if (!maybe_pair)
    return;

  auto &table_offset = maybe_pair->first;
  auto &hash_table_blob = maybe_pair->second;

  // Write the header followed by the body.
  DataEncoder encoder;
  auto uuid = module->GetUUID().GetBytes();
  // Append the uuid size followed by the uuid itself.
  encoder.AppendU16(uuid.size());
  encoder.AppendData(uuid);


  auto size_so_far = encoder.GetByteSize();
  // The on disk hash table must have a 4-byte alignment, so
  // write 0 bytes until we get to the required alignemnt.
  auto padding = llvm::alignTo(size_so_far, 4) - size_so_far;
  while (padding-- > 0)
    encoder.AppendU8(0);

  encoder.AppendU32(table_offset);
  encoder.AppendData(hash_table_blob);

  auto filename = getTyperefCacheFileNameForModule(module);

  m_data_file_cache->SetCachedData(filename, encoder.GetData());
  LLDB_LOGV(log, "[SwiftMetadataCache] Cache file written for module {0}.",
            module->GetFileSpec().GetFilename());
}

std::optional<swift::remote::FieldDescriptorLocator>
SwiftMetadataCache::getFieldDescriptorLocator(const std::string &Name) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);
  // Compute hash outside of loop as an optimization.
  auto hash = m_info.ComputeHash(Name);
  for (auto &pair : m_reflection_info_to_module) {
    auto &cache_hash_table = pair.second.cache_hash_table;
    // No cache for this reflection module.
    if (!cache_hash_table)
      continue;
    auto it = cache_hash_table->find_hashed(Name, hash, &m_info);
    if (it != cache_hash_table->end()) {
      LLDB_LOGV(log,
                "[SwiftMetadataCache] Returning field descriptor for mangled "
                "name {0}",
                Name);
      auto info_id = pair.first;
      return {{info_id, *it}};
    }
  }
  return {};
}

bool SwiftMetadataCache::isReflectionInfoCached(uint64_t info_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  auto it = m_reflection_info_to_module.find(info_id);
  // First check if we've registered the reflection info with that id.
  if (it != m_reflection_info_to_module.end())
    // Then check whether we have a cache for it or not.
    return it->second.cache_hash_table.get() != nullptr;
  return false;
}

std::string SwiftMetadataCache::getTyperefCacheFileNameForModule(
    const lldb::ModuleSP &module) {
  // We hash the lldb string version (so we don't run into the risk of two lldbs
  // invalidating each other's cache), and the modules path (so we clean up
  // stale caches when the module changes) as the typeref cache file name.
  llvm::BLAKE3 blake3;
  const char *version = lldb_private::GetVersion();
  blake3.update(version);
  blake3.update(module->GetFileSpec().GetPath());
  auto hashed_result = llvm::toHex(blake3.final());
  return "typeref-" + hashed_result;
}
