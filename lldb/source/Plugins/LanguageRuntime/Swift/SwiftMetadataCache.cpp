#include "SwiftMetadataCache.h"

#include "lldb/Utility/DataEncoder.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Version/Version.h"
#include "llvm/CodeGen/AccelTable.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/CachePruning.h"
#include "llvm/Support/Compression.h"

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

bool SwiftMetadataCache::is_enabled() {
  return llvm::zlib::isAvailable() && m_data_file_cache.hasValue();
}

void SwiftMetadataCache::registerModuleWithReflectionInfoID(ModuleSP module,
                                                            uint64_t info_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);

  if (!is_enabled())
    return;

  m_info_to_module[info_id] = {module, false};

  // Attempt to load the cached file.
  auto module_name = getTyperefCacheFileNameForModule(module);
  auto mem_buffer_up = m_data_file_cache->GetCachedData(module_name);

  // Nothing cached.
  if (!mem_buffer_up) {
    LLDB_LOGV(log, "[SwiftMetadataCache] No cached file found for module {0}.",
              module->GetFileSpec().GetFilename());
    return;
  }

  // Extractor used to extract the header information (see the .h file for
  // details on the format).
  DataExtractor header_extractor(mem_buffer_up->getBufferStart(),
                                 mem_buffer_up->getBufferSize(),
                                 module->GetObjectFile()->GetByteOrder(),
                                 module->GetObjectFile()->GetAddressByteSize());

  lldb::offset_t read_offset = 0;

  std::string UUID = module->GetUUID().GetAsString();
  std::string cached_UUID = header_extractor.GetCStr(&read_offset);
  // If no uuid in the file something is wrong with the cache.
  if (cached_UUID.empty()) {
    LLDB_LOG(log,
             "[SwiftMetadataCache] Failed to read cached UUID for module {0}.",
             module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  // If the UUIDs don't match this is most likely a stale cache.
  if (cached_UUID != UUID) {
    LLDB_LOGV(log, "[SwiftMetadataCache] Module UUID mismatch for {0}.",
              module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  uint64_t expanded_size = 0;
  if (!header_extractor.GetU64(&read_offset, &expanded_size, 1)) {
    LLDB_LOGV(log,
              "[SwiftMetadataCache] Failed to read decompressed cache size for "
              "module {0}.",
              module->GetFileSpec().GetFilename());
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  const auto *start = (const char *)header_extractor.GetData(&read_offset, 0);
  // Create a reference to the compressed data.
  llvm::StringRef string_buffer(start, (uint64_t)mem_buffer_up->getBufferEnd() -
                                           (uint64_t)start);

  llvm::SmallString<0> decompressed;
  auto error =
      llvm::zlib::uncompress(string_buffer, decompressed, expanded_size);
  if (error) {
    auto error_string = llvm::toString(std::move(error));
    LLDB_LOG(log,
             "[SwiftMetadataCache] Cache decompression failed with error: {0}. "
             "Deleting cached file.",
             error_string);
    m_data_file_cache->RemoveCacheFile(module_name);
    return;
  }

  // Extractor to extract the body of the cached file (see SwiftMetadataCache.h
  // for more details of the format).
  DataExtractor body_extractor(decompressed.data(), decompressed.size(),
                               module->GetObjectFile()->GetByteOrder(),
                               module->GetObjectFile()->GetAddressByteSize());
  read_offset = 0;
  auto num_entries = body_extractor.GetU64(&read_offset);

  // Map to extract the encoded data to. Since extraction can fail we don't want
  // to insert values into the final map in case we have to abort midway.
  llvm::StringMap<swift::remote::FieldDescriptorLocator> temp_map;
  for (size_t i = 0; i < num_entries; i++) {
    const auto *mangled_name = body_extractor.GetCStr(&read_offset);
    if (!mangled_name) {
      LLDB_LOG(log,
               "[SwiftMetadataCache] Failed to read mangled name {0} at offset "
               "{1} for module {2}.",
               i, read_offset, module->GetFileSpec().GetFilename());
      m_data_file_cache->RemoveCacheFile(module_name);
      return;
    }
    uint64_t offset = 0;
    if (!body_extractor.GetU64(&read_offset, &offset, 1)) {
      LLDB_LOG(log,
               "[SwiftMetadataCache] Failed to read mangled name {0} at offset "
               "{1} for module {2}.",
               i, read_offset, module->GetFileSpec().GetFilename());
      m_data_file_cache->RemoveCacheFile(module_name);
      return;
    }
    temp_map[mangled_name] = {info_id, offset};
  }

  // Move the values to the actual map now that we know that it's safe.
  for (auto &p : temp_map)
    m_mangled_name_to_offset.try_emplace(p.getKey(), p.second);

  // Mark this reflection info as processed.
  m_info_to_module[info_id] = {module, true};
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

bool SwiftMetadataCache::writeMangledNamesAndOffsetsToEncoder(
    uint64_t info_id, const swift::reflection::FieldSection &field_descriptors,
    const std::vector<std::string> &mangled_names, DataEncoder &encoder) {
  Log *log = GetLog(LLDBLog::Types);
  auto num_entries = mangled_names.size();
  encoder.AppendU64(num_entries);

  // If the amount of mangled names and field descriptors don't match something
  // unexpected happened.
  if (!areMangledNamesAndFieldSectionSameSize(field_descriptors,
                                              mangled_names)) {
    LLDB_LOG(log, "[SwiftMetadataCache] Mismatch between number of mangled "
                  "names and field descriptors passed in.");
    return false;
  }

  for (auto pair : llvm::zip(field_descriptors, mangled_names)) {
    auto field_descriptor = std::get<0>(pair);
    auto &mangled_name = std::get<1>(pair);
    if (mangled_name.empty())
      continue;
    auto offset = field_descriptor.getAddressData() -
                  field_descriptors.startAddress().getAddressData();
    encoder.AppendCString(mangled_name.data());
    encoder.AppendU64(offset);
  }
  return true;
}

void SwiftMetadataCache::cacheFieldDescriptors(
    uint64_t info_id, const swift::reflection::FieldSection &field_descriptors,
    llvm::ArrayRef<std::string> mangled_names) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);

  if (!is_enabled())
    return;

  auto it = m_info_to_module.find(info_id);
  if (it == m_info_to_module.end()) {
    LLDB_LOGV(log, "[SwiftMetadataCache] No module found with module id {0}.",
              info_id);
    return;
  }

  auto module = std::get<ModuleSP>(it->second);
  // Write the data to the body encoder with the format expected by the current
  // cache version.
  DataEncoder body_encoder;
  if (!writeMangledNamesAndOffsetsToEncoder(info_id, field_descriptors,
                                            mangled_names, body_encoder))
    return;

  uint64_t typeref_buffer_size = body_encoder.GetData().size();
  llvm::StringRef typeref_buffer((const char *)body_encoder.GetData().data(),
                                 typeref_buffer_size);

  llvm::SmallString<0> compressed_buffer;
  llvm::zlib::compress(typeref_buffer, compressed_buffer);

  // Write the header followed by the body.
  DataEncoder encoder;
  encoder.AppendCString(module->GetUUID().GetAsString());
  encoder.AppendU64(typeref_buffer_size);
  encoder.AppendData(compressed_buffer);

  auto filename = getTyperefCacheFileNameForModule(module);

  m_data_file_cache->SetCachedData(filename, encoder.GetData());
  LLDB_LOGV(log, "[SwiftMetadataCache] Cache file written for module {0}.",
            module->GetFileSpec().GetFilename());
}

llvm::Optional<swift::remote::FieldDescriptorLocator>
SwiftMetadataCache::getFieldDescriptorLocator(const std::string &Name) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  Log *log = GetLog(LLDBLog::Types);
  auto it = m_mangled_name_to_offset.find(Name);
  if (it != m_mangled_name_to_offset.end()) {
    LLDB_LOGV(
        log,
        "[SwiftMetadataCache] Returning field descriptor for mangled name {0}",
        Name);
    return it->second;
  }
  return {};
}

bool SwiftMetadataCache::isReflectionInfoCached(uint64_t info_id) {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  auto it = m_info_to_module.find(info_id);
  // First check if we've registered the reflection info with that id.
  if (it != m_info_to_module.end())
    // Then check whether we've already parsed it or not.
    return std::get<bool>(it->second);
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
