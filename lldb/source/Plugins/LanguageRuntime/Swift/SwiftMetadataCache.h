
#ifndef liblldb_TypeRefCacher_h_
#define liblldb_TypeRefCacher_h_

#include <mutex>
#include <unordered_map>

#include "lldb/Core/DataFileCache.h"
#include "lldb/Core/Module.h"

#include "llvm/ADT/STLExtras.h"

#include "swift/Reflection/ReflectionContext.h"
#include "swift/Remote/ExternalTypeRefCache.h"

namespace lldb_private {

/// A cache for data used to speed up retrieving swift metadata.
/// Currently we only cache typeref information.
/// This caches files in the directory specified by the SwiftMetadataCachePath
/// setting. The file format is the following: A header consisting of:
/// - Version number (uint16_t).
/// - Module signature.
/// - Size of the remainder of the contents when decompressed (uint64_t).
/// A body that is zlib compressed containing:
/// - The number (N) of name/offset pairs (uint64_t).
/// - N pairs composed of a c-string mangled name followed by the offset
///   (uint32_t) where the corresponding field descriptor can be found in
//    the fieldmd section.
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

  llvm::Optional<swift::remote::FieldDescriptorLocator>
  getFieldDescriptorLocator(const std::string &mangled_name) override;

  bool isReflectionInfoCached(uint64_t info_id) override;

private:
  bool writeMangledNamesAndOffsetsToEncoder(
      uint64_t info_id,
      const swift::reflection::FieldSection &field_descriptors,
      const std::vector<std::string> &mangled_names, DataEncoder &encoder);

  void
  fillMangledNameToOffsetMap(uint64_t info_id,
                             const swift::reflection::ReflectionInfo &info,
                             const std::vector<std::string> &mangled_names);

  /// Gets the file name that the cache file should use for a given module.
  std::string getTyperefCacheFileNameForModule(const lldb::ModuleSP &module);

  /// A map from mangled names to field descriptor locators.
  llvm::StringMap<swift::remote::FieldDescriptorLocator>
      m_mangled_name_to_offset;

  /// A map from reflection infos ids to a pair constituting of its
  /// corresponding module and whether or not we've inserted the cached metadata
  /// for that reflection info already.
  llvm::DenseMap<uint32_t, std::pair<lldb::ModuleSP, bool>> m_info_to_module;

  std::recursive_mutex m_mutex;

  llvm::Optional<DataFileCache> m_data_file_cache;
};
} // namespace lldb_private
#endif
