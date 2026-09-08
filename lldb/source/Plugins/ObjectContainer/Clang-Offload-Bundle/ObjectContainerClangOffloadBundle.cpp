//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerClangOffloadBundle.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/MemoryBuffer.h"

#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectContainerClangOffloadBundle)

void ObjectContainerClangOffloadBundle::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      GetModuleSpecifications, /*create_memory_callback=*/nullptr);
}

void ObjectContainerClangOffloadBundle::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectContainerClangOffloadBundle::ObjectContainerClangOffloadBundle(
    const ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t length)
    : ObjectContainer(module_sp, file, file_offset, length, data_sp,
                      data_offset) {}

ObjectContainerClangOffloadBundle::~ObjectContainerClangOffloadBundle() =
    default;

bool ObjectContainerClangOffloadBundle::MagicBytesMatch(
    const DataExtractor &data) {
  llvm::StringRef bytes(reinterpret_cast<const char *>(data.GetDataStart()),
                        data.GetByteSize());
  switch (llvm::identify_magic(bytes)) {
  case llvm::file_magic::elf:
  case llvm::file_magic::elf_relocatable:
  case llvm::file_magic::elf_executable:
  case llvm::file_magic::elf_shared_object:
  case llvm::file_magic::elf_core:
    return true;
  default:
    return false;
  }
}

static ArchSpec ParseArchFromBundleEntryID(llvm::StringRef id) {
  // Bundle entry IDs use the format: <offload-kind>-<target-triple>
  // e.g. "hip-amdgpu-amd-amdhsa--gfx906", "host-x86_64-unknown-linux-gnu"
  llvm::StringRef triple = id.split('-').second;
  if (triple.empty())
    return {};
  return ArchSpec(triple);
}

bool ObjectContainerClangOffloadBundle::FindBundleEntries(
    const FileSpec &file, DataExtractorSP extractor_sp,
    lldb::offset_t file_offset, lldb::offset_t file_size,
    std::vector<Entry> &entries) {
  using CacheKey = std::tuple<std::string, lldb::offset_t, uint64_t>;
  struct CacheValue {
    llvm::sys::TimePoint<> mod_time;
    std::vector<Entry> entries;
  };
  static std::mutex cache_mutex;
  static std::map<CacheKey, CacheValue> cache;

  const std::string path = file.GetPath();
  CacheKey cache_key(path, file_offset, file_size);
  llvm::sys::TimePoint<> mod_time;
  // Prefer and cache on-disk data when the FileSpec names a real file.
  bool cacheable = FileSystem::Instance().Exists(file);
  if (cacheable) {
    mod_time = FileSystem::Instance().GetModificationTime(file);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(cache_key);
    if (it != cache.end() && it->second.mod_time == mod_time) {
      entries = it->second.entries;
      return !entries.empty();
    }
  }

  const uint8_t *data = nullptr;
  size_t data_size = 0;
  DataBufferSP mapped_data_sp;
  if (cacheable) {
    mapped_data_sp =
        FileSystem::Instance().CreateDataBuffer(file, file_size, file_offset);
    if (mapped_data_sp) {
      data = mapped_data_sp->GetBytes();
      data_size = mapped_data_sp->GetByteSize();
    } else {
      cacheable = false;
    }
  }
  if (!data && extractor_sp && extractor_sp->HasData() &&
      extractor_sp->GetByteSize() >= file_size) {
    data = extractor_sp->GetDataStart();
    data_size = file_size ? file_size : extractor_sp->GetByteSize();
  }
  if (!data)
    return false;

  llvm::StringRef bytes(reinterpret_cast<const char *>(data), data_size);
  llvm::MemoryBufferRef buffer(bytes, path);
  auto parse = [&]() -> std::vector<Entry> {
    std::vector<Entry> result;
    auto object_or_error = llvm::object::ObjectFile::createObjectFile(buffer);
    if (!object_or_error) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Object), object_or_error.takeError(),
                     "unable to parse clang offload bundle container: {0}");
      return result;
    }

    llvm::SmallVector<llvm::object::OffloadBundleFatBin> bundles;
    if (llvm::Error error = llvm::object::extractOffloadBundleFatBinary(
            **object_or_error, bundles)) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(error),
                     "unable to extract clang offload bundle: {0}");
      return result;
    }

    for (llvm::object::OffloadBundleFatBin &bundle : bundles) {
      for (const llvm::object::OffloadBundleEntry &bundle_entry :
           bundle.getEntries()) {
        if (bundle_entry.Size == 0)
          continue;

        DataExtractorSP entry_extractor_sp;
        uint64_t entry_offset = 0;
        if (bundle.isDecompressed()) {
          if (!bundle.DecompressedBuffer)
            continue;

          llvm::StringRef decompressed = bundle.DecompressedBuffer->getBuffer();
          if (bundle_entry.Offset > decompressed.size() ||
              bundle_entry.Size > decompressed.size() - bundle_entry.Offset)
            continue;

          auto entry_data_sp = std::make_shared<DataBufferHeap>(
              decompressed.data() + bundle_entry.Offset, bundle_entry.Size);
          entry_extractor_sp = std::make_shared<DataExtractor>(entry_data_sp);
        } else {
          entry_offset = file_offset + bundle_entry.Offset;
        }

        ArchSpec arch = ParseArchFromBundleEntryID(bundle_entry.ID);
        if (!arch.IsValid())
          continue;

        result.push_back({std::move(arch), entry_offset, bundle_entry.Size,
                          std::move(entry_extractor_sp)});
      }
    }
    return result;
  };

  std::vector<Entry> parsed = parse();

  if (cacheable) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[std::move(cache_key)] = CacheValue{mod_time, parsed};
  }

  entries = std::move(parsed);
  return !entries.empty();
}

ObjectContainer *ObjectContainerClangOffloadBundle::CreateInstance(
    const ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t length) {
  if (!data_sp || !file)
    return nullptr;

  DataExtractor data;
  data.SetData(data_sp, data_offset, length);
  if (!MagicBytesMatch(data))
    return nullptr;

  auto container_up = std::make_unique<ObjectContainerClangOffloadBundle>(
      module_sp, data_sp, data_offset, file, file_offset, length);
  if (!container_up->ParseHeader())
    return nullptr;
  return container_up.release();
}

bool ObjectContainerClangOffloadBundle::ParseHeader() {
  m_entries.clear();
  return FindBundleEntries(m_file, m_extractor_sp, m_offset, m_length,
                           m_entries);
}

size_t ObjectContainerClangOffloadBundle::GetNumArchitectures() const {
  return m_entries.size();
}

bool ObjectContainerClangOffloadBundle::GetArchitectureAtIndex(
    uint32_t idx, ArchSpec &arch) const {
  if (idx >= m_entries.size())
    return false;
  arch = m_entries[idx].arch;
  return true;
}

ModuleSpecList ObjectContainerClangOffloadBundle::GetModuleSpecifications(
    const FileSpec &file, DataExtractorSP &extractor_sp,
    lldb::offset_t file_offset, lldb::offset_t file_size) {
  if (!extractor_sp || !MagicBytesMatch(*extractor_sp))
    return {};

  std::vector<Entry> entries;
  if (!FindBundleEntries(file, extractor_sp, file_offset, file_size, entries))
    return {};

  ModuleSpecList specs;
  for (const Entry &entry : entries) {
    ModuleSpec spec = entry.extractor_sp
                          ? ModuleSpec(file, UUID(), entry.extractor_sp)
                          : ModuleSpec(file, entry.arch);
    if (entry.extractor_sp)
      spec.GetArchitecture() = entry.arch;
    spec.SetObjectOffset(entry.offset);
    spec.SetObjectSize(entry.size);
    specs.Append(spec);
  }
  return specs;
}

ObjectFileSP
ObjectContainerClangOffloadBundle::GetObjectFile(const FileSpec *file) {
  ModuleSP module_sp = GetModule();
  if (!module_sp)
    return {};

  ArchSpec arch = module_sp->GetArchitecture();
  if (!arch.IsValid()) {
    arch = Target::GetDefaultArchitecture();
    if (!arch.IsValid())
      arch.SetTriple(LLDB_ARCH_DEFAULT);
  }

  for (int pass = 0; pass < 2; ++pass) {
    for (const Entry &entry : m_entries) {
      bool match = (pass == 0) ? arch.IsExactMatch(entry.arch)
                               : arch.IsCompatibleMatch(entry.arch);
      if (match) {
        DataExtractorSP extractor_sp = entry.extractor_sp;
        lldb::offset_t data_offset = 0;
        return ObjectFile::FindPlugin(module_sp, file, entry.offset, entry.size,
                                      extractor_sp, data_offset);
      }
    }
  }

  return {};
}
