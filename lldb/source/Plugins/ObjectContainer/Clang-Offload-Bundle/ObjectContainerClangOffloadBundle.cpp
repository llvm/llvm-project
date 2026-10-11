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
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/Chrono.h"
#include <mutex>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectContainerClangOffloadBundle)

void ObjectContainerClangOffloadBundle::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                GetModuleSpecifications, nullptr);
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
  llvm::file_magic magic = llvm::identify_magic(bytes);
  switch (magic) {
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

static ArchSpec ParseArchFromBundleEntryID(llvm::StringRef ID) {
  // Bundle entry IDs use the format: <offload-kind>-<target-triple>
  // e.g. "hip-amdgcn-amd-amdhsa--gfx906", "host-x86_64-unknown-linux-gnu"
  auto [Kind, Triple] = ID.split('-');
  if (Triple.empty())
    return ArchSpec();
  return ArchSpec(Triple);
}

bool ObjectContainerClangOffloadBundle::FindBundleEntries(
    const FileSpec &file, std::vector<Entry> &entries) {
  std::string path = file.GetPath();
  if (path.empty())
    return false;

  // Cache the parse per file so a bundle with many code objects isn't rescanned
  // once per object. Keyed by path and validated by mtime, so a changed file
  // re-parses and overwrites. Locked for concurrent module loads.
  struct CacheValue {
    llvm::sys::TimePoint<> mod_time;
    std::vector<Entry> entries;
  };
  static std::mutex cache_mutex;
  static llvm::StringMap<CacheValue> cache;

  llvm::sys::TimePoint<> mod_time =
      FileSystem::Instance().GetModificationTime(file);

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(path);
    if (it != cache.end() && it->second.mod_time == mod_time) {
      entries = it->second.entries;
      return !entries.empty();
    }
  }

  // Parse the offload bundle (helper keeps this separate from the caching).
  auto parse = [&path]() -> std::vector<Entry> {
    std::vector<Entry> result;
    auto obj_or_err = llvm::object::ObjectFile::createObjectFile(path);
    if (!obj_or_err) {
      llvm::consumeError(obj_or_err.takeError());
      return result;
    }

    llvm::SmallVector<llvm::object::OffloadBundleFatBin> bundles;
    if (auto err = llvm::object::extractOffloadBundleFatBinary(
            *obj_or_err->getBinary(), bundles)) {
      llvm::consumeError(std::move(err));
      return result;
    }

    for (auto &bundle : bundles) {
      for (auto &bundle_entry : bundle.getEntries()) {
        if (bundle_entry.Size == 0)
          continue;

        DataExtractorSP entry_extractor_sp;
        uint64_t offset = bundle_entry.Offset;
        if (bundle.isDecompressed()) {
          if (!bundle.DecompressedBuffer)
            continue;

          llvm::StringRef decompressed = bundle.DecompressedBuffer->getBuffer();
          if (offset > decompressed.size() ||
              bundle_entry.Size > decompressed.size() - offset)
            continue;

          auto entry_data_sp = std::make_shared<DataBufferHeap>(
              decompressed.data() + offset, bundle_entry.Size);
          entry_extractor_sp = std::make_shared<DataExtractor>(entry_data_sp);
          // Compressed entries have no corresponding offset in the containing
          // file. The owned buffer above contains only the selected object.
          offset = 0;
        }

        Entry entry;
        entry.arch = ParseArchFromBundleEntryID(bundle_entry.ID);
        entry.offset = offset;
        entry.size = bundle_entry.Size;
        entry.extractor_sp = std::move(entry_extractor_sp);
        if (entry.arch.IsValid())
          result.push_back(std::move(entry));
      }
    }
    return result;
  };

  std::vector<Entry> parsed = parse();

  // Store/overwrite this path's entry; cache empty results too so non-bundle
  // files aren't re-parsed.
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache[path] = CacheValue{mod_time, parsed};
  }

  entries = std::move(parsed);
  return !entries.empty();
}

ObjectContainer *ObjectContainerClangOffloadBundle::CreateInstance(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
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
  if (!FindBundleEntries(m_file, m_entries))
    return false;
  return true;
}

size_t ObjectContainerClangOffloadBundle::GetNumArchitectures() const {
  return m_entries.size();
}

bool ObjectContainerClangOffloadBundle::GetArchitectureAtIndex(
    uint32_t idx, ArchSpec &arch) const {
  if (idx < m_entries.size()) {
    arch = m_entries[idx].arch;
    return true;
  }
  return false;
}

ModuleSpecList ObjectContainerClangOffloadBundle::GetModuleSpecifications(
    const FileSpec &file, DataExtractorSP &extractor_sp,
    lldb::offset_t /*file_offset*/, lldb::offset_t /*file_size*/) {
  if (!extractor_sp || !MagicBytesMatch(*extractor_sp))
    return {};

  std::vector<Entry> entries;
  if (!FindBundleEntries(file, entries))
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
  ModuleSP module_sp(GetModule());
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
