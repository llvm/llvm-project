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
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Object/OffloadBundle.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"

#include <limits>
#include <string>
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
  // Bundle entry IDs contain an offload kind, a target triple, and optionally
  // a target ID and target features.
  llvm::StringRef triple = id.split('-').second;
  // AMDGPU target features follow the processor name after a colon. ArchSpec
  // models the processor, but not these target features.
  triple = triple.split(':').first;
  if (triple.empty())
    return {};

  // Clang offload bundle IDs can use legacy or processor-qualified
  // architecture spellings, while LLDB's canonical name is "amdgpu".
  auto [architecture, triple_suffix] = triple.split('-');
  llvm::Triple target_triple(triple);
  if (target_triple.isAMDGCN()) {
    llvm::StringRef processor =
        llvm::AMDGPU::getArchNameFromSubArch(target_triple.getSubArch());
    if (!processor.empty()) {
      std::string normalized_triple = "amdgpu-";
      normalized_triple.append(triple_suffix.split("--").first);
      normalized_triple.append("--");
      normalized_triple.append(processor);
      return ArchSpec(normalized_triple);
    }
  }
  if (architecture == "amdgcn") {
    std::string normalized_triple = "amdgpu-";
    normalized_triple.append(triple_suffix);
    return ArchSpec(normalized_triple);
  }
  return ArchSpec(triple);
}

bool ObjectContainerClangOffloadBundle::FindBundleEntries(
    const FileSpec &file, DataExtractorSP extractor_sp,
    lldb::offset_t file_offset, lldb::offset_t file_size,
    std::vector<Entry> &entries) {
  const uint8_t *data = nullptr;
  size_t data_size = 0;
  DataBufferSP mapped_data_sp;
  if (extractor_sp && extractor_sp->HasData() &&
      extractor_sp->GetByteSize() >= file_size) {
    data = extractor_sp->GetDataStart();
    data_size = file_size ? file_size : extractor_sp->GetByteSize();
  } else if (file) {
    mapped_data_sp =
        FileSystem::Instance().CreateDataBuffer(file, file_size, file_offset);
    if (mapped_data_sp) {
      data = mapped_data_sp->GetBytes();
      data_size = mapped_data_sp->GetByteSize();
    }
  }
  if (!data)
    return false;

  llvm::StringRef bytes(reinterpret_cast<const char *>(data), data_size);
  const std::string path = file.GetPath();
  llvm::MemoryBufferRef buffer(bytes, path);
  auto object_or_error = llvm::object::ObjectFile::createObjectFile(buffer);
  if (!object_or_error) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Object), object_or_error.takeError(),
                   "unable to parse clang offload bundle container: {0}");
    return false;
  }

  llvm::SmallVector<llvm::object::OffloadBundleFatBin> bundles;
  if (llvm::Error error = llvm::object::extractOffloadBundleFatBinary(
          **object_or_error, bundles)) {
    LLDB_LOG_ERROR(GetLog(LLDBLog::Object), std::move(error),
                   "unable to extract clang offload bundle: {0}");
    return false;
  }

  for (llvm::object::OffloadBundleFatBin &bundle : bundles) {
    // Compressed bundle entries refer to the decompressed buffer, not to
    // offsets in the containing object file.
    if (bundle.isDecompressed()) {
      LLDB_LOG(GetLog(LLDBLog::Object),
               "skipping compressed clang offload bundle in {0}", path);
      continue;
    }

    for (const llvm::object::OffloadBundleEntry &bundle_entry :
         bundle.getEntries()) {
      if (bundle_entry.Size == 0 || bundle_entry.Offset == 0 ||
          bundle_entry.Offset > data_size ||
          bundle_entry.Size > data_size - bundle_entry.Offset ||
          bundle_entry.Offset >
              std::numeric_limits<lldb::offset_t>::max() - file_offset ||
          bundle_entry.Size > std::numeric_limits<lldb::offset_t>::max() -
                                  file_offset - bundle_entry.Offset)
        continue;

      ArchSpec arch = ParseArchFromBundleEntryID(bundle_entry.ID);
      if (!arch.IsValid())
        continue;

      entries.push_back({std::move(arch), file_offset + bundle_entry.Offset,
                         bundle_entry.Size});
    }
  }

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
    ModuleSpec spec(file, entry.arch);
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
      const bool matches = pass == 0 ? arch.IsExactMatch(entry.arch)
                                     : arch.IsCompatibleMatch(entry.arch);
      if (!matches)
        continue;

      DataExtractorSP extractor_sp;
      lldb::offset_t data_offset = 0;
      if (ObjectFileSP object_file_sp =
              ObjectFile::FindPlugin(module_sp, file, entry.offset, entry.size,
                                     extractor_sp, data_offset))
        return object_file_sp;
    }
  }

  return {};
}
