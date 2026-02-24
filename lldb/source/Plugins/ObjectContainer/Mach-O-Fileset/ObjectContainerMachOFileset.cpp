//===-- ObjectContainerMachOFileset.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectContainerMachOFileset.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/Stream.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace llvm::MachO;

LLDB_PLUGIN_DEFINE(ObjectContainerMachOFileset)

void ObjectContainerMachOFileset::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                GetModuleSpecifications, CreateMemoryInstance);
}

void ObjectContainerMachOFileset::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectContainerMachOFileset::ObjectContainerMachOFileset(
    const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, const lldb_private::FileSpec *file,
    lldb::offset_t offset, lldb::offset_t length)
    : ObjectContainer(module_sp, file, offset, length, data_sp, data_offset),
      m_memory_addr(LLDB_INVALID_ADDRESS) {}

ObjectContainerMachOFileset::ObjectContainerMachOFileset(
    const lldb::ModuleSP &module_sp, lldb::WritableDataBufferSP data_sp,
    const lldb::ProcessSP &process_sp, lldb::addr_t header_addr)
    : ObjectContainer(module_sp, nullptr, 0, data_sp->GetByteSize(), data_sp,
                      0),
      m_process_wp(process_sp), m_memory_addr(header_addr) {}

ObjectContainer *ObjectContainerMachOFileset::CreateInstance(
    const lldb::ModuleSP &module_sp, DataBufferSP &data_sp,
    lldb::offset_t data_offset, const FileSpec *file,
    lldb::offset_t file_offset, lldb::offset_t length) {
  if (!data_sp)
    return {};

  DataExtractor extractor;
  extractor.SetData(data_sp, data_offset, length);
  if (!MagicBytesMatch(extractor))
    return {};

  auto container_up = std::make_unique<ObjectContainerMachOFileset>(
      module_sp, data_sp, data_offset, file, file_offset, length);
  if (!container_up->ParseHeader())
    return {};

  return container_up.release();
}

ObjectContainer *ObjectContainerMachOFileset::CreateMemoryInstance(
    const lldb::ModuleSP &module_sp, lldb::WritableDataBufferSP data_sp,
    const lldb::ProcessSP &process_sp, lldb::addr_t header_addr) {
  if (!MagicBytesMatch(data_sp, 0, data_sp->GetByteSize()))
    return {};

  auto container_up = std::make_unique<ObjectContainerMachOFileset>(
      module_sp, data_sp, process_sp, header_addr);
  if (!container_up->ParseHeader())
    return {};

  return container_up.release();
}

ObjectContainerMachOFileset::~ObjectContainerMachOFileset() = default;

static uint32_t MachHeaderSizeFromMagic(uint32_t magic) {
  switch (magic) {
  case MH_MAGIC:
  case MH_CIGAM:
    return sizeof(struct mach_header);
  case MH_MAGIC_64:
  case MH_CIGAM_64:
    return sizeof(struct mach_header_64);
  default:
    return 0;
  }
}

static std::optional<mach_header> ParseMachOHeader(DataExtractor &extractor) {
  lldb::offset_t offset = 0;
  mach_header header;
  header.magic = extractor.GetU32(&offset);
  switch (header.magic) {
  case MH_MAGIC:
    extractor.SetByteOrder(endian::InlHostByteOrder());
    extractor.SetAddressByteSize(4);
    break;
  case MH_MAGIC_64:
    extractor.SetByteOrder(endian::InlHostByteOrder());
    extractor.SetAddressByteSize(8);
    break;
  case MH_CIGAM:
    extractor.SetByteOrder(endian::InlHostByteOrder() == eByteOrderBig
                               ? eByteOrderLittle
                               : eByteOrderBig);
    extractor.SetAddressByteSize(4);
    break;
  case MH_CIGAM_64:
    extractor.SetByteOrder(endian::InlHostByteOrder() == eByteOrderBig
                               ? eByteOrderLittle
                               : eByteOrderBig);
    extractor.SetAddressByteSize(8);
    break;
  default:
    return {};
  }

  header.cputype = extractor.GetU32(&offset);
  header.cpusubtype = extractor.GetU32(&offset);
  header.filetype = extractor.GetU32(&offset);
  header.ncmds = extractor.GetU32(&offset);
  header.sizeofcmds = extractor.GetU32(&offset);
  return header;
}

static bool
ParseFileset(DataExtractor &extractor, mach_header header,
             std::vector<ObjectContainerMachOFileset::Entry> &entries,
             std::optional<lldb::addr_t> load_addr = std::nullopt) {
  lldb::offset_t offset = MachHeaderSizeFromMagic(header.magic);
  lldb::offset_t slide = 0;
  for (uint32_t i = 0; i < header.ncmds; ++i) {
    const lldb::offset_t load_cmd_offset = offset;
    load_command lc = {};
    if (extractor.GetU32(&offset, &lc.cmd, 2) == nullptr)
      break;

    // If we know the load address we can compute the slide.
    if (load_addr) {
      if (lc.cmd == llvm::MachO::LC_SEGMENT_64) {
        segment_command_64 segment;
        extractor.CopyData(load_cmd_offset, sizeof(segment_command_64),
                           &segment);
        if (llvm::StringRef(segment.segname) == "__TEXT")
          slide = *load_addr - segment.vmaddr;
      }
    }

    if (lc.cmd == LC_FILESET_ENTRY) {
      fileset_entry_command entry;
      extractor.CopyData(load_cmd_offset, sizeof(fileset_entry_command),
                         &entry);
      lldb::offset_t entry_id_offset = load_cmd_offset + entry.entry_id.offset;
      if (const char *id = extractor.GetCStr(&entry_id_offset))
        entries.emplace_back(entry.vmaddr + slide, entry.fileoff,
                             std::string(id));
    }

    offset = load_cmd_offset + lc.cmdsize;
  }

  return true;
}

bool ObjectContainerMachOFileset::ParseHeader(
    DataExtractor &extractor, const lldb_private::FileSpec &file,
    lldb::offset_t file_offset, std::vector<Entry> &entries) {
  std::optional<mach_header> header = ParseMachOHeader(extractor);

  if (!header)
    return false;

  const size_t header_size = MachHeaderSizeFromMagic(header->magic);
  const size_t header_and_lc_size = header_size + header->sizeofcmds;

  if (extractor.GetByteSize() < header_and_lc_size) {
    DataBufferSP data_sp =
        ObjectFile::MapFileData(file, header_and_lc_size, file_offset);
    extractor.SetData(data_sp);
  }

  return ParseFileset(extractor, *header, entries);
}

bool ObjectContainerMachOFileset::ParseHeader() {
  ModuleSP module_sp(GetModule());
  if (!module_sp)
    return false;

  std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());

  std::optional<mach_header> header = ParseMachOHeader(*m_extractor_sp);
  if (!header)
    return false;

  const size_t header_size = MachHeaderSizeFromMagic(header->magic);
  const size_t header_and_lc_size = header_size + header->sizeofcmds;

  if (m_extractor_sp->GetByteSize() < header_and_lc_size) {
    ProcessSP process_sp(m_process_wp.lock());
    DataBufferSP data_sp =
        process_sp
            ? ObjectFile::ReadMemory(process_sp, m_memory_addr,
                                     header_and_lc_size)
            : ObjectFile::MapFileData(m_file, header_and_lc_size, m_offset);
    m_extractor_sp->SetData(data_sp);
  }

  return ParseFileset(*m_extractor_sp, *header, m_entries, m_memory_addr);
}

size_t ObjectContainerMachOFileset::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataExtractorSP &extractor_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t file_size, lldb_private::ModuleSpecList &specs) {
  const size_t initial_count = specs.GetSize();
  if (!extractor_sp)
    return initial_count;

  DataExtractorSP data_extractor_sp =
      extractor_sp->GetSubsetExtractorSP(data_offset);
  if (!data_extractor_sp)
    return initial_count;
  if (MagicBytesMatch(*data_extractor_sp)) {
    std::vector<Entry> entries;
    if (ParseHeader(*data_extractor_sp, file, file_offset, entries)) {
      for (const Entry &entry : entries) {
        const lldb::offset_t entry_offset = entry.fileoff + file_offset;
        if (ObjectFile::GetModuleSpecifications(
                file, entry_offset, file_size - entry_offset, specs)) {
          ModuleSpec &spec = specs.GetModuleSpecRefAtIndex(specs.GetSize() - 1);
          spec.GetObjectName() = ConstString(entry.id);
        }
      }
    }
  }
  return specs.GetSize() - initial_count;
}

bool ObjectContainerMachOFileset::MagicBytesMatch(DataBufferSP data_sp,
                                                  lldb::addr_t data_offset,
                                                  lldb::addr_t data_length) {
  DataExtractor extractor;
  extractor.SetData(data_sp, data_offset, data_length);
  return MagicBytesMatch(extractor);
}

bool ObjectContainerMachOFileset::MagicBytesMatch(
    const DataExtractor &extractor) {
  lldb::offset_t offset = 0;
  uint32_t magic = extractor.GetU32(&offset);
  switch (magic) {
  case MH_MAGIC:
  case MH_CIGAM:
  case MH_MAGIC_64:
  case MH_CIGAM_64:
    break;
  default:
    return false;
  }
  offset += 4; // cputype
  offset += 4; // cpusubtype
  uint32_t filetype = extractor.GetU32(&offset);
  return filetype == MH_FILESET;
}

ObjectFileSP
ObjectContainerMachOFileset::GetObjectFile(const lldb_private::FileSpec *file) {
  ModuleSP module_sp(GetModule());
  if (!module_sp)
    return {};

  ConstString object_name = module_sp->GetObjectName();
  if (!object_name)
    return {};

  Entry *entry = FindEntry(object_name.GetCString());
  if (!entry)
    return {};

  DataExtractorSP extractor_sp;
  lldb::offset_t data_offset = 0;
  return ObjectFile::FindPlugin(module_sp, file, m_offset + entry->fileoff,
                                m_extractor_sp->GetByteSize() - entry->fileoff,
                                extractor_sp, data_offset);
}

ObjectContainerMachOFileset::Entry *
ObjectContainerMachOFileset::FindEntry(llvm::StringRef id) {
  for (Entry &entry : m_entries) {
    if (entry.id == id)
      return &entry;
  }
  return nullptr;
}
