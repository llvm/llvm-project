//===-- ObjectFileJSON.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/JSON/ObjectFileJSON.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Symbol/Symbol.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/DenseSet.h"
#include <optional>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectFileJSON)

char ObjectFileJSON::ID;

void ObjectFileJSON::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                CreateMemoryInstance, GetModuleSpecifications);
}

void ObjectFileJSON::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectFile *
ObjectFileJSON::CreateInstance(const ModuleSP &module_sp, DataBufferSP data_sp,
                               offset_t data_offset, const FileSpec *file,
                               offset_t file_offset, offset_t length) {
  if (!data_sp) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }

  if (!MagicBytesMatch(data_sp, 0, data_sp->GetByteSize()))
    return nullptr;

  // Update the data to contain the entire file if it doesn't already.
  if (data_sp->GetByteSize() < length) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }

  Log *log = GetLog(LLDBLog::Symbols);

  auto text =
      llvm::StringRef(reinterpret_cast<const char *>(data_sp->GetBytes()));

  Expected<json::Value> json = json::parse(text);
  if (!json) {
    LLDB_LOG_ERROR(log, json.takeError(),
                   "failed to parse JSON object file: {0}");
    return nullptr;
  }

  json::Path::Root root;
  Header header;
  if (!fromJSON(*json, header, root)) {
    LLDB_LOG_ERROR(log, root.getError(),
                   "failed to parse JSON object file header: {0}");
    return nullptr;
  }

  ArchSpec arch(header.triple);
  UUID uuid;
  uuid.SetFromStringRef(header.uuid);
  Type type = header.type.value_or(eTypeDebugInfo);

  Body body;
  if (!fromJSON(*json, body, root)) {
    LLDB_LOG_ERROR(log, root.getError(),
                   "failed to parse JSON object file body: {0}");
    return nullptr;
  }

  return new ObjectFileJSON(module_sp, data_sp, data_offset, file, file_offset,
                            length, std::move(arch), std::move(uuid), type,
                            std::move(body.symbols), std::move(body.sections));
}

ObjectFile *ObjectFileJSON::CreateMemoryInstance(const ModuleSP &module_sp,
                                                 WritableDataBufferSP data_sp,
                                                 const ProcessSP &process_sp,
                                                 addr_t header_addr) {
  return nullptr;
}

size_t ObjectFileJSON::GetModuleSpecifications(
    const FileSpec &file, DataBufferSP &data_sp, offset_t data_offset,
    offset_t file_offset, offset_t length, ModuleSpecList &specs) {
  if (!MagicBytesMatch(data_sp, data_offset, data_sp->GetByteSize()))
    return 0;

  // Update the data to contain the entire file if it doesn't already.
  if (data_sp->GetByteSize() < length) {
    data_sp = MapFileData(file, length, file_offset);
    if (!data_sp)
      return 0;
    data_offset = 0;
  }

  Log *log = GetLog(LLDBLog::Symbols);

  auto text =
      llvm::StringRef(reinterpret_cast<const char *>(data_sp->GetBytes()));

  Expected<json::Value> json = json::parse(text);
  if (!json) {
    LLDB_LOG_ERROR(log, json.takeError(),
                   "failed to parse JSON object file: {0}");
    return 0;
  }

  json::Path::Root root;
  Header header;
  if (!fromJSON(*json, header, root)) {
    LLDB_LOG_ERROR(log, root.getError(),
                   "failed to parse JSON object file header: {0}");
    return 0;
  }

  ArchSpec arch(header.triple);
  UUID uuid;
  uuid.SetFromStringRef(header.uuid);

  ModuleSpec spec(file, std::move(arch));
  spec.GetUUID() = std::move(uuid);
  specs.Append(spec);
  return 1;
}

ObjectFileJSON::ObjectFileJSON(const ModuleSP &module_sp, DataBufferSP &data_sp,
                               offset_t data_offset, const FileSpec *file,
                               offset_t offset, offset_t length, ArchSpec arch,
                               UUID uuid, Type type,
                               std::vector<JSONSymbol> symbols,
                               std::vector<JSONSection> sections)
    : ObjectFile(module_sp, file, offset, length, data_sp, data_offset),
      m_arch(std::move(arch)), m_uuid(std::move(uuid)), m_type(type),
      m_symbols(std::move(symbols)), m_sections(std::move(sections)) {}

bool ObjectFileJSON::ParseHeader() {
  // We already parsed the header during initialization.
  return true;
}

void ObjectFileJSON::ParseSymtab(Symtab &symtab) {
  Log *log = GetLog(LLDBLog::Symbols);
  SectionList *section_list = GetModule()->GetSectionList();
  for (JSONSymbol json_symbol : m_symbols) {
    llvm::Expected<Symbol> symbol = Symbol::FromJSON(json_symbol, section_list);
    if (!symbol) {
      LLDB_LOG_ERROR(log, symbol.takeError(), "invalid symbol: {0}");
      continue;
    }
    symtab.AddSymbol(*symbol);
  }
  symtab.Finalize();
}

void ObjectFileJSON::CreateSections(SectionList &unified_section_list) {
  if (m_sections_up)
    return;
  m_sections_up = std::make_unique<SectionList>();

  lldb::user_id_t id = 0;
  for (const auto &json_section : m_sections) {
    auto make_section = [this, &id](const JSONSection &section,
                                    SectionSP parent_section_sp =
                                        nullptr) -> SectionSP {
      SectionSP section_sp;
      auto sect_id = section.user_id.value_or(id + 1);
      if (!section.user_id.has_value())
        ++id;
      const auto name = ConstString(section.name);
      const auto sect_type = section.type.value_or(eSectionTypeCode);
      const auto vm_addr = section.address.value_or(0);
      const auto vm_size = section.size.value_or(0);
      const auto file_offset = section.file_offset.value_or(0);
      const auto file_size = section.file_size.value_or(0);
      const auto log2align = section.log2align.value_or(0);
      const auto flags = section.flags.value_or(0);
      if (parent_section_sp) {
        section_sp = std::make_shared<Section>(
            parent_section_sp, GetModule(), this, sect_id, name, sect_type,
            vm_addr - parent_section_sp->GetFileAddress(), vm_size, file_offset,
            file_size, log2align, flags);

      } else {
        section_sp = std::make_shared<Section>(
            GetModule(), this, sect_id, name, sect_type, vm_addr, vm_size,
            file_offset, file_size, log2align, flags);
      }
      // Set permissions
      uint32_t permissions = 0;
      if (section.read.value_or(0))
        permissions |= lldb::ePermissionsReadable;
      if (section.write.value_or(0))
        permissions |= lldb::ePermissionsWritable;
      if (section.execute.value_or(0))
        permissions |= lldb::ePermissionsExecutable;
      if (permissions)
        section_sp->SetPermissions(permissions);
      section_sp->SetIsFake(section.fake.value_or(false));
      section_sp->SetIsEncrypted(section.encrypted.value_or(false));
      section_sp->SetIsThreadSpecific(section.thread_specific.value_or(false));
      return section_sp;
    };
    auto section_sp = make_section(json_section);
    for (const auto &subsection : json_section.subsections) {
      SectionSP subsection_sp = make_section(subsection, section_sp);
      section_sp->GetChildren().AddSection(subsection_sp);
    }

    m_sections_up->AddSection(section_sp);
    unified_section_list.AddSection(section_sp);
  }
}

bool ObjectFileJSON::SetLoadAddress(Target &target, lldb::addr_t value,
                                    bool value_is_offset) {
  Log *log(GetLog(LLDBLog::DynamicLoader));
  if (!m_sections_up)
    return true;

  addr_t slide = value;
  if (!value_is_offset) {
    addr_t lowest_addr = LLDB_INVALID_ADDRESS;
    for (const SectionSP &section_sp : *m_sections_up) {
      addr_t section_load_addr = section_sp->GetFileAddress();
      lowest_addr = std::min(lowest_addr, section_load_addr);
    }
    if (lowest_addr == LLDB_INVALID_ADDRESS)
      return false;
    slide = value - lowest_addr;
  }

  // Apply slide to each section's file address.
  for (const SectionSP &section_sp : *m_sections_up) {
    addr_t section_load_addr = section_sp->GetFileAddress();
    if (section_load_addr != LLDB_INVALID_ADDRESS) {
      LLDB_LOGF(
          log,
          "ObjectFileJSON::SetLoadAddress section %s to load addr 0x%" PRIx64,
          section_sp->GetName().AsCString(), section_load_addr + slide);
      target.SetSectionLoadAddress(section_sp, section_load_addr + slide,
                                   /*warn_multiple=*/true);
    }
  }

  return true;
}

bool ObjectFileJSON::MagicBytesMatch(DataBufferSP data_sp,
                                     lldb::addr_t data_offset,
                                     lldb::addr_t data_length) {
  DataExtractor data;
  data.SetData(data_sp, data_offset, data_length);
  lldb::offset_t offset = 0;
  uint32_t magic = data.GetU8(&offset);
  return magic == '{';
}

namespace lldb_private {

bool fromJSON(const json::Value &value, ObjectFileJSON::Header &header,
              json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.map("triple", header.triple) && o.map("uuid", header.uuid) &&
         o.map("type", header.type);
}

bool fromJSON(const json::Value &value, ObjectFileJSON::Body &body,
              json::Path path) {
  json::ObjectMapper o(value, path);
  return o && o.mapOptional("symbols", body.symbols) &&
         o.mapOptional("sections", body.sections);
}

} // namespace lldb_private
