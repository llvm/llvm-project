//===-- ObjectFileWASM.cpp -------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Plugins/ObjectFile/WASM/ObjectFileWasm.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "llvm/ADT/ArrayRef.h"

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::wasm;

// Binary encoding of the module header.
constexpr uint32_t kWasmMagic = 0x6d736100; // '\0asm'
constexpr uint32_t kWasmVersion = 0x01;
static const uint32_t kHeaderSize = sizeof(kWasmMagic) + sizeof(kWasmVersion);
static const uint32_t kWasmCodeSectionId = 10;

void ObjectFileWASM::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                CreateMemoryInstance, GetModuleSpecifications);
}

void ObjectFileWASM::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ConstString ObjectFileWASM::GetPluginNameStatic() {
  static ConstString g_name("wasm");
  return g_name;
}

ObjectFile *
ObjectFileWASM::CreateInstance(const ModuleSP &module_sp, DataBufferSP &data_sp,
                               offset_t data_offset, const FileSpec *file,
                               offset_t file_offset, offset_t length) {
  return new ObjectFileWASM(module_sp, data_sp, data_offset, file, file_offset,
                            length);
}

ObjectFile *ObjectFileWASM::CreateMemoryInstance(const ModuleSP &module_sp,
                                                 DataBufferSP &data_sp,
                                                 const ProcessSP &process_sp,
                                                 addr_t header_addr) {
  if (data_sp && data_sp->GetByteSize() > 8) {
    const uint32_t *magic =
        reinterpret_cast<const uint32_t *>(data_sp->GetBytes());
    const uint32_t *version = reinterpret_cast<const uint32_t *>(
        data_sp->GetBytes() + sizeof(kWasmMagic));
    if (*magic == kWasmMagic && *version == kWasmVersion) {
      std::unique_ptr<ObjectFileWASM> objfile_up(
          new ObjectFileWASM(module_sp, data_sp, process_sp, header_addr));
      ArchSpec spec = objfile_up->GetArchitecture();
      if (spec && objfile_up->SetModulesArchitecture(spec)) {
        return objfile_up.release();
      }
    }
  }
  return nullptr;
}

// static
bool ObjectFileWASM::GetVaruint7(DataExtractor &section_header_data,
                                 lldb::offset_t *offset_ptr, uint8_t *result) {
  lldb::offset_t initial_offset = *offset_ptr;
  uint64_t value = section_header_data.GetULEB128(offset_ptr);
  if (*offset_ptr == initial_offset || value > 127) {
    return false;
  }
  *result = static_cast<uint8_t>(value);
  return true;
}

// static
bool ObjectFileWASM::GetVaruint32(DataExtractor &section_header_data,
                                  lldb::offset_t *offset_ptr,
                                  uint32_t *result) {
  lldb::offset_t initial_offset = *offset_ptr;
  uint64_t value = section_header_data.GetULEB128(offset_ptr);
  if (*offset_ptr == initial_offset || value > uint64_t(1) << 32) {
    return false;
  }
  *result = static_cast<uint32_t>(value);
  return true;
}

bool ObjectFileWASM::DecodeNextSection(lldb::offset_t *offset_ptr) {
  static ConstString g_sect_name_source_mapping_url("sourceMappingURL");

  // Buffer sufficient to read a section header and find the pointer to the next
  // section.
  const uint32_t kBufferSize = 1024;
  DataExtractor section_header_data = ReadImageData(*offset_ptr, kBufferSize);
  size_t len = section_header_data.BytesLeft(0);
  if (len > 0) {
    const uint8_t *data =
        section_header_data.PeekData(0, section_header_data.BytesLeft(0));
    m_hash.update(llvm::ArrayRef<uint8_t>(data, len));
  }

  lldb::offset_t offset = 0;
  uint8_t section_id = 0;
  uint32_t payload_len = 0;
  if (GetVaruint7(section_header_data, &offset, &section_id) &&
      GetVaruint32(section_header_data, &offset, &payload_len)) {

    if (section_id == 0) {
      uint32_t name_len = 0;
      lldb::offset_t prev_offset = offset;
      if (GetVaruint32(section_header_data, &offset, &name_len)) {
        uint32_t name_len_uleb_size = offset - prev_offset;
        std::string sect_name(section_header_data.PeekCStr(offset), name_len);
        offset += name_len;

        if (g_sect_name_source_mapping_url == sect_name.c_str()) {
          uint32_t url_len = 0;
          prev_offset = offset;
          if (GetVaruint32(section_header_data, &offset, &url_len)) {
            uint32_t url_len_uleb_size = offset - prev_offset;
            m_symbols_url =
                ConstString(section_header_data.PeekCStr(offset), url_len);
            GetModule()->SetSymbolFileFileSpec(
                FileSpec(m_symbols_url.GetStringRef()));

            uint32_t section_length =
                payload_len - name_len - name_len_uleb_size - url_len_uleb_size;
            offset += section_length;
          }
        } else {
          uint32_t section_length = payload_len - name_len - name_len_uleb_size;
          m_sect_infos.push_back(section_info{
              *offset_ptr + offset, section_length, section_id, sect_name});
          offset += section_length;
        }
      }
    } else if (section_id <= 11) {
      m_sect_infos.push_back(
          section_info{*offset_ptr + offset, payload_len, section_id, ""});
      offset += payload_len;
    } else {
      // Invalid section id
      return false;
    }
    *offset_ptr += offset;
    return true;
  }
  return false;
}

bool ObjectFileWASM::DecodeSections(lldb::addr_t load_address) {
  lldb::offset_t offset = load_address + kHeaderSize;
  while (DecodeNextSection(&offset))
    ;

  // The UUID should be retrieved by a custom section in the Wasm module (still
  // to be standardized). For the moment, just calculate a UUID from a MD5 hash.
  llvm::MD5::MD5Result md5_res;
  m_hash.final(md5_res);
  m_uuid = UUID::fromData(md5_res.Bytes.data(), md5_res.Bytes.size());

  return true;
}

size_t ObjectFileWASM::GetModuleSpecifications(
    const FileSpec &file, DataBufferSP &data_sp, offset_t data_offset,
    offset_t file_offset, offset_t length, ModuleSpecList &specs) {
  if (data_sp->GetByteSize() < sizeof(kWasmMagic)) {
    return 0;
  }

  uint32_t magic_number = *(uint32_t *)data_sp->GetBytes();
  if (magic_number != kWasmMagic) {
    return 0;
  }

  ModuleSpec spec(file, ArchSpec("wasm32-unknown-unknown-wasm"));
  specs.Append(spec);
  return 1;
}

ObjectFileWASM::ObjectFileWASM(const ModuleSP &module_sp, DataBufferSP &data_sp,
                               offset_t data_offset, const FileSpec *file,
                               offset_t offset, offset_t length)
    : ObjectFile(module_sp, file, offset, length, data_sp, data_offset),
      m_arch("wasm32-unknown-unknown-wasm"), m_code_section_offset(0) {
}

ObjectFileWASM::ObjectFileWASM(const lldb::ModuleSP &module_sp,
                               lldb::DataBufferSP &header_data_sp,
                               const lldb::ProcessSP &process_sp,
                               lldb::addr_t header_addr)
    : ObjectFile(module_sp, process_sp, header_addr, header_data_sp),
      m_arch("wasm32-unknown-unknown-wasm"), m_code_section_offset(0) {
}

bool ObjectFileWASM::ParseHeader() {
  // We already parsed the header during initialization.
  return true;
}

Symtab *ObjectFileWASM::GetSymtab() {
  return nullptr;
}

void ObjectFileWASM::CreateSections(SectionList &unified_section_list) {
  static ConstString g_sect_name_dwarf_debug_abbrev(".debug_abbrev");
  static ConstString g_sect_name_dwarf_debug_aranges(".debug_aranges");
  static ConstString g_sect_name_dwarf_debug_frame(".debug_frame");
  static ConstString g_sect_name_dwarf_debug_info(".debug_info");
  static ConstString g_sect_name_dwarf_debug_line(".debug_line");
  static ConstString g_sect_name_dwarf_debug_loc(".debug_loc");
  static ConstString g_sect_name_dwarf_debug_loclists(".debug_loclists");
  static ConstString g_sect_name_dwarf_debug_macinfo(".debug_macinfo");
  static ConstString g_sect_name_dwarf_debug_names(".debug_names");
  static ConstString g_sect_name_dwarf_debug_pubnames(".debug_pubnames");
  static ConstString g_sect_name_dwarf_debug_pubtypes(".debug_pubtypes");
  static ConstString g_sect_name_dwarf_debug_ranges(".debug_ranges");
  static ConstString g_sect_name_dwarf_debug_str(".debug_str");
  static ConstString g_sect_name_dwarf_debug_types(".debug_types");
  static ConstString g_sect_name_dwarf_debug_addr("dwarf-addr");
  static ConstString g_sect_name_dwarf_debug_cuindex("dwarf-cu-index");
  static ConstString g_sect_name_dwarf_debug_macro("dwarf-macro");
  static ConstString g_sect_name_dwarf_debug_stroffsets("dwarf-str-offsets");

  if (m_sections_up)
    return;

  m_sections_up = std::make_unique<SectionList>();

  int index = 1;
  for (SectionInfoCollConstIter it = m_sect_infos.begin();
       it != m_sect_infos.end(); ++it) {
    const section_info &sect_info = *it;

    SectionType section_type = eSectionTypeInvalid;

    if (kWasmCodeSectionId == sect_info.id)
      section_type = eSectionTypeCode;
    else if (g_sect_name_dwarf_debug_abbrev == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugAbbrev;
    else if (g_sect_name_dwarf_debug_aranges == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugAranges;
    else if (g_sect_name_dwarf_debug_frame == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugFrame;
    else if (g_sect_name_dwarf_debug_info == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugInfo;
    else if (g_sect_name_dwarf_debug_line == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugLine;
    else if (g_sect_name_dwarf_debug_loc == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugLoc;
    else if (g_sect_name_dwarf_debug_loclists == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugLocLists;
    else if (g_sect_name_dwarf_debug_macinfo == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugMacInfo;
    else if (g_sect_name_dwarf_debug_names == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugNames;
    else if (g_sect_name_dwarf_debug_pubnames == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugPubNames;
    else if (g_sect_name_dwarf_debug_pubtypes == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugPubTypes;
    else if (g_sect_name_dwarf_debug_ranges == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugRanges;
    else if (g_sect_name_dwarf_debug_str == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugStr;
    else if (g_sect_name_dwarf_debug_types == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugTypes;
    else if (g_sect_name_dwarf_debug_addr == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugAddr;
    else if (g_sect_name_dwarf_debug_cuindex == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugCuIndex;
    else if (g_sect_name_dwarf_debug_macro == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugMacro;
    else if (g_sect_name_dwarf_debug_stroffsets == sect_info.name.c_str())
      section_type = eSectionTypeDWARFDebugStrOffsets;

    if (section_type == eSectionTypeCode) {
      m_code_section_offset = sect_info.offset & 0xffffffff;
      SectionSP section_sp(new Section(
          GetModule(),         // Module to which this section belongs.
          this,                // ObjectFile to which this section belongs and
                               // should read section data from.
          index++,             // Section ID.
          ConstString("code"), // Section name.
          section_type,        // Section type.
          0,                   // sect_info.offset & 0xffffffff, // VM address.
          sect_info.size,      // VM size in bytes of this section.
          0, // sect_info.offset & 0xffffffff, // Offset of this section
             // in the file.
          sect_info.size, // Size of the section as found in
                          // the file.
          0,              // Alignment of the section
          0,              // Flags for this section.
          1));            // Number of host bytes per target byte
      m_sections_up->AddSection(section_sp);
    } else if (section_type != eSectionTypeInvalid) {
      SectionSP section_sp(new Section(
          GetModule(), // Module to which this section belongs.
          this,        // ObjectFile to which this section belongs and
                       // should read section data from.
          index++,     // Section ID.
          ConstString(sect_info.name.c_str()), // Section name.
          section_type,                        // Section type.
          sect_info.offset & 0xffffffff,       // VM address.
          sect_info.size,                // VM size in bytes of this section.
          sect_info.offset & 0xffffffff, // Offset of this section in the file.
          sect_info.size,                // Size of the section as found in
                                         // the file.
          0,                             // Alignment of the section
          0,                             // Flags for this section.
          1));                           // Number of host bytes per target byte
      m_sections_up->AddSection(section_sp);
    }
  }

  unified_section_list = *m_sections_up;
}

bool ObjectFileWASM::SetLoadAddress(Target &target, lldb::addr_t value,
                                    bool value_is_offset) {
  ModuleSP module_sp = GetModule();
  if (module_sp) {
    DecodeSections(value);

    size_t num_loaded_sections = 0;
    SectionList *section_list = GetSectionList();
    if (section_list) {
      const size_t num_sections = section_list->GetSize();
      size_t sect_idx = 0;

      for (sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
        SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
        if (target.GetSectionLoadList().SetSectionLoadAddress(
                section_sp, value | section_sp->GetFileOffset())) {
          ++num_loaded_sections;
        }
      }
      return num_loaded_sections > 0;
    }
  }
  return false;
}

DataExtractor ObjectFileWASM::ReadImageData(uint64_t offset, size_t size) {
  if (m_file) {
    auto buffer_sp = MapFileData(m_file, size, offset);
    return DataExtractor(buffer_sp, GetByteOrder(), GetAddressByteSize());
  }
  ProcessSP process_sp(m_process_wp.lock());
  DataExtractor data;
  if (process_sp) {
    auto data_up = std::make_unique<DataBufferHeap>(size, 0);
    Status readmem_error;
    size_t bytes_read =
        process_sp->ReadMemory(offset, data_up->GetBytes(),
                               data_up->GetByteSize(), readmem_error);
    if (bytes_read > 0) {
      DataBufferSP buffer_sp(data_up.release());
      data.SetData(buffer_sp, 0, buffer_sp->GetByteSize());
    }
  }
  return data;
}

void ObjectFileWASM::Dump(Stream *s) {
  ModuleSP module_sp(GetModule());
  if (module_sp) {
    std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());
    s->Printf("%p: ", static_cast<void *>(this));
    s->Indent();
    s->PutCString("ObjectFileWASM");

    ArchSpec header_arch = GetArchitecture();

    *s << ", file = '" << m_file
       << "', arch = " << header_arch.GetArchitectureName() << "\n";

    SectionList *sections = GetSectionList();
    if (sections) {
      sections->Dump(s, nullptr, true, UINT32_MAX);
    }
    s->EOL();
    DumpSectionHeaders(s);
    s->EOL();
  }
}

// Dump a single Wasm section header to the specified output stream.
void ObjectFileWASM::DumpSectionHeader(Stream *s, const section_info_t &sh) {
  s->Printf("%-16s 0x%8.8x 0x%8.8x 0x%4.4x\n", sh.name.c_str(), sh.offset,
            sh.size, sh.id);
}

lldb::offset_t offset;
uint32_t size;
uint32_t id;
std::string name;


// Dump all of the Wasm section header to the specified output stream.
void ObjectFileWASM::DumpSectionHeaders(Stream *s) {
  s->PutCString("Section Headers\n");
  s->PutCString("IDX  name             addr       size       id\n");
  s->PutCString("==== ---------------- ---------- ---------- ------\n");

  uint32_t idx = 0;
  SectionInfoCollIter pos, end = m_sect_infos.end();
  for (pos = m_sect_infos.begin(); pos != end; ++pos, ++idx) {
    s->Printf("[%2u] ", idx);
    ObjectFileWASM::DumpSectionHeader(s, *pos);
  }
}
