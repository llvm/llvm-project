//===-- ObjectFileWasm.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectFileWasm.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/SectionLoadList.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Magic.h"
#include "llvm/BinaryFormat/Wasm.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include <optional>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::wasm;

LLDB_PLUGIN_DEFINE(ObjectFileWasm)

static const uint32_t kWasmHeaderSize =
    sizeof(llvm::wasm::WasmMagic) + sizeof(llvm::wasm::WasmVersion);

/// Helper to read a 32-bit ULEB using LLDB's DataExtractor.
static inline llvm::Expected<uint32_t> GetULEB32(DataExtractor &data,
                                                 lldb::offset_t &offset) {
  const uint64_t value = data.GetULEB128(&offset);
  if (value > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError("ULEB exceeds 32 bits");
  return value;
}

/// Helper to read a 32-bit ULEB using LLVM's DataExtractor.
static inline llvm::Expected<uint32_t>
GetULEB32(llvm::DataExtractor &data, llvm::DataExtractor::Cursor &c) {
  const uint64_t value = data.getULEB128(c);
  if (!c)
    return c.takeError();
  if (value > std::numeric_limits<uint32_t>::max())
    return llvm::createStringError("ULEB exceeds 32 bits");
  return value;
}

/// Helper to read a Wasm string, whcih is encoded as a vector of UTF-8 codes.
static inline llvm::Expected<std::string>
GetWasmString(llvm::DataExtractor &data, llvm::DataExtractor::Cursor &c) {
  llvm::Expected<uint32_t> len = GetULEB32(data, c);
  if (!len)
    return len.takeError();

  llvm::SmallVector<uint8_t, 32> str_storage;
  data.getU8(c, str_storage, *len);
  if (!c)
    return c.takeError();

  return std::string(toStringRef(llvm::ArrayRef(str_storage)));
}

/// An "init expr" refers to a constant expression used to determine the initial
/// value of certain elements within a module during instantiation. These
/// expressions are restricted to operations that can be evaluated at module
/// instantiation time. Currently we only support simple constant opcodes.
static lldb::offset_t GetWasmOffsetFromInitExpr(DataExtractor &data,
                                                lldb::offset_t &offset) {
  lldb::offset_t init_expr_offset = LLDB_INVALID_OFFSET;

  uint8_t opcode = data.GetU8(&offset);
  switch (opcode) {
  case llvm::wasm::WASM_OPCODE_I32_CONST:
  case llvm::wasm::WASM_OPCODE_I64_CONST:
    init_expr_offset = data.GetSLEB128(&offset);
    break;
  case llvm::wasm::WASM_OPCODE_GLOBAL_GET:
    init_expr_offset = data.GetULEB128(&offset);
    break;
  case llvm::wasm::WASM_OPCODE_F32_CONST:
  case llvm::wasm::WASM_OPCODE_F64_CONST:
    // Not a meaningful offset.
    data.GetFloat(&offset);
    break;
  case llvm::wasm::WASM_OPCODE_REF_NULL:
    // Not a meaningful offset.
    data.GetULEB128(&offset);
    break;
  }

  // Make sure the opcodes we read aren't part of an extended init expr.
  opcode = data.GetU8(&offset);
  if (opcode == llvm::wasm::WASM_OPCODE_END)
    return init_expr_offset;

  // Extended init expressions are not supported, but we still have to parse
  // them to skip over them and read the next segment.
  do {
    opcode = data.GetU8(&offset);
  } while (opcode != llvm::wasm::WASM_OPCODE_END);
  return LLDB_INVALID_OFFSET;
}

/// Checks whether the data buffer starts with a valid Wasm module header.
static bool ValidateModuleHeader(const DataBufferSP &data_sp) {
  if (!data_sp || data_sp->GetByteSize() < kWasmHeaderSize)
    return false;

  if (llvm::identify_magic(toStringRef(data_sp->GetData())) !=
      llvm::file_magic::wasm_object)
    return false;

  const uint8_t *Ptr = data_sp->GetBytes() + sizeof(llvm::wasm::WasmMagic);

  uint32_t version = llvm::support::endian::read32le(Ptr);
  return version == llvm::wasm::WasmVersion;
}

char ObjectFileWasm::ID;

void ObjectFileWasm::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                CreateMemoryInstance, GetModuleSpecifications);
}

void ObjectFileWasm::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectFile *
ObjectFileWasm::CreateInstance(const ModuleSP &module_sp, DataBufferSP data_sp,
                               offset_t data_offset, const FileSpec *file,
                               offset_t file_offset, offset_t length) {
  Log *log = GetLog(LLDBLog::Object);

  if (!data_sp) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp) {
      LLDB_LOGF(log, "Failed to create ObjectFileWasm instance for file %s",
                file->GetPath().c_str());
      return nullptr;
    }
    data_offset = 0;
  }

  assert(data_sp);
  if (!ValidateModuleHeader(data_sp)) {
    LLDB_LOGF(log,
              "Failed to create ObjectFileWasm instance: invalid Wasm header");
    return nullptr;
  }

  // Update the data to contain the entire file if it doesn't contain it
  // already.
  if (data_sp->GetByteSize() < length) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp) {
      LLDB_LOGF(log,
                "Failed to create ObjectFileWasm instance: cannot read file %s",
                file->GetPath().c_str());
      return nullptr;
    }
    data_offset = 0;
  }

  std::unique_ptr<ObjectFileWasm> objfile_up(new ObjectFileWasm(
      module_sp, data_sp, data_offset, file, file_offset, length));
  ArchSpec spec = objfile_up->GetArchitecture();
  if (spec && objfile_up->SetModulesArchitecture(spec)) {
    LLDB_LOGF(log,
              "%p ObjectFileWasm::CreateInstance() module = %p (%s), file = %s",
              static_cast<void *>(objfile_up.get()),
              static_cast<void *>(objfile_up->GetModule().get()),
              objfile_up->GetModule()->GetSpecificationDescription().c_str(),
              file ? file->GetPath().c_str() : "<NULL>");
    return objfile_up.release();
  }

  LLDB_LOGF(log, "Failed to create ObjectFileWasm instance");
  return nullptr;
}

ObjectFile *ObjectFileWasm::CreateMemoryInstance(const ModuleSP &module_sp,
                                                 WritableDataBufferSP data_sp,
                                                 const ProcessSP &process_sp,
                                                 addr_t header_addr) {
  if (!ValidateModuleHeader(data_sp))
    return nullptr;

  std::unique_ptr<ObjectFileWasm> objfile_up(
      new ObjectFileWasm(module_sp, data_sp, process_sp, header_addr));
  ArchSpec spec = objfile_up->GetArchitecture();
  if (spec && objfile_up->SetModulesArchitecture(spec))
    return objfile_up.release();
  return nullptr;
}

bool ObjectFileWasm::DecodeNextSection(lldb::offset_t *offset_ptr) {
  // Buffer sufficient to read a section header and find the pointer to the next
  // section.
  const uint32_t kBufferSize = 1024;
  DataExtractor section_header_data = ReadImageData(*offset_ptr, kBufferSize);

  llvm::DataExtractor data = section_header_data.GetAsLLVM();
  llvm::DataExtractor::Cursor c(0);

  // Each section consists of:
  // - a one-byte section id,
  // - the u32 size of the contents, in bytes,
  // - the actual contents.
  uint8_t section_id = data.getU8(c);
  uint64_t payload_len = data.getULEB128(c);
  if (!c)
    return !llvm::errorToBool(c.takeError());

  if (payload_len > std::numeric_limits<uint32_t>::max())
    return false;

  if (section_id == llvm::wasm::WASM_SEC_CUSTOM) {
    // Custom sections have the id 0. Their contents consist of a name
    // identifying the custom section, followed by an uninterpreted sequence
    // of bytes.
    lldb::offset_t prev_offset = c.tell();
    llvm::Expected<std::string> sect_name = GetWasmString(data, c);
    if (!sect_name) {
      LLDB_LOG_ERROR(GetLog(LLDBLog::Object), sect_name.takeError(),
                     "failed to parse section name: {0}");
      return false;
    }

    if (payload_len < c.tell() - prev_offset)
      return false;

    uint32_t section_length = payload_len - (c.tell() - prev_offset);
    m_sect_infos.push_back(section_info{*offset_ptr + c.tell(), section_length,
                                        section_id, ConstString(*sect_name)});
    *offset_ptr += (c.tell() + section_length);
  } else if (section_id <= llvm::wasm::WASM_SEC_LAST_KNOWN) {
    m_sect_infos.push_back(section_info{*offset_ptr + c.tell(),
                                        static_cast<uint32_t>(payload_len),
                                        section_id, ConstString()});
    *offset_ptr += (c.tell() + payload_len);
  } else {
    // Invalid section id.
    return false;
  }
  return true;
}

bool ObjectFileWasm::DecodeSections() {
  lldb::offset_t offset = kWasmHeaderSize;
  if (IsInMemory()) {
    offset += m_memory_addr;
  }

  while (DecodeNextSection(&offset))
    ;
  return true;
}

size_t ObjectFileWasm::GetModuleSpecifications(
    const FileSpec &file, DataBufferSP &data_sp, offset_t data_offset,
    offset_t file_offset, offset_t length, ModuleSpecList &specs) {
  if (!ValidateModuleHeader(data_sp)) {
    return 0;
  }

  ModuleSpec spec(file, ArchSpec("wasm32-unknown-unknown-wasm"));
  specs.Append(spec);
  return 1;
}

ObjectFileWasm::ObjectFileWasm(const ModuleSP &module_sp, DataBufferSP data_sp,
                               offset_t data_offset, const FileSpec *file,
                               offset_t offset, offset_t length)
    : ObjectFile(module_sp, file, offset, length, data_sp, data_offset),
      m_arch("wasm32-unknown-unknown-wasm") {
  m_data.SetAddressByteSize(4);
}

ObjectFileWasm::ObjectFileWasm(const lldb::ModuleSP &module_sp,
                               lldb::WritableDataBufferSP header_data_sp,
                               const lldb::ProcessSP &process_sp,
                               lldb::addr_t header_addr)
    : ObjectFile(module_sp, process_sp, header_addr, header_data_sp),
      m_arch("wasm32-unknown-unknown-wasm") {}

bool ObjectFileWasm::ParseHeader() {
  // We already parsed the header during initialization.
  return true;
}

struct WasmFunction {
  lldb::offset_t section_offset = LLDB_INVALID_OFFSET;
  uint32_t size = 0;
};

static llvm::Expected<std::vector<WasmFunction>>
ParseFunctions(DataExtractor &data) {
  lldb::offset_t offset = 0;

  llvm::Expected<uint32_t> function_count = GetULEB32(data, offset);
  if (!function_count)
    return function_count.takeError();

  std::vector<WasmFunction> functions;
  functions.reserve(*function_count);

  for (uint32_t i = 0; i < *function_count; ++i) {
    llvm::Expected<uint32_t> function_size = GetULEB32(data, offset);
    if (!function_size)
      return function_size.takeError();
    // llvm-objdump considers the ULEB with the function size to be part of the
    // function. We can't do that here because that would break symbolic
    // breakpoints, as that address is never executed.
    functions.push_back({offset, *function_size});

    std::optional<lldb::offset_t> next_offset =
        llvm::checkedAddUnsigned<lldb::offset_t>(offset, *function_size);
    if (!next_offset)
      return llvm::createStringError("function offset overflows 64 bits");
    offset = *next_offset;
  }

  return functions;
}

struct WasmSegment {
  enum SegmentType {
    Active,
    Passive,
  };

  std::string name;
  SegmentType type = Passive;
  lldb::offset_t section_offset = LLDB_INVALID_OFFSET;
  uint32_t size = 0;
  uint32_t memory_index = 0;
  lldb::offset_t init_expr_offset = 0;

  lldb::offset_t GetFileOffset() const { return section_offset & 0xffffffff; }
};

static llvm::Expected<std::vector<WasmSegment>> ParseData(DataExtractor &data) {
  lldb::offset_t offset = 0;

  llvm::Expected<uint32_t> segment_count = GetULEB32(data, offset);
  if (!segment_count)
    return segment_count.takeError();

  std::vector<WasmSegment> segments;
  segments.reserve(*segment_count);

  for (uint32_t i = 0; i < *segment_count; ++i) {
    llvm::Expected<uint32_t> flags = GetULEB32(data, offset);
    if (!flags)
      return flags.takeError();

    WasmSegment segment;

    // Data segments have a mode that identifies them as either passive or
    // active. An active data segment copies its contents into a memory during
    // instantiation, as specified by a memory index and a constant expression
    // defining an offset into that memory.
    segment.type = (*flags & llvm::wasm::WASM_DATA_SEGMENT_IS_PASSIVE)
                       ? WasmSegment::Passive
                       : WasmSegment::Active;

    if (*flags & llvm::wasm::WASM_DATA_SEGMENT_HAS_MEMINDEX) {
      assert(segment.type == WasmSegment::Active);
      llvm::Expected<uint32_t> memidx = GetULEB32(data, offset);
      if (!memidx)
        return memidx.takeError();
      segment.memory_index = *memidx;
    }

    if (segment.type == WasmSegment::Active)
      segment.init_expr_offset = GetWasmOffsetFromInitExpr(data, offset);

    llvm::Expected<uint32_t> segment_size = GetULEB32(data, offset);
    if (!segment_size)
      return segment_size.takeError();

    segment.section_offset = offset;
    segment.size = *segment_size;
    segments.push_back(segment);

    std::optional<lldb::offset_t> next_offset =
        llvm::checkedAddUnsigned<lldb::offset_t>(offset, *segment_size);
    if (!next_offset)
      return llvm::createStringError("segment offset overflows 64 bits");
    offset = *next_offset;
  }

  return segments;
}

static llvm::Expected<std::vector<Symbol>>
ParseNames(SectionSP code_section_sp, DataExtractor &name_data,
           const std::vector<WasmFunction> &functions,
           std::vector<WasmSegment> &segments) {

  llvm::DataExtractor data = name_data.GetAsLLVM();
  llvm::DataExtractor::Cursor c(0);
  std::vector<Symbol> symbols;
  while (c && c.tell() < data.size()) {
    const uint8_t type = data.getU8(c);
    llvm::Expected<uint32_t> size = GetULEB32(data, c);
    if (!size)
      return size.takeError();

    switch (type) {
    case llvm::wasm::WASM_NAMES_FUNCTION: {
      const uint64_t count = data.getULEB128(c);
      if (count > std::numeric_limits<uint32_t>::max())
        return llvm::createStringError("function count overflows uint32_t");

      for (uint64_t i = 0; c && i < count; ++i) {
        llvm::Expected<uint32_t> idx = GetULEB32(data, c);
        if (!idx)
          return idx.takeError();
        llvm::Expected<std::string> name = GetWasmString(data, c);
        if (!name)
          return name.takeError();
        if (*idx >= functions.size())
          continue;
        symbols.emplace_back(
            symbols.size(), *name, lldb::eSymbolTypeCode,
            /*external=*/false, /*is_debug=*/false, /*is_trampoline=*/false,
            /*is_artificial=*/false, code_section_sp,
            functions[i].section_offset, functions[i].size,
            /*size_is_valid=*/true, /*contains_linker_annotations=*/false,
            /*flags=*/0);
      }
    } break;
    case llvm::wasm::WASM_NAMES_DATA_SEGMENT: {
      llvm::Expected<uint32_t> count = GetULEB32(data, c);
      if (!count)
        return count.takeError();
      for (uint32_t i = 0; c && i < *count; ++i) {
        llvm::Expected<uint32_t> idx = GetULEB32(data, c);
        if (!idx)
          return idx.takeError();
        llvm::Expected<std::string> name = GetWasmString(data, c);
        if (!name)
          return name.takeError();
        if (*idx >= segments.size())
          continue;
        // Update the segment name.
        segments[i].name = *name;
      }

    } break;
    case llvm::wasm::WASM_NAMES_GLOBAL:
    case llvm::wasm::WASM_NAMES_LOCAL:
    default:
      std::optional<lldb::offset_t> offset =
          llvm::checkedAddUnsigned<lldb::offset_t>(c.tell(), *size);
      if (!offset)
        return llvm::createStringError("offset overflows 64 bits");
      c.seek(*offset);
    }
  }

  if (!c)
    return c.takeError();

  return symbols;
}

void ObjectFileWasm::ParseSymtab(Symtab &symtab) {
  for (const Symbol &symbol : m_symbols)
    symtab.AddSymbol(symbol);

  symtab.Finalize();
  m_symbols.clear();
}

static SectionType GetSectionTypeFromName(llvm::StringRef Name) {
  if (Name == "name")
    return lldb::eSectionTypeWasmName;
  if (Name.consume_front(".debug_") || Name.consume_front(".zdebug_"))
    return ObjectFile::GetDWARFSectionTypeFromName(Name);
  return eSectionTypeOther;
}

std::optional<ObjectFileWasm::section_info>
ObjectFileWasm::GetSectionInfo(uint32_t section_id) {
  for (const section_info &sect_info : m_sect_infos) {
    if (sect_info.id == section_id)
      return sect_info;
  }
  return std::nullopt;
}

std::optional<ObjectFileWasm::section_info>
ObjectFileWasm::GetSectionInfo(llvm::StringRef section_name) {
  for (const section_info &sect_info : m_sect_infos) {
    if (sect_info.name == section_name)
      return sect_info;
  }
  return std::nullopt;
}

void ObjectFileWasm::CreateSections(SectionList &unified_section_list) {
  Log *log = GetLog(LLDBLog::Object);

  if (m_sections_up)
    return;

  m_sections_up = std::make_unique<SectionList>();

  if (m_sect_infos.empty()) {
    DecodeSections();
  }

  for (const section_info &sect_info : m_sect_infos) {
    SectionType section_type = eSectionTypeOther;
    ConstString section_name;
    offset_t file_offset = sect_info.offset & 0xffffffff;
    addr_t vm_addr = sect_info.offset;
    size_t vm_size = sect_info.size;

    if (llvm::wasm::WASM_SEC_CODE == sect_info.id) {
      section_type = eSectionTypeCode;
      section_name = ConstString("code");

      // A code address in DWARF for WebAssembly is the offset of an
      // instruction relative within the Code section of the WebAssembly file.
      // For this reason Section::GetFileAddress() must return zero for the
      // Code section.
      vm_addr = 0;
    } else {
      section_type = GetSectionTypeFromName(sect_info.name.GetStringRef());
      if (section_type == eSectionTypeOther)
        continue;
      section_name = sect_info.name;
      if (!IsInMemory()) {
        vm_size = 0;
        vm_addr = 0;
      }
    }

    SectionSP section_sp = std::make_shared<Section>(
        GetModule(),    // Module to which this section belongs.
        this,           // ObjectFile to which this section belongs and
                        // should read section data from.
        section_type,   // Section ID.
        section_name,   // Section name.
        section_type,   // Section type.
        vm_addr,        // VM address.
        vm_size,        // VM size in bytes of this section.
        file_offset,    // Offset of this section in the file.
        sect_info.size, // Size of the section as found in the file.
        0,              // Alignment of the section
        0,              // Flags for this section.
        1);             // Number of host bytes per target byte
    m_sections_up->AddSection(section_sp);
    unified_section_list.AddSection(section_sp);
  }

  // The name section contains names and indexes. First parse the data from the
  // relevant sections so we can access it by its index.
  std::vector<WasmFunction> functions;
  std::vector<WasmSegment> segments;

  // Parse the code section.
  if (std::optional<section_info> info =
          GetSectionInfo(llvm::wasm::WASM_SEC_CODE)) {
    DataExtractor code_data = ReadImageData(info->offset, info->size);
    llvm::Expected<std::vector<WasmFunction>> maybe_functions =
        ParseFunctions(code_data);
    if (!maybe_functions) {
      LLDB_LOG_ERROR(log, maybe_functions.takeError(),
                     "Failed to parse Wasm code section: {0}");
    } else {
      functions = *maybe_functions;
    }
  }

  // Parse the data section.
  std::optional<section_info> data_info =
      GetSectionInfo(llvm::wasm::WASM_SEC_DATA);
  if (data_info) {
    DataExtractor data_data = ReadImageData(data_info->offset, data_info->size);
    llvm::Expected<std::vector<WasmSegment>> maybe_segments =
        ParseData(data_data);
    if (!maybe_segments) {
      LLDB_LOG_ERROR(log, maybe_segments.takeError(),
                     "Failed to parse Wasm data section: {0}");
    } else {
      segments = *maybe_segments;
    }
  }

  if (std::optional<section_info> info = GetSectionInfo("name")) {
    DataExtractor names_data = ReadImageData(info->offset, info->size);
    llvm::Expected<std::vector<Symbol>> symbols = ParseNames(
        m_sections_up->FindSectionByType(lldb::eSectionTypeCode, false),
        names_data, functions, segments);
    if (!symbols) {
      LLDB_LOG_ERROR(log, symbols.takeError(),
                     "Failed to parse Wasm names: {0}");
    } else {
      m_symbols = *symbols;
    }
  }

  lldb::user_id_t segment_id = 0;
  for (const WasmSegment &segment : segments) {
    if (segment.type == WasmSegment::Active) {
      // FIXME: Support segments with a memory index.
      if (segment.memory_index != 0) {
        LLDB_LOG(log, "Skipping segment {0}: non-zero memory index is "
                      "currently unsupported");
        continue;
      }

      if (segment.init_expr_offset == LLDB_INVALID_OFFSET) {
        LLDB_LOG(log, "Skipping segment {0}: unsupported init expression");
        continue;
      }
    }

    const lldb::addr_t file_vm_addr =
        segment.type == WasmSegment::Active
            ? segment.init_expr_offset
            : data_info->offset + segment.section_offset;
    const lldb::offset_t file_offset =
        data_info->GetFileOffset() + segment.GetFileOffset();
    SectionSP segment_sp = std::make_shared<Section>(
        GetModule(),
        /*obj_file=*/this,
        ++segment_id << 8, // 1-based segment index, shifted by 8 bits to avoid
                           // collision with section IDs.
        ConstString(segment.name), eSectionTypeData,
        /*file_vm_addr=*/file_vm_addr,
        /*vm_size=*/segment.size,
        /*file_offset=*/file_offset,
        /*file_size=*/segment.size,
        /*log2align=*/0, /*flags=*/0);
    m_sections_up->AddSection(segment_sp);
    GetModule()->GetSectionList()->AddSection(segment_sp);
  }
}

bool ObjectFileWasm::SetLoadAddress(Target &target, lldb::addr_t load_address,
                                    bool value_is_offset) {
  /// In WebAssembly, linear memory is disjointed from code space. The VM can
  /// load multiple instances of a module, which logically share the same code.
  /// We represent a wasm32 code address with 64-bits, like:
  /// 63            32 31             0
  /// +---------------+---------------+
  /// +   module_id   |     offset    |
  /// +---------------+---------------+
  /// where the lower 32 bits represent a module offset (relative to the module
  /// start not to the beginning of the code section) and the higher 32 bits
  /// uniquely identify the module in the WebAssembly VM.
  /// In other words, we assume that each WebAssembly module is loaded by the
  /// engine at a 64-bit address that starts at the boundary of 4GB pages, like
  /// 0x0000000400000000 for module_id == 4.
  /// These 64-bit addresses will be used to request code ranges for a specific
  /// module from the WebAssembly engine.

  assert(m_memory_addr == LLDB_INVALID_ADDRESS ||
         m_memory_addr == load_address);

  ModuleSP module_sp = GetModule();
  if (!module_sp)
    return false;

  DecodeSections();

  size_t num_loaded_sections = 0;
  SectionList *section_list = GetSectionList();
  if (!section_list)
    return false;

  const size_t num_sections = section_list->GetSize();
  for (size_t sect_idx = 0; sect_idx < num_sections; ++sect_idx) {
    SectionSP section_sp(section_list->GetSectionAtIndex(sect_idx));
    if (target.SetSectionLoadAddress(
            section_sp, load_address | section_sp->GetFileOffset())) {
      ++num_loaded_sections;
    }
  }

  return num_loaded_sections > 0;
}

DataExtractor ObjectFileWasm::ReadImageData(offset_t offset, uint32_t size) {
  DataExtractor data;
  if (m_file) {
    if (offset < GetByteSize()) {
      size = std::min(static_cast<uint64_t>(size), GetByteSize() - offset);
      auto buffer_sp = MapFileData(m_file, size, offset);
      return DataExtractor(buffer_sp, GetByteOrder(), GetAddressByteSize());
    }
  } else {
    ProcessSP process_sp(m_process_wp.lock());
    if (process_sp) {
      auto data_up = std::make_unique<DataBufferHeap>(size, 0);
      Status readmem_error;
      size_t bytes_read = process_sp->ReadMemory(
          offset, data_up->GetBytes(), data_up->GetByteSize(), readmem_error);
      if (bytes_read > 0) {
        DataBufferSP buffer_sp(data_up.release());
        data.SetData(buffer_sp, 0, buffer_sp->GetByteSize());
      }
    } else if (offset < m_data.GetByteSize()) {
      size =
          std::min(static_cast<uint64_t>(size), m_data.GetByteSize() - offset);
      return DataExtractor(m_data.GetDataStart() + offset, size, GetByteOrder(),
                           GetAddressByteSize());
    }
  }
  data.SetByteOrder(GetByteOrder());
  return data;
}

std::optional<FileSpec> ObjectFileWasm::GetExternalDebugInfoFileSpec() {
  static ConstString g_sect_name_external_debug_info("external_debug_info");

  for (const section_info &sect_info : m_sect_infos) {
    if (g_sect_name_external_debug_info == sect_info.name) {
      const uint32_t kBufferSize = 1024;
      DataExtractor section_header_data =
          ReadImageData(sect_info.offset, kBufferSize);

      llvm::DataExtractor data = section_header_data.GetAsLLVM();
      llvm::DataExtractor::Cursor c(0);
      llvm::Expected<std::string> symbols_url = GetWasmString(data, c);
      if (!symbols_url) {
        llvm::consumeError(symbols_url.takeError());
        return std::nullopt;
      }
      return FileSpec(*symbols_url);
    }
  }
  return std::nullopt;
}

void ObjectFileWasm::Dump(Stream *s) {
  ModuleSP module_sp(GetModule());
  if (!module_sp)
    return;

  std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());

  llvm::raw_ostream &ostream = s->AsRawOstream();
  ostream << static_cast<void *>(this) << ": ";
  s->Indent();
  ostream << "ObjectFileWasm, file = '";
  m_file.Dump(ostream);
  ostream << "', arch = ";
  ostream << GetArchitecture().GetArchitectureName() << "\n";

  SectionList *sections = GetSectionList();
  if (sections) {
    sections->Dump(s->AsRawOstream(), s->GetIndentLevel(), nullptr, true,
                   UINT32_MAX);
  }
  ostream << "\n";
  DumpSectionHeaders(ostream);
  ostream << "\n";
}

void ObjectFileWasm::DumpSectionHeader(llvm::raw_ostream &ostream,
                                       const section_info &sh) {
  ostream << llvm::left_justify(sh.name.GetStringRef(), 16) << " "
          << llvm::format_hex(sh.offset, 10) << " "
          << llvm::format_hex(sh.size, 10) << " " << llvm::format_hex(sh.id, 6)
          << "\n";
}

void ObjectFileWasm::DumpSectionHeaders(llvm::raw_ostream &ostream) {
  ostream << "Section Headers\n";
  ostream << "IDX  name             addr       size       id\n";
  ostream << "==== ---------------- ---------- ---------- ------\n";

  uint32_t idx = 0;
  for (auto pos = m_sect_infos.begin(); pos != m_sect_infos.end();
       ++pos, ++idx) {
    ostream << "[" << llvm::format_decimal(idx, 2) << "] ";
    ObjectFileWasm::DumpSectionHeader(ostream, *pos);
  }
}
