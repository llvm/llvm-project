//===-- ObjectFileXCOFF.cpp
//-------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectFileXCOFF.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/FileSpecList.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/Utility/Status.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <unordered_map>

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectFileXCOFF)
// FIXME: target 64bit at this moment.

// Static methods.
void ObjectFileXCOFF::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                CreateMemoryInstance, GetModuleSpecifications);
}

void ObjectFileXCOFF::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectFile *ObjectFileXCOFF::CreateInstance(const lldb::ModuleSP &module_sp,
                                            DataBufferSP data_sp,
                                            lldb::offset_t data_offset,
                                            const lldb_private::FileSpec *file,
                                            lldb::offset_t file_offset,
                                            lldb::offset_t length) {
  if (!data_sp) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }
  if (!ObjectFileXCOFF::MagicBytesMatch(data_sp, data_offset, length))
    return nullptr;
  // Update the data to contain the entire file if it doesn't already
  if (data_sp->GetByteSize() < length) {
    data_sp = MapFileData(*file, length, file_offset);
    if (!data_sp)
      return nullptr;
    data_offset = 0;
  }
  auto objfile_up = std::make_unique<ObjectFileXCOFF>(
      module_sp, data_sp, data_offset, file, file_offset, length);
  if (!objfile_up)
    return nullptr;

  // Cache xcoff binary.
  if (!objfile_up->CreateBinary())
    return nullptr;

  if (!objfile_up->ParseHeader())
    return nullptr;

  return objfile_up.release();
}

bool ObjectFileXCOFF::CreateBinary() {
  if (m_binary)
    return true;

  Log *log = GetLog(LLDBLog::Object);

  auto memory_ref = llvm::MemoryBufferRef(toStringRef(m_data.GetData()),
                                          m_file.GetFilename().GetStringRef());
  llvm::file_magic magic = llvm::identify_magic(memory_ref.getBuffer());

  auto binary = llvm::object::ObjectFile::createObjectFile(memory_ref, magic);
  if (!binary) {
    LLDB_LOG_ERROR(log, binary.takeError(),
                   "Failed to create binary for file ({1}): {0}", m_file);
    return false;
  }
  // Make sure we only handle XCOFF format.
  m_binary =
      llvm::unique_dyn_cast<llvm::object::XCOFFObjectFile>(std::move(*binary));
  if (!m_binary)
    return false;

  LLDB_LOG(log, "this = {0}, module = {1} ({2}), file = {3}, binary = {4}",
           this, GetModule().get(), GetModule()->GetSpecificationDescription(),
           m_file.GetPath(), m_binary.get());

  return true;
}

ObjectFile *ObjectFileXCOFF::CreateMemoryInstance(
    const lldb::ModuleSP &module_sp, WritableDataBufferSP data_sp,
    const lldb::ProcessSP &process_sp, lldb::addr_t header_addr) {
  return nullptr;
}

size_t ObjectFileXCOFF::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t length, lldb_private::ModuleSpecList &specs) {
  const size_t initial_count = specs.GetSize();

  if (ObjectFileXCOFF::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize())) {
    ArchSpec arch_spec =
        ArchSpec(eArchTypeXCOFF, XCOFF::TCPU_PPC64, LLDB_INVALID_CPUTYPE);
    ModuleSpec spec(file, arch_spec);
    spec.GetArchitecture().SetArchitecture(eArchTypeXCOFF, XCOFF::TCPU_PPC64,
                                           LLDB_INVALID_CPUTYPE,
                                           llvm::Triple::AIX);
    specs.Append(spec);
  }
  return specs.GetSize() - initial_count;
}

static uint32_t XCOFFHeaderSizeFromMagic(uint32_t magic) {
  switch (magic) {
  case XCOFF::XCOFF32:
    return sizeof(struct llvm::object::XCOFFFileHeader32);
    break;
  case XCOFF::XCOFF64:
    return sizeof(struct llvm::object::XCOFFFileHeader64);
    break;

  default:
    break;
  }
  return 0;
}

bool ObjectFileXCOFF::MagicBytesMatch(DataBufferSP &data_sp,
                                      lldb::addr_t data_offset,
                                      lldb::addr_t data_length) {
  lldb_private::DataExtractor data;
  data.SetData(data_sp, data_offset, data_length);
  // Need to set this as XCOFF is only compatible with Big Endian
  data.SetByteOrder(eByteOrderBig);
  lldb::offset_t offset = 0;
  uint16_t magic = data.GetU16(&offset);
  return XCOFFHeaderSizeFromMagic(magic) != 0;
}

bool ObjectFileXCOFF::ParseHeader() {
  if (m_binary->is64Bit())
    return m_binary->fileHeader64()->Magic == XCOFF::XCOFF64;
  return m_binary->fileHeader32()->Magic == XCOFF::XCOFF32;
}

ByteOrder ObjectFileXCOFF::GetByteOrder() const { return eByteOrderBig; }

bool ObjectFileXCOFF::IsExecutable() const { return true; }

uint32_t ObjectFileXCOFF::GetAddressByteSize() const {
  if (m_binary->is64Bit())
    return 8;
  return 4;
}

AddressClass ObjectFileXCOFF::GetAddressClass(addr_t file_addr) {
  return AddressClass::eUnknown;
}

static lldb::SymbolType MapSymbolType(llvm::object::SymbolRef::Type sym_type) {
  switch (sym_type) {
  case llvm::object::SymbolRef::ST_Function:
    return lldb::eSymbolTypeCode;
  case llvm::object::SymbolRef::ST_Data:
    return lldb::eSymbolTypeData;
  case llvm::object::SymbolRef::ST_File:
    return lldb::eSymbolTypeSourceFile;
  default:
    return lldb::eSymbolTypeInvalid;
  }
}

void ObjectFileXCOFF::ParseSymtab(Symtab &lldb_symtab) {
  Log *log = GetLog(LLDBLog::Object);
  SectionList *sectionList = GetSectionList();

  for (const auto &symbol_ref : m_binary->symbols()) {
    llvm::object::XCOFFSymbolRef xcoff_sym_ref(symbol_ref);

    llvm::Expected<llvm::StringRef> name_or_err = xcoff_sym_ref.getName();
    if (!name_or_err) {
      LLDB_LOG_ERROR(log, name_or_err.takeError(),
                     "Unable to extract name from the xcoff symbol ref object");
      continue;
    }

    llvm::StringRef symbolName = name_or_err.get();
    // Remove the . prefix added during compilation. This prefix is usually
    // added to differentiate between reference to the code and function
    // descriptor. For instance, Adding .func will only allow user to put bp on
    // .func, which is not known to the user, instead of func.
    llvm::StringRef name_no_dot =
        symbolName.starts_with(".") ? symbolName.drop_front() : symbolName;
    auto storageClass = xcoff_sym_ref.getStorageClass();
    // C_HIDEXT symbols are not needed to be exposed, with the exception of TOC
    // which is responsible for storing references to global data
    if (storageClass == XCOFF::C_HIDEXT && symbolName != "TOC") {

      // Zero or muliple aux entries may suggest ambiguous data
      if (xcoff_sym_ref.getNumberOfAuxEntries() != 1)
        continue;

      auto aux_csect_or_err = xcoff_sym_ref.getXCOFFCsectAuxRef();
      if (!aux_csect_or_err) {
        LLDB_LOG_ERROR(log, aux_csect_or_err.takeError(),
                       "Unable to access xcoff csect aux ref object");
        continue;
      }

      const llvm::object::XCOFFCsectAuxRef csect_aux = aux_csect_or_err.get();

      // Only add hidden ext entries which come under Program Code, skip others
      // as they are not useful as debugging data.
      if (csect_aux.getStorageMappingClass() != XCOFF::XMC_PR)
        continue;

      // This does not apply to 32-bit,
      // Only add csect symbols identified by the aux entry, as they are
      // needed to reference section information. Skip others
      if (m_binary->is64Bit())
        if (csect_aux.getAuxType64() != XCOFF::AUX_CSECT)
          continue;
    }

    Symbol symbol;
    symbol.GetMangled().SetValue(ConstString(name_no_dot));

    int16_t sectionNumber = xcoff_sym_ref.getSectionNumber();
    // Note that XCOFF section headers are numbered from 1 and not 0.
    size_t sectionIndex = static_cast<size_t>(sectionNumber - 1);
    if (sectionNumber > 0) {
      if (sectionIndex < sectionList->GetSize()) {

        lldb::SectionSP section_sp =
            sectionList->GetSectionAtIndex(sectionIndex);
        if (!section_sp || section_sp->GetFileAddress() == LLDB_INVALID_ADDRESS)
          continue;

        lldb::addr_t file_addr = section_sp->GetFileAddress();
        lldb::addr_t symbolValue = xcoff_sym_ref.getValue();
        if (symbolValue < file_addr)
          continue;

        symbol.GetAddressRef() = Address(section_sp, symbolValue - file_addr);
      }
    }

    Expected<llvm::object::SymbolRef::Type> sym_type_or_err =
        symbol_ref.getType();
    if (!sym_type_or_err) {
      LLDB_LOG_ERROR(log, sym_type_or_err.takeError(),
                     "Unable to access xcoff symbol type");
      continue;
    }

    symbol.SetType(MapSymbolType(sym_type_or_err.get()));

    lldb_symtab.AddSymbol(symbol);
  }
}

bool ObjectFileXCOFF::IsStripped() { return false; }

void ObjectFileXCOFF::CreateSections(SectionList &unified_section_list) {

  if (m_sections_up)
    return;

  m_sections_up = std::make_unique<SectionList>();
  if (m_binary->is64Bit())
    CreateSectionsWithBitness<XCOFF64>(unified_section_list);
  else
    CreateSectionsWithBitness<XCOFF32>(unified_section_list);
}

template <typename T>
static auto GetSections(llvm::object::XCOFFObjectFile *binary) {
  if constexpr (T::Is64Bit)
    return binary->sections64();
  else
    return binary->sections32();
}

template <typename T>
void ObjectFileXCOFF::CreateSectionsWithBitness(
    SectionList &unified_section_list) {
  ModuleSP module_sp(GetModule());
  if (!module_sp)
    return;

  std::lock_guard<std::recursive_mutex> guard(module_sp->GetMutex());

  int idx = 0;
  for (const typename T::SectionHeader &section :
       GetSections<T>(m_binary.get())) {

    ConstString const_sect_name(section.Name);

    SectionType section_type = lldb::eSectionTypeOther;
    if (section.Flags & XCOFF::STYP_TEXT)
      section_type = eSectionTypeCode;
    else if (section.Flags & XCOFF::STYP_DATA)
      section_type = eSectionTypeData;
    else if (section.Flags & XCOFF::STYP_BSS)
      section_type = eSectionTypeZeroFill;
    else if (section.Flags & XCOFF::STYP_DWARF) {
      section_type = llvm::StringSwitch<SectionType>(section.Name)
                         .Case(".dwinfo", eSectionTypeDWARFDebugInfo)
                         .Case(".dwline", eSectionTypeDWARFDebugLine)
                         .Case(".dwabrev", eSectionTypeDWARFDebugAbbrev)
                         .Case(".dwrnges", eSectionTypeDWARFDebugRanges)
                         .Default(eSectionTypeInvalid);
    }

    SectionSP section_sp(new Section(
        module_sp, this, ++idx, const_sect_name, section_type,
        section.VirtualAddress, section.SectionSize,
        section.FileOffsetToRawData, section.SectionSize, 0, section.Flags));

    uint32_t permissions = ePermissionsReadable;
    if (section.Flags & (XCOFF::STYP_DATA | XCOFF::STYP_BSS))
      permissions |= ePermissionsWritable;
    if (section.Flags & XCOFF::STYP_TEXT)
      permissions |= ePermissionsExecutable;

    section_sp->SetPermissions(permissions);
    m_sections_up->AddSection(section_sp);
    unified_section_list.AddSection(section_sp);
  }
}

void ObjectFileXCOFF::Dump(Stream *s) {}

ArchSpec ObjectFileXCOFF::GetArchitecture() {
  ArchSpec arch_spec =
      ArchSpec(eArchTypeXCOFF, XCOFF::TCPU_PPC64, LLDB_INVALID_CPUTYPE);
  return arch_spec;
}

UUID ObjectFileXCOFF::GetUUID() { return UUID(); }

uint32_t ObjectFileXCOFF::GetDependentModules(FileSpecList &files) { return 0; }

ObjectFile::Type ObjectFileXCOFF::CalculateType() {

  const auto flags = m_binary->is64Bit() ? m_binary->fileHeader64()->Flags
                                         : m_binary->fileHeader32()->Flags;

  if (flags & XCOFF::F_EXEC)
    return eTypeExecutable;
  else if (flags & XCOFF::F_SHROBJ)
    return eTypeSharedLibrary;
  return eTypeUnknown;
}

ObjectFile::Strata ObjectFileXCOFF::CalculateStrata() { return eStrataUnknown; }

lldb::WritableDataBufferSP
ObjectFileXCOFF::MapFileDataWritable(const FileSpec &file, uint64_t Size,
                                     uint64_t Offset) {
  return FileSystem::Instance().CreateWritableDataBuffer(file.GetPath(), Size,
                                                         Offset);
}

ObjectFileXCOFF::ObjectFileXCOFF(const lldb::ModuleSP &module_sp,
                                 DataBufferSP data_sp,
                                 lldb::offset_t data_offset,
                                 const FileSpec *file,
                                 lldb::offset_t file_offset,
                                 lldb::offset_t length)
    : ObjectFile(module_sp, file, file_offset, length, data_sp, data_offset) {
  if (file)
    m_file = *file;
}

ObjectFileXCOFF::ObjectFileXCOFF(const lldb::ModuleSP &module_sp,
                                 DataBufferSP header_data_sp,
                                 const lldb::ProcessSP &process_sp,
                                 addr_t header_addr)
    : ObjectFile(module_sp, process_sp, header_addr, header_data_sp) {}
