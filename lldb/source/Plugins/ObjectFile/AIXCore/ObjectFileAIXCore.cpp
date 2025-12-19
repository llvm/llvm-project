//===-- ObjectFileAIXCore.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ObjectFileAIXCore.h"

#include <algorithm>
#include <cassert>
#include <unordered_map>
#include <string.h>

#include "lldb/Utility/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBufferHeap.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Stream.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/XCOFF.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Object/XCOFFObjectFile.h"

using namespace llvm;
using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(ObjectFileAIXCore)

enum CoreVersion : uint64_t {AIXCORE32 = 0xFEEDDB1, AIXCORE64 = 0xFEEDDB2};

bool m_is_core = false;

// Static methods.
void ObjectFileAIXCore::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(), CreateInstance,
                                CreateMemoryInstance, GetModuleSpecifications);
}

void ObjectFileAIXCore::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

ObjectFile *ObjectFileAIXCore::CreateInstance(const lldb::ModuleSP &module_sp,
                                          DataExtractorSP extractor_sp,
                                          lldb::offset_t data_offset,
                                          const lldb_private::FileSpec *file,
                                          lldb::offset_t file_offset,
                                          lldb::offset_t length) {

  if(m_is_core)
  {

      bool mapped_writable = false;
      if (!extractor_sp || !extractor_sp->HasData()) {
          DataBufferSP data_sp = MapFileDataWritable(*file, length, file_offset);
          if (!data_sp)
              return nullptr;
          data_offset = 0;
          mapped_writable = true;
          extractor_sp = std::make_shared<lldb_private::DataExtractor>(data_sp);
      }

      //assert(data_sp);

      // Update the data to contain the entire file if it doesn't already
      if (extractor_sp->GetByteSize() < length) {
          DataBufferSP data_sp = MapFileDataWritable(*file, length, file_offset);
          if (!data_sp)
              return nullptr;
          data_offset = 0;
          mapped_writable = true;
      }

      // If we didn't map the data as writable take ownership of the buffer.
      if (!mapped_writable) {
          DataBufferSP data_sp = std::make_shared<DataBufferHeap>(data_sp->GetBytes(),
                  data_sp->GetByteSize());
          data_offset = 0;
      }

      std::unique_ptr<ObjectFileAIXCore> objfile_up(new ObjectFileAIXCore(
                  module_sp, extractor_sp, data_offset, file, file_offset, length));
      ArchSpec spec = objfile_up->GetArchitecture();
      objfile_up->SetModulesArchitecture(spec);
      return objfile_up.release();

  }
}

ObjectFile *ObjectFileAIXCore::CreateMemoryInstance(
    const lldb::ModuleSP &module_sp, WritableDataBufferSP data_sp,
    const lldb::ProcessSP &process_sp, lldb::addr_t header_addr) {
  return nullptr;
}

size_t ObjectFileAIXCore::GetModuleSpecifications(
    const lldb_private::FileSpec &file, lldb::DataBufferSP &data_sp,
    lldb::offset_t data_offset, lldb::offset_t file_offset,
    lldb::offset_t length, lldb_private::ModuleSpecList &specs) {
  const size_t initial_count = specs.GetSize();

  if (ObjectFileAIXCore::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize())) {
    // Need new ArchType???
    ArchSpec arch_spec = ArchSpec(eArchTypeXCOFF, XCOFF::TCPU_PPC64, LLDB_INVALID_CPUTYPE);
    ModuleSpec spec(file, arch_spec);
    spec.GetArchitecture().SetArchitecture(eArchTypeXCOFF, XCOFF::TCPU_PPC64, LLDB_INVALID_CPUTYPE, llvm::Triple::AIX);
    specs.Append(spec);
  }
  return specs.GetSize() - initial_count;
}

static bool AIXCoreHeaderCheckFromMagic(uint32_t magic) {

    Log *log = GetLog(LLDBLog::Modules);
    bool ret = false;
    switch (magic) {
        case AIXCORE32: 
        case AIXCORE64:
            m_is_core = true;
            ret = true; 
            break;
        default:
            break;
    }
    return ret;
}

bool ObjectFileAIXCore::MagicBytesMatch(DataBufferSP &data_sp,
                                    lldb::addr_t data_offset,
                                    lldb::addr_t data_length) {
  lldb_private::DataExtractor data; 
  data.SetData(data_sp, data_offset, data_length);
  lldb::offset_t offset = 0;
  offset += 4; // Skipping to the coredump version
  uint32_t magic = data.GetU32(&offset);
  return AIXCoreHeaderCheckFromMagic(magic);
}

bool ObjectFileAIXCore::ParseHeader() {

  return false;
}

ByteOrder ObjectFileAIXCore::GetByteOrder() const {
  return eByteOrderBig;
}

bool ObjectFileAIXCore::IsExecutable() const {
  return false;
}

uint32_t ObjectFileAIXCore::GetAddressByteSize() const {
    return 8;
}

AddressClass ObjectFileAIXCore::GetAddressClass(addr_t file_addr) {
  return AddressClass::eUnknown;
}

lldb::SymbolType ObjectFileAIXCore::MapSymbolType(llvm::object::SymbolRef::Type sym_type) {
  if (sym_type == llvm::object::SymbolRef::ST_Function)
    return lldb::eSymbolTypeCode;
  else if (sym_type == llvm::object::SymbolRef::ST_Data)
    return lldb::eSymbolTypeData;
  return lldb::eSymbolTypeInvalid;
}

void ObjectFileAIXCore::ParseSymtab(Symtab &lldb_symtab) {
}

bool ObjectFileAIXCore::IsStripped() {
  return false;
}

void ObjectFileAIXCore::CreateSections(SectionList &unified_section_list) {
}

void ObjectFileAIXCore::Dump(Stream *s) {
}

ArchSpec ObjectFileAIXCore::GetArchitecture() {
  ArchSpec arch_spec = ArchSpec(eArchTypeXCOFF, XCOFF::TCPU_PPC64, LLDB_INVALID_CPUTYPE);
  return arch_spec;
}

UUID ObjectFileAIXCore::GetUUID() {
  return UUID();
}

uint32_t ObjectFileAIXCore::GetDependentModules(FileSpecList &files) {
  
    auto original_size = files.GetSize();
    return files.GetSize() - original_size;
}

Address ObjectFileAIXCore::GetImageInfoAddress(Target *target) {
  return Address();
}

lldb_private::Address ObjectFileAIXCore::GetBaseAddress() {
  return lldb_private::Address();
}
ObjectFile::Type ObjectFileAIXCore::CalculateType() {
  return eTypeCoreFile;
}

ObjectFile::Strata ObjectFileAIXCore::CalculateStrata() {
  return eStrataUnknown;
}

std::vector<ObjectFile::LoadableData>
ObjectFileAIXCore::GetLoadableData(Target &target) {
  std::vector<LoadableData> loadables;
  return loadables;
}

lldb::WritableDataBufferSP
ObjectFileAIXCore::MapFileDataWritable(const FileSpec &file, uint64_t Size,
                                   uint64_t Offset) {
  return FileSystem::Instance().CreateWritableDataBuffer(file.GetPath(), Size,
                                                         Offset);
}

ObjectFileAIXCore::ObjectFileAIXCore(const lldb::ModuleSP &module_sp,
                             DataExtractorSP extractor_sp, lldb::offset_t data_offset,
                             const FileSpec *file, lldb::offset_t file_offset,
                             lldb::offset_t length)
    : ObjectFile(module_sp, file, file_offset, length, extractor_sp, data_offset)
      {
  if (file)
    m_file = *file;
}

ObjectFileAIXCore::ObjectFileAIXCore(const lldb::ModuleSP &module_sp,
                             DataBufferSP header_data_sp,
                             const lldb::ProcessSP &process_sp,
                             addr_t header_addr)
    : ObjectFile(module_sp, process_sp, header_addr, 
                 std::make_shared<lldb_private::DataExtractor>(header_data_sp))
      {
}
