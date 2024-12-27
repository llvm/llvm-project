//===-- ObjectFileXCOFF.h --------------------------------------- -*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_XCOFF_OBJECTFILEXCOFF_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_XCOFF_OBJECTFILEXCOFF_H

#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/UUID.h"
#include "lldb/lldb-private.h"
#include "llvm/Object/XCOFFObjectFile.h"
#include <cstdint>
#include <vector>

/// \class ObjectFileXCOFF
/// Generic XCOFF object file reader.
///
/// This class provides a generic XCOFF (32/64 bit) reader plugin implementing
/// the ObjectFile protocol.
class ObjectFileXCOFF : public lldb_private::ObjectFile {
public:
  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "xcoff"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "XCOFF object file reader.";
  }

  static lldb_private::ObjectFile *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP data_sp,
                 lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                 lldb::offset_t file_offset, lldb::offset_t length);

  static lldb_private::ObjectFile *CreateMemoryInstance(
      const lldb::ModuleSP &module_sp, lldb::WritableDataBufferSP data_sp,
      const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

  static size_t GetModuleSpecifications(const lldb_private::FileSpec &file,
                                        lldb::DataBufferSP &data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        lldb_private::ModuleSpecList &specs);

  static bool MagicBytesMatch(lldb::DataBufferSP &data_sp, lldb::addr_t offset,
                              lldb::addr_t length);

  static lldb::SymbolType MapSymbolType(llvm::object::SymbolRef::Type sym_type);

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // LLVM RTTI support
  static char ID;
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || ObjectFile::isA(ClassID);
  }
  static bool classof(const ObjectFile *obj) { return obj->isA(&ID); }

  // ObjectFile Protocol.
  bool ParseHeader() override;

  bool SetLoadAddress(lldb_private::Target &target, lldb::addr_t value,
                      bool value_is_offset) override;

  bool SetLoadAddressByType(lldb_private::Target &target, lldb::addr_t value,
                              bool value_is_offset, int type_id) override;

  lldb::ByteOrder GetByteOrder() const override;

  bool IsExecutable() const override;

  uint32_t GetAddressByteSize() const override;

  lldb_private::AddressClass GetAddressClass(lldb::addr_t file_addr) override;

  void ParseSymtab(lldb_private::Symtab &symtab) override;

  bool IsStripped() override;

  void CreateSections(lldb_private::SectionList &unified_section_list) override;

  void Dump(lldb_private::Stream *s) override;

  lldb_private::ArchSpec GetArchitecture() override;

  lldb_private::UUID GetUUID() override;

  /// Return the contents of the .gnu_debuglink section, if the object file
  /// contains it.
  std::optional<lldb_private::FileSpec> GetDebugLink();

  uint32_t GetDependentModules(lldb_private::FileSpecList &files) override;

  lldb_private::Address
  GetImageInfoAddress(lldb_private::Target *target) override;

  lldb_private::Address GetEntryPointAddress() override;

  lldb_private::Address GetBaseAddress() override;

  ObjectFile::Type CalculateType() override;

  ObjectFile::Strata CalculateStrata() override;

  llvm::StringRef
  StripLinkerSymbolAnnotations(llvm::StringRef symbol_name) const override;

  void RelocateSection(lldb_private::Section *section) override;

  lldb_private::DataExtractor ReadImageData(uint32_t offset, size_t size);

  ObjectFileXCOFF(const lldb::ModuleSP &module_sp, lldb::DataBufferSP data_sp,
                lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                lldb::offset_t offset, lldb::offset_t length);

  ObjectFileXCOFF(const lldb::ModuleSP &module_sp,
                lldb::DataBufferSP header_data_sp,
                const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

protected:

  typedef struct xcoff_header {
    uint16_t magic;
    uint16_t nsects;
    uint32_t modtime;
    uint64_t symoff;
    uint32_t nsyms;
    uint16_t auxhdrsize;
    uint16_t flags;
  } xcoff_header_t;

  typedef struct xcoff_aux_header {
    uint16_t AuxMagic;
    uint16_t Version;
    uint32_t ReservedForDebugger;
    uint64_t TextStartAddr;
    uint64_t DataStartAddr;
    uint64_t TOCAnchorAddr;
    uint16_t SecNumOfEntryPoint;
    uint16_t SecNumOfText;
    uint16_t SecNumOfData;
    uint16_t SecNumOfTOC;
    uint16_t SecNumOfLoader;
    uint16_t SecNumOfBSS;
    uint16_t MaxAlignOfText;
    uint16_t MaxAlignOfData;
    uint16_t ModuleType;
    uint8_t CpuFlag;
    uint8_t CpuType;
    uint8_t TextPageSize;
    uint8_t DataPageSize;
    uint8_t StackPageSize;
    uint8_t FlagAndTDataAlignment;
    uint64_t TextSize;
    uint64_t InitDataSize;
    uint64_t BssDataSize;
    uint64_t EntryPointAddr;
    uint64_t MaxStackSize;
    uint64_t MaxDataSize;
    uint16_t SecNumOfTData;
    uint16_t SecNumOfTBSS;
    uint16_t XCOFF64Flag;
  } xcoff_aux_header_t;

  typedef struct section_header {
    char name[8];
    uint64_t phyaddr; // Physical Addr
    uint64_t vmaddr;  // Virtual Addr
    uint64_t size;    // Section size
    uint64_t offset;  // File offset to raw data
    uint64_t reloff;  // Offset to relocations
    uint64_t lineoff; // Offset to line table entries
    uint32_t nreloc;  // Number of relocation entries
    uint32_t nline;   // Number of line table entries
    uint32_t flags;
  } section_header_t;

  typedef struct xcoff_symbol {
    uint64_t value;
    uint32_t offset;
    uint16_t sect;
    uint16_t type;
    uint8_t storage;
    uint8_t naux;
  } xcoff_symbol_t;

  typedef struct xcoff_sym_csect_aux_entry {
    uint32_t section_or_len_low_byte;
    uint32_t parameter_hash_index;
    uint16_t type_check_sect_num;
    uint8_t symbol_alignment_and_type;
    uint8_t storage_mapping_class;
    uint32_t section_or_len_high_byte;
    uint8_t pad;
    uint8_t aux_type;
  } xcoff_sym_csect_aux_entry_t;

  static bool ParseXCOFFHeader(lldb_private::DataExtractor &data,
                              lldb::offset_t *offset_ptr,
                              xcoff_header_t &xcoff_header);
  bool ParseXCOFFOptionalHeader(lldb_private::DataExtractor &data,
                                lldb::offset_t *offset_ptr);
  bool ParseSectionHeaders(uint32_t offset);

  std::vector<LoadableData>
  GetLoadableData(lldb_private::Target &target) override;

  static lldb::WritableDataBufferSP
  MapFileDataWritable(const lldb_private::FileSpec &file, uint64_t Size,
                      uint64_t Offset);
  llvm::StringRef GetSectionName(const section_header_t &sect);
  static lldb::SectionType GetSectionType(llvm::StringRef sect_name,
                                          const section_header_t &sect);

  uint32_t ParseDependentModules();
  typedef std::vector<section_header_t> SectionHeaderColl;

private:
  bool CreateBinary();

  xcoff_header_t m_xcoff_header;
  xcoff_aux_header_t m_xcoff_aux_header;
  SectionHeaderColl m_sect_headers;
  std::unique_ptr<llvm::object::XCOFFObjectFile> m_binary;
  lldb_private::Address m_entry_point_address;
  std::optional<lldb_private::FileSpecList> m_deps_filespec;
  std::map<std::string, std::vector<std::string>> m_deps_base_members;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_XCOFF_OBJECTFILE_H
