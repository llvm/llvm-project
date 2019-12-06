//===-- ObjectFileWasm.h ---------------------------------- -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_OBJECTFILE_WASM_OBJECTFILEWASM_H
#define LLDB_PLUGINS_OBJECTFILE_WASM_OBJECTFILEWASM_H

#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/ArchSpec.h"
#include "llvm/Support/MD5.h"

namespace lldb_private {
namespace wasm {

class ObjectFileWASM : public ObjectFile {
public:
  // Static Functions
  static void Initialize();
  static void Terminate();

  static ConstString GetPluginNameStatic();
  static const char *GetPluginDescriptionStatic() {
    return "WebAssembly object file reader.";
  }

  static ObjectFile *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const FileSpec *file,
                 lldb::offset_t file_offset, lldb::offset_t length);

  static ObjectFile *CreateMemoryInstance(const lldb::ModuleSP &module_sp,
                                          lldb::DataBufferSP &data_sp,
                                          const lldb::ProcessSP &process_sp,
                                          lldb::addr_t header_addr);

  static size_t GetModuleSpecifications(const FileSpec &file,
                                        lldb::DataBufferSP &data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        ModuleSpecList &specs);

  // PluginInterface protocol
  ConstString GetPluginName() override { return GetPluginNameStatic(); }

  uint32_t GetPluginVersion() override { return 1; }

  // ObjectFile Protocol.

  bool ParseHeader() override;

  lldb::ByteOrder GetByteOrder() const override {
    return m_arch.GetByteOrder();
  }

  bool IsExecutable() const override { return false; }

  uint32_t GetAddressByteSize() const override {
    return m_arch.GetAddressByteSize();
  }

  AddressClass GetAddressClass(lldb::addr_t file_addr) override {
    return AddressClass::eInvalid;
  }

  Symtab *GetSymtab() override;

  bool IsStripped() override { return true; }

  void CreateSections(SectionList &unified_section_list) override;

  void Dump(Stream *s) override;

  ArchSpec GetArchitecture() override { return m_arch; }

  UUID GetUUID() override { return m_uuid; }

  uint32_t GetDependentModules(FileSpecList &files) override { return 0; }

  Type CalculateType() override { return eTypeExecutable; }

  Strata CalculateStrata() override { return eStrataUser; }

  bool SetLoadAddress(lldb_private::Target &target, lldb::addr_t value,
                      bool value_is_offset) override;

  lldb_private::Address GetBaseAddress() override {
    return Address(m_memory_addr + m_code_section_offset);
  }

private:
  ObjectFileWASM(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const FileSpec *file,
                 lldb::offset_t offset, lldb::offset_t length);
  ObjectFileWASM(const lldb::ModuleSP &module_sp,
                 lldb::DataBufferSP &header_data_sp,
                 const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

  static bool GetVaruint7(DataExtractor &section_header_data,
                          lldb::offset_t *offset_ptr, uint8_t *result);
  static bool GetVaruint32(DataExtractor &section_header_data,
                           lldb::offset_t *offset_ptr, uint32_t *result);

  bool DecodeNextSection(lldb::offset_t *offset_ptr);
  bool DecodeSections(lldb::addr_t load_address);

  DataExtractor ReadImageData(uint64_t offset, size_t size);

  typedef struct section_info {
    lldb::offset_t offset;
    uint32_t size;
    uint32_t id;
    std::string name;
  } section_info_t;

  void DumpSectionHeader(Stream *s, const section_info_t &sh);
  void DumpSectionHeaders(Stream *s);

  typedef std::vector<section_info_t> SectionInfoColl;
  typedef SectionInfoColl::iterator SectionInfoCollIter;
  typedef SectionInfoColl::const_iterator SectionInfoCollConstIter;
  SectionInfoColl m_sect_infos;

  ArchSpec m_arch;
  llvm::MD5 m_hash;
  UUID m_uuid;
  ConstString m_symbols_url;
  uint32_t m_code_section_offset;
};

} // namespace wasm
} // namespace lldb_private
#endif // LLDB_PLUGINS_OBJECTFILE_WASM_OBJECTFILEWASM_H
