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

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

  // ObjectFile Protocol.
  bool ParseHeader() override;

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

  uint32_t GetDependentModules(lldb_private::FileSpecList &files) override;

  ObjectFile::Type CalculateType() override;

  ObjectFile::Strata CalculateStrata() override;

  ObjectFileXCOFF(const lldb::ModuleSP &module_sp, lldb::DataBufferSP data_sp,
                  lldb::offset_t data_offset,
                  const lldb_private::FileSpec *file, lldb::offset_t offset,
                  lldb::offset_t length);

  ObjectFileXCOFF(const lldb::ModuleSP &module_sp,
                  lldb::DataBufferSP header_data_sp,
                  const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

protected:
  static lldb::WritableDataBufferSP
  MapFileDataWritable(const lldb_private::FileSpec &file, uint64_t Size,
                      uint64_t Offset);

private:
  bool CreateBinary();

  std::unique_ptr<llvm::object::XCOFFObjectFile> m_binary;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_XCOFF_OBJECTFILE_H
