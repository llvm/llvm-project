//===-- ObjectFileMachO.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileMachO_h_
#define liblldb_ObjectFileMachO_h_

#include "lldb/Core/Address.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Host/SafeMachO.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/UUID.h"

//----------------------------------------------------------------------
// This class needs to be hidden as eventually belongs in a plugin that
// will export the ObjectFile protocol
//----------------------------------------------------------------------
class ObjectFileMachO : public lldb_private::ObjectFile {
public:
  ObjectFileMachO(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                  lldb::offset_t data_offset,
                  const lldb_private::FileSpec *file, lldb::offset_t offset,
                  lldb::offset_t length);

  ObjectFileMachO(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                  const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

  ~ObjectFileMachO() override = default;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static lldb_private::ObjectFile *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                 lldb::offset_t file_offset, lldb::offset_t length);

  static lldb_private::ObjectFile *CreateMemoryInstance(
      const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
      const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

  static size_t GetModuleSpecifications(const lldb_private::FileSpec &file,
                                        lldb::DataBufferSP &data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        lldb_private::ModuleSpecList &specs);

  static bool SaveCore(const lldb::ProcessSP &process_sp,
                       const lldb_private::FileSpec &outfile,
                       lldb_private::Status &error);

  static bool MagicBytesMatch(lldb::DataBufferSP &data_sp, lldb::addr_t offset,
                              lldb::addr_t length);

  //------------------------------------------------------------------
  // Member Functions
  //------------------------------------------------------------------
  bool ParseHeader() override;

  bool SetLoadAddress(lldb_private::Target &target, lldb::addr_t value,
                      bool value_is_offset) override;

  lldb::ByteOrder GetByteOrder() const override;

  bool IsExecutable() const override;

  uint32_t GetAddressByteSize() const override;

  lldb_private::AddressClass GetAddressClass(lldb::addr_t file_addr) override;

  lldb_private::Symtab *GetSymtab() override;

  bool IsStripped() override;

  void CreateSections(lldb_private::SectionList &unified_section_list) override;

  void Dump(lldb_private::Stream *s) override;

  lldb_private::ArchSpec GetArchitecture() override;

  lldb_private::UUID GetUUID() override;

  uint32_t GetDependentModules(lldb_private::FileSpecList &files) override;

  lldb_private::FileSpecList GetReExportedLibraries() override {
    return m_reexported_dylibs;
  }

  lldb_private::Address GetEntryPointAddress() override;

  lldb_private::Address GetBaseAddress() override;

  uint32_t GetNumThreadContexts() override;

  std::string GetIdentifierString() override;

  bool GetCorefileMainBinaryInfo (lldb::addr_t &address, lldb_private::UUID &uuid) override;

  lldb::RegisterContextSP
  GetThreadContextAtIndex(uint32_t idx, lldb_private::Thread &thread) override;

  ObjectFile::Type CalculateType() override;

  ObjectFile::Strata CalculateStrata() override;

  llvm::VersionTuple GetVersion() override;

  llvm::VersionTuple GetMinimumOSVersion() override;

  uint32_t GetSDKVersion(uint32_t *versions, uint32_t num_versions) override;

  bool GetIsDynamicLinkEditor() override;

  static bool ParseHeader(lldb_private::DataExtractor &data,
                          lldb::offset_t *data_offset_ptr,
                          llvm::MachO::mach_header &header);

  bool AllowAssemblyEmulationUnwindPlans() override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

protected:
  static lldb_private::UUID
  GetUUID(const llvm::MachO::mach_header &header,
          const lldb_private::DataExtractor &data,
          lldb::offset_t lc_offset); // Offset to the first load command

  static lldb_private::ArchSpec
  GetArchitecture(const llvm::MachO::mach_header &header,
                  const lldb_private::DataExtractor &data,
                  lldb::offset_t lc_offset);

  // Intended for same-host arm device debugging where lldb needs to
  // detect libraries in the shared cache and augment the nlist entries
  // with an on-disk dyld_shared_cache file.  The process will record
  // the shared cache UUID so the on-disk cache can be matched or rejected
  // correctly.
  void GetProcessSharedCacheUUID(lldb_private::Process *, lldb::addr_t &base_addr, lldb_private::UUID &uuid);

  // Intended for same-host arm device debugging where lldb will read
  // shared cache libraries out of its own memory instead of the remote
  // process' memory as an optimization.  If lldb's shared cache UUID
  // does not match the process' shared cache UUID, this optimization
  // should not be used.
  void GetLLDBSharedCacheUUID(lldb::addr_t &base_addir, lldb_private::UUID &uuid);

  lldb_private::Section *GetMachHeaderSection();

  lldb::addr_t CalculateSectionLoadAddressForMemoryImage(
      lldb::addr_t mach_header_load_address,
      const lldb_private::Section *mach_header_section,
      const lldb_private::Section *section);

  lldb_private::UUID
  GetSharedCacheUUID(lldb_private::FileSpec dyld_shared_cache,
                     const lldb::ByteOrder byte_order,
                     const uint32_t addr_byte_size);

  size_t ParseSymtab();

  typedef lldb_private::RangeArray<uint32_t, uint32_t, 8> EncryptedFileRanges;
  EncryptedFileRanges GetEncryptedFileRanges();

  struct SegmentParsingContext;
  void ProcessDysymtabCommand(const llvm::MachO::load_command &load_cmd,
                              lldb::offset_t offset);
  void ProcessSegmentCommand(const llvm::MachO::load_command &load_cmd,
                             lldb::offset_t offset, uint32_t cmd_idx,
                             SegmentParsingContext &context);
  void SanitizeSegmentCommand(llvm::MachO::segment_command_64 &seg_cmd,
                              uint32_t cmd_idx);

  bool SectionIsLoadable(const lldb_private::Section *section);

  llvm::MachO::mach_header m_header;
  static lldb_private::ConstString GetSegmentNameTEXT();
  static lldb_private::ConstString GetSegmentNameDATA();
  static lldb_private::ConstString GetSegmentNameDATA_DIRTY();
  static lldb_private::ConstString GetSegmentNameDATA_CONST();
  static lldb_private::ConstString GetSegmentNameOBJC();
  static lldb_private::ConstString GetSegmentNameLINKEDIT();
  static lldb_private::ConstString GetSegmentNameDWARF();
  static lldb_private::ConstString GetSectionNameEHFrame();

  llvm::MachO::dysymtab_command m_dysymtab;
  std::vector<llvm::MachO::segment_command_64> m_mach_segments;
  std::vector<llvm::MachO::section_64> m_mach_sections;
  llvm::Optional<llvm::VersionTuple> m_min_os_version;
  std::vector<uint32_t> m_sdk_versions;
  typedef lldb_private::RangeVector<uint32_t, uint32_t> FileRangeArray;
  lldb_private::Address m_entry_point_address;
  FileRangeArray m_thread_context_offsets;
  bool m_thread_context_offsets_valid;
  lldb_private::FileSpecList m_reexported_dylibs;
  bool m_allow_assembly_emulation_unwind_plans;
};

#endif // liblldb_ObjectFileMachO_h_
