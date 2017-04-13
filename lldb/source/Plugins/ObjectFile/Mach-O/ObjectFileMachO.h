//===-- ObjectFileMachO.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileMachO_h_
#define liblldb_ObjectFileMachO_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/Address.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/RangeMap.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/SafeMachO.h"
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
                       lldb_private::Error &error);

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

  lldb::AddressClass GetAddressClass(lldb::addr_t file_addr) override;

  lldb_private::Symtab *GetSymtab() override;

  bool IsStripped() override;

  void CreateSections(lldb_private::SectionList &unified_section_list) override;

  void Dump(lldb_private::Stream *s) override;

  bool GetArchitecture(lldb_private::ArchSpec &arch) override;

  bool GetUUID(lldb_private::UUID *uuid) override;

  uint32_t GetDependentModules(lldb_private::FileSpecList &files) override;

  lldb_private::FileSpecList GetReExportedLibraries() override {
    return m_reexported_dylibs;
  }

  lldb_private::Address GetEntryPointAddress() override;

  lldb_private::Address GetHeaderAddress() override;

  uint32_t GetNumThreadContexts() override;

  std::string GetIdentifierString() override;

  bool GetCorefileMainBinaryInfo (lldb::addr_t &address, lldb_private::UUID &uuid) override;

  lldb::RegisterContextSP
  GetThreadContextAtIndex(uint32_t idx, lldb_private::Thread &thread) override;

  ObjectFile::Type CalculateType() override;

  ObjectFile::Strata CalculateStrata() override;

  uint32_t GetVersion(uint32_t *versions, uint32_t num_versions) override;

  uint32_t GetMinimumOSVersion(uint32_t *versions,
                               uint32_t num_versions) override;

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
  static bool
  GetUUID(const llvm::MachO::mach_header &header,
          const lldb_private::DataExtractor &data,
          lldb::offset_t lc_offset, // Offset to the first load command
          lldb_private::UUID &uuid);

  static bool GetArchitecture(const llvm::MachO::mach_header &header,
                              const lldb_private::DataExtractor &data,
                              lldb::offset_t lc_offset,
                              lldb_private::ArchSpec &arch);

  // Intended for same-host arm device debugging where lldb needs to
  // detect libraries in the shared cache and augment the nlist entries
  // with an on-disk dyld_shared_cache file.  The process will record
  // the shared cache UUID so the on-disk cache can be matched or rejected
  // correctly.
  lldb_private::UUID GetProcessSharedCacheUUID(lldb_private::Process *);

  // Intended for same-host arm device debugging where lldb will read
  // shared cache libraries out of its own memory instead of the remote
  // process' memory as an optimization.  If lldb's shared cache UUID
  // does not match the process' shared cache UUID, this optimization
  // should not be used.
  lldb_private::UUID GetLLDBSharedCacheUUID();

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

  llvm::MachO::mach_header m_header;
  static const lldb_private::ConstString &GetSegmentNameTEXT();
  static const lldb_private::ConstString &GetSegmentNameDATA();
  static const lldb_private::ConstString &GetSegmentNameDATA_DIRTY();
  static const lldb_private::ConstString &GetSegmentNameDATA_CONST();
  static const lldb_private::ConstString &GetSegmentNameOBJC();
  static const lldb_private::ConstString &GetSegmentNameLINKEDIT();
  static const lldb_private::ConstString &GetSectionNameEHFrame();

  llvm::MachO::dysymtab_command m_dysymtab;
  std::vector<llvm::MachO::segment_command_64> m_mach_segments;
  std::vector<llvm::MachO::section_64> m_mach_sections;
  std::vector<uint32_t> m_min_os_versions;
  std::vector<uint32_t> m_sdk_versions;
  typedef lldb_private::RangeVector<uint32_t, uint32_t> FileRangeArray;
  lldb_private::Address m_entry_point_address;
  FileRangeArray m_thread_context_offsets;
  bool m_thread_context_offsets_valid;
  lldb_private::FileSpecList m_reexported_dylibs;
  bool m_allow_assembly_emulation_unwind_plans;
};

#endif // liblldb_ObjectFileMachO_h_
