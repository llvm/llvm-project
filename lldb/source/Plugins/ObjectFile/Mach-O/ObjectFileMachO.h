//===-- ObjectFileMachO.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_MACH_O_OBJECTFILEMACHO_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_MACH_O_OBJECTFILEMACHO_H

#include "lldb/Core/Address.h"
#include "lldb/Host/SafeMachO.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SaveCoreOptions.h"
#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/FileSpecList.h"
#include "lldb/Utility/RangeMap.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/UUID.h"
#include <optional>

// This class needs to be hidden as eventually belongs in a plugin that
// will export the ObjectFile protocol
class ObjectFileMachO : public lldb_private::ObjectFile {
public:
  ObjectFileMachO(const lldb::ModuleSP &module_sp, lldb::DataBufferSP data_sp,
                  lldb::offset_t data_offset,
                  const lldb_private::FileSpec *file, lldb::offset_t offset,
                  lldb::offset_t length);

  ObjectFileMachO(const lldb::ModuleSP &module_sp,
                  lldb::WritableDataBufferSP data_sp,
                  const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

  ~ObjectFileMachO() override = default;

  // Static Functions
  static void Initialize();

  static void Terminate();

  static llvm::StringRef GetPluginNameStatic() { return "mach-o"; }

  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Mach-o object file reader (32 and 64 bit)";
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

  static bool SaveCore(const lldb::ProcessSP &process_sp,
                       lldb_private::SaveCoreOptions &options,
                       lldb_private::Status &error);

  static bool MagicBytesMatch(lldb::DataBufferSP data_sp, lldb::addr_t offset,
                              lldb::addr_t length);

  // LLVM RTTI support
  static char ID;
  bool isA(const void *ClassID) const override {
    return ClassID == &ID || ObjectFile::isA(ClassID);
  }
  static bool classof(const ObjectFile *obj) { return obj->isA(&ID); }

  // Member Functions
  bool ParseHeader() override;

  bool SetLoadAddress(lldb_private::Target &target, lldb::addr_t value,
                      bool value_is_offset) override;

  lldb::ByteOrder GetByteOrder() const override;

  bool IsExecutable() const override;

  bool IsDynamicLoader() const;

  bool IsSharedCacheBinary() const;

  bool IsKext() const;

  uint32_t GetAddressByteSize() const override;

  lldb_private::AddressClass GetAddressClass(lldb::addr_t file_addr) override;

  void ParseSymtab(lldb_private::Symtab &symtab) override;

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

  std::vector<std::tuple<lldb::offset_t, lldb::offset_t>>
  FindLC_NOTEByName(std::string name);

  std::string GetIdentifierString() override;

  lldb_private::AddressableBits GetAddressableBits() override;

  bool GetCorefileMainBinaryInfo(lldb::addr_t &value, bool &value_is_offset,
                                 lldb_private::UUID &uuid,
                                 ObjectFile::BinaryType &type) override;

  bool GetCorefileThreadExtraInfos(std::vector<lldb::tid_t> &tids) override;

  lldb_private::StructuredData::ObjectSP GetCorefileProcessMetadata() override;

  bool LoadCoreFileImages(lldb_private::Process &process) override;

  lldb::RegisterContextSP
  GetThreadContextAtIndex(uint32_t idx, lldb_private::Thread &thread) override;

  ObjectFile::Type CalculateType() override;

  ObjectFile::Strata CalculateStrata() override;

  llvm::VersionTuple GetVersion() override;

  llvm::VersionTuple GetMinimumOSVersion() override;

  llvm::VersionTuple GetSDKVersion() override;

  bool GetIsDynamicLinkEditor() override;

  bool CanTrustAddressRanges() override;

  static bool ParseHeader(lldb_private::DataExtractor &data,
                          lldb::offset_t *data_offset_ptr,
                          llvm::MachO::mach_header &header);

  bool AllowAssemblyEmulationUnwindPlans() override;

  lldb_private::Section *GetMachHeaderSection();

  bool IsGOTSection(const lldb_private::Section &section) const override;

  // PluginInterface protocol
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }

protected:
  static lldb_private::UUID
  GetUUID(const llvm::MachO::mach_header &header,
          const lldb_private::DataExtractor &data,
          lldb::offset_t lc_offset); // Offset to the first load command

  static lldb_private::ArchSpec GetArchitecture(
      lldb::ModuleSP module_sp, const llvm::MachO::mach_header &header,
      const lldb_private::DataExtractor &data, lldb::offset_t lc_offset);

  /// Enumerate all ArchSpecs supported by this Mach-O file.
  ///
  /// On macOS one Mach-O slice can contain multiple load commands:
  /// One load command for being loaded into a macOS process and one
  /// load command for being loaded into a macCatalyst process. In
  /// contrast to ObjectContainerUniversalMachO, this is the same
  /// binary that can be loaded into different contexts.
  static void GetAllArchSpecs(const llvm::MachO::mach_header &header,
                              const lldb_private::DataExtractor &data,
                              lldb::offset_t lc_offset,
                              lldb_private::ModuleSpec &base_spec,
                              lldb_private::ModuleSpecList &all_specs);

  /// Intended for same-host arm device debugging where lldb needs to
  /// detect libraries in the shared cache and augment the nlist entries
  /// with an on-disk dyld_shared_cache file.  The process will record
  /// the shared cache UUID so the on-disk cache can be matched or rejected
  /// correctly.
  void GetProcessSharedCacheUUID(lldb_private::Process *,
                                 lldb::addr_t &base_addr,
                                 lldb_private::UUID &uuid);

  /// Intended for same-host arm device debugging where lldb will read
  /// shared cache libraries out of its own memory instead of the remote
  /// process' memory as an optimization.  If lldb's shared cache UUID
  /// does not match the process' shared cache UUID, this optimization
  /// should not be used.
  void GetLLDBSharedCacheUUID(lldb::addr_t &base_addir, lldb_private::UUID &uuid);

  lldb::addr_t CalculateSectionLoadAddressForMemoryImage(
      lldb::addr_t mach_header_load_address,
      const lldb_private::Section *mach_header_section,
      const lldb_private::Section *section);

  lldb_private::UUID
  GetSharedCacheUUID(lldb_private::FileSpec dyld_shared_cache,
                     const lldb::ByteOrder byte_order,
                     const uint32_t addr_byte_size);

  size_t ParseSymtab();

  typedef lldb_private::RangeVector<uint32_t, uint32_t, 8> EncryptedFileRanges;
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

  /// A corefile may include metadata about all of the binaries that were
  /// present in the process when the corefile was taken.  This is only
  /// implemented for Mach-O files for now; we'll generalize it when we
  /// have other systems that can include the same.
  struct MachOCorefileImageEntry {
    std::string filename;
    lldb_private::UUID uuid;
    lldb::addr_t load_address = LLDB_INVALID_ADDRESS;
    lldb::addr_t slide = 0;
    bool currently_executing = false;
    std::vector<std::tuple<lldb_private::ConstString, lldb::addr_t>>
        segment_load_addresses;
  };

  struct LCNoteEntry {
    LCNoteEntry(uint32_t addr_byte_size, lldb::ByteOrder byte_order)
        : payload(lldb_private::Stream::eBinary, addr_byte_size, byte_order) {}

    std::string name;
    lldb::addr_t payload_file_offset = 0;
    lldb_private::StreamString payload;
  };

  struct MachOCorefileAllImageInfos {
    std::vector<MachOCorefileImageEntry> all_image_infos;
    bool IsValid() { return all_image_infos.size() > 0; }
  };

  // The LC_SYMTAB's symtab_command structure uses 32-bit file offsets
  // for two fields, but ObjectFileMachO needs to calculate the offsets
  // in virtual address layout from the start of the TEXT segment, and
  // that span may be larger than 4GB.
  struct SymtabCommandLargeOffsets {
    SymtabCommandLargeOffsets() {}
    SymtabCommandLargeOffsets(const llvm::MachO::symtab_command &in)
        : cmd(in.cmd), cmdsize(in.cmdsize), symoff(in.symoff), nsyms(in.nsyms),
          stroff(in.stroff), strsize(in.strsize) {}
    void operator=(const llvm::MachO::symtab_command &in) {
      cmd = in.cmd;
      cmdsize = in.cmdsize;
      symoff = in.symoff;
      nsyms = in.nsyms;
      stroff = in.stroff;
      strsize = in.strsize;
    }
    uint32_t cmd = 0;          /* LC_SYMTAB */
    uint32_t cmdsize = 0;      /* sizeof(struct symtab_command) */
    lldb::offset_t symoff = 0; /* symbol table offset */
    uint32_t nsyms = 0;        /* number of symbol table entries */
    lldb::offset_t stroff = 0; /* string table offset */
    uint32_t strsize = 0;      /* string table size in bytes */
  };

  // The LC_DYLD_INFO's dyld_info_command has 32-bit file offsets
  // that we will use as virtual address offsets, and may need to span
  // more than 4GB in virtual memory.
  struct DyldInfoCommandLargeOffsets {
    DyldInfoCommandLargeOffsets() {}
    DyldInfoCommandLargeOffsets(const llvm::MachO::dyld_info_command &in)
        : cmd(in.cmd), cmdsize(in.cmdsize), rebase_off(in.rebase_off),
          rebase_size(in.rebase_size), bind_off(in.bind_off),
          bind_size(in.bind_size), weak_bind_off(in.weak_bind_off),
          weak_bind_size(in.weak_bind_size), lazy_bind_off(in.lazy_bind_off),
          lazy_bind_size(in.lazy_bind_size), export_off(in.export_off),
          export_size(in.export_size) {}

    void operator=(const llvm::MachO::dyld_info_command &in) {
      cmd = in.cmd;
      cmdsize = in.cmdsize;
      rebase_off = in.rebase_off;
      rebase_size = in.rebase_size;
      bind_off = in.bind_off;
      bind_size = in.bind_size;
      weak_bind_off = in.weak_bind_off;
      weak_bind_size = in.weak_bind_size;
      lazy_bind_off = in.lazy_bind_off;
      lazy_bind_size = in.lazy_bind_size;
      export_off = in.export_off;
      export_size = in.export_size;
    };

    /// LC_DYLD_INFO or LC_DYLD_INFO_ONLY
    uint32_t cmd = 0;
    uint32_t cmdsize = 0;             /* sizeof(struct dyld_info_command) */
    lldb::offset_t rebase_off = 0;    /* file offset to rebase info  */
    uint32_t rebase_size = 0;         /* size of rebase info   */
    lldb::offset_t bind_off = 0;      /* file offset to binding info   */
    uint32_t bind_size = 0;           /* size of binding info  */
    lldb::offset_t weak_bind_off = 0; /* file offset to weak binding info   */
    uint32_t weak_bind_size = 0;      /* size of weak binding info  */
    lldb::offset_t lazy_bind_off = 0; /* file offset to lazy binding info */
    uint32_t lazy_bind_size = 0;      /* size of lazy binding infs */
    lldb::offset_t export_off = 0;    /* file offset to lazy binding info */
    uint32_t export_size = 0;         /* size of lazy binding infs */
  };

  /// The LC_DYSYMTAB's dysymtab_command has 32-bit file offsets
  /// that we will use as virtual address offsets, and may need to span
  /// more than 4GB in virtual memory.
  struct DysymtabCommandLargeOffsets {
    DysymtabCommandLargeOffsets() {}
    DysymtabCommandLargeOffsets(const llvm::MachO::dysymtab_command &in)
        : cmd(in.cmd), cmdsize(in.cmdsize), ilocalsym(in.ilocalsym),
          nlocalsym(in.nlocalsym), iextdefsym(in.iextdefsym),
          nextdefsym(in.nextdefsym), iundefsym(in.iundefsym),
          nundefsym(in.nundefsym), tocoff(in.tocoff), ntoc(in.ntoc),
          modtaboff(in.modtaboff), nmodtab(in.nmodtab),
          extrefsymoff(in.extrefsymoff), nextrefsyms(in.nextrefsyms),
          indirectsymoff(in.indirectsymoff), nindirectsyms(in.nindirectsyms),
          extreloff(in.extreloff), nextrel(in.nextrel), locreloff(in.locreloff),
          nlocrel(in.nlocrel) {}

    void operator=(const llvm::MachO::dysymtab_command &in) {
      cmd = in.cmd;
      cmdsize = in.cmdsize;
      ilocalsym = in.ilocalsym;
      nlocalsym = in.nlocalsym;
      iextdefsym = in.iextdefsym;
      nextdefsym = in.nextdefsym;
      iundefsym = in.iundefsym;
      nundefsym = in.nundefsym;
      tocoff = in.tocoff;
      ntoc = in.ntoc;
      modtaboff = in.modtaboff;
      nmodtab = in.nmodtab;
      extrefsymoff = in.extrefsymoff;
      nextrefsyms = in.nextrefsyms;
      indirectsymoff = in.indirectsymoff;
      nindirectsyms = in.nindirectsyms;
      extreloff = in.extreloff;
      nextrel = in.nextrel;
      locreloff = in.locreloff;
      nlocrel = in.nlocrel;
    };

    uint32_t cmd = 0;             /* LC_DYSYMTAB */
    uint32_t cmdsize = 0;         /* sizeof(struct dysymtab_command) */
    uint32_t ilocalsym = 0;       /* index to local symbols */
    uint32_t nlocalsym = 0;       /* number of local symbols */
    uint32_t iextdefsym = 0;      /* index to externally defined symbols */
    uint32_t nextdefsym = 0;      /* number of externally defined symbols */
    uint32_t iundefsym = 0;       /* index to undefined symbols */
    uint32_t nundefsym = 0;       /* number of undefined symbols */
    lldb::offset_t tocoff = 0;    /* file offset to table of contents */
    uint32_t ntoc = 0;            /* number of entries in table of contents */
    lldb::offset_t modtaboff = 0; /* file offset to module table */
    uint32_t nmodtab = 0;         /* number of module table entries */
    lldb::offset_t extrefsymoff = 0; /* offset to referenced symbol table */
    uint32_t nextrefsyms = 0; /* number of referenced symbol table entries */
    lldb::offset_t indirectsymoff =
        0;                        /* file offset to the indirect symbol table */
    uint32_t nindirectsyms = 0;   /* number of indirect symbol table entries */
    lldb::offset_t extreloff = 0; /* offset to external relocation entries */
    uint32_t nextrel = 0;         /* number of external relocation entries */
    lldb::offset_t locreloff = 0; /* offset to local relocation entries */
    uint32_t nlocrel = 0;         /* number of local relocation entries */
  };

  // The linkedit_data_command is used in several load commands including
  // LC_FUNCTION_STARTS and LC_DYLD_EXPORTS_TRIE.  It has a 32-bit file offset
  // that may need to span more than 4GB in real virtual addresses.
  struct LinkeditDataCommandLargeOffsets {
    LinkeditDataCommandLargeOffsets() {}
    LinkeditDataCommandLargeOffsets(
        const llvm::MachO::linkedit_data_command &in)
        : cmd(in.cmd), cmdsize(in.cmdsize), dataoff(in.dataoff),
          datasize(in.datasize) {}
    void operator=(const llvm::MachO::linkedit_data_command &in) {
      cmd = in.cmd;
      cmdsize = in.cmdsize;
      dataoff = in.dataoff;
      datasize = in.datasize;
    }
    uint32_t cmd = 0;     /* LC_FUNCTION_STARTS, LC_DYLD_EXPORTS_TRIE, etc */
    uint32_t cmdsize = 0; /* sizeof(struct linkedit_data_command) */
    lldb::offset_t dataoff = 0; /* file offset of data in __LINKEDIT segment */
    uint32_t datasize = 0;      /* file size of data in __LINKEDIT segment  */
  };

  /// Get the list of binary images that were present in the process
  /// when the corefile was produced.
  /// \return
  ///     The MachOCorefileAllImageInfos object returned will have
  ///     IsValid() == false if the information is unavailable.
  MachOCorefileAllImageInfos GetCorefileAllImageInfos();

  llvm::MachO::mach_header m_header;
  static lldb_private::ConstString GetSegmentNameTEXT();
  static lldb_private::ConstString GetSegmentNameDATA();
  static lldb_private::ConstString GetSegmentNameDATA_DIRTY();
  static lldb_private::ConstString GetSegmentNameDATA_CONST();
  static lldb_private::ConstString GetSegmentNameOBJC();
  static lldb_private::ConstString GetSegmentNameLINKEDIT();
  static lldb_private::ConstString GetSegmentNameDWARF();
  static lldb_private::ConstString GetSegmentNameLLVM_COV();
  static lldb_private::ConstString GetSectionNameEHFrame();
  static lldb_private::ConstString GetSectionNameLLDBNoNlist();

  llvm::MachO::dysymtab_command m_dysymtab;
  std::vector<llvm::MachO::section_64> m_mach_sections;
  std::optional<llvm::VersionTuple> m_min_os_version;
  std::optional<llvm::VersionTuple> m_sdk_versions;
  typedef lldb_private::RangeVector<uint32_t, uint32_t> FileRangeArray;
  lldb_private::Address m_entry_point_address;
  FileRangeArray m_thread_context_offsets;
  lldb::offset_t m_linkedit_original_offset = 0;
  lldb::addr_t m_text_address = LLDB_INVALID_ADDRESS;
  bool m_thread_context_offsets_valid;
  lldb_private::FileSpecList m_reexported_dylibs;
  bool m_allow_assembly_emulation_unwind_plans;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_MACH_O_OBJECTFILEMACHO_H
