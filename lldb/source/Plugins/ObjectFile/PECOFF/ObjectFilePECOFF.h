//===-- ObjectFilePECOFF.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFilePECOFF_h_
#define liblldb_ObjectFilePECOFF_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/Symbol/ObjectFile.h"

class ObjectFilePECOFF : public lldb_private::ObjectFile {
public:
  typedef enum MachineType {
    MachineUnknown = 0x0,
    MachineAm33 = 0x1d3,
    MachineAmd64 = 0x8664,
    MachineArm = 0x1c0,
    MachineArmNt = 0x1c4,
    MachineArm64 = 0xaa64,
    MachineEbc = 0xebc,
    MachineX86 = 0x14c,
    MachineIA64 = 0x200,
    MachineM32R = 0x9041,
    MachineMips16 = 0x266,
    MachineMipsFpu = 0x366,
    MachineMipsFpu16 = 0x466,
    MachinePowerPc = 0x1f0,
    MachinePowerPcfp = 0x1f1,
    MachineR4000 = 0x166,
    MachineSh3 = 0x1a2,
    MachineSh3dsp = 0x1a3,
    MachineSh4 = 0x1a6,
    MachineSh5 = 0x1a8,
    MachineThumb = 0x1c2,
    MachineWcemIpsv2 = 0x169
  } MachineType;

  ObjectFilePECOFF(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                   lldb::offset_t data_offset,
                   const lldb_private::FileSpec *file,
                   lldb::offset_t file_offset, lldb::offset_t length);

  ObjectFilePECOFF(const lldb::ModuleSP &module_sp,
                   lldb::DataBufferSP &header_data_sp,
                   const lldb::ProcessSP &process_sp, lldb::addr_t header_addr);

  ~ObjectFilePECOFF() override;

  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();

  static void Terminate();

  static lldb_private::ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic();

  static ObjectFile *
  CreateInstance(const lldb::ModuleSP &module_sp, lldb::DataBufferSP &data_sp,
                 lldb::offset_t data_offset, const lldb_private::FileSpec *file,
                 lldb::offset_t offset, lldb::offset_t length);

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

  static bool MagicBytesMatch(lldb::DataBufferSP &data_sp);

  static lldb::SymbolType MapSymbolType(uint16_t coff_symbol_type);

  bool ParseHeader() override;

  bool SetLoadAddress(lldb_private::Target &target, lldb::addr_t value,
                      bool value_is_offset) override;

  lldb::ByteOrder GetByteOrder() const override;

  bool IsExecutable() const override;

  uint32_t GetAddressByteSize() const override;

  //    virtual lldb_private::AddressClass
  //    GetAddressClass (lldb::addr_t file_addr);

  lldb_private::Symtab *GetSymtab() override;

  bool IsStripped() override;

  void CreateSections(lldb_private::SectionList &unified_section_list) override;

  void Dump(lldb_private::Stream *s) override;

  bool GetArchitecture(lldb_private::ArchSpec &arch) override;

  bool GetUUID(lldb_private::UUID *uuid) override;

  uint32_t GetDependentModules(lldb_private::FileSpecList &files) override;

  virtual lldb_private::Address GetEntryPointAddress() override;

  ObjectFile::Type CalculateType() override;

  ObjectFile::Strata CalculateStrata() override;

  //------------------------------------------------------------------
  // PluginInterface protocol
  //------------------------------------------------------------------
  lldb_private::ConstString GetPluginName() override;

  uint32_t GetPluginVersion() override;

  bool IsWindowsSubsystem();

  lldb_private::DataExtractor ReadImageData(uint32_t offset, size_t size);

protected:
  bool NeedsEndianSwap() const;

  typedef struct dos_header { // DOS .EXE header
    uint16_t e_magic;         // Magic number
    uint16_t e_cblp;          // Bytes on last page of file
    uint16_t e_cp;            // Pages in file
    uint16_t e_crlc;          // Relocations
    uint16_t e_cparhdr;       // Size of header in paragraphs
    uint16_t e_minalloc;      // Minimum extra paragraphs needed
    uint16_t e_maxalloc;      // Maximum extra paragraphs needed
    uint16_t e_ss;            // Initial (relative) SS value
    uint16_t e_sp;            // Initial SP value
    uint16_t e_csum;          // Checksum
    uint16_t e_ip;            // Initial IP value
    uint16_t e_cs;            // Initial (relative) CS value
    uint16_t e_lfarlc;        // File address of relocation table
    uint16_t e_ovno;          // Overlay number
    uint16_t e_res[4];        // Reserved words
    uint16_t e_oemid;         // OEM identifier (for e_oeminfo)
    uint16_t e_oeminfo;       // OEM information; e_oemid specific
    uint16_t e_res2[10];      // Reserved words
    uint32_t e_lfanew;        // File address of new exe header
  } dos_header_t;

  typedef struct coff_header {
    uint16_t machine;
    uint16_t nsects;
    uint32_t modtime;
    uint32_t symoff;
    uint32_t nsyms;
    uint16_t hdrsize;
    uint16_t flags;
  } coff_header_t;

  typedef struct data_directory {
    uint32_t vmaddr;
    uint32_t vmsize;
  } data_directory_t;

  typedef struct coff_opt_header {
    uint16_t magic;
    uint8_t major_linker_version;
    uint8_t minor_linker_version;
    uint32_t code_size;
    uint32_t data_size;
    uint32_t bss_size;
    uint32_t entry;
    uint32_t code_offset;
    uint32_t data_offset;

    uint64_t image_base;
    uint32_t sect_alignment;
    uint32_t file_alignment;
    uint16_t major_os_system_version;
    uint16_t minor_os_system_version;
    uint16_t major_image_version;
    uint16_t minor_image_version;
    uint16_t major_subsystem_version;
    uint16_t minor_subsystem_version;
    uint32_t reserved1;
    uint32_t image_size;
    uint32_t header_size;
    uint32_t checksum;
    uint16_t subsystem;
    uint16_t dll_flags;
    uint64_t stack_reserve_size;
    uint64_t stack_commit_size;
    uint64_t heap_reserve_size;
    uint64_t heap_commit_size;
    uint32_t loader_flags;
    //    uint32_t	num_data_dir_entries;
    std::vector<data_directory>
        data_dirs; // will contain num_data_dir_entries entries
  } coff_opt_header_t;

  typedef enum coff_data_dir_type {
    coff_data_dir_export_table = 0,
    coff_data_dir_import_table = 1,
  } coff_data_dir_type;

  typedef struct section_header {
    char name[8];
    uint32_t vmsize;  // Virtual Size
    uint32_t vmaddr;  // Virtual Addr
    uint32_t size;    // File size
    uint32_t offset;  // File offset
    uint32_t reloff;  // Offset to relocations
    uint32_t lineoff; // Offset to line table entries
    uint16_t nreloc;  // Number of relocation entries
    uint16_t nline;   // Number of line table entries
    uint32_t flags;
  } section_header_t;

  typedef struct coff_symbol {
    char name[8];
    uint32_t value;
    uint16_t sect;
    uint16_t type;
    uint8_t storage;
    uint8_t naux;
  } coff_symbol_t;

  typedef struct export_directory_entry {
    uint32_t characteristics;
    uint32_t time_date_stamp;
    uint16_t major_version;
    uint16_t minor_version;
    uint32_t name;
    uint32_t base;
    uint32_t number_of_functions;
    uint32_t number_of_names;
    uint32_t address_of_functions;
    uint32_t address_of_names;
    uint32_t address_of_name_ordinals;
  } export_directory_entry;

  static bool ParseDOSHeader(lldb_private::DataExtractor &data,
                             dos_header_t &dos_header);
  static bool ParseCOFFHeader(lldb_private::DataExtractor &data,
                              lldb::offset_t *offset_ptr,
                              coff_header_t &coff_header);
  bool ParseCOFFOptionalHeader(lldb::offset_t *offset_ptr);
  bool ParseSectionHeaders(uint32_t offset);

  static void DumpDOSHeader(lldb_private::Stream *s,
                            const dos_header_t &header);
  static void DumpCOFFHeader(lldb_private::Stream *s,
                             const coff_header_t &header);
  static void DumpOptCOFFHeader(lldb_private::Stream *s,
                                const coff_opt_header_t &header);
  void DumpSectionHeaders(lldb_private::Stream *s);
  void DumpSectionHeader(lldb_private::Stream *s, const section_header_t &sh);
  bool GetSectionName(std::string &sect_name, const section_header_t &sect);

  typedef std::vector<section_header_t> SectionHeaderColl;
  typedef SectionHeaderColl::iterator SectionHeaderCollIter;
  typedef SectionHeaderColl::const_iterator SectionHeaderCollConstIter;

private:
  dos_header_t m_dos_header;
  coff_header_t m_coff_header;
  coff_opt_header_t m_coff_header_opt;
  SectionHeaderColl m_sect_headers;
  lldb::addr_t m_image_base;
  lldb_private::Address m_entry_point_address;
};

#endif // liblldb_ObjectFilePECOFF_h_
