//===-- ObjectFileELF.h --------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjectFileELF_h_
#define liblldb_ObjectFileELF_h_

#include <stdint.h>
#include <vector>

#include "lldb/lldb-private.h"
#include "lldb/Host/FileSpec.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Core/UUID.h"

#include "ELFHeader.h"

//------------------------------------------------------------------------------
/// @class ObjectFileELF
/// @brief Generic ELF object file reader.
///
/// This class provides a generic ELF (32/64 bit) reader plugin implementing the
/// ObjectFile protocol.
class ObjectFileELF :
    public lldb_private::ObjectFile
{
public:
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static lldb_private::ConstString
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::ObjectFile *
    CreateInstance(const lldb::ModuleSP &module_sp,
                   lldb::DataBufferSP& data_sp,
                   lldb::offset_t data_offset,
                   const lldb_private::FileSpec* file,
                   lldb::offset_t file_offset,
                   lldb::offset_t length);

    static lldb_private::ObjectFile *
    CreateMemoryInstance (const lldb::ModuleSP &module_sp, 
                          lldb::DataBufferSP& data_sp, 
                          const lldb::ProcessSP &process_sp, 
                          lldb::addr_t header_addr);

    static size_t
    GetModuleSpecifications (const lldb_private::FileSpec& file,
                             lldb::DataBufferSP& data_sp,
                             lldb::offset_t data_offset,
                             lldb::offset_t file_offset,
                             lldb::offset_t length,
                             lldb_private::ModuleSpecList &specs);

    static bool
    MagicBytesMatch (lldb::DataBufferSP& data_sp,
                     lldb::addr_t offset, 
                     lldb::addr_t length);

    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();

    virtual uint32_t
    GetPluginVersion();

    //------------------------------------------------------------------
    // ObjectFile Protocol.
    //------------------------------------------------------------------
    virtual
    ~ObjectFileELF();

    virtual bool
    ParseHeader();

    virtual lldb::ByteOrder
    GetByteOrder() const;

    virtual bool
    IsExecutable () const;

    virtual uint32_t
    GetAddressByteSize() const;

    virtual lldb_private::Symtab *
    GetSymtab();

    virtual lldb_private::SectionList *
    GetSectionList();

    virtual void
    Dump(lldb_private::Stream *s);

    virtual bool
    GetArchitecture (lldb_private::ArchSpec &arch);

    virtual bool
    GetUUID(lldb_private::UUID* uuid);

    virtual uint32_t
    GetDependentModules(lldb_private::FileSpecList& files);

    virtual lldb_private::Address
    GetImageInfoAddress();
    
    virtual lldb_private::Address
    GetEntryPointAddress ();
    
    virtual ObjectFile::Type
    CalculateType();
    
    virtual ObjectFile::Strata
    CalculateStrata();

private:
    ObjectFileELF(const lldb::ModuleSP &module_sp,
                  lldb::DataBufferSP& data_sp,
                  lldb::offset_t data_offset,
                  const lldb_private::FileSpec* file,
                  lldb::offset_t offset,
                  lldb::offset_t length);

    typedef std::vector<elf::ELFProgramHeader>  ProgramHeaderColl;
    typedef ProgramHeaderColl::iterator         ProgramHeaderCollIter;
    typedef ProgramHeaderColl::const_iterator   ProgramHeaderCollConstIter;

    typedef std::vector<elf::ELFSectionHeader>  SectionHeaderColl;
    typedef SectionHeaderColl::iterator         SectionHeaderCollIter;
    typedef SectionHeaderColl::const_iterator   SectionHeaderCollConstIter;

    typedef std::vector<elf::ELFDynamic>        DynamicSymbolColl;
    typedef DynamicSymbolColl::iterator         DynamicSymbolCollIter;
    typedef DynamicSymbolColl::const_iterator   DynamicSymbolCollConstIter;

    /// Version of this reader common to all plugins based on this class.
    static const uint32_t m_plugin_version = 1;

    /// ELF file header.
    elf::ELFHeader m_header;

    /// ELF build ID
    lldb_private::UUID m_uuid;

    /// Collection of program headers.
    ProgramHeaderColl m_program_headers;

    /// Collection of section headers.
    SectionHeaderColl m_section_headers;

    /// Collection of symbols from the dynamic table.
    DynamicSymbolColl m_dynamic_symbols;

    /// List of file specifications corresponding to the modules (shared
    /// libraries) on which this object file depends.
    mutable std::unique_ptr<lldb_private::FileSpecList> m_filespec_ap;

    /// Data extractor holding the string table used to resolve section names.
    lldb_private::DataExtractor m_shstr_data;

    /// Cached value of the entry point for this module.
    lldb_private::Address  m_entry_point_address;

    /// Returns a 1 based index of the given section header.
    size_t
    SectionIndex(const SectionHeaderCollIter &I);

    /// Returns a 1 based index of the given section header.
    size_t
    SectionIndex(const SectionHeaderCollConstIter &I) const;

    /// Parses all section headers present in this object file and populates
    /// m_program_headers.  This method will compute the header list only once.
    /// Returns the number of headers parsed.
    size_t
    ParseProgramHeaders();

    /// Parses all section headers present in this object file and populates
    /// m_section_headers.  This method will compute the header list only once.
    /// Returns the number of headers parsed.
    size_t
    ParseSectionHeaders();

    /// Scans the dynamic section and locates all dependent modules (shared
    /// libraries) populating m_filespec_ap.  This method will compute the
    /// dependent module list only once.  Returns the number of dependent
    /// modules parsed.
    size_t
    ParseDependentModules();

    /// Parses the dynamic symbol table and populates m_dynamic_symbols.  The
    /// vector retains the order as found in the object file.  Returns the
    /// number of dynamic symbols parsed.
    size_t
    ParseDynamicSymbols();

    /// Populates m_symtab_ap will all non-dynamic linker symbols.  This method
    /// will parse the symbols only once.  Returns the number of symbols parsed.
    unsigned
    ParseSymbolTable(lldb_private::Symtab *symbol_table,
                     lldb::user_id_t start_id,
                     const elf::ELFSectionHeader *symtab_section,
                     lldb::user_id_t symtab_id);

    /// Scans the relocation entries and adds a set of artificial symbols to the
    /// given symbol table for each PLT slot.  Returns the number of symbols
    /// added.
    unsigned
    ParseTrampolineSymbols(lldb_private::Symtab *symbol_table, 
                           lldb::user_id_t start_id,
                           const elf::ELFSectionHeader *rela_hdr,
                           lldb::user_id_t section_id);

    /// Loads the section name string table into m_shstr_data.  Returns the
    /// number of bytes constituting the table.
    size_t
    GetSectionHeaderStringTable();

    /// Utility method for looking up a section given its name.  Returns the
    /// index of the corresponding section or zero if no section with the given
    /// name can be found (note that section indices are always 1 based, and so
    /// section index 0 is never valid).
    lldb::user_id_t
    GetSectionIndexByName(const char *name);

    // Returns the ID of the first section that has the given type.
    lldb::user_id_t
    GetSectionIndexByType(unsigned type);

    /// Returns the section header with the given id or NULL.
    const elf::ELFSectionHeader *
    GetSectionHeaderByIndex(lldb::user_id_t id);

    /// @name  ELF header dump routines
    //@{
    static void
    DumpELFHeader(lldb_private::Stream *s, const elf::ELFHeader& header);

    static void
    DumpELFHeader_e_ident_EI_DATA(lldb_private::Stream *s,
                                  unsigned char ei_data);

    static void
    DumpELFHeader_e_type(lldb_private::Stream *s, elf::elf_half e_type);
    //@}

    /// @name ELF program header dump routines
    //@{
    void
    DumpELFProgramHeaders(lldb_private::Stream *s);

    static void
    DumpELFProgramHeader(lldb_private::Stream *s, 
                         const elf::ELFProgramHeader &ph);

    static void
    DumpELFProgramHeader_p_type(lldb_private::Stream *s, elf::elf_word p_type);

    static void
    DumpELFProgramHeader_p_flags(lldb_private::Stream *s, 
                                 elf::elf_word p_flags);
    //@}

    /// @name ELF section header dump routines
    //@{
    void
    DumpELFSectionHeaders(lldb_private::Stream *s);

    static void
    DumpELFSectionHeader(lldb_private::Stream *s, 
                         const elf::ELFSectionHeader& sh);

    static void
    DumpELFSectionHeader_sh_type(lldb_private::Stream *s, 
                                 elf::elf_word sh_type);

    static void
    DumpELFSectionHeader_sh_flags(lldb_private::Stream *s, 
                                  elf::elf_xword sh_flags);
    //@}

    /// ELF dependent module dump routine.
    void
    DumpDependentModules(lldb_private::Stream *s);

    const elf::ELFDynamic *
    FindDynamicSymbol(unsigned tag);
        
    lldb_private::Section *
    PLTSection();

    unsigned
    PLTRelocationType();
};

#endif // #ifndef liblldb_ObjectFileELF_h_
