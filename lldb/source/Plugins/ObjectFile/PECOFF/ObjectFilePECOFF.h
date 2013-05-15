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

#include <vector>

#include "lldb/Symbol/ObjectFile.h"

class ObjectFilePECOFF : 
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
    
    static ObjectFile *
    CreateInstance (const lldb::ModuleSP &module_sp,
                    lldb::DataBufferSP& data_sp,
                    lldb::offset_t data_offset,
                    const lldb_private::FileSpec* file,
                    lldb::offset_t offset,
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
    MagicBytesMatch (lldb::DataBufferSP& data_sp);
    
    
    ObjectFilePECOFF (const lldb::ModuleSP &module_sp,
                      lldb::DataBufferSP& data_sp,
                      lldb::offset_t data_offset,
                      const lldb_private::FileSpec* file,
                      lldb::offset_t file_offset,
                      lldb::offset_t length);
    
	virtual 
    ~ObjectFilePECOFF();
    
    virtual bool
    ParseHeader ();
    
    virtual lldb::ByteOrder
    GetByteOrder () const;
    
    virtual bool
    IsExecutable () const;
    
    virtual uint32_t
    GetAddressByteSize ()  const;
    
//    virtual lldb_private::AddressClass
//    GetAddressClass (lldb::addr_t file_addr);
//    
    virtual lldb_private::Symtab *
    GetSymtab();
    
    virtual lldb_private::SectionList *
    GetSectionList();
    
    virtual void
    Dump (lldb_private::Stream *s);
    
    virtual bool
    GetArchitecture (lldb_private::ArchSpec &arch);
    
    virtual bool
    GetUUID (lldb_private::UUID* uuid);
    
    virtual uint32_t
    GetDependentModules (lldb_private::FileSpecList& files);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual lldb_private::ConstString
    GetPluginName();
    
    virtual uint32_t
    GetPluginVersion();
//    
//    virtual lldb_private::Address
//    GetEntryPointAddress ();
    
    virtual ObjectFile::Type
    CalculateType();
    
    virtual ObjectFile::Strata
    CalculateStrata();
    
protected:
	bool NeedsEndianSwap() const;    
    
	typedef struct dos_header  {  // DOS .EXE header
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
    
	typedef struct coff_opt_header 
	{
		uint16_t	magic;
		uint8_t		major_linker_version;
		uint8_t		minor_linker_version;
		uint32_t	code_size;
		uint32_t	data_size;
		uint32_t	bss_size;
		uint32_t	entry;
		uint32_t	code_offset;
		uint32_t	data_offset;
        
		uint64_t	image_base;
		uint32_t	sect_alignment;
		uint32_t	file_alignment;
		uint16_t	major_os_system_version;
		uint16_t	minor_os_system_version;
		uint16_t	major_image_version;
		uint16_t	minor_image_version;
		uint16_t	major_subsystem_version;
		uint16_t	minor_subsystem_version;
		uint32_t	reserved1;
		uint32_t	image_size;
		uint32_t	header_size;
		uint32_t	checksum;
		uint16_t	subsystem;
		uint16_t	dll_flags;
		uint64_t	stack_reserve_size;
		uint64_t	stack_commit_size;
		uint64_t	heap_reserve_size;
		uint64_t	heap_commit_size;
		uint32_t	loader_flags;
        //    uint32_t	num_data_dir_entries;
		std::vector<data_directory> data_dirs;	// will contain num_data_dir_entries entries
	} coff_opt_header_t;
    
	typedef struct section_header {
		char		name[8];
		uint32_t	vmsize;	// Virtual Size
		uint32_t	vmaddr;	// Virtual Addr
		uint32_t	size;	// File size
		uint32_t	offset;	// File offset
		uint32_t	reloff;	// Offset to relocations
		uint32_t	lineoff;// Offset to line table entries
		uint16_t	nreloc;	// Number of relocation entries
		uint16_t	nline;	// Number of line table entries
		uint32_t	flags;
	} section_header_t;
    
	typedef struct coff_symbol {
		char		name[8];
		uint32_t	value;
		uint16_t	sect;
		uint16_t	type;
		uint8_t		storage;
		uint8_t		naux;		
	} coff_symbol_t;
    
	bool ParseDOSHeader ();
	bool ParseCOFFHeader (lldb::offset_t *offset_ptr);
	bool ParseCOFFOptionalHeader (lldb::offset_t *offset_ptr);
	bool ParseSectionHeaders (uint32_t offset);
	
	static	void DumpDOSHeader(lldb_private::Stream *s, const dos_header_t& header);
	static	void DumpCOFFHeader(lldb_private::Stream *s, const coff_header_t& header);
	static	void DumpOptCOFFHeader(lldb_private::Stream *s, const coff_opt_header_t& header);
    void DumpSectionHeaders(lldb_private::Stream *s);
    void DumpSectionHeader(lldb_private::Stream *s, const section_header_t& sh);
    bool GetSectionName(std::string& sect_name, const section_header_t& sect);
    
	typedef std::vector<section_header_t>		SectionHeaderColl;
	typedef SectionHeaderColl::iterator			SectionHeaderCollIter;
	typedef SectionHeaderColl::const_iterator	SectionHeaderCollConstIter;
private:
	dos_header_t		m_dos_header;
	coff_header_t		m_coff_header;
	coff_opt_header_t	m_coff_header_opt;
	SectionHeaderColl	m_sect_headers;
};

#endif  // #ifndef liblldb_ObjectFilePECOFF_h_
