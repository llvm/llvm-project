//===-- ObjectFileELF.cpp ------------------------------------- -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ObjectFileELF.h"

#include <cassert>
#include <algorithm>

#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/DataBuffer.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/SymbolContext.h"
#include "lldb/Host/Host.h"

#include "llvm/ADT/PointerUnion.h"

#define CASE_AND_STREAM(s, def, width)                  \
    case def: s->Printf("%-*s", width, #def); break;

using namespace lldb;
using namespace lldb_private;
using namespace elf;
using namespace llvm::ELF;

namespace {
//===----------------------------------------------------------------------===//
/// @class ELFRelocation
/// @brief Generic wrapper for ELFRel and ELFRela.
///
/// This helper class allows us to parse both ELFRel and ELFRela relocation
/// entries in a generic manner.
class ELFRelocation
{
public:

    /// Constructs an ELFRelocation entry with a personality as given by @p
    /// type.
    ///
    /// @param type Either DT_REL or DT_RELA.  Any other value is invalid.
    ELFRelocation(unsigned type);
 
    ~ELFRelocation();

    bool
    Parse(const lldb_private::DataExtractor &data, lldb::offset_t *offset);

    static unsigned
    RelocType32(const ELFRelocation &rel);

    static unsigned
    RelocType64(const ELFRelocation &rel);

    static unsigned
    RelocSymbol32(const ELFRelocation &rel);

    static unsigned
    RelocSymbol64(const ELFRelocation &rel);

private:
    typedef llvm::PointerUnion<ELFRel*, ELFRela*> RelocUnion;

    RelocUnion reloc;
};

ELFRelocation::ELFRelocation(unsigned type)
{ 
    if (type == DT_REL)
        reloc = new ELFRel();
    else if (type == DT_RELA)
        reloc = new ELFRela();
    else {
        assert(false && "unexpected relocation type");
        reloc = static_cast<ELFRel*>(NULL);
    }
}

ELFRelocation::~ELFRelocation()
{
    if (reloc.is<ELFRel*>())
        delete reloc.get<ELFRel*>();
    else
        delete reloc.get<ELFRela*>();            
}

bool
ELFRelocation::Parse(const lldb_private::DataExtractor &data, lldb::offset_t *offset)
{
    if (reloc.is<ELFRel*>())
        return reloc.get<ELFRel*>()->Parse(data, offset);
    else
        return reloc.get<ELFRela*>()->Parse(data, offset);
}

unsigned
ELFRelocation::RelocType32(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocType32(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocType32(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocType64(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocType64(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocType64(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocSymbol32(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocSymbol32(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocSymbol32(*rel.reloc.get<ELFRela*>());
}

unsigned
ELFRelocation::RelocSymbol64(const ELFRelocation &rel)
{
    if (rel.reloc.is<ELFRel*>())
        return ELFRel::RelocSymbol64(*rel.reloc.get<ELFRel*>());
    else
        return ELFRela::RelocSymbol64(*rel.reloc.get<ELFRela*>());
}

} // end anonymous namespace

//------------------------------------------------------------------
// Static methods.
//------------------------------------------------------------------
void
ObjectFileELF::Initialize()
{
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(),
                                  CreateInstance,
                                  CreateMemoryInstance,
                                  GetModuleSpecifications);
}

void
ObjectFileELF::Terminate()
{
    PluginManager::UnregisterPlugin(CreateInstance);
}

lldb_private::ConstString
ObjectFileELF::GetPluginNameStatic()
{
    static ConstString g_name("elf");
    return g_name;
}

const char *
ObjectFileELF::GetPluginDescriptionStatic()
{
    return "ELF object file reader.";
}

ObjectFile *
ObjectFileELF::CreateInstance (const lldb::ModuleSP &module_sp,
                               DataBufferSP &data_sp,
                               lldb::offset_t data_offset,
                               const lldb_private::FileSpec* file,
                               lldb::offset_t file_offset,
                               lldb::offset_t length)
{
    if (!data_sp)
    {
        data_sp = file->MemoryMapFileContents(file_offset, length);
        data_offset = 0;
    }

    if (data_sp && data_sp->GetByteSize() > (llvm::ELF::EI_NIDENT + data_offset))
    {
        const uint8_t *magic = data_sp->GetBytes() + data_offset;
        if (ELFHeader::MagicBytesMatch(magic))
        {
            // Update the data to contain the entire file if it doesn't already
            if (data_sp->GetByteSize() < length) {
                data_sp = file->MemoryMapFileContents(file_offset, length);
                data_offset = 0;
                magic = data_sp->GetBytes();
            }
            unsigned address_size = ELFHeader::AddressSizeInBytes(magic);
            if (address_size == 4 || address_size == 8)
            {
                std::unique_ptr<ObjectFileELF> objfile_ap(new ObjectFileELF(module_sp, data_sp, data_offset, file, file_offset, length));
                ArchSpec spec;
                if (objfile_ap->GetArchitecture(spec) &&
                    objfile_ap->SetModulesArchitecture(spec))
                    return objfile_ap.release();
            }
        }
    }
    return NULL;
}


ObjectFile*
ObjectFileELF::CreateMemoryInstance (const lldb::ModuleSP &module_sp, 
                                     DataBufferSP& data_sp, 
                                     const lldb::ProcessSP &process_sp, 
                                     lldb::addr_t header_addr)
{
    return NULL;
}

bool
ObjectFileELF::MagicBytesMatch (DataBufferSP& data_sp,
                                  lldb::addr_t data_offset,
                                  lldb::addr_t data_length)
{
    if (data_sp && data_sp->GetByteSize() > (llvm::ELF::EI_NIDENT + data_offset))
    {
        const uint8_t *magic = data_sp->GetBytes() + data_offset;
        return ELFHeader::MagicBytesMatch(magic);
    }
    return false;
}

size_t
ObjectFileELF::GetModuleSpecifications (const lldb_private::FileSpec& file,
                                        lldb::DataBufferSP& data_sp,
                                        lldb::offset_t data_offset,
                                        lldb::offset_t file_offset,
                                        lldb::offset_t length,
                                        lldb_private::ModuleSpecList &specs)
{
    const size_t initial_count = specs.GetSize();

    if (ObjectFileELF::MagicBytesMatch(data_sp, 0, data_sp->GetByteSize()))
    {
        DataExtractor data;
        data.SetData(data_sp);
        elf::ELFHeader header;
        if (header.Parse(data, &data_offset))
        {
            if (data_sp)
            {
                ModuleSpec spec;
                spec.GetFileSpec() = file;
                spec.GetArchitecture().SetArchitecture(eArchTypeELF,
                                                       header.e_machine,
                                                       LLDB_INVALID_CPUTYPE);
                if (spec.GetArchitecture().IsValid())
                {
                    // We could parse the ABI tag information (in .note, .notes, or .note.ABI-tag) to get the
                    // machine information. However, we'd have to read a good bit of the rest of the file,
                    // and this info isn't guaranteed to exist or be correct. More details here:
                    //  http://refspecs.linuxfoundation.org/LSB_1.2.0/gLSB/noteabitag.html
                    // Instead of passing potentially incorrect information down the pipeline, grab
                    // the host information and use it.
                    spec.GetArchitecture().GetTriple().setOSName (Host::GetOSString().GetCString());
                    spec.GetArchitecture().GetTriple().setVendorName(Host::GetVendorString().GetCString());
                    specs.Append(spec);
                }
            }
        }
    }

    return specs.GetSize() - initial_count;
}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
lldb_private::ConstString
ObjectFileELF::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ObjectFileELF::GetPluginVersion()
{
    return m_plugin_version;
}
//------------------------------------------------------------------
// ObjectFile protocol
//------------------------------------------------------------------

ObjectFileELF::ObjectFileELF (const lldb::ModuleSP &module_sp, 
                              DataBufferSP& data_sp,
                              lldb::offset_t data_offset,
                              const FileSpec* file, 
                              lldb::offset_t file_offset,
                              lldb::offset_t length) : 
    ObjectFile(module_sp, file, file_offset, length, data_sp, data_offset),
    m_header(),
    m_program_headers(),
    m_section_headers(),
    m_filespec_ap(),
    m_shstr_data()
{
    if (file)
        m_file = *file;
    ::memset(&m_header, 0, sizeof(m_header));
}

ObjectFileELF::~ObjectFileELF()
{
}

bool
ObjectFileELF::IsExecutable() const
{
    return m_header.e_entry != 0;
}

ByteOrder
ObjectFileELF::GetByteOrder() const
{
    if (m_header.e_ident[EI_DATA] == ELFDATA2MSB)
        return eByteOrderBig;
    if (m_header.e_ident[EI_DATA] == ELFDATA2LSB)
        return eByteOrderLittle;
    return eByteOrderInvalid;
}

uint32_t
ObjectFileELF::GetAddressByteSize() const
{
    return m_data.GetAddressByteSize();
}

size_t
ObjectFileELF::SectionIndex(const SectionHeaderCollIter &I)
{
    return std::distance(m_section_headers.begin(), I) + 1u;
}

size_t
ObjectFileELF::SectionIndex(const SectionHeaderCollConstIter &I) const
{
    return std::distance(m_section_headers.begin(), I) + 1u;
}

bool
ObjectFileELF::ParseHeader()
{
    lldb::offset_t offset = 0;
    return m_header.Parse(m_data, &offset);
}

bool
ObjectFileELF::GetUUID(lldb_private::UUID* uuid)
{
    if (m_uuid.IsValid())
    {
        *uuid = m_uuid;
        return true;
    }
    // FIXME: Return MD5 sum here. See comment in ObjectFile.h.
    return false;
}

uint32_t
ObjectFileELF::GetDependentModules(FileSpecList &files)
{
    size_t num_modules = ParseDependentModules();
    uint32_t num_specs = 0;

    for (unsigned i = 0; i < num_modules; ++i)
    {
        if (files.AppendIfUnique(m_filespec_ap->GetFileSpecAtIndex(i)))
            num_specs++;
    }

    return num_specs;
}

user_id_t
ObjectFileELF::GetSectionIndexByType(unsigned type)
{
    if (!ParseSectionHeaders())
        return 0;

    for (SectionHeaderCollIter sh_pos = m_section_headers.begin();
         sh_pos != m_section_headers.end(); ++sh_pos) 
    {
        if (sh_pos->sh_type == type)
            return SectionIndex(sh_pos);
    }

    return 0;
}

Address
ObjectFileELF::GetImageInfoAddress()
{
    if (!ParseDynamicSymbols())
        return Address();

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return Address();

    user_id_t dynsym_id = GetSectionIndexByType(SHT_DYNAMIC);
    if (!dynsym_id)
        return Address();

    const ELFSectionHeader *dynsym_hdr = GetSectionHeaderByIndex(dynsym_id);
    if (!dynsym_hdr)
        return Address();

    SectionSP dynsym_section_sp (section_list->FindSectionByID(dynsym_id));
    if (dynsym_section_sp)
    {
        for (size_t i = 0; i < m_dynamic_symbols.size(); ++i)
        {
            ELFDynamic &symbol = m_dynamic_symbols[i];

            if (symbol.d_tag == DT_DEBUG)
            {
                // Compute the offset as the number of previous entries plus the
                // size of d_tag.
                addr_t offset = i * dynsym_hdr->sh_entsize + GetAddressByteSize();
                return Address(dynsym_section_sp, offset);
            }
        }
    }

    return Address();
}

lldb_private::Address
ObjectFileELF::GetEntryPointAddress () 
{
    SectionList *sections;
    addr_t offset;

    if (m_entry_point_address.IsValid())
        return m_entry_point_address;

    if (!ParseHeader() || !IsExecutable())
        return m_entry_point_address;

    sections = GetSectionList();
    offset = m_header.e_entry;

    if (!sections) 
    {
        m_entry_point_address.SetOffset(offset);
        return m_entry_point_address;
    }

    m_entry_point_address.ResolveAddressUsingFileSections(offset, sections);

    return m_entry_point_address;
}

//----------------------------------------------------------------------
// ParseDependentModules
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseDependentModules()
{
    if (m_filespec_ap.get())
        return m_filespec_ap->GetSize();

    m_filespec_ap.reset(new FileSpecList());

    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return 0;

    // Locate the dynamic table.
    user_id_t dynsym_id = 0;
    user_id_t dynstr_id = 0;
    for (SectionHeaderCollIter sh_pos = m_section_headers.begin();
         sh_pos != m_section_headers.end(); ++sh_pos)
    {
        if (sh_pos->sh_type == SHT_DYNAMIC)
        {
            dynsym_id = SectionIndex(sh_pos);
            dynstr_id = sh_pos->sh_link + 1; // Section ID's are 1 based.
            break;
        }
    }

    if (!(dynsym_id && dynstr_id))
        return 0;

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    // Resolve and load the dynamic table entries and corresponding string
    // table.
    Section *dynsym = section_list->FindSectionByID(dynsym_id).get();
    Section *dynstr = section_list->FindSectionByID(dynstr_id).get();
    if (!(dynsym && dynstr))
        return 0;

    DataExtractor dynsym_data;
    DataExtractor dynstr_data;
    if (ReadSectionData(dynsym, dynsym_data) &&
        ReadSectionData(dynstr, dynstr_data))
    {
        ELFDynamic symbol;
        const lldb::offset_t section_size = dynsym_data.GetByteSize();
        lldb::offset_t offset = 0;

        // The only type of entries we are concerned with are tagged DT_NEEDED,
        // yielding the name of a required library.
        while (offset < section_size)
        {
            if (!symbol.Parse(dynsym_data, &offset))
                break;

            if (symbol.d_tag != DT_NEEDED)
                continue;

            uint32_t str_index = static_cast<uint32_t>(symbol.d_val);
            const char *lib_name = dynstr_data.PeekCStr(str_index);
            m_filespec_ap->Append(FileSpec(lib_name, true));
        }
    }

    return m_filespec_ap->GetSize();
}

//----------------------------------------------------------------------
// ParseProgramHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseProgramHeaders()
{
    // We have already parsed the program headers
    if (!m_program_headers.empty())
        return m_program_headers.size();

    // If there are no program headers to read we are done.
    if (m_header.e_phnum == 0)
        return 0;

    m_program_headers.resize(m_header.e_phnum);
    if (m_program_headers.size() != m_header.e_phnum)
        return 0;

    const size_t ph_size = m_header.e_phnum * m_header.e_phentsize;
    const elf_off ph_offset = m_header.e_phoff;
    DataExtractor data;
    if (GetData (ph_offset, ph_size, data) != ph_size)
        return 0;

    uint32_t idx;
    lldb::offset_t offset;
    for (idx = 0, offset = 0; idx < m_header.e_phnum; ++idx)
    {
        if (m_program_headers[idx].Parse(data, &offset) == false)
            break;
    }

    if (idx < m_program_headers.size())
        m_program_headers.resize(idx);

    return m_program_headers.size();
}

//----------------------------------------------------------------------
// ParseSectionHeaders
//----------------------------------------------------------------------
size_t
ObjectFileELF::ParseSectionHeaders()
{
    // We have already parsed the section headers
    if (!m_section_headers.empty())
        return m_section_headers.size();

    // If there are no section headers we are done.
    if (m_header.e_shnum == 0)
        return 0;

    m_section_headers.resize(m_header.e_shnum);
    if (m_section_headers.size() != m_header.e_shnum)
        return 0;

    const size_t sh_size = m_header.e_shnum * m_header.e_shentsize;
    const elf_off sh_offset = m_header.e_shoff;
    DataExtractor data;
    if (GetData (sh_offset, sh_size, data) != sh_size)
        return 0;

    uint32_t idx;
    lldb::offset_t offset;
    for (idx = 0, offset = 0; idx < m_header.e_shnum; ++idx)
    {
        if (m_section_headers[idx].Parse(data, &offset) == false)
            break;
    }
    if (idx < m_section_headers.size())
        m_section_headers.resize(idx);

    return m_section_headers.size();
}

size_t
ObjectFileELF::GetSectionHeaderStringTable()
{
    if (m_shstr_data.GetByteSize() == 0)
    {
        const unsigned strtab_idx = m_header.e_shstrndx;

        if (strtab_idx && strtab_idx < m_section_headers.size())
        {
            const ELFSectionHeader &sheader = m_section_headers[strtab_idx];
            const size_t byte_size = sheader.sh_size;
            const Elf64_Off offset = sheader.sh_offset;
            m_shstr_data.SetData (m_data, offset, byte_size);

            if (m_shstr_data.GetByteSize() != byte_size)
                return 0;
        }
    }
    return m_shstr_data.GetByteSize();
}

lldb::user_id_t
ObjectFileELF::GetSectionIndexByName(const char *name)
{
    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return 0;

    // Search the collection of section headers for one with a matching name.
    for (SectionHeaderCollIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I)
    {
        const char *sectionName = m_shstr_data.PeekCStr(I->sh_name);

        if (!sectionName)
            return 0;

        if (strcmp(name, sectionName) != 0)
            continue;

        return SectionIndex(I);
    }

    return 0;
}

const elf::ELFSectionHeader *
ObjectFileELF::GetSectionHeaderByIndex(lldb::user_id_t id)
{
    if (!ParseSectionHeaders() || !id)
        return NULL;

    if (--id < m_section_headers.size())
        return &m_section_headers[id];

    return NULL;
}

static bool
ParseNoteGNUBuildID(DataExtractor& data, lldb_private::UUID& uuid)
{
    // Try to parse the note section (ie .note.gnu.build-id|.notes|.note|...) and get the build id.
    // BuildID documentation: https://fedoraproject.org/wiki/Releases/FeatureBuildId
    struct
    {
        uint32_t name_len;  // Length of note name
        uint32_t desc_len;  // Length of note descriptor
        uint32_t type;      // Type of note (1 is ABI_TAG, 3 is BUILD_ID)
    } notehdr;
    lldb::offset_t offset = 0;
    static const uint32_t g_gnu_build_id = 3; // NT_GNU_BUILD_ID from elf.h

    while (true)
    {
        if (data.GetU32 (&offset, &notehdr, 3) == NULL)
            return false;

        notehdr.name_len = llvm::RoundUpToAlignment (notehdr.name_len, 4);
        notehdr.desc_len = llvm::RoundUpToAlignment (notehdr.desc_len, 4);

        lldb::offset_t offset_next_note = offset + notehdr.name_len + notehdr.desc_len;

        // 16 bytes is UUID|MD5, 20 bytes is SHA1
        if ((notehdr.type == g_gnu_build_id) && (notehdr.name_len == 4) &&
            (notehdr.desc_len == 16 || notehdr.desc_len == 20))
        {
            char name[4];
            if (data.GetU8 (&offset, name, 4) == NULL)
                return false;
            if (!strcmp(name, "GNU"))
            {
                uint8_t uuidbuf[20]; 
                if (data.GetU8 (&offset, &uuidbuf, notehdr.desc_len) == NULL)
                    return false;
                uuid.SetBytes (uuidbuf, notehdr.desc_len);
                return true;
            }
        }
        offset = offset_next_note;
    }
    return false;
}
 
SectionList *
ObjectFileELF::GetSectionList()
{
    if (m_sections_ap.get())
        return m_sections_ap.get();

    if (ParseSectionHeaders() && GetSectionHeaderStringTable())
    {
        m_sections_ap.reset(new SectionList());

        for (SectionHeaderCollIter I = m_section_headers.begin();
             I != m_section_headers.end(); ++I)
        {
            const ELFSectionHeader &header = *I;

            ConstString name(m_shstr_data.PeekCStr(header.sh_name));
            const uint64_t file_size = header.sh_type == SHT_NOBITS ? 0 : header.sh_size;
            const uint64_t vm_size = header.sh_flags & SHF_ALLOC ? header.sh_size : 0;

            static ConstString g_sect_name_text (".text");
            static ConstString g_sect_name_data (".data");
            static ConstString g_sect_name_bss (".bss");
            static ConstString g_sect_name_tdata (".tdata");
            static ConstString g_sect_name_tbss (".tbss");
            static ConstString g_sect_name_dwarf_debug_abbrev (".debug_abbrev");
            static ConstString g_sect_name_dwarf_debug_aranges (".debug_aranges");
            static ConstString g_sect_name_dwarf_debug_frame (".debug_frame");
            static ConstString g_sect_name_dwarf_debug_info (".debug_info");
            static ConstString g_sect_name_dwarf_debug_line (".debug_line");
            static ConstString g_sect_name_dwarf_debug_loc (".debug_loc");
            static ConstString g_sect_name_dwarf_debug_macinfo (".debug_macinfo");
            static ConstString g_sect_name_dwarf_debug_pubnames (".debug_pubnames");
            static ConstString g_sect_name_dwarf_debug_pubtypes (".debug_pubtypes");
            static ConstString g_sect_name_dwarf_debug_ranges (".debug_ranges");
            static ConstString g_sect_name_dwarf_debug_str (".debug_str");
            static ConstString g_sect_name_eh_frame (".eh_frame");

            SectionType sect_type = eSectionTypeOther;

            bool is_thread_specific = false;
            
            if      (name == g_sect_name_text)                  sect_type = eSectionTypeCode;
            else if (name == g_sect_name_data)                  sect_type = eSectionTypeData;
            else if (name == g_sect_name_bss)                   sect_type = eSectionTypeZeroFill;
            else if (name == g_sect_name_tdata)
            {
                sect_type = eSectionTypeData;
                is_thread_specific = true;   
            }
            else if (name == g_sect_name_tbss)
            {
                sect_type = eSectionTypeZeroFill;   
                is_thread_specific = true;   
            }
            else if (name == g_sect_name_dwarf_debug_abbrev)    sect_type = eSectionTypeDWARFDebugAbbrev;
            else if (name == g_sect_name_dwarf_debug_aranges)   sect_type = eSectionTypeDWARFDebugAranges;
            else if (name == g_sect_name_dwarf_debug_frame)     sect_type = eSectionTypeDWARFDebugFrame;
            else if (name == g_sect_name_dwarf_debug_info)      sect_type = eSectionTypeDWARFDebugInfo;
            else if (name == g_sect_name_dwarf_debug_line)      sect_type = eSectionTypeDWARFDebugLine;
            else if (name == g_sect_name_dwarf_debug_loc)       sect_type = eSectionTypeDWARFDebugLoc;
            else if (name == g_sect_name_dwarf_debug_macinfo)   sect_type = eSectionTypeDWARFDebugMacInfo;
            else if (name == g_sect_name_dwarf_debug_pubnames)  sect_type = eSectionTypeDWARFDebugPubNames;
            else if (name == g_sect_name_dwarf_debug_pubtypes)  sect_type = eSectionTypeDWARFDebugPubTypes;
            else if (name == g_sect_name_dwarf_debug_ranges)    sect_type = eSectionTypeDWARFDebugRanges;
            else if (name == g_sect_name_dwarf_debug_str)       sect_type = eSectionTypeDWARFDebugStr;
            else if (name == g_sect_name_eh_frame)              sect_type = eSectionTypeEHFrame;
            else if (header.sh_type == SHT_NOTE)
            {
                if (!m_uuid.IsValid())
                {
                    DataExtractor data;
                    if (vm_size && (GetData (header.sh_offset, vm_size, data) == vm_size))
                    {
                        ParseNoteGNUBuildID (data, m_uuid);
                    }
                }
            }
            
            SectionSP section_sp(new Section(
                GetModule(),        // Module to which this section belongs.
                SectionIndex(I),    // Section ID.
                name,               // Section name.
                sect_type,          // Section type.
                header.sh_addr,     // VM address.
                vm_size,            // VM size in bytes of this section.
                header.sh_offset,   // Offset of this section in the file.
                file_size,          // Size of the section as found in the file.
                header.sh_flags));  // Flags for this section.

            if (is_thread_specific)
                section_sp->SetIsThreadSpecific (is_thread_specific);
            m_sections_ap->AddSection(section_sp);
        }
        
        m_sections_ap->Finalize(); // Now that we're done adding sections, finalize to build fast-lookup caches
    }

    return m_sections_ap.get();
}

static unsigned
ParseSymbols(Symtab *symtab, 
             user_id_t start_id,
             SectionList *section_list,
             const ELFSectionHeader *symtab_shdr,
             const DataExtractor &symtab_data,
             const DataExtractor &strtab_data)
{
    ELFSymbol symbol;
    lldb::offset_t offset = 0;
    const size_t num_symbols = symtab_data.GetByteSize() / symtab_shdr->sh_entsize;

    static ConstString text_section_name(".text");
    static ConstString init_section_name(".init");
    static ConstString fini_section_name(".fini");
    static ConstString ctors_section_name(".ctors");
    static ConstString dtors_section_name(".dtors");

    static ConstString data_section_name(".data");
    static ConstString rodata_section_name(".rodata");
    static ConstString rodata1_section_name(".rodata1");
    static ConstString data2_section_name(".data1");
    static ConstString bss_section_name(".bss");

    //StreamFile strm(stdout, false);
    unsigned i;
    for (i = 0; i < num_symbols; ++i)
    {
        if (symbol.Parse(symtab_data, &offset) == false)
            break;
        
        const char *symbol_name = strtab_data.PeekCStr(symbol.st_name);

        // No need to add symbols that have no names
        if (symbol_name == NULL || symbol_name[0] == '\0')
            continue;

        //symbol.Dump (&strm, i, &strtab_data, section_list);

        SectionSP symbol_section_sp;
        SymbolType symbol_type = eSymbolTypeInvalid;
        Elf64_Half symbol_idx = symbol.st_shndx;

        switch (symbol_idx)
        {
        case SHN_ABS:
            symbol_type = eSymbolTypeAbsolute;
            break;
        case SHN_UNDEF:
            symbol_type = eSymbolTypeUndefined;
            break;
        default:
            symbol_section_sp = section_list->GetSectionAtIndex(symbol_idx);
            break;
        }

        // If a symbol is undefined do not process it further even if it has a STT type
        if (symbol_type != eSymbolTypeUndefined)
        {
            switch (symbol.getType())
            {
            default:
            case STT_NOTYPE:
                // The symbol's type is not specified.
                break;

            case STT_OBJECT:
                // The symbol is associated with a data object, such as a variable,
                // an array, etc.
                symbol_type = eSymbolTypeData;
                break;

            case STT_FUNC:
                // The symbol is associated with a function or other executable code.
                symbol_type = eSymbolTypeCode;
                break;

            case STT_SECTION:
                // The symbol is associated with a section. Symbol table entries of
                // this type exist primarily for relocation and normally have
                // STB_LOCAL binding.
                break;

            case STT_FILE:
                // Conventionally, the symbol's name gives the name of the source
                // file associated with the object file. A file symbol has STB_LOCAL
                // binding, its section index is SHN_ABS, and it precedes the other
                // STB_LOCAL symbols for the file, if it is present.
                symbol_type = eSymbolTypeSourceFile;
                break;

            case STT_GNU_IFUNC:
                // The symbol is associated with an indirect function. The actual
                // function will be resolved if it is referenced.
                symbol_type = eSymbolTypeResolver;
                break;
            }
        }

        if (symbol_type == eSymbolTypeInvalid)
        {
            if (symbol_section_sp)
            {
                const ConstString &sect_name = symbol_section_sp->GetName();
                if (sect_name == text_section_name ||
                    sect_name == init_section_name ||
                    sect_name == fini_section_name ||
                    sect_name == ctors_section_name ||
                    sect_name == dtors_section_name)
                {
                    symbol_type = eSymbolTypeCode;
                }
                else if (sect_name == data_section_name ||
                         sect_name == data2_section_name ||
                         sect_name == rodata_section_name ||
                         sect_name == rodata1_section_name ||
                         sect_name == bss_section_name)
                {
                    symbol_type = eSymbolTypeData;
                }
            }
        }

        uint64_t symbol_value = symbol.st_value;
        if (symbol_section_sp)
            symbol_value -= symbol_section_sp->GetFileAddress();
        bool is_global = symbol.getBinding() == STB_GLOBAL;
        uint32_t flags = symbol.st_other << 8 | symbol.st_info;
        bool is_mangled = symbol_name ? (symbol_name[0] == '_' && symbol_name[1] == 'Z') : false;
        Symbol dc_symbol(
            i + start_id,       // ID is the original symbol table index.
            symbol_name,        // Symbol name.
            is_mangled,         // Is the symbol name mangled?
            symbol_type,        // Type of this symbol
            is_global,          // Is this globally visible?
            false,              // Is this symbol debug info?
            false,              // Is this symbol a trampoline?
            false,              // Is this symbol artificial?
            symbol_section_sp,  // Section in which this symbol is defined or null.
            symbol_value,       // Offset in section or symbol value.
            symbol.st_size,     // Size in bytes of this symbol.
            true,               // Size is valid
            flags);             // Symbol flags.
        symtab->AddSymbol(dc_symbol);
    }

    return i;
}

unsigned
ObjectFileELF::ParseSymbolTable(Symtab *symbol_table, user_id_t start_id,
                                const ELFSectionHeader *symtab_hdr,
                                user_id_t symtab_id)
{
    assert(symtab_hdr->sh_type == SHT_SYMTAB || 
           symtab_hdr->sh_type == SHT_DYNSYM);

    // Parse in the section list if needed.
    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    // Section ID's are ones based.
    user_id_t strtab_id = symtab_hdr->sh_link + 1;

    Section *symtab = section_list->FindSectionByID(symtab_id).get();
    Section *strtab = section_list->FindSectionByID(strtab_id).get();
    unsigned num_symbols = 0;
    if (symtab && strtab)
    {
        DataExtractor symtab_data;
        DataExtractor strtab_data;
        if (ReadSectionData(symtab, symtab_data) &&
            ReadSectionData(strtab, strtab_data))
        {
            num_symbols = ParseSymbols(symbol_table, start_id, 
                                       section_list, symtab_hdr,
                                       symtab_data, strtab_data);
        }
    }

    return num_symbols;
}

size_t
ObjectFileELF::ParseDynamicSymbols()
{
    if (m_dynamic_symbols.size())
        return m_dynamic_symbols.size();

    user_id_t dyn_id = GetSectionIndexByType(SHT_DYNAMIC);
    if (!dyn_id)
        return 0;

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    Section *dynsym = section_list->FindSectionByID(dyn_id).get();
    if (!dynsym)
        return 0;

    ELFDynamic symbol;
    DataExtractor dynsym_data;
    if (ReadSectionData(dynsym, dynsym_data))
    {
        const lldb::offset_t section_size = dynsym_data.GetByteSize();
        lldb::offset_t cursor = 0;

        while (cursor < section_size)
        {
            if (!symbol.Parse(dynsym_data, &cursor))
                break;

            m_dynamic_symbols.push_back(symbol);
        }
    }

    return m_dynamic_symbols.size();
}

const ELFDynamic *
ObjectFileELF::FindDynamicSymbol(unsigned tag)
{
    if (!ParseDynamicSymbols())
        return NULL;

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    DynamicSymbolCollIter I = m_dynamic_symbols.begin();
    DynamicSymbolCollIter E = m_dynamic_symbols.end();
    for ( ; I != E; ++I)
    {
        ELFDynamic *symbol = &*I;

        if (symbol->d_tag == tag)
            return symbol;
    }

    return NULL;
}

Section *
ObjectFileELF::PLTSection()
{
    const ELFDynamic *symbol = FindDynamicSymbol(DT_JMPREL);
    SectionList *section_list = GetSectionList();

    if (symbol && section_list)
    {
        addr_t addr = symbol->d_ptr;
        return section_list->FindSectionContainingFileAddress(addr).get();
    }

    return NULL;
}

unsigned
ObjectFileELF::PLTRelocationType()
{
    const ELFDynamic *symbol = FindDynamicSymbol(DT_PLTREL);

    if (symbol)
        return symbol->d_val;

    return 0;
}

static unsigned
ParsePLTRelocations(Symtab *symbol_table,
                    user_id_t start_id,
                    unsigned rel_type,
                    const ELFHeader *hdr,
                    const ELFSectionHeader *rel_hdr,
                    const ELFSectionHeader *plt_hdr,
                    const ELFSectionHeader *sym_hdr,
                    const lldb::SectionSP &plt_section_sp,
                    DataExtractor &rel_data,
                    DataExtractor &symtab_data,
                    DataExtractor &strtab_data)
{
    ELFRelocation rel(rel_type);
    ELFSymbol symbol;
    lldb::offset_t offset = 0;
    const elf_xword plt_entsize = plt_hdr->sh_entsize;
    const elf_xword num_relocations = rel_hdr->sh_size / rel_hdr->sh_entsize;

    typedef unsigned (*reloc_info_fn)(const ELFRelocation &rel);
    reloc_info_fn reloc_type;
    reloc_info_fn reloc_symbol;

    if (hdr->Is32Bit())
    {
        reloc_type = ELFRelocation::RelocType32;
        reloc_symbol = ELFRelocation::RelocSymbol32;
    }
    else
    {
        reloc_type = ELFRelocation::RelocType64;
        reloc_symbol = ELFRelocation::RelocSymbol64;
    }

    unsigned slot_type = hdr->GetRelocationJumpSlotType();
    unsigned i;
    for (i = 0; i < num_relocations; ++i)
    {
        if (rel.Parse(rel_data, &offset) == false)
            break;

        if (reloc_type(rel) != slot_type)
            continue;

        lldb::offset_t symbol_offset = reloc_symbol(rel) * sym_hdr->sh_entsize;
        uint64_t plt_index = (i + 1) * plt_entsize;

        if (!symbol.Parse(symtab_data, &symbol_offset))
            break;

        const char *symbol_name = strtab_data.PeekCStr(symbol.st_name);
        bool is_mangled = symbol_name ? (symbol_name[0] == '_' && symbol_name[1] == 'Z') : false;

        Symbol jump_symbol(
            i + start_id,    // Symbol table index
            symbol_name,     // symbol name.
            is_mangled,      // is the symbol name mangled?
            eSymbolTypeTrampoline, // Type of this symbol
            false,           // Is this globally visible?
            false,           // Is this symbol debug info?
            true,            // Is this symbol a trampoline?
            true,            // Is this symbol artificial?
            plt_section_sp,  // Section in which this symbol is defined or null.
            plt_index,       // Offset in section or symbol value.
            plt_entsize,     // Size in bytes of this symbol.
            true,            // Size is valid
            0);              // Symbol flags.

        symbol_table->AddSymbol(jump_symbol);
    }

    return i;
}

unsigned
ObjectFileELF::ParseTrampolineSymbols(Symtab *symbol_table,
                                      user_id_t start_id,
                                      const ELFSectionHeader *rel_hdr,
                                      user_id_t rel_id)
{
    assert(rel_hdr->sh_type == SHT_RELA || rel_hdr->sh_type == SHT_REL);

    // The link field points to the asscoiated symbol table.  The info field
    // points to the section holding the plt.
    user_id_t symtab_id = rel_hdr->sh_link;
    user_id_t plt_id = rel_hdr->sh_info;

    if (!symtab_id || !plt_id)
        return 0;

    // Section ID's are ones based;
    symtab_id++;
    plt_id++;

    const ELFSectionHeader *plt_hdr = GetSectionHeaderByIndex(plt_id);
    if (!plt_hdr)
        return 0;

    const ELFSectionHeader *sym_hdr = GetSectionHeaderByIndex(symtab_id);
    if (!sym_hdr)
        return 0;

    SectionList *section_list = GetSectionList();
    if (!section_list)
        return 0;

    Section *rel_section = section_list->FindSectionByID(rel_id).get();
    if (!rel_section)
        return 0;

    SectionSP plt_section_sp (section_list->FindSectionByID(plt_id));
    if (!plt_section_sp)
        return 0;

    Section *symtab = section_list->FindSectionByID(symtab_id).get();
    if (!symtab)
        return 0;

    Section *strtab = section_list->FindSectionByID(sym_hdr->sh_link + 1).get();
    if (!strtab)
        return 0;

    DataExtractor rel_data;
    if (!ReadSectionData(rel_section, rel_data))
        return 0;

    DataExtractor symtab_data;
    if (!ReadSectionData(symtab, symtab_data))
        return 0;

    DataExtractor strtab_data;
    if (!ReadSectionData(strtab, strtab_data))
        return 0;

    unsigned rel_type = PLTRelocationType();
    if (!rel_type)
        return 0;

    return ParsePLTRelocations (symbol_table, 
                                start_id, 
                                rel_type,
                                &m_header, 
                                rel_hdr, 
                                plt_hdr, 
                                sym_hdr,
                                plt_section_sp, 
                                rel_data, 
                                symtab_data, 
                                strtab_data);
}

Symtab *
ObjectFileELF::GetSymtab()
{
    if (m_symtab_ap.get())
        return m_symtab_ap.get();

    Symtab *symbol_table = new Symtab(this);
    m_symtab_ap.reset(symbol_table);

    Mutex::Locker locker(symbol_table->GetMutex());
    
    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return symbol_table;

    // Locate and parse all linker symbol tables.
    uint64_t symbol_id = 0;
    for (SectionHeaderCollIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I)
    {
        if (I->sh_type == SHT_SYMTAB || I->sh_type == SHT_DYNSYM)
        {
            const ELFSectionHeader &symtab_header = *I;
            user_id_t section_id = SectionIndex(I);
            symbol_id += ParseSymbolTable(symbol_table, symbol_id,
                                          &symtab_header, section_id);
        }
    }
    
    // Synthesize trampoline symbols to help navigate the PLT.
    Section *reloc_section = PLTSection();
    if (reloc_section) 
    {
        user_id_t reloc_id = reloc_section->GetID();
        const ELFSectionHeader *reloc_header = GetSectionHeaderByIndex(reloc_id);
        assert(reloc_header);

        ParseTrampolineSymbols(symbol_table, symbol_id, reloc_header, reloc_id);
    }

    return symbol_table;
}

//===----------------------------------------------------------------------===//
// Dump
//
// Dump the specifics of the runtime file container (such as any headers
// segments, sections, etc).
//----------------------------------------------------------------------
void
ObjectFileELF::Dump(Stream *s)
{
    DumpELFHeader(s, m_header);
    s->EOL();
    DumpELFProgramHeaders(s);
    s->EOL();
    DumpELFSectionHeaders(s);
    s->EOL();
    SectionList *section_list = GetSectionList();
    if (section_list)
        section_list->Dump(s, NULL, true, UINT32_MAX);
    Symtab *symtab = GetSymtab();
    if (symtab)
        symtab->Dump(s, NULL, eSortOrderNone);
    s->EOL();
    DumpDependentModules(s);
    s->EOL();
}

//----------------------------------------------------------------------
// DumpELFHeader
//
// Dump the ELF header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader(Stream *s, const ELFHeader &header)
{
    s->PutCString("ELF Header\n");
    s->Printf("e_ident[EI_MAG0   ] = 0x%2.2x\n", header.e_ident[EI_MAG0]);
    s->Printf("e_ident[EI_MAG1   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG1], header.e_ident[EI_MAG1]);
    s->Printf("e_ident[EI_MAG2   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG2], header.e_ident[EI_MAG2]);
    s->Printf("e_ident[EI_MAG3   ] = 0x%2.2x '%c'\n",
              header.e_ident[EI_MAG3], header.e_ident[EI_MAG3]);

    s->Printf("e_ident[EI_CLASS  ] = 0x%2.2x\n", header.e_ident[EI_CLASS]);
    s->Printf("e_ident[EI_DATA   ] = 0x%2.2x ", header.e_ident[EI_DATA]);
    DumpELFHeader_e_ident_EI_DATA(s, header.e_ident[EI_DATA]);
    s->Printf ("\ne_ident[EI_VERSION] = 0x%2.2x\n", header.e_ident[EI_VERSION]);
    s->Printf ("e_ident[EI_PAD    ] = 0x%2.2x\n", header.e_ident[EI_PAD]);

    s->Printf("e_type      = 0x%4.4x ", header.e_type);
    DumpELFHeader_e_type(s, header.e_type);
    s->Printf("\ne_machine   = 0x%4.4x\n", header.e_machine);
    s->Printf("e_version   = 0x%8.8x\n", header.e_version);
    s->Printf("e_entry     = 0x%8.8" PRIx64 "\n", header.e_entry);
    s->Printf("e_phoff     = 0x%8.8" PRIx64 "\n", header.e_phoff);
    s->Printf("e_shoff     = 0x%8.8" PRIx64 "\n", header.e_shoff);
    s->Printf("e_flags     = 0x%8.8x\n", header.e_flags);
    s->Printf("e_ehsize    = 0x%4.4x\n", header.e_ehsize);
    s->Printf("e_phentsize = 0x%4.4x\n", header.e_phentsize);
    s->Printf("e_phnum     = 0x%4.4x\n", header.e_phnum);
    s->Printf("e_shentsize = 0x%4.4x\n", header.e_shentsize);
    s->Printf("e_shnum     = 0x%4.4x\n", header.e_shnum);
    s->Printf("e_shstrndx  = 0x%4.4x\n", header.e_shstrndx);
}

//----------------------------------------------------------------------
// DumpELFHeader_e_type
//
// Dump an token value for the ELF header member e_type
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader_e_type(Stream *s, elf_half e_type)
{
    switch (e_type)
    {
    case ET_NONE:   *s << "ET_NONE"; break;
    case ET_REL:    *s << "ET_REL"; break;
    case ET_EXEC:   *s << "ET_EXEC"; break;
    case ET_DYN:    *s << "ET_DYN"; break;
    case ET_CORE:   *s << "ET_CORE"; break;
    default:
        break;
    }
}

//----------------------------------------------------------------------
// DumpELFHeader_e_ident_EI_DATA
//
// Dump an token value for the ELF header member e_ident[EI_DATA]
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFHeader_e_ident_EI_DATA(Stream *s, unsigned char ei_data)
{
    switch (ei_data)
    {
    case ELFDATANONE:   *s << "ELFDATANONE"; break;
    case ELFDATA2LSB:   *s << "ELFDATA2LSB - Little Endian"; break;
    case ELFDATA2MSB:   *s << "ELFDATA2MSB - Big Endian"; break;
    default:
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFProgramHeader
//
// Dump a single ELF program header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader(Stream *s, const ELFProgramHeader &ph)
{
    DumpELFProgramHeader_p_type(s, ph.p_type);
    s->Printf(" %8.8" PRIx64 " %8.8" PRIx64 " %8.8" PRIx64, ph.p_offset, ph.p_vaddr, ph.p_paddr);
    s->Printf(" %8.8" PRIx64 " %8.8" PRIx64 " %8.8x (", ph.p_filesz, ph.p_memsz, ph.p_flags);

    DumpELFProgramHeader_p_flags(s, ph.p_flags);
    s->Printf(") %8.8" PRIx64, ph.p_align);
}

//----------------------------------------------------------------------
// DumpELFProgramHeader_p_type
//
// Dump an token value for the ELF program header member p_type which
// describes the type of the program header
// ----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader_p_type(Stream *s, elf_word p_type)
{
    const int kStrWidth = 15;
    switch (p_type)
    {
    CASE_AND_STREAM(s, PT_NULL        , kStrWidth);
    CASE_AND_STREAM(s, PT_LOAD        , kStrWidth);
    CASE_AND_STREAM(s, PT_DYNAMIC     , kStrWidth);
    CASE_AND_STREAM(s, PT_INTERP      , kStrWidth);
    CASE_AND_STREAM(s, PT_NOTE        , kStrWidth);
    CASE_AND_STREAM(s, PT_SHLIB       , kStrWidth);
    CASE_AND_STREAM(s, PT_PHDR        , kStrWidth);
    CASE_AND_STREAM(s, PT_TLS         , kStrWidth);
    CASE_AND_STREAM(s, PT_GNU_EH_FRAME, kStrWidth);
    default:
        s->Printf("0x%8.8x%*s", p_type, kStrWidth - 10, "");
        break;
    }
}


//----------------------------------------------------------------------
// DumpELFProgramHeader_p_flags
//
// Dump an token value for the ELF program header member p_flags
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeader_p_flags(Stream *s, elf_word p_flags)
{
    *s  << ((p_flags & PF_X) ? "PF_X" : "    ")
        << (((p_flags & PF_X) && (p_flags & PF_W)) ? '+' : ' ')
        << ((p_flags & PF_W) ? "PF_W" : "    ")
        << (((p_flags & PF_W) && (p_flags & PF_R)) ? '+' : ' ')
        << ((p_flags & PF_R) ? "PF_R" : "    ");
}

//----------------------------------------------------------------------
// DumpELFProgramHeaders
//
// Dump all of the ELF program header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFProgramHeaders(Stream *s)
{
    if (ParseProgramHeaders())
    {
        s->PutCString("Program Headers\n");
        s->PutCString("IDX  p_type          p_offset p_vaddr  p_paddr  "
                      "p_filesz p_memsz  p_flags                   p_align\n");
        s->PutCString("==== --------------- -------- -------- -------- "
                      "-------- -------- ------------------------- --------\n");

        uint32_t idx = 0;
        for (ProgramHeaderCollConstIter I = m_program_headers.begin();
             I != m_program_headers.end(); ++I, ++idx)
        {
            s->Printf("[%2u] ", idx);
            ObjectFileELF::DumpELFProgramHeader(s, *I);
            s->EOL();
        }
    }
}

//----------------------------------------------------------------------
// DumpELFSectionHeader
//
// Dump a single ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader(Stream *s, const ELFSectionHeader &sh)
{
    s->Printf("%8.8x ", sh.sh_name);
    DumpELFSectionHeader_sh_type(s, sh.sh_type);
    s->Printf(" %8.8" PRIx64 " (", sh.sh_flags);
    DumpELFSectionHeader_sh_flags(s, sh.sh_flags);
    s->Printf(") %8.8" PRIx64 " %8.8" PRIx64 " %8.8" PRIx64, sh.sh_addr, sh.sh_offset, sh.sh_size);
    s->Printf(" %8.8x %8.8x", sh.sh_link, sh.sh_info);
    s->Printf(" %8.8" PRIx64 " %8.8" PRIx64, sh.sh_addralign, sh.sh_entsize);
}

//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_type
//
// Dump an token value for the ELF section header member sh_type which
// describes the type of the section
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader_sh_type(Stream *s, elf_word sh_type)
{
    const int kStrWidth = 12;
    switch (sh_type)
    {
    CASE_AND_STREAM(s, SHT_NULL     , kStrWidth);
    CASE_AND_STREAM(s, SHT_PROGBITS , kStrWidth);
    CASE_AND_STREAM(s, SHT_SYMTAB   , kStrWidth);
    CASE_AND_STREAM(s, SHT_STRTAB   , kStrWidth);
    CASE_AND_STREAM(s, SHT_RELA     , kStrWidth);
    CASE_AND_STREAM(s, SHT_HASH     , kStrWidth);
    CASE_AND_STREAM(s, SHT_DYNAMIC  , kStrWidth);
    CASE_AND_STREAM(s, SHT_NOTE     , kStrWidth);
    CASE_AND_STREAM(s, SHT_NOBITS   , kStrWidth);
    CASE_AND_STREAM(s, SHT_REL      , kStrWidth);
    CASE_AND_STREAM(s, SHT_SHLIB    , kStrWidth);
    CASE_AND_STREAM(s, SHT_DYNSYM   , kStrWidth);
    CASE_AND_STREAM(s, SHT_LOPROC   , kStrWidth);
    CASE_AND_STREAM(s, SHT_HIPROC   , kStrWidth);
    CASE_AND_STREAM(s, SHT_LOUSER   , kStrWidth);
    CASE_AND_STREAM(s, SHT_HIUSER   , kStrWidth);
    default:
        s->Printf("0x%8.8x%*s", sh_type, kStrWidth - 10, "");
        break;
    }
}

//----------------------------------------------------------------------
// DumpELFSectionHeader_sh_flags
//
// Dump an token value for the ELF section header member sh_flags
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeader_sh_flags(Stream *s, elf_xword sh_flags)
{
    *s  << ((sh_flags & SHF_WRITE) ? "WRITE" : "     ")
        << (((sh_flags & SHF_WRITE) && (sh_flags & SHF_ALLOC)) ? '+' : ' ')
        << ((sh_flags & SHF_ALLOC) ? "ALLOC" : "     ")
        << (((sh_flags & SHF_ALLOC) && (sh_flags & SHF_EXECINSTR)) ? '+' : ' ')
        << ((sh_flags & SHF_EXECINSTR) ? "EXECINSTR" : "         ");
}

//----------------------------------------------------------------------
// DumpELFSectionHeaders
//
// Dump all of the ELF section header to the specified output stream
//----------------------------------------------------------------------
void
ObjectFileELF::DumpELFSectionHeaders(Stream *s)
{
    if (!(ParseSectionHeaders() && GetSectionHeaderStringTable()))
        return;

    s->PutCString("Section Headers\n");
    s->PutCString("IDX  name     type         flags                            "
                  "addr     offset   size     link     info     addralgn "
                  "entsize  Name\n");
    s->PutCString("==== -------- ------------ -------------------------------- "
                  "-------- -------- -------- -------- -------- -------- "
                  "-------- ====================\n");

    uint32_t idx = 0;
    for (SectionHeaderCollConstIter I = m_section_headers.begin();
         I != m_section_headers.end(); ++I, ++idx)
    {
        s->Printf("[%2u] ", idx);
        ObjectFileELF::DumpELFSectionHeader(s, *I);
        const char* section_name = m_shstr_data.PeekCStr(I->sh_name);
        if (section_name)
            *s << ' ' << section_name << "\n";
    }
}

void
ObjectFileELF::DumpDependentModules(lldb_private::Stream *s)
{
    size_t num_modules = ParseDependentModules();

    if (num_modules > 0)
    {
        s->PutCString("Dependent Modules:\n");
        for (unsigned i = 0; i < num_modules; ++i)
        {
            const FileSpec &spec = m_filespec_ap->GetFileSpecAtIndex(i);
            s->Printf("   %s\n", spec.GetFilename().GetCString());
        }
    }
}

bool
ObjectFileELF::GetArchitecture (ArchSpec &arch)
{
    if (!ParseHeader())
        return false;

    arch.SetArchitecture (eArchTypeELF, m_header.e_machine, LLDB_INVALID_CPUTYPE);
    arch.GetTriple().setOSName (Host::GetOSString().GetCString());
    arch.GetTriple().setVendorName(Host::GetVendorString().GetCString());
    return true;
}

ObjectFile::Type
ObjectFileELF::CalculateType()
{
    switch (m_header.e_type)
    {
        case llvm::ELF::ET_NONE:
            // 0 - No file type
            return eTypeUnknown;

        case llvm::ELF::ET_REL:
            // 1 - Relocatable file
            return eTypeObjectFile;

        case llvm::ELF::ET_EXEC:
            // 2 - Executable file
            return eTypeExecutable;

        case llvm::ELF::ET_DYN:
            // 3 - Shared object file
            return eTypeSharedLibrary;

        case ET_CORE:
            // 4 - Core file
            return eTypeCoreFile;

        default:
            break;
    }
    return eTypeUnknown;
}

ObjectFile::Strata
ObjectFileELF::CalculateStrata()
{
    switch (m_header.e_type)
    {
        case llvm::ELF::ET_NONE:    
            // 0 - No file type
            return eStrataUnknown;

        case llvm::ELF::ET_REL:
            // 1 - Relocatable file
            return eStrataUnknown;

        case llvm::ELF::ET_EXEC:
            // 2 - Executable file
            // TODO: is there any way to detect that an executable is a kernel
            // related executable by inspecting the program headers, section 
            // headers, symbols, or any other flag bits???
            return eStrataUser;

        case llvm::ELF::ET_DYN:
            // 3 - Shared object file
            // TODO: is there any way to detect that an shared library is a kernel
            // related executable by inspecting the program headers, section 
            // headers, symbols, or any other flag bits???
            return eStrataUnknown;

        case ET_CORE:
            // 4 - Core file
            // TODO: is there any way to detect that an core file is a kernel
            // related executable by inspecting the program headers, section 
            // headers, symbols, or any other flag bits???
            return eStrataUnknown;

        default:
            break;
    }
    return eStrataUnknown;
}

