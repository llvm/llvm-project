//===-- DWARFCompileUnit.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFCompileUnit_h_
#define SymbolFileDWARF_DWARFCompileUnit_h_

#include "DWARFDebugInfoEntry.h"
#include "SymbolFileDWARF.h"

class NameToDIE;

class DWARFCompileUnit
{
public:
    enum Producer 
    {
        eProducerInvalid = 0,
        eProducerClang,
        eProducerGCC,
        eProducerLLVMGCC,
        eProcucerOther
    };

    DWARFCompileUnit(SymbolFileDWARF* dwarf2Data);

    bool        Extract(const lldb_private::DataExtractor &debug_info, lldb::offset_t *offset_ptr);
    dw_offset_t Extract(lldb::offset_t offset, const lldb_private::DataExtractor& debug_info_data, const DWARFAbbreviationDeclarationSet* abbrevs);
    size_t      ExtractDIEsIfNeeded (bool cu_die_only);
    bool        LookupAddress(
                    const dw_addr_t address,
                    DWARFDebugInfoEntry** function_die,
                    DWARFDebugInfoEntry** block_die);

    size_t      AppendDIEsWithTag (const dw_tag_t tag, DWARFDIECollection& matching_dies, uint32_t depth = UINT32_MAX) const;
    void        Clear();
    bool        Verify(lldb_private::Stream *s) const;
    void        Dump(lldb_private::Stream *s) const;
    dw_offset_t GetOffset() const { return m_offset; }
    uint32_t    Size() const { return 11; /* Size in bytes of the compile unit header */ }
    bool        ContainsDIEOffset(dw_offset_t die_offset) const { return die_offset >= GetFirstDIEOffset() && die_offset < GetNextCompileUnitOffset(); }
    dw_offset_t GetFirstDIEOffset() const { return m_offset + Size(); }
    dw_offset_t GetNextCompileUnitOffset() const { return m_offset + m_length + 4; }
    size_t      GetDebugInfoSize() const { return m_length + 4 - Size(); /* Size in bytes of the .debug_info data associated with this compile unit. */ }
    uint32_t    GetLength() const { return m_length; }
    uint16_t    GetVersion() const { return m_version; }
    const DWARFAbbreviationDeclarationSet*  GetAbbreviations() const { return m_abbrevs; }
    dw_offset_t GetAbbrevOffset() const;
    uint8_t     GetAddressByteSize() const { return m_addr_size; }
    dw_addr_t   GetBaseAddress() const { return m_base_addr; }
    void        ClearDIEs(bool keep_compile_unit_die);
    void        BuildAddressRangeTable (SymbolFileDWARF* dwarf2Data,
                                        DWARFDebugAranges* debug_aranges,
                                        bool clear_dies_if_already_not_parsed);

    void
    SetBaseAddress(dw_addr_t base_addr)
    {
        m_base_addr = base_addr;
    }

    const DWARFDebugInfoEntry*
    GetCompileUnitDIEOnly()
    {
        ExtractDIEsIfNeeded (true);
        if (m_die_array.empty())
            return NULL;
        return &m_die_array[0];
    }

    const DWARFDebugInfoEntry*
    DIE()
    {
        ExtractDIEsIfNeeded (false);
        if (m_die_array.empty())
            return NULL;
        return &m_die_array[0];
    }

    void
    AddDIE (DWARFDebugInfoEntry& die)
    {
        // The average bytes per DIE entry has been seen to be
        // around 14-20 so lets pre-reserve half of that since
        // we are now stripping the NULL tags. 

        // Only reserve the memory if we are adding children of
        // the main compile unit DIE. The compile unit DIE is always
        // the first entry, so if our size is 1, then we are adding
        // the first compile unit child DIE and should reserve
        // the memory.
        if (m_die_array.empty())
            m_die_array.reserve(GetDebugInfoSize() / 24);
        m_die_array.push_back(die);
    }

    bool
    HasDIEsParsed () const
    {
        return m_die_array.size() > 1;
    }

    DWARFDebugInfoEntry*
    GetDIEAtIndexUnchecked (uint32_t idx)
    {
        return &m_die_array[idx];
    }

    DWARFDebugInfoEntry*
    GetDIEPtr (dw_offset_t die_offset);

    const DWARFDebugInfoEntry*
    GetDIEPtrContainingOffset (dw_offset_t die_offset);

    static uint8_t
    GetAddressByteSize(const DWARFCompileUnit* cu);

    static uint8_t
    GetDefaultAddressSize();

    static void
    SetDefaultAddressSize(uint8_t addr_size);

    void *
    GetUserData() const
    {
        return m_user_data;
    }

    void
    SetUserData(void *d)
    {
        m_user_data = d;
    }

    bool
    Supports_DW_AT_APPLE_objc_complete_type ();

    bool
    DW_AT_decl_file_attributes_are_invalid();

    bool
    Supports_unnamed_objc_bitfields ();

//    void
//    AddGlobalDIEByIndex (uint32_t die_idx);
//
//    void
//    AddGlobal (const DWARFDebugInfoEntry* die);
//
    void
    Index (const uint32_t cu_idx,
           NameToDIE& func_basenames,
           NameToDIE& func_fullnames,
           NameToDIE& func_methods,
           NameToDIE& func_selectors,
           NameToDIE& objc_class_selectors,
           NameToDIE& globals,
           NameToDIE& types,
           NameToDIE& namespaces);

    const DWARFDebugAranges &
    GetFunctionAranges ();

    SymbolFileDWARF*
    GetSymbolFileDWARF () const
    {
        return m_dwarf2Data;
    }
    
    Producer
    GetProducer ();
    
    uint32_t
    GetProducerVersionMajor();

    uint32_t
    GetProducerVersionMinor();
    
    uint32_t
    GetProducerVersionUpdate();

protected:
    SymbolFileDWARF*    m_dwarf2Data;
    const DWARFAbbreviationDeclarationSet *m_abbrevs;
    void *              m_user_data;
    DWARFDebugInfoEntry::collection m_die_array;    // The compile unit debug information entry item
    std::unique_ptr<DWARFDebugAranges> m_func_aranges_ap;   // A table similar to the .debug_aranges table, but this one points to the exact DW_TAG_subprogram DIEs
    dw_addr_t           m_base_addr;
    dw_offset_t         m_offset;
    uint32_t            m_length;
    uint16_t            m_version;
    uint8_t             m_addr_size;
    Producer            m_producer;
    uint32_t            m_producer_version_major;
    uint32_t            m_producer_version_minor;
    uint32_t            m_producer_version_update;
    
    void
    ParseProducerInfo ();
private:
    DISALLOW_COPY_AND_ASSIGN (DWARFCompileUnit);
};

#endif  // SymbolFileDWARF_DWARFCompileUnit_h_
