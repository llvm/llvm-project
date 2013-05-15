//===-- DWARFCompileUnit.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFCompileUnit.h"

#include "lldb/Core/Mangled.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDIECollection.h"
#include "DWARFFormValue.h"
#include "LogChannelDWARF.h"
#include "NameToDIE.h"
#include "SymbolFileDWARF.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;


extern int g_verbose;

DWARFCompileUnit::DWARFCompileUnit(SymbolFileDWARF* dwarf2Data) :
    m_dwarf2Data    (dwarf2Data),
    m_abbrevs       (NULL),
    m_user_data     (NULL),
    m_die_array     (),
    m_func_aranges_ap (),
    m_base_addr     (0),
    m_offset        (DW_INVALID_OFFSET),
    m_length        (0),
    m_version       (0),
    m_addr_size     (DWARFCompileUnit::GetDefaultAddressSize()),
    m_producer      (eProducerInvalid),
    m_producer_version_major (0),
    m_producer_version_minor (0),
    m_producer_version_update (0)
{
}

void
DWARFCompileUnit::Clear()
{
    m_offset        = DW_INVALID_OFFSET;
    m_length        = 0;
    m_version       = 0;
    m_abbrevs       = NULL;
    m_addr_size     = DWARFCompileUnit::GetDefaultAddressSize();
    m_base_addr     = 0;
    m_die_array.clear();
    m_func_aranges_ap.reset();
    m_user_data     = NULL;
    m_producer      = eProducerInvalid;
}

bool
DWARFCompileUnit::Extract(const DataExtractor &debug_info, lldb::offset_t *offset_ptr)
{
    Clear();

    m_offset = *offset_ptr;

    if (debug_info.ValidOffset(*offset_ptr))
    {
        dw_offset_t abbr_offset;
        const DWARFDebugAbbrev *abbr = m_dwarf2Data->DebugAbbrev();
        m_length        = debug_info.GetU32(offset_ptr);
        m_version       = debug_info.GetU16(offset_ptr);
        abbr_offset     = debug_info.GetU32(offset_ptr);
        m_addr_size     = debug_info.GetU8 (offset_ptr);

        bool length_OK = debug_info.ValidOffset(GetNextCompileUnitOffset()-1);
        bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
        bool abbr_offset_OK = m_dwarf2Data->get_debug_abbrev_data().ValidOffset(abbr_offset);
        bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));

        if (length_OK && version_OK && addr_size_OK && abbr_offset_OK && abbr != NULL)
        {
            m_abbrevs = abbr->GetAbbreviationDeclarationSet(abbr_offset);
            return true;
        }

        // reset the offset to where we tried to parse from if anything went wrong
        *offset_ptr = m_offset;
    }

    return false;
}


dw_offset_t
DWARFCompileUnit::Extract(lldb::offset_t offset, const DataExtractor& debug_info_data, const DWARFAbbreviationDeclarationSet* abbrevs)
{
    Clear();

    m_offset = offset;

    if (debug_info_data.ValidOffset(offset))
    {
        m_length        = debug_info_data.GetU32(&offset);
        m_version       = debug_info_data.GetU16(&offset);
        bool abbrevs_OK = debug_info_data.GetU32(&offset) == abbrevs->GetOffset();
        m_abbrevs       = abbrevs;
        m_addr_size     = debug_info_data.GetU8 (&offset);

        bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
        bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));

        if (version_OK && addr_size_OK && abbrevs_OK && debug_info_data.ValidOffset(offset))
            return offset;
    }
    return DW_INVALID_OFFSET;
}

void
DWARFCompileUnit::ClearDIEs(bool keep_compile_unit_die)
{
    if (m_die_array.size() > 1)
    {
        // std::vectors never get any smaller when resized to a smaller size,
        // or when clear() or erase() are called, the size will report that it
        // is smaller, but the memory allocated remains intact (call capacity()
        // to see this). So we need to create a temporary vector and swap the
        // contents which will cause just the internal pointers to be swapped
        // so that when "tmp_array" goes out of scope, it will destroy the
        // contents.

        // Save at least the compile unit DIE
        DWARFDebugInfoEntry::collection tmp_array;
        m_die_array.swap(tmp_array);
        if (keep_compile_unit_die)
            m_die_array.push_back(tmp_array.front());
    }
}

//----------------------------------------------------------------------
// ParseCompileUnitDIEsIfNeeded
//
// Parses a compile unit and indexes its DIEs if it hasn't already been
// done.
//----------------------------------------------------------------------
size_t
DWARFCompileUnit::ExtractDIEsIfNeeded (bool cu_die_only)
{
    const size_t initial_die_array_size = m_die_array.size();
    if ((cu_die_only && initial_die_array_size > 0) || initial_die_array_size > 1)
        return 0; // Already parsed

    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "%8.8x: DWARFCompileUnit::ExtractDIEsIfNeeded( cu_die_only = %i )",
                        m_offset,
                        cu_die_only);

    // Set the offset to that of the first DIE and calculate the start of the
    // next compilation unit header.
    lldb::offset_t offset = GetFirstDIEOffset();
    lldb::offset_t next_cu_offset = GetNextCompileUnitOffset();

    DWARFDebugInfoEntry die;
        // Keep a flat array of the DIE for binary lookup by DIE offset
    if (!cu_die_only)
    {
        Log *log (LogChannelDWARF::GetLogIfAny(DWARF_LOG_DEBUG_INFO | DWARF_LOG_LOOKUPS));
        if (log)
        {
            m_dwarf2Data->GetObjectFile()->GetModule()->LogMessageVerboseBacktrace (log,
                                                                                    "DWARFCompileUnit::ExtractDIEsIfNeeded () for compile unit at .debug_info[0x%8.8x]",
                                                                                    GetOffset());
        }
    }

    uint32_t depth = 0;
    // We are in our compile unit, parse starting at the offset
    // we were told to parse
    const DataExtractor& debug_info_data = m_dwarf2Data->get_debug_info_data();
    std::vector<uint32_t> die_index_stack;
    die_index_stack.reserve(32);
    die_index_stack.push_back(0);
    bool prev_die_had_children = false;
    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (GetAddressByteSize());
    while (offset < next_cu_offset &&
           die.FastExtract (debug_info_data, this, fixed_form_sizes, &offset))
    {
//        if (log)
//            log->Printf("0x%8.8x: %*.*s%s%s",
//                        die.GetOffset(),
//                        depth * 2, depth * 2, "",
//                        DW_TAG_value_to_name (die.Tag()),
//                        die.HasChildren() ? " *" : "");

        const bool null_die = die.IsNULL();
        if (depth == 0)
        {
            uint64_t base_addr = die.GetAttributeValueAsUnsigned(m_dwarf2Data, this, DW_AT_low_pc, LLDB_INVALID_ADDRESS);
            if (base_addr == LLDB_INVALID_ADDRESS)
                base_addr = die.GetAttributeValueAsUnsigned(m_dwarf2Data, this, DW_AT_entry_pc, 0);
            SetBaseAddress (base_addr);
            if (initial_die_array_size == 0)
                AddDIE (die);
            if (cu_die_only)
                return 1;
        }
        else
        {
            if (null_die)
            {
                if (prev_die_had_children)
                {
                    // This will only happen if a DIE says is has children
                    // but all it contains is a NULL tag. Since we are removing
                    // the NULL DIEs from the list (saves up to 25% in C++ code),
                    // we need a way to let the DIE know that it actually doesn't
                    // have children.
                    if (!m_die_array.empty())
                        m_die_array.back().SetEmptyChildren(true);
                }
            }
            else
            {
                die.SetParentIndex(m_die_array.size() - die_index_stack[depth-1]);

                if (die_index_stack.back())
                    m_die_array[die_index_stack.back()].SetSiblingIndex(m_die_array.size()-die_index_stack.back());
                
                // Only push the DIE if it isn't a NULL DIE
                    m_die_array.push_back(die);
            }
        }

        if (null_die)
        {
            // NULL DIE.
            if (!die_index_stack.empty())
                die_index_stack.pop_back();

            if (depth > 0)
                --depth;
            if (depth == 0)
                break;  // We are done with this compile unit!

            prev_die_had_children = false;
        }
        else
        {
            die_index_stack.back() = m_die_array.size() - 1;
            // Normal DIE
            const bool die_has_children = die.HasChildren();
            if (die_has_children)
            {
                die_index_stack.push_back(0);
                ++depth;
            }
            prev_die_had_children = die_has_children;
        }
    }

    // Give a little bit of info if we encounter corrupt DWARF (our offset
    // should always terminate at or before the start of the next compilation
    // unit header).
    if (offset > next_cu_offset)
    {
        m_dwarf2Data->GetObjectFile()->GetModule()->ReportWarning ("DWARF compile unit extends beyond its bounds cu 0x%8.8x at 0x%8.8" PRIx64 "\n",
                                                                   GetOffset(), 
                                                                   offset);
    }

    // Since std::vector objects will double their size, we really need to
    // make a new array with the perfect size so we don't end up wasting
    // space. So here we copy and swap to make sure we don't have any extra
    // memory taken up.
    
    if (m_die_array.size () < m_die_array.capacity())
    {
        DWARFDebugInfoEntry::collection exact_size_die_array (m_die_array.begin(), m_die_array.end());
        exact_size_die_array.swap (m_die_array);
    }
    Log *log (LogChannelDWARF::GetLogIfAll (DWARF_LOG_DEBUG_INFO | DWARF_LOG_VERBOSE));
    if (log)
    {
        StreamString strm;
        DWARFDebugInfoEntry::DumpDIECollection (strm, m_die_array);
        log->PutCString (strm.GetString().c_str());
    }

    return m_die_array.size();
}


dw_offset_t
DWARFCompileUnit::GetAbbrevOffset() const
{
    return m_abbrevs ? m_abbrevs->GetOffset() : DW_INVALID_OFFSET;
}



bool
DWARFCompileUnit::Verify(Stream *s) const
{
    const DataExtractor& debug_info = m_dwarf2Data->get_debug_info_data();
    bool valid_offset = debug_info.ValidOffset(m_offset);
    bool length_OK = debug_info.ValidOffset(GetNextCompileUnitOffset()-1);
    bool version_OK = SymbolFileDWARF::SupportedVersion(m_version);
    bool abbr_offset_OK = m_dwarf2Data->get_debug_abbrev_data().ValidOffset(GetAbbrevOffset());
    bool addr_size_OK = ((m_addr_size == 4) || (m_addr_size == 8));
    bool verbose = s->GetVerbose();
    if (valid_offset && length_OK && version_OK && addr_size_OK && abbr_offset_OK)
    {
        if (verbose)
            s->Printf("    0x%8.8x: OK\n", m_offset);
        return true;
    }
    else
    {
        s->Printf("    0x%8.8x: ", m_offset);

        m_dwarf2Data->get_debug_info_data().Dump (s, m_offset, lldb::eFormatHex, 1, Size(), 32, LLDB_INVALID_ADDRESS, 0, 0);
        s->EOL();
        if (valid_offset)
        {
            if (!length_OK)
                s->Printf("        The length (0x%8.8x) for this compile unit is too large for the .debug_info provided.\n", m_length);
            if (!version_OK)
                s->Printf("        The 16 bit compile unit header version is not supported.\n");
            if (!abbr_offset_OK)
                s->Printf("        The offset into the .debug_abbrev section (0x%8.8x) is not valid.\n", GetAbbrevOffset());
            if (!addr_size_OK)
                s->Printf("        The address size is unsupported: 0x%2.2x\n", m_addr_size);
        }
        else
            s->Printf("        The start offset of the compile unit header in the .debug_info is invalid.\n");
    }
    return false;
}


void
DWARFCompileUnit::Dump(Stream *s) const
{
    s->Printf("0x%8.8x: Compile Unit: length = 0x%8.8x, version = 0x%4.4x, abbr_offset = 0x%8.8x, addr_size = 0x%2.2x (next CU at {0x%8.8x})\n",
                m_offset, m_length, m_version, GetAbbrevOffset(), m_addr_size, GetNextCompileUnitOffset());
}


static uint8_t g_default_addr_size = 4;

uint8_t
DWARFCompileUnit::GetAddressByteSize(const DWARFCompileUnit* cu)
{
    if (cu)
        return cu->GetAddressByteSize();
    return DWARFCompileUnit::GetDefaultAddressSize();
}

uint8_t
DWARFCompileUnit::GetDefaultAddressSize()
{
    return g_default_addr_size;
}

void
DWARFCompileUnit::SetDefaultAddressSize(uint8_t addr_size)
{
    g_default_addr_size = addr_size;
}

void
DWARFCompileUnit::BuildAddressRangeTable (SymbolFileDWARF* dwarf2Data,
                                          DWARFDebugAranges* debug_aranges,
                                          bool clear_dies_if_already_not_parsed)
{
    // This function is usually called if there in no .debug_aranges section
    // in order to produce a compile unit level set of address ranges that
    // is accurate. If the DIEs weren't parsed, then we don't want all dies for
    // all compile units to stay loaded when they weren't needed. So we can end
    // up parsing the DWARF and then throwing them all away to keep memory usage
    // down.
    const bool clear_dies = ExtractDIEsIfNeeded (false) > 1;
    
    const DWARFDebugInfoEntry* die = DIE();
    if (die)
        die->BuildAddressRangeTable(dwarf2Data, this, debug_aranges);
    
    if (debug_aranges->IsEmpty())
    {
        // We got nothing from the functions, maybe we have a line tables only
        // situation. Check the line tables and build the arange table from this.
        SymbolContext sc;
        sc.comp_unit = dwarf2Data->GetCompUnitForDWARFCompUnit(this);
        if (sc.comp_unit)
        {
            LineTable *line_table = sc.comp_unit->GetLineTable();

            if (line_table)
            {
                LineTable::FileAddressRanges file_ranges;
                const bool append = true;
                const size_t num_ranges = line_table->GetContiguousFileAddressRanges (file_ranges, append);
                for (uint32_t idx=0; idx<num_ranges; ++idx)
                {
                    const LineTable::FileAddressRanges::Entry &range = file_ranges.GetEntryRef(idx);
                    debug_aranges->AppendRange(GetOffset(), range.GetRangeBase(), range.GetRangeEnd());
                    printf ("0x%8.8x: [0x%16.16" PRIx64 " - 0x%16.16" PRIx64 ")\n", GetOffset(), range.GetRangeBase(), range.GetRangeEnd());
                }
            }
        }
    }
    
    // Keep memory down by clearing DIEs if this generate function
    // caused them to be parsed
    if (clear_dies)
        ClearDIEs (true);

}


const DWARFDebugAranges &
DWARFCompileUnit::GetFunctionAranges ()
{
    if (m_func_aranges_ap.get() == NULL)
    {
        m_func_aranges_ap.reset (new DWARFDebugAranges());
        Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_ARANGES));

        if (log)
        {
            m_dwarf2Data->GetObjectFile()->GetModule()->LogMessage (log,
                                                                    "DWARFCompileUnit::GetFunctionAranges() for compile unit at .debug_info[0x%8.8x]",
                                                                    GetOffset());
        }
        const DWARFDebugInfoEntry* die = DIE();
        if (die)
            die->BuildFunctionAddressRangeTable (m_dwarf2Data, this, m_func_aranges_ap.get());
        const bool minimize = false;
        m_func_aranges_ap->Sort(minimize);
    }
    return *m_func_aranges_ap.get();
}

bool
DWARFCompileUnit::LookupAddress
(
    const dw_addr_t address,
    DWARFDebugInfoEntry** function_die_handle,
    DWARFDebugInfoEntry** block_die_handle
)
{
    bool success = false;

    if (function_die_handle != NULL && DIE())
    {

        const DWARFDebugAranges &func_aranges = GetFunctionAranges ();

        // Re-check the aranges auto pointer contents in case it was created above
        if (!func_aranges.IsEmpty())
        {
            *function_die_handle = GetDIEPtr(func_aranges.FindAddress(address));
            if (*function_die_handle != NULL)
            {
                success = true;
                if (block_die_handle != NULL)
                {
                    DWARFDebugInfoEntry* child = (*function_die_handle)->GetFirstChild();
                    while (child)
                    {
                        if (child->LookupAddress(address, m_dwarf2Data, this, NULL, block_die_handle))
                            break;
                        child = child->GetSibling();
                    }
                }
            }
        }
    }
    return success;
}

//----------------------------------------------------------------------
// Compare function DWARFDebugAranges::Range structures
//----------------------------------------------------------------------
static bool CompareDIEOffset (const DWARFDebugInfoEntry& die1, const DWARFDebugInfoEntry& die2)
{
    return die1.GetOffset() < die2.GetOffset();
}

//----------------------------------------------------------------------
// GetDIEPtr()
//
// Get the DIE (Debug Information Entry) with the specified offset.
//----------------------------------------------------------------------
DWARFDebugInfoEntry*
DWARFCompileUnit::GetDIEPtr(dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        ExtractDIEsIfNeeded (false);
        DWARFDebugInfoEntry compare_die;
        compare_die.SetOffset(die_offset);
        DWARFDebugInfoEntry::iterator end = m_die_array.end();
        DWARFDebugInfoEntry::iterator pos = lower_bound(m_die_array.begin(), end, compare_die, CompareDIEOffset);
        if (pos != end)
        {
            if (die_offset == (*pos).GetOffset())
                return &(*pos);
        }
    }
    return NULL;    // Not found in any compile units
}

//----------------------------------------------------------------------
// GetDIEPtrContainingOffset()
//
// Get the DIE (Debug Information Entry) that contains the specified
// .debug_info offset.
//----------------------------------------------------------------------
const DWARFDebugInfoEntry*
DWARFCompileUnit::GetDIEPtrContainingOffset(dw_offset_t die_offset)
{
    if (die_offset != DW_INVALID_OFFSET)
    {
        ExtractDIEsIfNeeded (false);
        DWARFDebugInfoEntry compare_die;
        compare_die.SetOffset(die_offset);
        DWARFDebugInfoEntry::iterator end = m_die_array.end();
        DWARFDebugInfoEntry::iterator pos = lower_bound(m_die_array.begin(), end, compare_die, CompareDIEOffset);
        if (pos != end)
        {
            if (die_offset >= (*pos).GetOffset())
            {
                DWARFDebugInfoEntry::iterator next = pos + 1;
                if (next != end)
                {
                    if (die_offset < (*next).GetOffset())
                        return &(*pos);
                }
            }
        }
    }
    return NULL;    // Not found in any compile units
}



size_t
DWARFCompileUnit::AppendDIEsWithTag (const dw_tag_t tag, DWARFDIECollection& dies, uint32_t depth) const
{
    size_t old_size = dies.Size();
    DWARFDebugInfoEntry::const_iterator pos;
    DWARFDebugInfoEntry::const_iterator end = m_die_array.end();
    for (pos = m_die_array.begin(); pos != end; ++pos)
    {
        if (pos->Tag() == tag)
            dies.Append (&(*pos));
    }

    // Return the number of DIEs added to the collection
    return dies.Size() - old_size;
}

//void
//DWARFCompileUnit::AddGlobalDIEByIndex (uint32_t die_idx)
//{
//    m_global_die_indexes.push_back (die_idx);
//}
//
//
//void
//DWARFCompileUnit::AddGlobal (const DWARFDebugInfoEntry* die)
//{
//    // Indexes to all file level global and static variables
//    m_global_die_indexes;
//    
//    if (m_die_array.empty())
//        return;
//    
//    const DWARFDebugInfoEntry* first_die = &m_die_array[0];
//    const DWARFDebugInfoEntry* end = first_die + m_die_array.size();
//    if (first_die <= die && die < end)
//        m_global_die_indexes.push_back (die - first_die);
//}


void
DWARFCompileUnit::Index (const uint32_t cu_idx,
                         NameToDIE& func_basenames,
                         NameToDIE& func_fullnames,
                         NameToDIE& func_methods,
                         NameToDIE& func_selectors,
                         NameToDIE& objc_class_selectors,
                         NameToDIE& globals,
                         NameToDIE& types,
                         NameToDIE& namespaces)
{
    const DataExtractor* debug_str = &m_dwarf2Data->get_debug_str_data();

    const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (GetAddressByteSize());

    Log *log (LogChannelDWARF::GetLogIfAll (DWARF_LOG_LOOKUPS));
    
    if (log)
    {
        m_dwarf2Data->GetObjectFile()->GetModule()->LogMessage (log, 
                                                                "DWARFCompileUnit::Index() for compile unit at .debug_info[0x%8.8x]",
                                                                GetOffset());
    }

    DWARFDebugInfoEntry::const_iterator pos;
    DWARFDebugInfoEntry::const_iterator begin = m_die_array.begin();
    DWARFDebugInfoEntry::const_iterator end = m_die_array.end();
    for (pos = begin; pos != end; ++pos)
    {
        const DWARFDebugInfoEntry &die = *pos;
        
        const dw_tag_t tag = die.Tag();
    
        switch (tag)
        {
        case DW_TAG_subprogram:
        case DW_TAG_inlined_subroutine:
        case DW_TAG_base_type:
        case DW_TAG_class_type:
        case DW_TAG_constant:
        case DW_TAG_enumeration_type:
        case DW_TAG_string_type:
        case DW_TAG_subroutine_type:
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_typedef:
        case DW_TAG_namespace:
        case DW_TAG_variable:
        case DW_TAG_unspecified_type:
            break;
            
        default:
            continue;
        }

        DWARFDebugInfoEntry::Attributes attributes;
        const char *name = NULL;
        const char *mangled_cstr = NULL;
        bool is_declaration = false;
        //bool is_artificial = false;
        bool has_address = false;
        bool has_location = false;
        bool is_global_or_static_variable = false;
        
        dw_offset_t specification_die_offset = DW_INVALID_OFFSET;
        const size_t num_attributes = die.GetAttributes(m_dwarf2Data, this, fixed_form_sizes, attributes);
        if (num_attributes > 0)
        {
            for (uint32_t i=0; i<num_attributes; ++i)
            {
                dw_attr_t attr = attributes.AttributeAtIndex(i);
                DWARFFormValue form_value;
                switch (attr)
                {
                case DW_AT_name:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        name = form_value.AsCString(debug_str);
                    break;

                case DW_AT_declaration:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        is_declaration = form_value.Unsigned() != 0;
                    break;

//                case DW_AT_artificial:
//                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
//                        is_artificial = form_value.Unsigned() != 0;
//                    break;

                case DW_AT_MIPS_linkage_name:
                case DW_AT_linkage_name:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        mangled_cstr = form_value.AsCString(debug_str);                        
                    break;

                case DW_AT_low_pc:
                case DW_AT_high_pc:
                case DW_AT_ranges:
                    has_address = true;
                    break;

                case DW_AT_entry_pc:
                    has_address = true;
                    break;

                case DW_AT_location:
                    has_location = true;
                    if (tag == DW_TAG_variable)
                    {
                        const DWARFDebugInfoEntry* parent_die = die.GetParent();
                        while ( parent_die != NULL )
                        {
                            switch (parent_die->Tag())
                            {
                            case DW_TAG_subprogram:
                            case DW_TAG_lexical_block:
                            case DW_TAG_inlined_subroutine:
                                // Even if this is a function level static, we don't add it. We could theoretically
                                // add these if we wanted to by introspecting into the DW_AT_location and seeing
                                // if the location describes a hard coded address, but we dont want the performance
                                // penalty of that right now.
                                is_global_or_static_variable = false;
//                              if (attributes.ExtractFormValueAtIndex(dwarf2Data, i, form_value))
//                              {
//                                  // If we have valid block data, then we have location expression bytes
//                                  // that are fixed (not a location list).
//                                  const uint8_t *block_data = form_value.BlockData();
//                                  if (block_data)
//                                  {
//                                      uint32_t block_length = form_value.Unsigned();
//                                      if (block_length == 1 + attributes.CompileUnitAtIndex(i)->GetAddressByteSize())
//                                      {
//                                          if (block_data[0] == DW_OP_addr)
//                                              add_die = true;
//                                      }
//                                  }
//                              }
                                parent_die = NULL;  // Terminate the while loop.
                                break;

                            case DW_TAG_compile_unit:
                                is_global_or_static_variable = true;
                                parent_die = NULL;  // Terminate the while loop.
                                break;

                            default:
                                parent_die = parent_die->GetParent();   // Keep going in the while loop.
                                break;
                            }
                        }
                    }
                    break;
                    
                case DW_AT_specification:
                    if (attributes.ExtractFormValueAtIndex(m_dwarf2Data, i, form_value))
                        specification_die_offset = form_value.Reference(this);
                    break;
                }
            }
        }

        switch (tag)
        {
        case DW_TAG_subprogram:
            if (has_address)
            {
                if (name)
                {
                    // Note, this check is also done in ParseMethodName, but since this is a hot loop, we do the
                    // simple inlined check outside the call.
                    ObjCLanguageRuntime::MethodName objc_method(name, true);
                    if (objc_method.IsValid(true))
                    {
                        ConstString objc_class_name_with_category (objc_method.GetClassNameWithCategory());
                        ConstString objc_selector_name (objc_method.GetSelector());
                        ConstString objc_fullname_no_category_name (objc_method.GetFullNameWithoutCategory(true));
                        ConstString objc_class_name_no_category (objc_method.GetClassName());
                        func_fullnames.Insert (ConstString(name), die.GetOffset());
                        if (objc_class_name_with_category)
                            objc_class_selectors.Insert(objc_class_name_with_category, die.GetOffset());
                        if (objc_class_name_no_category && objc_class_name_no_category != objc_class_name_with_category)
                            objc_class_selectors.Insert(objc_class_name_no_category, die.GetOffset());
                        if (objc_selector_name)
                            func_selectors.Insert (objc_selector_name, die.GetOffset());
                        if (objc_fullname_no_category_name)
                            func_fullnames.Insert (objc_fullname_no_category_name, die.GetOffset());
                    }
                    // If we have a mangled name, then the DW_AT_name attribute
                    // is usually the method name without the class or any parameters
                    const DWARFDebugInfoEntry *parent = die.GetParent();
                    bool is_method = false;
                    if (parent)
                    {
                        dw_tag_t parent_tag = parent->Tag();
                        if (parent_tag == DW_TAG_class_type || parent_tag == DW_TAG_structure_type)
                        {
                            is_method = true;
                        }
                        else
                        {
                            if (specification_die_offset != DW_INVALID_OFFSET)
                            {
                                const DWARFDebugInfoEntry *specification_die = m_dwarf2Data->DebugInfo()->GetDIEPtr (specification_die_offset, NULL);
                                if (specification_die)
                                {
                                    parent = specification_die->GetParent();
                                    if (parent)
                                    {
                                        parent_tag = parent->Tag();
                                    
                                        if (parent_tag == DW_TAG_class_type || parent_tag == DW_TAG_structure_type)
                                            is_method = true;
                                    }
                                }
                            }
                        }
                    }


                    if (is_method)
                        func_methods.Insert (ConstString(name), die.GetOffset());
                    else
                        func_basenames.Insert (ConstString(name), die.GetOffset());

                    if (!is_method && !mangled_cstr && !objc_method.IsValid(true))
                        func_fullnames.Insert (ConstString(name), die.GetOffset());
                }
                if (mangled_cstr)
                {
                    // Make sure our mangled name isn't the same string table entry
                    // as our name. If it starts with '_', then it is ok, else compare
                    // the string to make sure it isn't the same and we don't end up
                    // with duplicate entries
                    if (name != mangled_cstr && ((mangled_cstr[0] == '_') || (name && ::strcmp(name, mangled_cstr) != 0)))
                    {
                        Mangled mangled (ConstString(mangled_cstr), true);
                        func_fullnames.Insert (mangled.GetMangledName(), die.GetOffset());
                        if (mangled.GetDemangledName())
                            func_fullnames.Insert (mangled.GetDemangledName(), die.GetOffset());
                    }
                }
            }
            break;

        case DW_TAG_inlined_subroutine:
            if (has_address)
            {
                if (name)
                    func_basenames.Insert (ConstString(name), die.GetOffset());
                if (mangled_cstr)
                {
                    // Make sure our mangled name isn't the same string table entry
                    // as our name. If it starts with '_', then it is ok, else compare
                    // the string to make sure it isn't the same and we don't end up
                    // with duplicate entries
                    if (name != mangled_cstr && ((mangled_cstr[0] == '_') || (::strcmp(name, mangled_cstr) != 0)))
                    {
                        Mangled mangled (ConstString(mangled_cstr), true);
                        func_fullnames.Insert (mangled.GetMangledName(), die.GetOffset());
                        if (mangled.GetDemangledName())
                            func_fullnames.Insert (mangled.GetDemangledName(), die.GetOffset());
                    }
                }
                else
                    func_fullnames.Insert (ConstString(name), die.GetOffset());
            }
            break;
        
        case DW_TAG_base_type:
        case DW_TAG_class_type:
        case DW_TAG_constant:
        case DW_TAG_enumeration_type:
        case DW_TAG_string_type:
        case DW_TAG_subroutine_type:
        case DW_TAG_structure_type:
        case DW_TAG_union_type:
        case DW_TAG_typedef:
        case DW_TAG_unspecified_type:
            if (name && is_declaration == false)
            {
                types.Insert (ConstString(name), die.GetOffset());
            }
            break;

        case DW_TAG_namespace:
            if (name)
                namespaces.Insert (ConstString(name), die.GetOffset());
            break;

        case DW_TAG_variable:
            if (name && has_location && is_global_or_static_variable)
            {
                globals.Insert (ConstString(name), die.GetOffset());
                // Be sure to include variables by their mangled and demangled
                // names if they have any since a variable can have a basename
                // "i", a mangled named "_ZN12_GLOBAL__N_11iE" and a demangled 
                // mangled name "(anonymous namespace)::i"...
                
                // Make sure our mangled name isn't the same string table entry
                // as our name. If it starts with '_', then it is ok, else compare
                // the string to make sure it isn't the same and we don't end up
                // with duplicate entries
                if (mangled_cstr && name != mangled_cstr && ((mangled_cstr[0] == '_') || (::strcmp(name, mangled_cstr) != 0)))
                {
                    Mangled mangled (ConstString(mangled_cstr), true);
                    globals.Insert (mangled.GetMangledName(), die.GetOffset());
                    if (mangled.GetDemangledName())
                        globals.Insert (mangled.GetDemangledName(), die.GetOffset());
                }
            }
            break;
            
        default:
            continue;
        }
    }
}

bool
DWARFCompileUnit::Supports_unnamed_objc_bitfields ()
{
    if (GetProducer() == eProducerClang)
    {
        const uint32_t major_version = GetProducerVersionMajor();
        if (major_version > 425 || (major_version == 425 && GetProducerVersionUpdate() >= 13))
            return true;
        else
            return false;
    }
    return true; // Assume all other compilers didn't have incorrect ObjC bitfield info
}

bool
DWARFCompileUnit::Supports_DW_AT_APPLE_objc_complete_type ()
{
    if (GetProducer() == eProducerLLVMGCC)
        return false;
    return true;
}

bool
DWARFCompileUnit::DW_AT_decl_file_attributes_are_invalid()
{
    // llvm-gcc makes completely invalid decl file attributes and won't ever
    // be fixed, so we need to know to ignore these.
    return GetProducer() == eProducerLLVMGCC;
}

void
DWARFCompileUnit::ParseProducerInfo ()
{
    m_producer_version_major = UINT32_MAX;
    m_producer_version_minor = UINT32_MAX;
    m_producer_version_update = UINT32_MAX;

    const DWARFDebugInfoEntry *die = GetCompileUnitDIEOnly();
    if (die)
    {

        const char *producer_cstr = die->GetAttributeValueAsString(m_dwarf2Data, this, DW_AT_producer, NULL);
        if (producer_cstr)
        {
            RegularExpression llvm_gcc_regex("^4\\.[012]\\.[01] \\(Based on Apple Inc\\. build [0-9]+\\) \\(LLVM build [\\.0-9]+\\)$");
            if (llvm_gcc_regex.Execute (producer_cstr))
            {
                m_producer = eProducerLLVMGCC;
            }
            else if (strstr(producer_cstr, "clang"))
            {
                static RegularExpression g_clang_version_regex("clang-([0-9]+)\\.([0-9]+)\\.([0-9]+)");
                RegularExpression::Match regex_match(3);
                if (g_clang_version_regex.Execute (producer_cstr, &regex_match))
                {
                    std::string str;
                    if (regex_match.GetMatchAtIndex (producer_cstr, 1, str))
                        m_producer_version_major = Args::StringToUInt32(str.c_str(), UINT32_MAX, 10);
                    if (regex_match.GetMatchAtIndex (producer_cstr, 2, str))
                        m_producer_version_minor = Args::StringToUInt32(str.c_str(), UINT32_MAX, 10);
                    if (regex_match.GetMatchAtIndex (producer_cstr, 3, str))
                        m_producer_version_update = Args::StringToUInt32(str.c_str(), UINT32_MAX, 10);
                }
                m_producer = eProducerClang;
            }
            else if (strstr(producer_cstr, "GNU"))
                m_producer = eProducerGCC;
        }
    }
    if (m_producer == eProducerInvalid)
        m_producer = eProcucerOther;
}

DWARFCompileUnit::Producer
DWARFCompileUnit::GetProducer ()
{
    if (m_producer == eProducerInvalid)
        ParseProducerInfo ();
    return m_producer;
}


uint32_t
DWARFCompileUnit::GetProducerVersionMajor()
{
    if (m_producer_version_major == 0)
        ParseProducerInfo ();
    return m_producer_version_major;
}

uint32_t
DWARFCompileUnit::GetProducerVersionMinor()
{
    if (m_producer_version_minor == 0)
        ParseProducerInfo ();
    return m_producer_version_minor;
}

uint32_t
DWARFCompileUnit::GetProducerVersionUpdate()
{
    if (m_producer_version_update == 0)
        ParseProducerInfo ();
    return m_producer_version_update;
}

