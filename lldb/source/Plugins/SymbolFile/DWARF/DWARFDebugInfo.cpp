//===-- DWARFDebugInfo.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARF.h"

#include <algorithm>
#include <set>

#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/ObjectFile.h"

#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFCompileUnit.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfoEntry.h"
#include "DWARFFormValue.h"
#include "LogChannelDWARF.h"

using namespace lldb;
using namespace lldb_private;
using namespace std;

//----------------------------------------------------------------------
// Constructor
//----------------------------------------------------------------------
DWARFDebugInfo::DWARFDebugInfo() :
    m_dwarf2Data(NULL),
    m_compile_units(),
    m_cu_aranges_ap ()
{
}

//----------------------------------------------------------------------
// SetDwarfData
//----------------------------------------------------------------------
void
DWARFDebugInfo::SetDwarfData(SymbolFileDWARF* dwarf2Data)
{
    m_dwarf2Data = dwarf2Data;
    m_compile_units.clear();
}


DWARFDebugAranges &
DWARFDebugInfo::GetCompileUnitAranges ()
{
    if (m_cu_aranges_ap.get() == NULL && m_dwarf2Data)
    {
        Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_ARANGES));

        m_cu_aranges_ap.reset (new DWARFDebugAranges());
        const DataExtractor &debug_aranges_data = m_dwarf2Data->get_debug_aranges_data();
        if (debug_aranges_data.GetByteSize() > 0)
        {
            if (log)
                log->Printf ("DWARFDebugInfo::GetCompileUnitAranges() for \"%s\" from .debug_aranges",
                             m_dwarf2Data->GetObjectFile()->GetFileSpec().GetPath().c_str());
            m_cu_aranges_ap->Extract (debug_aranges_data);
            
        }
        else
        {
            if (log)
                log->Printf ("DWARFDebugInfo::GetCompileUnitAranges() for \"%s\" by parsing", 
                             m_dwarf2Data->GetObjectFile()->GetFileSpec().GetPath().c_str());
            const size_t num_compile_units = GetNumCompileUnits();
            const bool clear_dies_if_already_not_parsed = true;
            for (size_t idx = 0; idx < num_compile_units; ++idx)
            {
                DWARFCompileUnit* cu = GetCompileUnitAtIndex(idx);
                if (cu)
                    cu->BuildAddressRangeTable (m_dwarf2Data, m_cu_aranges_ap.get(), clear_dies_if_already_not_parsed);
            }
        }

        const bool minimize = true;
        m_cu_aranges_ap->Sort (minimize);
    }
    return *m_cu_aranges_ap.get();
}


//----------------------------------------------------------------------
// LookupAddress
//----------------------------------------------------------------------
bool
DWARFDebugInfo::LookupAddress
(
    const dw_addr_t address,
    const dw_offset_t hint_die_offset,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry** function_die,
    DWARFDebugInfoEntry** block_die
)
{

    if (hint_die_offset != DW_INVALID_OFFSET)
        cu_sp = GetCompileUnit(hint_die_offset);
    else
    {
        DWARFDebugAranges &cu_aranges = GetCompileUnitAranges ();
        const dw_offset_t cu_offset = cu_aranges.FindAddress (address);
        cu_sp = GetCompileUnit(cu_offset);
    }

    if (cu_sp.get())
    {
        if (cu_sp->LookupAddress(address, function_die, block_die))
            return true;
        cu_sp.reset();
    }
    else
    {
        // The hint_die_offset may have been a pointer to the actual item that
        // we are looking for
        DWARFDebugInfoEntry* die_ptr = GetDIEPtr(hint_die_offset, &cu_sp);
        if (die_ptr)
        {
            if (cu_sp.get())
            {
                if (function_die || block_die)
                    return die_ptr->LookupAddress(address, m_dwarf2Data, cu_sp.get(), function_die, block_die);

                // We only wanted the compile unit that contained this address
                return true;
            }
        }
    }
    return false;
}


void
DWARFDebugInfo::ParseCompileUnitHeadersIfNeeded()
{
    if (m_compile_units.empty())
    {
        if (m_dwarf2Data != NULL)
        {
            lldb::offset_t offset = 0;
            const DataExtractor &debug_info_data = m_dwarf2Data->get_debug_info_data();
            while (debug_info_data.ValidOffset(offset))
            {
                DWARFCompileUnitSP cu_sp(new DWARFCompileUnit(m_dwarf2Data));
                // Out of memory?
                if (cu_sp.get() == NULL)
                    break;

                if (cu_sp->Extract(debug_info_data, &offset) == false)
                    break;

                m_compile_units.push_back(cu_sp);

                offset = cu_sp->GetNextCompileUnitOffset();
            }
        }
    }
}

size_t
DWARFDebugInfo::GetNumCompileUnits()
{
    ParseCompileUnitHeadersIfNeeded();
    return m_compile_units.size();
}

DWARFCompileUnit*
DWARFDebugInfo::GetCompileUnitAtIndex(uint32_t idx)
{
    DWARFCompileUnit* cu = NULL;
    if (idx < GetNumCompileUnits())
        cu = m_compile_units[idx].get();
    return cu;
}

bool
DWARFDebugInfo::ContainsCompileUnit (const DWARFCompileUnit *cu) const
{
    // Not a verify efficient function, but it is handy for use in assertions
    // to make sure that a compile unit comes from a debug information file.
    CompileUnitColl::const_iterator end_pos = m_compile_units.end();
    CompileUnitColl::const_iterator pos;
    
    for (pos = m_compile_units.begin(); pos != end_pos; ++pos)
    {
        if (pos->get() == cu)
            return true;
    }
    return false;
}


static bool CompileUnitOffsetLessThan (const DWARFCompileUnitSP& a, const DWARFCompileUnitSP& b)
{
    return a->GetOffset() < b->GetOffset();
}


static int
CompareDWARFCompileUnitSPOffset (const void *key, const void *arrmem)
{
    const dw_offset_t key_cu_offset = *(dw_offset_t*) key;
    const dw_offset_t cu_offset = ((DWARFCompileUnitSP *)arrmem)->get()->GetOffset();
    if (key_cu_offset < cu_offset)
        return -1;
    if (key_cu_offset > cu_offset)
        return 1;
    return 0;
}

DWARFCompileUnitSP
DWARFDebugInfo::GetCompileUnit(dw_offset_t cu_offset, uint32_t* idx_ptr)
{
    DWARFCompileUnitSP cu_sp;
    uint32_t cu_idx = DW_INVALID_INDEX;
    if (cu_offset != DW_INVALID_OFFSET)
    {
        ParseCompileUnitHeadersIfNeeded();

        DWARFCompileUnitSP* match = (DWARFCompileUnitSP*)bsearch(&cu_offset, &m_compile_units[0], m_compile_units.size(), sizeof(DWARFCompileUnitSP), CompareDWARFCompileUnitSPOffset);
        if (match)
        {
            cu_sp = *match;
            cu_idx = match - &m_compile_units[0];
        }
    }
    if (idx_ptr)
        *idx_ptr = cu_idx;
    return cu_sp;
}

DWARFCompileUnitSP
DWARFDebugInfo::GetCompileUnitContainingDIE(dw_offset_t die_offset)
{
    DWARFCompileUnitSP cu_sp;
    if (die_offset != DW_INVALID_OFFSET)
    {
        ParseCompileUnitHeadersIfNeeded();

        CompileUnitColl::const_iterator end_pos = m_compile_units.end();
        CompileUnitColl::const_iterator pos;

        for (pos = m_compile_units.begin(); pos != end_pos; ++pos)
        {
            dw_offset_t cu_start_offset = (*pos)->GetOffset();
            dw_offset_t cu_end_offset = (*pos)->GetNextCompileUnitOffset();
            if (cu_start_offset <= die_offset && die_offset < cu_end_offset)
            {
                cu_sp = *pos;
                break;
            }
        }
    }
    return cu_sp;
}

//----------------------------------------------------------------------
// Compare function DWARFDebugAranges::Range structures
//----------------------------------------------------------------------
static bool CompareDIEOffset (const DWARFDebugInfoEntry& die1, const DWARFDebugInfoEntry& die2)
{
    return die1.GetOffset() < die2.GetOffset();
}


//----------------------------------------------------------------------
// GetDIE()
//
// Get the DIE (Debug Information Entry) with the specified offset.
//----------------------------------------------------------------------
DWARFDebugInfoEntry*
DWARFDebugInfo::GetDIEPtr(dw_offset_t die_offset, DWARFCompileUnitSP* cu_sp_ptr)
{
    DWARFCompileUnitSP cu_sp(GetCompileUnitContainingDIE(die_offset));
    if (cu_sp_ptr)
        *cu_sp_ptr = cu_sp;
    if (cu_sp.get())
        return cu_sp->GetDIEPtr(die_offset);
    return NULL;    // Not found in any compile units
}

DWARFDebugInfoEntry*
DWARFDebugInfo::GetDIEPtrWithCompileUnitHint (dw_offset_t die_offset, DWARFCompileUnit**cu_handle)
{
    assert (cu_handle);
    DWARFDebugInfoEntry* die = NULL;
    if (*cu_handle)
        die = (*cu_handle)->GetDIEPtr(die_offset);

    if (die == NULL)
    {
        DWARFCompileUnitSP cu_sp (GetCompileUnitContainingDIE(die_offset));
        if (cu_sp.get())
        {
            *cu_handle = cu_sp.get();
            die = cu_sp->GetDIEPtr(die_offset);
        }
    }
    if (die == NULL)
        *cu_handle = NULL;
    return die;
}


const DWARFDebugInfoEntry*
DWARFDebugInfo::GetDIEPtrContainingOffset(dw_offset_t die_offset, DWARFCompileUnitSP* cu_sp_ptr)
{
    DWARFCompileUnitSP cu_sp(GetCompileUnitContainingDIE(die_offset));
    if (cu_sp_ptr)
        *cu_sp_ptr = cu_sp;
    if (cu_sp.get())
        return cu_sp->GetDIEPtrContainingOffset(die_offset);

    return NULL;    // Not found in any compile units

}

//----------------------------------------------------------------------
// DWARFDebugInfo_ParseCallback
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets parses all compile units and DIE's into an internate
// representation for further modification.
//----------------------------------------------------------------------

static dw_offset_t
DWARFDebugInfo_ParseCallback
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    DWARFDebugInfo* debug_info = (DWARFDebugInfo*)userData;
    DWARFCompileUnit* cu = cu_sp.get();
    if (die)
    {
        cu->AddDIE(*die);
    }
    else if (cu)
    {
        debug_info->AddCompileUnit(cu_sp);
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}

//----------------------------------------------------------------------
// AddCompileUnit
//----------------------------------------------------------------------
void
DWARFDebugInfo::AddCompileUnit(DWARFCompileUnitSP& cu)
{
    m_compile_units.push_back(cu);
}

/*
void
DWARFDebugInfo::AddDIE(DWARFDebugInfoEntry& die)
{
    m_die_array.push_back(die);
}
*/




//----------------------------------------------------------------------
// Parse
//
// Parses the .debug_info section and uses the .debug_abbrev section
// and various other sections in the SymbolFileDWARF class and calls the
// supplied callback function each time a compile unit header, or debug
// information entry is successfully parsed. This function can be used
// for different tasks such as parsing the file contents into a
// structured data, dumping, verifying and much more.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Parse(SymbolFileDWARF* dwarf2Data, Callback callback, void* userData)
{
    if (dwarf2Data)
    {
        lldb::offset_t offset = 0;
        uint32_t depth = 0;
        DWARFCompileUnitSP cu(new DWARFCompileUnit(dwarf2Data));
        if (cu.get() == NULL)
            return;
        DWARFDebugInfoEntry die;

        while (cu->Extract(dwarf2Data->get_debug_info_data(), &offset))
        {
            const dw_offset_t next_cu_offset = cu->GetNextCompileUnitOffset();

            depth = 0;
            // Call the callback function with no DIE pointer for the compile unit
            // and get the offset that we are to continue to parse from
            offset = callback(dwarf2Data, cu, NULL, offset, depth, userData);

            // Make sure we are within our compile unit
            if (offset < next_cu_offset)
            {
                // We are in our compile unit, parse starting at the offset
                // we were told to parse
                bool done = false;
                while (!done && die.Extract(dwarf2Data, cu.get(), &offset))
                {
                    // Call the callback function with DIE pointer that falls within the compile unit
                    offset = callback(dwarf2Data, cu, &die, offset, depth, userData);

                    if (die.IsNULL())
                    {
                        if (depth)
                            --depth;
                        else
                            done = true;    // We are done with this compile unit!
                    }
                    else if (die.HasChildren())
                        ++depth;
                }
            }

            // Make sure the offset returned is valid, and if not stop parsing.
            // Returning DW_INVALID_OFFSET from this callback is a good way to end
            // all parsing
            if (!dwarf2Data->get_debug_info_data().ValidOffset(offset))
                break;

            // See if during the callback anyone retained a copy of the compile
            // unit other than ourselves and if so, let whomever did own the object
            // and create a new one for our own use!
            if (!cu.unique())
                cu.reset(new DWARFCompileUnit(dwarf2Data));


            // Make sure we start on a proper
            offset = next_cu_offset;
        }
    }
}

typedef struct DumpInfo
{
    DumpInfo(Stream* init_strm, uint32_t off, uint32_t depth) :
        strm(init_strm),
        die_offset(off),
        recurse_depth(depth),
        found_depth(UINT32_MAX),
        found_die(false),
        ancestors()
    {
    }
    Stream* strm;
    const uint32_t die_offset;
    const uint32_t recurse_depth;
    uint32_t found_depth;
    bool found_die;
    std::vector<DWARFDebugInfoEntry> ancestors;

    DISALLOW_COPY_AND_ASSIGN(DumpInfo);
} DumpInfo;

//----------------------------------------------------------------------
// DumpCallback
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets called each time a compile unit header or debug information
// entry is successfully parsed.
//
// This function dump DWARF information and obey recurse depth and
// whether a single DIE is to be dumped (or all of the data).
//----------------------------------------------------------------------
static dw_offset_t DumpCallback
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    DumpInfo* dumpInfo = (DumpInfo*)userData;

    const DWARFCompileUnit* cu = cu_sp.get();

    Stream *s = dumpInfo->strm;
    bool show_parents = s->GetFlags().Test(DWARFDebugInfo::eDumpFlag_ShowAncestors);

    if (die)
    {
        // Are we dumping everything?
        if (dumpInfo->die_offset == DW_INVALID_OFFSET)
        {
            // Yes we are dumping everything. Obey our recurse level though
            if (curr_depth < dumpInfo->recurse_depth)
                die->Dump(dwarf2Data, cu, *s, 0);
        }
        else
        {
            // We are dumping a specific DIE entry by offset
            if (dumpInfo->die_offset == die->GetOffset())
            {
                // We found the DIE we were looking for, dump it!
                if (show_parents)
                {
                    s->SetIndentLevel(0);
                    const uint32_t num_ancestors = dumpInfo->ancestors.size();
                    if (num_ancestors > 0)
                    {
                        for (uint32_t i=0; i<num_ancestors-1; ++i)
                        {
                            dumpInfo->ancestors[i].Dump(dwarf2Data, cu, *s, 0);
                            s->IndentMore();
                        }
                    }
                }

                dumpInfo->found_depth = curr_depth;

                die->Dump(dwarf2Data, cu, *s, 0);

                // Note that we found the DIE we were looking for
                dumpInfo->found_die = true;

                // Since we are dumping a single DIE, if there are no children we are done!
                if (!die->HasChildren() || dumpInfo->recurse_depth == 0)
                    return DW_INVALID_OFFSET;   // Return an invalid address to end parsing
            }
            else if (dumpInfo->found_die)
            {
                // Are we done with all the children?
                if (curr_depth <= dumpInfo->found_depth)
                    return DW_INVALID_OFFSET;

                // We have already found our DIE and are printing it's children. Obey
                // our recurse depth and return an invalid offset if we get done
                // dumping all the the children
                if (dumpInfo->recurse_depth == UINT32_MAX || curr_depth <= dumpInfo->found_depth + dumpInfo->recurse_depth)
                    die->Dump(dwarf2Data, cu, *s, 0);
            }
            else if (dumpInfo->die_offset > die->GetOffset())
            {
                if (show_parents)
                    dumpInfo->ancestors.back() = *die;
            }
        }

        // Keep up with our indent level
        if (die->IsNULL())
        {
            if (show_parents)
                dumpInfo->ancestors.pop_back();

            if (curr_depth <= 1)
                return cu->GetNextCompileUnitOffset();
            else
                s->IndentLess();
        }
        else if (die->HasChildren())
        {
            if (show_parents)
            {
                DWARFDebugInfoEntry null_die;
                dumpInfo->ancestors.push_back(null_die);
            }
            s->IndentMore();
        }
    }
    else
    {
        if (cu == NULL)
            s->PutCString("NULL - cu");
        // We have a compile unit, reset our indent level to zero just in case
        s->SetIndentLevel(0);

        // See if we are dumping everything?
        if (dumpInfo->die_offset == DW_INVALID_OFFSET)
        {
            // We are dumping everything
            cu->Dump(s);
            return cu->GetFirstDIEOffset(); // Return true to parse all DIEs in this Compile Unit
        }
        else
        {
            if (show_parents)
            {
                dumpInfo->ancestors.clear();
                dumpInfo->ancestors.resize(1);
            }

            // We are dumping only a single DIE possibly with it's children and
            // we must find it's compile unit before we can dump it properly
            if (dumpInfo->die_offset < cu->GetFirstDIEOffset())
            {
                // Not found, maybe the DIE offset provided wasn't correct?
            //  *ostrm_ptr << "DIE at offset " << HEX32 << dumpInfo->die_offset << " was not found." << endl;
                return DW_INVALID_OFFSET;
            }
            else
            {
                // See if the DIE is in this compile unit?
                if (dumpInfo->die_offset < cu->GetNextCompileUnitOffset())
                {
                    // This DIE is in this compile unit!
                    if (s->GetVerbose())
                        cu->Dump(s); // Dump the compile unit for the DIE in verbose mode

                    return next_offset;
                //  // We found our compile unit that contains our DIE, just skip to dumping the requested DIE...
                //  return dumpInfo->die_offset;
                }
                else
                {
                    // Skip to the next compile unit as the DIE isn't in the current one!
                    return cu->GetNextCompileUnitOffset();
                }
            }
        }
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}

//----------------------------------------------------------------------
// Dump
//
// Dump the information in the .debug_info section to the specified
// ostream. If die_offset is valid, a single DIE will be dumped. If the
// die_offset is invalid, all the DWARF information will be dumped. Both
// cases will obey a "recurse_depth" or how deep to traverse into the
// children of each DIE entry. A recurse_depth of zero will dump all
// compile unit headers. A recurse_depth of 1 will dump all compile unit
// headers and the DW_TAG_compile unit tags. A depth of 2 will also
// dump all types and functions.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Dump
(
    Stream *s,
    SymbolFileDWARF* dwarf2Data,
    const uint32_t die_offset,
    const uint32_t recurse_depth
)
{
    DumpInfo dumpInfo(s, die_offset, recurse_depth);
    s->PutCString(".debug_info contents");
    if (dwarf2Data->get_debug_info_data().GetByteSize() > 0)
    {
        if (die_offset == DW_INVALID_OFFSET)
            s->PutCString(":\n");
        else
        {
            s->Printf(" for DIE entry at .debug_info[0x%8.8x]", die_offset);
            if (recurse_depth != UINT32_MAX)
                s->Printf(" recursing %u levels deep.", recurse_depth);
            s->EOL();
        }
    }
    else
    {
        s->PutCString(": < EMPTY >\n");
        return;
    }
    DWARFDebugInfo::Parse(dwarf2Data, DumpCallback, &dumpInfo);
}


//----------------------------------------------------------------------
// Dump
//
// Dump the contents of this DWARFDebugInfo object as has been parsed
// and/or modified after it has been parsed.
//----------------------------------------------------------------------
void
DWARFDebugInfo::Dump (Stream *s, const uint32_t die_offset, const uint32_t recurse_depth)
{
    DumpInfo dumpInfo(s, die_offset, recurse_depth);

    s->PutCString("Dumping .debug_info section from internal representation\n");

    CompileUnitColl::const_iterator pos;
    uint32_t curr_depth = 0;
    ParseCompileUnitHeadersIfNeeded();
    for (pos = m_compile_units.begin(); pos != m_compile_units.end(); ++pos)
    {
        const DWARFCompileUnitSP& cu_sp = *pos;
        DumpCallback(m_dwarf2Data, (DWARFCompileUnitSP&)cu_sp, NULL, 0, curr_depth, &dumpInfo);
        
        const DWARFDebugInfoEntry* die = cu_sp->DIE();
        if (die)
            die->Dump(m_dwarf2Data, cu_sp.get(), *s, recurse_depth);
    }
}


//----------------------------------------------------------------------
// FindCallbackString
//
// A callback function for the static DWARFDebugInfo::Parse() function
// that gets called each time a compile unit header or debug information
// entry is successfully parsed.
//
// This function will find the die_offset of any items whose DW_AT_name
// matches the given string
//----------------------------------------------------------------------
typedef struct FindCallbackStringInfoTag
{
    const char* name;
    bool ignore_case;
    RegularExpression* regex;
    vector<dw_offset_t>& die_offsets;
} FindCallbackStringInfo;

static dw_offset_t FindCallbackString
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnitSP& cu_sp,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    FindCallbackStringInfo* info = (FindCallbackStringInfo*)userData;
    const DWARFCompileUnit* cu = cu_sp.get();

    if (die)
    {
        const char* die_name = die->GetName(dwarf2Data, cu);
        if (die_name)
        {
            if (info->regex)
            {
                if (info->regex->Execute(die_name))
                    info->die_offsets.push_back(die->GetOffset());
            }
            else
            {
                if ((info->ignore_case ? strcasecmp(die_name, info->name) : strcmp(die_name, info->name)) == 0)
                    info->die_offsets.push_back(die->GetOffset());
            }
        }
    }

    // Just return the current offset to parse the next CU or DIE entry
    return next_offset;
}

//----------------------------------------------------------------------
// Find
//
// Finds all DIE that have a specific DW_AT_name attribute by manually
// searching through the debug information (not using the
// .debug_pubnames section). The string must match the entire name
// and case sensitive searches are an option.
//----------------------------------------------------------------------
bool
DWARFDebugInfo::Find(const char* name, bool ignore_case, vector<dw_offset_t>& die_offsets) const
{
    die_offsets.clear();
    if (name && name[0])
    {
        FindCallbackStringInfo info = { name, ignore_case, NULL, die_offsets };
        DWARFDebugInfo::Parse(m_dwarf2Data, FindCallbackString, &info);
    }
    return !die_offsets.empty();
}

//----------------------------------------------------------------------
// Find
//
// Finds all DIE that have a specific DW_AT_name attribute by manually
// searching through the debug information (not using the
// .debug_pubnames section). The string must match the supplied regular
// expression.
//----------------------------------------------------------------------
bool
DWARFDebugInfo::Find(RegularExpression& re, vector<dw_offset_t>& die_offsets) const
{
    die_offsets.clear();
    FindCallbackStringInfo info = { NULL, false, &re, die_offsets };
    DWARFDebugInfo::Parse(m_dwarf2Data, FindCallbackString, &info);
    return !die_offsets.empty();
}
