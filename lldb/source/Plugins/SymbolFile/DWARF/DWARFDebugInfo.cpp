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
        const DWARFDataExtractor &debug_aranges_data = m_dwarf2Data->get_debug_aranges_data();
        if (debug_aranges_data.GetByteSize() > 0)
        {
            if (log)
                log->Printf ("DWARFDebugInfo::GetCompileUnitAranges() for \"%s\" from .debug_aranges",
                             m_dwarf2Data->GetObjectFile()->GetFileSpec().GetPath().c_str());
            m_cu_aranges_ap->Extract (debug_aranges_data);
            
        }

        // Make a list of all CUs represented by the arange data in the file.
        std::set<dw_offset_t> cus_with_data;
        for (size_t n=0;n<m_cu_aranges_ap.get()->GetNumRanges();n++)
        {
            dw_offset_t offset = m_cu_aranges_ap.get()->OffsetAtIndex(n);
            if (offset != DW_INVALID_OFFSET)
                cus_with_data.insert (offset);
        }

        // Manually build arange data for everything that wasn't in the .debug_aranges table.
        bool printed = false;
        const size_t num_compile_units = GetNumCompileUnits();
        for (size_t idx = 0; idx < num_compile_units; ++idx)
        {
            DWARFCompileUnit* cu = GetCompileUnitAtIndex(idx);

            dw_offset_t offset = cu->GetOffset();
            if (cus_with_data.find(offset) == cus_with_data.end())
            {
                if (log)
                {
                    if (!printed)
                        log->Printf ("DWARFDebugInfo::GetCompileUnitAranges() for \"%s\" by parsing",
                                     m_dwarf2Data->GetObjectFile()->GetFileSpec().GetPath().c_str());
                    printed = true;
                }
                cu->BuildAddressRangeTable (m_dwarf2Data, m_cu_aranges_ap.get());
            }
        }

        const bool minimize = true;
        m_cu_aranges_ap->Sort (minimize);
    }
    return *m_cu_aranges_ap.get();
}

void
DWARFDebugInfo::ParseCompileUnitHeadersIfNeeded()
{
    if (m_compile_units.empty())
    {
        if (m_dwarf2Data != NULL)
        {
            lldb::offset_t offset = 0;
            const DWARFDataExtractor &debug_info_data = m_dwarf2Data->get_debug_info_data();
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

bool
DWARFDebugInfo::OffsetLessThanCompileUnitOffset (dw_offset_t offset, const DWARFCompileUnitSP& cu_sp)
{
    return offset < cu_sp->GetOffset();
}

DWARFCompileUnit *
DWARFDebugInfo::GetCompileUnit(dw_offset_t cu_offset, uint32_t* idx_ptr)
{
    DWARFCompileUnitSP cu_sp;
    uint32_t cu_idx = DW_INVALID_INDEX;
    if (cu_offset != DW_INVALID_OFFSET)
    {
        ParseCompileUnitHeadersIfNeeded();

        // Watch out for single compile unit executable as they are pretty common
        const size_t num_cus = m_compile_units.size();
        if (num_cus == 1)
        {
            if (m_compile_units[0]->GetOffset() == cu_offset)
            {
                cu_sp = m_compile_units[0];
                cu_idx = 0;
            }
        }
        else if (num_cus)
        {
            CompileUnitColl::const_iterator end_pos = m_compile_units.end();
            CompileUnitColl::const_iterator begin_pos = m_compile_units.begin();
            CompileUnitColl::const_iterator pos = std::upper_bound(begin_pos, end_pos, cu_offset, OffsetLessThanCompileUnitOffset);
            if (pos != begin_pos)
            {
                --pos;
                if ((*pos)->GetOffset() == cu_offset)
                {
                    cu_sp = *pos;
                    cu_idx = std::distance(begin_pos, pos);
                }
            }
        }
    }
    if (idx_ptr)
        *idx_ptr = cu_idx;
    return cu_sp.get();
}

DWARFCompileUnit *
DWARFDebugInfo::GetCompileUnit (const DIERef& die_ref)
{
    if (die_ref.cu_offset == DW_INVALID_OFFSET)
        return GetCompileUnitContainingDIEOffset(die_ref.die_offset);
    else
        return GetCompileUnit(die_ref.cu_offset);
}

DWARFCompileUnit*
DWARFDebugInfo::GetCompileUnitContainingDIEOffset(dw_offset_t die_offset)
{
    ParseCompileUnitHeadersIfNeeded();

    DWARFCompileUnitSP cu_sp;

    // Watch out for single compile unit executable as they are pretty common
    const size_t num_cus = m_compile_units.size();
    if (num_cus == 1)
    {
        if (m_compile_units[0]->ContainsDIEOffset(die_offset))
            return m_compile_units[0].get();
    }
    else if (num_cus)
    {
        CompileUnitColl::const_iterator end_pos = m_compile_units.end();
        CompileUnitColl::const_iterator begin_pos = m_compile_units.begin();
        CompileUnitColl::const_iterator pos = std::upper_bound(begin_pos, end_pos, die_offset, OffsetLessThanCompileUnitOffset);
        if (pos != begin_pos)
        {
            --pos;
            if ((*pos)->ContainsDIEOffset(die_offset))
                return (*pos).get();
        }
    }

    return nullptr;
}

DWARFDIE
DWARFDebugInfo::GetDIEForDIEOffset(dw_offset_t die_offset)
{
    DWARFCompileUnit* cu = GetCompileUnitContainingDIEOffset(die_offset);
    if (cu)
        return cu->GetDIE(die_offset);
    return DWARFDIE();
}

//----------------------------------------------------------------------
// GetDIE()
//
// Get the DIE (Debug Information Entry) with the specified offset.
//----------------------------------------------------------------------
DWARFDIE
DWARFDebugInfo::GetDIE(const DIERef& die_ref)
{
    DWARFCompileUnit *cu = GetCompileUnit(die_ref);
    if (cu)
        return cu->GetDIE (die_ref.die_offset);
    return DWARFDIE();    // Not found
}

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
            offset = callback(dwarf2Data, cu.get(), NULL, offset, depth, userData);

            // Make sure we are within our compile unit
            if (offset < next_cu_offset)
            {
                // We are in our compile unit, parse starting at the offset
                // we were told to parse
                bool done = false;
                while (!done && die.Extract(dwarf2Data, cu.get(), &offset))
                {
                    // Call the callback function with DIE pointer that falls within the compile unit
                    offset = callback(dwarf2Data, cu.get(), &die, offset, depth, userData);

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
    DWARFCompileUnit* cu,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    DumpInfo* dumpInfo = (DumpInfo*)userData;
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
                // dumping all of the children
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
            if (cu)
            {
                cu->Dump(s);
                return cu->GetFirstDIEOffset(); // Return true to parse all DIEs in this Compile Unit
            }
            else
            {
                return DW_INVALID_OFFSET;
            }
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
            if (cu && dumpInfo->die_offset < cu->GetFirstDIEOffset())
            {
                // Not found, maybe the DIE offset provided wasn't correct?
            //  *ostrm_ptr << "DIE at offset " << HEX32 << dumpInfo->die_offset << " was not found." << endl;
                return DW_INVALID_OFFSET;
            }
            else
            {
                // See if the DIE is in this compile unit?
                if (cu && dumpInfo->die_offset < cu->GetNextCompileUnitOffset())
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
                    if (cu)
                    {
                        return cu->GetNextCompileUnitOffset();
                    }
                    else
                    {
                        return DW_INVALID_OFFSET;
                    }
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
        DWARFCompileUnit *cu = pos->get();
        DumpCallback(m_dwarf2Data, cu, NULL, 0, curr_depth, &dumpInfo);
        
        const DWARFDIE die = cu->DIE();
        if (die)
            die.Dump(s, recurse_depth);
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
    DWARFCompileUnit* cu,
    DWARFDebugInfoEntry* die,
    const dw_offset_t next_offset,
    const uint32_t curr_depth,
    void* userData
)
{
    FindCallbackStringInfo* info = (FindCallbackStringInfo*)userData;

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
