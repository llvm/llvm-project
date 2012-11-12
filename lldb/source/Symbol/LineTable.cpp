//===-- LineTable.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Address.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Stream.h"
#include "lldb/Symbol/CompileUnit.h"
#include "lldb/Symbol/LineTable.h"
#include <algorithm>

using namespace lldb;
using namespace lldb_private;

//----------------------------------------------------------------------
// LineTable constructor
//----------------------------------------------------------------------
LineTable::LineTable(CompileUnit* comp_unit) :
    m_comp_unit(comp_unit),
    m_section_list(),
    m_entries()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LineTable::~LineTable()
{
}

//void
//LineTable::AddLineEntry(const LineEntry& entry)
//{
//  // Do a binary search for the correct entry and insert it
//  m_line_entries.insert(std::upper_bound(m_line_entries.begin(), m_line_entries.end(), entry), entry);
//}

void
LineTable::AppendLineEntry
(
    const lldb::SectionSP& section_sp,
    lldb::addr_t section_offset,
    uint32_t line,
    uint16_t column,
    uint16_t file_idx,
    bool is_start_of_statement,
    bool is_start_of_basic_block,
    bool is_prologue_end,
    bool is_epilogue_begin,
    bool is_terminal_entry
)
{
    uint32_t sect_idx = m_section_list.AddUniqueSection (section_sp);
    Entry entry(sect_idx, section_offset, line, column, file_idx, is_start_of_statement, is_start_of_basic_block, is_prologue_end, is_epilogue_begin, is_terminal_entry);
    m_entries.push_back (entry);
}


void
LineTable::InsertLineEntry
(
    const SectionSP& section_sp,
    lldb::addr_t section_offset,
    uint32_t line,
    uint16_t column,
    uint16_t file_idx,
    bool is_start_of_statement,
    bool is_start_of_basic_block,
    bool is_prologue_end,
    bool is_epilogue_begin,
    bool is_terminal_entry
)
{
    SectionSP line_section_sp;
    SectionSP linked_section_sp (section_sp->GetLinkedSection());
    if (linked_section_sp)
    {
        section_offset += section_sp->GetLinkedOffset();
        line_section_sp = linked_section_sp;
    }
    else
    {
        line_section_sp = section_sp;
    }
    assert(line_section_sp.get());

    uint32_t sect_idx = m_section_list.AddUniqueSection (line_section_sp);
    Entry entry(sect_idx, section_offset, line, column, file_idx, is_start_of_statement, is_start_of_basic_block, is_prologue_end, is_epilogue_begin, is_terminal_entry);

    entry_collection::iterator begin_pos = m_entries.begin();
    entry_collection::iterator end_pos = m_entries.end();
    LineTable::Entry::LessThanBinaryPredicate less_than_bp(this);
    entry_collection::iterator pos = upper_bound(begin_pos, end_pos, entry, less_than_bp);

//  Stream s(stdout);
//  s << "\n\nBefore:\n";
//  Dump (&s, Address::DumpStyleFileAddress);
    m_entries.insert(pos, entry);
//  s << "After:\n";
//  Dump (&s, Address::DumpStyleFileAddress);
}

//----------------------------------------------------------------------
LineTable::Entry::LessThanBinaryPredicate::LessThanBinaryPredicate(LineTable *line_table) :
    m_line_table (line_table)
{
}

bool
LineTable::Entry::LessThanBinaryPredicate::operator() (const LineTable::Entry& a, const LineTable::Entry& b) const
{
    if (a.sect_idx == b.sect_idx)
    {
        #define LT_COMPARE(a,b) if (a != b) return a < b
        LT_COMPARE (a.sect_offset, b.sect_offset);
        // b and a reversed on purpose below.
        LT_COMPARE (b.is_terminal_entry, a.is_terminal_entry);
        LT_COMPARE (a.line, b.line);
        LT_COMPARE (a.column, b.column);
        LT_COMPARE (a.is_start_of_statement, b.is_start_of_statement);
        LT_COMPARE (a.is_start_of_basic_block, b.is_start_of_basic_block);
        // b and a reversed on purpose below.
        LT_COMPARE (b.is_prologue_end, a.is_prologue_end);
        LT_COMPARE (a.is_epilogue_begin, b.is_epilogue_begin);
        LT_COMPARE (a.file_idx, b.file_idx);
        return false;
        #undef LT_COMPARE
    }

    const Section *a_section = m_line_table->GetSectionForEntryIndex (a.sect_idx);
    const Section *b_section = m_line_table->GetSectionForEntryIndex (b.sect_idx);
    return Section::Compare(*a_section, *b_section) < 0;
}


Section *
LineTable::GetSectionForEntryIndex (uint32_t idx)
{
    if (idx < m_section_list.GetSize())
        return m_section_list.GetSectionAtIndex(idx).get();
    return NULL;
}

uint32_t
LineTable::GetSize() const
{
    return m_entries.size();
}

bool
LineTable::GetLineEntryAtIndex(uint32_t idx, LineEntry& line_entry)
{
    if (idx < m_entries.size())
    {
        ConvertEntryAtIndexToLineEntry (idx, line_entry);
        return true;
    }
    line_entry.Clear();
    return false;
}

bool
LineTable::FindLineEntryByAddress (const Address &so_addr, LineEntry& line_entry, uint32_t *index_ptr)
{
    if (index_ptr != NULL )
        *index_ptr = UINT32_MAX;

    bool success = false;
    uint32_t sect_idx = m_section_list.FindSectionIndex (so_addr.GetSection().get());
    if (sect_idx != UINT32_MAX)
    {
        Entry search_entry;
        search_entry.sect_idx = sect_idx;
        search_entry.sect_offset = so_addr.GetOffset();

        entry_collection::const_iterator begin_pos = m_entries.begin();
        entry_collection::const_iterator end_pos = m_entries.end();
        entry_collection::const_iterator pos = lower_bound(begin_pos, end_pos, search_entry, Entry::EntryAddressLessThan);
        if (pos != end_pos)
        {
            if (pos != begin_pos)
            {
                if (pos->sect_offset != search_entry.sect_offset)
                    --pos;
                else if (pos->sect_offset == search_entry.sect_offset)
                {
                    // If this is a termination entry, it should't match since
                    // entries with the "is_terminal_entry" member set to true 
                    // are termination entries that define the range for the 
                    // previous entry.
                    if (pos->is_terminal_entry)
                    {
                        // The matching entry is a terminal entry, so we skip
                        // ahead to the next entry to see if there is another
                        // entry following this one whose section/offset matches.
                        ++pos;
                        if (pos != end_pos)
                        {
                            if (pos->sect_offset != search_entry.sect_offset)
                                pos = end_pos;
                        }
                    }
                    
                    if (pos != end_pos)
                    {
                        // While in the same section/offset backup to find the first
                        // line entry that matches the address in case there are 
                        // multiple
                        while (pos != begin_pos)
                        {
                            entry_collection::const_iterator prev_pos = pos - 1;
                            if (prev_pos->sect_idx    == search_entry.sect_idx &&
                                prev_pos->sect_offset == search_entry.sect_offset &&
                                prev_pos->is_terminal_entry == false)
                                --pos;
                            else
                                break;
                        }
                    }
                }

            }
            
            // Make sure we have a valid match and that the match isn't a terminating
            // entry for a previous line...
            if (pos != end_pos && pos->is_terminal_entry == false)
            {
                uint32_t match_idx = std::distance (begin_pos, pos);
                success = ConvertEntryAtIndexToLineEntry(match_idx, line_entry);
                if (index_ptr != NULL && success)
                    *index_ptr = match_idx;
            }
        }
    }
    return success;
}


bool
LineTable::ConvertEntryAtIndexToLineEntry (uint32_t idx, LineEntry &line_entry)
{
    if (idx < m_entries.size())
    {
        const Entry& entry = m_entries[idx];
        line_entry.range.GetBaseAddress().SetSection(m_section_list.GetSectionAtIndex (entry.sect_idx));
        line_entry.range.GetBaseAddress().SetOffset(entry.sect_offset);
        if (!entry.is_terminal_entry && idx + 1 < m_entries.size())
        {
            const Entry& next_entry = m_entries[idx+1];
            if (next_entry.sect_idx == entry.sect_idx)
            {
                line_entry.range.SetByteSize(next_entry.sect_offset - entry.sect_offset);
            }
            else
            {
                Address next_line_addr(m_section_list.GetSectionAtIndex (next_entry.sect_idx), next_entry.sect_offset);
                line_entry.range.SetByteSize(next_line_addr.GetFileAddress() - line_entry.range.GetBaseAddress().GetFileAddress());
            }
        }
        else
            line_entry.range.SetByteSize(0);
        line_entry.file = m_comp_unit->GetSupportFiles().GetFileSpecAtIndex (entry.file_idx);
        line_entry.line = entry.line;
        line_entry.column = entry.column;
        line_entry.is_start_of_statement = entry.is_start_of_statement;
        line_entry.is_start_of_basic_block = entry.is_start_of_basic_block;
        line_entry.is_prologue_end = entry.is_prologue_end;
        line_entry.is_epilogue_begin = entry.is_epilogue_begin;
        line_entry.is_terminal_entry = entry.is_terminal_entry;
        return true;
    }
    return false;
}

uint32_t
LineTable::FindLineEntryIndexByFileIndex 
(
    uint32_t start_idx, 
    const std::vector<uint32_t> &file_indexes, 
    uint32_t line, 
    bool exact, 
    LineEntry* line_entry_ptr
)
{

    const size_t count = m_entries.size();
    std::vector<uint32_t>::const_iterator begin_pos = file_indexes.begin();
    std::vector<uint32_t>::const_iterator end_pos = file_indexes.end();
    size_t best_match = UINT32_MAX;

    for (size_t idx = start_idx; idx < count; ++idx)
    {
        // Skip line table rows that terminate the previous row (is_terminal_entry is non-zero)
        if (m_entries[idx].is_terminal_entry)
            continue;

        if (find (begin_pos, end_pos, m_entries[idx].file_idx) == end_pos)
            continue;

        // Exact match always wins.  Otherwise try to find the closest line > the desired
        // line.
        // FIXME: Maybe want to find the line closest before and the line closest after and
        // if they're not in the same function, don't return a match.

        if (m_entries[idx].line < line)
        {
            continue;
        }
        else if (m_entries[idx].line == line)
        {
            if (line_entry_ptr)
                ConvertEntryAtIndexToLineEntry (idx, *line_entry_ptr);
            return idx;
        }
        else if (!exact)
        {
            if (best_match == UINT32_MAX)
                best_match = idx;
            else if (m_entries[idx].line < m_entries[best_match].line)
                best_match = idx;
        }
    }

    if (best_match != UINT32_MAX)
    {
        if (line_entry_ptr)
            ConvertEntryAtIndexToLineEntry (best_match, *line_entry_ptr);
        return best_match;
    }
    return UINT32_MAX;
}

uint32_t
LineTable::FindLineEntryIndexByFileIndex (uint32_t start_idx, uint32_t file_idx, uint32_t line, bool exact, LineEntry* line_entry_ptr)
{
    const size_t count = m_entries.size();
    size_t best_match = UINT32_MAX;

    for (size_t idx = start_idx; idx < count; ++idx)
    {
        // Skip line table rows that terminate the previous row (is_terminal_entry is non-zero)
        if (m_entries[idx].is_terminal_entry)
            continue;

        if (m_entries[idx].file_idx != file_idx)
            continue;

        // Exact match always wins.  Otherwise try to find the closest line > the desired
        // line.
        // FIXME: Maybe want to find the line closest before and the line closest after and
        // if they're not in the same function, don't return a match.

        if (m_entries[idx].line < line)
        {
            continue;
        }
        else if (m_entries[idx].line == line)
        {
            if (line_entry_ptr)
                ConvertEntryAtIndexToLineEntry (idx, *line_entry_ptr);
            return idx;
        }
        else if (!exact)
        {
            if (best_match == UINT32_MAX)
                best_match = idx;
            else if (m_entries[idx].line < m_entries[best_match].line)
                best_match = idx;
        }
    }

    if (best_match != UINT32_MAX)
    {
        if (line_entry_ptr)
            ConvertEntryAtIndexToLineEntry (best_match, *line_entry_ptr);
        return best_match;
    }
    return UINT32_MAX;
}

size_t
LineTable::FineLineEntriesForFileIndex (uint32_t file_idx, 
                                        bool append,
                                        SymbolContextList &sc_list)
{
    
    if (!append)
        sc_list.Clear();

    size_t num_added = 0;
    const size_t count = m_entries.size();
    if (count > 0)
    {
        SymbolContext sc (m_comp_unit);

        for (size_t idx = 0; idx < count; ++idx)
        {
            // Skip line table rows that terminate the previous row (is_terminal_entry is non-zero)
            if (m_entries[idx].is_terminal_entry)
                continue;
            
            if (m_entries[idx].file_idx == file_idx)
            {
                if (ConvertEntryAtIndexToLineEntry (idx, sc.line_entry))
                {
                    ++num_added;
                    sc_list.Append(sc);
                }
            }
        }
    }
    return num_added;
}


void
LineTable::Dump (Stream *s, Target *target, Address::DumpStyle style, Address::DumpStyle fallback_style, bool show_line_ranges)
{
    const size_t count = m_entries.size();
    LineEntry line_entry;
    FileSpec prev_file;
    for (size_t idx = 0; idx < count; ++idx)
    {
        ConvertEntryAtIndexToLineEntry (idx, line_entry);
        line_entry.Dump (s, target, prev_file != line_entry.file, style, fallback_style, show_line_ranges);
        s->EOL();
        prev_file = line_entry.file;
    }
}


void
LineTable::GetDescription (Stream *s, Target *target, DescriptionLevel level)
{
    const size_t count = m_entries.size();
    LineEntry line_entry;
    for (size_t idx = 0; idx < count; ++idx)
    {
        ConvertEntryAtIndexToLineEntry (idx, line_entry);
        line_entry.GetDescription (s, level, m_comp_unit, target, true);
        s->EOL();
    }
}



