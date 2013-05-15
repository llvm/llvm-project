//===-- LineTable.cpp -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/Address.h"
#include "lldb/Core/Module.h"
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
    m_entries()
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
LineTable::~LineTable()
{
}

void
LineTable::InsertLineEntry
(
    lldb::addr_t file_addr,
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
    Entry entry(file_addr, line, column, file_idx, is_start_of_statement, is_start_of_basic_block, is_prologue_end, is_epilogue_begin, is_terminal_entry);

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

LineSequence::LineSequence()
{
}

void
LineTable::LineSequenceImpl::Clear()
{ 
    m_entries.clear();
}

LineSequence* LineTable::CreateLineSequenceContainer ()
{
    return new LineTable::LineSequenceImpl();
}

void
LineTable::AppendLineEntryToSequence
(
    LineSequence* sequence,
    lldb::addr_t file_addr,
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
    assert(sequence != NULL);
    LineSequenceImpl* seq = reinterpret_cast<LineSequenceImpl*>(sequence);
    Entry entry(file_addr, line, column, file_idx, is_start_of_statement, is_start_of_basic_block, is_prologue_end, is_epilogue_begin, is_terminal_entry);
    seq->m_entries.push_back (entry);
}

void
LineTable::InsertSequence (LineSequence* sequence)
{
    assert(sequence != NULL);
    LineSequenceImpl* seq = reinterpret_cast<LineSequenceImpl*>(sequence);
    if (seq->m_entries.empty())
        return;
    Entry& entry = seq->m_entries.front();
    
    // If the first entry address in this sequence is greater than or equal to
    // the address of the last item in our entry collection, just append.
    if (m_entries.empty() || !Entry::EntryAddressLessThan(entry, m_entries.back()))
    {
        m_entries.insert(m_entries.end(),
                         seq->m_entries.begin(),
                         seq->m_entries.end());
        return;
    }

    // Otherwise, find where this belongs in the collection
    entry_collection::iterator begin_pos = m_entries.begin();
    entry_collection::iterator end_pos = m_entries.end();
    LineTable::Entry::LessThanBinaryPredicate less_than_bp(this);
    entry_collection::iterator pos = upper_bound(begin_pos, end_pos, entry, less_than_bp);
#ifdef LLDB_CONFIGURATION_DEBUG
    // If we aren't inserting at the beginning, the previous entry should
    // terminate a sequence.
    if (pos != begin_pos)
    {
        entry_collection::iterator prev_pos = pos - 1;
        assert(prev_pos->is_terminal_entry);
    }
#endif
    m_entries.insert(pos, seq->m_entries.begin(), seq->m_entries.end());
}

//----------------------------------------------------------------------
LineTable::Entry::LessThanBinaryPredicate::LessThanBinaryPredicate(LineTable *line_table) :
    m_line_table (line_table)
{
}

bool
LineTable::Entry::LessThanBinaryPredicate::operator() (const LineTable::Entry& a, const LineTable::Entry& b) const
{
    #define LT_COMPARE(a,b) if (a != b) return a < b
    LT_COMPARE (a.file_addr, b.file_addr);
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

    if (so_addr.GetModule().get() == m_comp_unit->GetModule().get())
    {
        Entry search_entry;
        search_entry.file_addr = so_addr.GetFileAddress();
        if (search_entry.file_addr != LLDB_INVALID_ADDRESS)
        {
            entry_collection::const_iterator begin_pos = m_entries.begin();
            entry_collection::const_iterator end_pos = m_entries.end();
            entry_collection::const_iterator pos = lower_bound(begin_pos, end_pos, search_entry, Entry::EntryAddressLessThan);
            if (pos != end_pos)
            {
                if (pos != begin_pos)
                {
                    if (pos->file_addr != search_entry.file_addr)
                        --pos;
                    else if (pos->file_addr == search_entry.file_addr)
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
                                if (pos->file_addr != search_entry.file_addr)
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
                                if (prev_pos->file_addr == search_entry.file_addr &&
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
    }
    return success;
}


bool
LineTable::ConvertEntryAtIndexToLineEntry (uint32_t idx, LineEntry &line_entry)
{
    if (idx < m_entries.size())
    {
        const Entry& entry = m_entries[idx];
        ModuleSP module_sp (m_comp_unit->GetModule());
        if (module_sp && module_sp->ResolveFileAddress(entry.file_addr, line_entry.range.GetBaseAddress()))
        {
            if (!entry.is_terminal_entry && idx + 1 < m_entries.size())
                line_entry.range.SetByteSize(m_entries[idx+1].file_addr - entry.file_addr);
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

size_t
LineTable::GetContiguousFileAddressRanges (FileAddressRanges &file_ranges, bool append)
{
    if (!append)
        file_ranges.Clear();
    const size_t initial_count = file_ranges.GetSize();
    
    const size_t count = m_entries.size();
    LineEntry line_entry;
    FileAddressRanges::Entry range (LLDB_INVALID_ADDRESS, 0);
    for (size_t idx = 0; idx < count; ++idx)
    {
        const Entry& entry = m_entries[idx];

        if (entry.is_terminal_entry)
        {
            if (range.GetRangeBase() != LLDB_INVALID_ADDRESS)
            {
                range.SetRangeEnd(entry.file_addr);
                file_ranges.Append(range);
                range.Clear(LLDB_INVALID_ADDRESS);
            }
        }
        else if (range.GetRangeBase() == LLDB_INVALID_ADDRESS)
        {
            range.SetRangeBase(entry.file_addr);
        }
    }
    return file_ranges.GetSize() - initial_count;
}

LineTable *
LineTable::LinkLineTable (const FileRangeMap &file_range_map)
{
    std::unique_ptr<LineTable> line_table_ap (new LineTable (m_comp_unit));
    LineSequenceImpl sequence;
    const size_t count = m_entries.size();
    LineEntry line_entry;
    const FileRangeMap::Entry *file_range_entry = NULL;
    const FileRangeMap::Entry *prev_file_range_entry = NULL;
    lldb::addr_t prev_file_addr = LLDB_INVALID_ADDRESS;
    bool prev_entry_was_linked = false;
    bool range_changed = false;
    for (size_t idx = 0; idx < count; ++idx)
    {
        const Entry& entry = m_entries[idx];
        
        const bool end_sequence = entry.is_terminal_entry;
        const lldb::addr_t lookup_file_addr = entry.file_addr - (end_sequence ? 1 : 0);
        if (file_range_entry == NULL || !file_range_entry->Contains(lookup_file_addr))
        {
            prev_file_range_entry = file_range_entry;
            file_range_entry = file_range_map.FindEntryThatContains(lookup_file_addr);
            range_changed = true;
        }

        lldb::addr_t prev_end_entry_linked_file_addr = LLDB_INVALID_ADDRESS;
        lldb::addr_t entry_linked_file_addr = LLDB_INVALID_ADDRESS;

        bool terminate_previous_entry = false;
        if (file_range_entry)
        {
            entry_linked_file_addr = entry.file_addr - file_range_entry->GetRangeBase() + file_range_entry->data;
            // Determine if we need to terminate the previous entry when the previous
            // entry was not contguous with this one after being linked.
            if (range_changed && prev_file_range_entry)
            {
                prev_end_entry_linked_file_addr = std::min<lldb::addr_t>(entry.file_addr, prev_file_range_entry->GetRangeEnd()) - prev_file_range_entry->GetRangeBase() + prev_file_range_entry->data;
                if (prev_end_entry_linked_file_addr != entry_linked_file_addr)
                    terminate_previous_entry = prev_entry_was_linked;                
            }
        }
        else if (prev_entry_was_linked)
        {
            // This entry doesn't have a remapping and it needs to be removed.
            // Watch out in case we need to terminate a previous entry needs to
            // be terminated now that one line entry in a sequence is not longer valid.
            if (!entry.is_terminal_entry &&
                !sequence.m_entries.empty() &&
                !sequence.m_entries.back().is_terminal_entry)
            {
                terminate_previous_entry = true;
            }
        }
        
        if (terminate_previous_entry && !sequence.m_entries.empty())
        {
            assert (prev_file_addr != LLDB_INVALID_ADDRESS);
            sequence.m_entries.push_back(sequence.m_entries.back());
            if (prev_end_entry_linked_file_addr == LLDB_INVALID_ADDRESS)
                prev_end_entry_linked_file_addr = std::min<lldb::addr_t>(entry.file_addr,prev_file_range_entry->GetRangeEnd()) - prev_file_range_entry->GetRangeBase() + prev_file_range_entry->data;
            sequence.m_entries.back().file_addr = prev_end_entry_linked_file_addr;
            sequence.m_entries.back().is_terminal_entry = true;

            // Append the sequence since we just terminated the previous one
            line_table_ap->InsertSequence (&sequence);
            sequence.Clear();
            prev_entry_was_linked = false;
        }
        
        // Now link the current entry
        if (file_range_entry)
        {
            // This entry has an address remapping and it needs to have its address relinked
            sequence.m_entries.push_back(entry);
            sequence.m_entries.back().file_addr = entry_linked_file_addr;
        }

        // If we have items in the sequence and the last entry is a terminal entry,
        // insert this sequence into our new line table.
        if (!sequence.m_entries.empty() && sequence.m_entries.back().is_terminal_entry)
        {
            line_table_ap->InsertSequence (&sequence);
            sequence.Clear();
            prev_entry_was_linked = false;
        }
        else
        {
            prev_entry_was_linked = file_range_entry != NULL;
        }
        prev_file_addr = entry.file_addr;
        range_changed = false;
    }
    if (line_table_ap->m_entries.empty())
        return NULL;
    return line_table_ap.release();
}




