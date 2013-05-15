//===-- DWARFDebugPubnames.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugPubnames.h"

#include "lldb/Core/Stream.h"
#include "lldb/Core/Timer.h"

#include "DWARFDebugInfo.h"
#include "DWARFDIECollection.h"
#include "DWARFFormValue.h"
#include "DWARFCompileUnit.h"
#include "LogChannelDWARF.h"
#include "SymbolFileDWARF.h"


using namespace lldb;
using namespace lldb_private;

DWARFDebugPubnames::DWARFDebugPubnames() :
    m_sets()
{
}

bool
DWARFDebugPubnames::Extract(const DataExtractor& data)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "DWARFDebugPubnames::Extract (byte_size = %" PRIu64 ")",
                        (uint64_t)data.GetByteSize());
    Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_PUBNAMES));
    if (log)
        log->Printf("DWARFDebugPubnames::Extract (byte_size = %" PRIu64 ")", (uint64_t)data.GetByteSize());

    if (data.ValidOffset(0))
    {
        lldb::offset_t offset = 0;

        DWARFDebugPubnamesSet set;
        while (data.ValidOffset(offset))
        {
            if (set.Extract(data, &offset))
            {
                m_sets.push_back(set);
                offset = set.GetOffsetOfNextEntry();
            }
            else
                break;
        }
        if (log)
            Dump (log);
        return true;
    }
    return false;
}


bool
DWARFDebugPubnames::GeneratePubnames(SymbolFileDWARF* dwarf2Data)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "DWARFDebugPubnames::GeneratePubnames (data = %p)",
                        dwarf2Data);

    Log *log (LogChannelDWARF::GetLogIfAll(DWARF_LOG_DEBUG_PUBNAMES));
    if (log)
        log->Printf("DWARFDebugPubnames::GeneratePubnames (data = %p)", dwarf2Data);

    m_sets.clear();
    DWARFDebugInfo* debug_info = dwarf2Data->DebugInfo();
    if (debug_info)
    {

        const DataExtractor* debug_str = &dwarf2Data->get_debug_str_data();

        uint32_t cu_idx = 0;
        const uint32_t num_compile_units = dwarf2Data->GetNumCompileUnits();
        for (cu_idx = 0; cu_idx < num_compile_units; ++cu_idx)
        {

            DWARFCompileUnit* cu = debug_info->GetCompileUnitAtIndex(cu_idx);

            const uint8_t *fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize (cu->GetAddressByteSize());

            bool clear_dies = cu->ExtractDIEsIfNeeded (false) > 1;

            DWARFDIECollection dies;
            const size_t die_count = cu->AppendDIEsWithTag (DW_TAG_subprogram, dies) +
                                     cu->AppendDIEsWithTag (DW_TAG_variable, dies);

            dw_offset_t cu_offset = cu->GetOffset();
            DWARFDebugPubnamesSet pubnames_set(DW_INVALID_OFFSET, cu_offset, cu->GetNextCompileUnitOffset() - cu_offset);

            size_t die_idx;
            for (die_idx = 0; die_idx < die_count; ++die_idx)
            {
                const DWARFDebugInfoEntry *die = dies.GetDIEPtrAtIndex(die_idx);
                DWARFDebugInfoEntry::Attributes attributes;
                const char *name = NULL;
                const char *mangled = NULL;
                bool add_die = false;
                const size_t num_attributes = die->GetAttributes(dwarf2Data, cu, fixed_form_sizes, attributes);
                if (num_attributes > 0)
                {
                    uint32_t i;

                    dw_tag_t tag = die->Tag();
                    
                    for (i=0; i<num_attributes; ++i)
                    {
                        dw_attr_t attr = attributes.AttributeAtIndex(i);
                        DWARFFormValue form_value;
                        switch (attr)
                        {
                        case DW_AT_name:
                            if (attributes.ExtractFormValueAtIndex(dwarf2Data, i, form_value))
                                name = form_value.AsCString(debug_str);
                            break;

                        case DW_AT_MIPS_linkage_name:
                        case DW_AT_linkage_name:
                            if (attributes.ExtractFormValueAtIndex(dwarf2Data, i, form_value))
                                mangled = form_value.AsCString(debug_str);
                            break;

                        case DW_AT_low_pc:
                        case DW_AT_ranges:
                        case DW_AT_entry_pc:
                            if (tag == DW_TAG_subprogram)
                                add_die = true;
                            break;

                        case DW_AT_location:
                            if (tag == DW_TAG_variable)
                            {
                                const DWARFDebugInfoEntry* parent_die = die->GetParent();
                                while ( parent_die != NULL )
                                {
                                    switch (parent_die->Tag())
                                    {
                                    case DW_TAG_subprogram:
                                    case DW_TAG_lexical_block:
                                    case DW_TAG_inlined_subroutine:
                                        // Even if this is a function level static, we don't add it. We could theoretically
                                        // add these if we wanted to by introspecting into the DW_AT_location and seeing
                                        // if the location describes a hard coded address, but we don't want the performance
                                        // penalty of that right now.
                                        add_die = false;
//                                      if (attributes.ExtractFormValueAtIndex(dwarf2Data, i, form_value))
//                                      {
//                                          // If we have valid block data, then we have location expression bytes
//                                          // that are fixed (not a location list).
//                                          const uint8_t *block_data = form_value.BlockData();
//                                          if (block_data)
//                                          {
//                                              uint32_t block_length = form_value.Unsigned();
//                                              if (block_length == 1 + attributes.CompileUnitAtIndex(i)->GetAddressByteSize())
//                                              {
//                                                  if (block_data[0] == DW_OP_addr)
//                                                      add_die = true;
//                                              }
//                                          }
//                                      }
                                        parent_die = NULL;  // Terminate the while loop.
                                        break;

                                    case DW_TAG_compile_unit:
                                        add_die = true;
                                        parent_die = NULL;  // Terminate the while loop.
                                        break;

                                    default:
                                        parent_die = parent_die->GetParent();   // Keep going in the while loop.
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }

                if (add_die && (name || mangled))
                {
                    pubnames_set.AddDescriptor(die->GetOffset() - cu_offset, mangled ? mangled : name);
                }
            }

            if (pubnames_set.NumDescriptors() > 0)
            {
                m_sets.push_back(pubnames_set);
            }
            
            // Keep memory down by clearing DIEs if this generate function
            // caused them to be parsed
            if (clear_dies)
                cu->ClearDIEs (true);
        }
    }
    if (m_sets.empty())
        return false;
    if (log)
        Dump (log);
    return true;
}

bool
DWARFDebugPubnames::GeneratePubBaseTypes(SymbolFileDWARF* dwarf2Data)
{
    m_sets.clear();
    DWARFDebugInfo* debug_info = dwarf2Data->DebugInfo();
    if (debug_info)
    {
        uint32_t cu_idx = 0;
        const uint32_t num_compile_units = dwarf2Data->GetNumCompileUnits();
        for (cu_idx = 0; cu_idx < num_compile_units; ++cu_idx)
        {
            DWARFCompileUnit* cu = debug_info->GetCompileUnitAtIndex(cu_idx);
            DWARFDIECollection dies;
            const size_t die_count = cu->AppendDIEsWithTag (DW_TAG_base_type, dies);
            dw_offset_t cu_offset = cu->GetOffset();
            DWARFDebugPubnamesSet pubnames_set(DW_INVALID_OFFSET, cu_offset, cu->GetNextCompileUnitOffset() - cu_offset);

            size_t die_idx;
            for (die_idx = 0; die_idx < die_count; ++die_idx)
            {
                const DWARFDebugInfoEntry *die = dies.GetDIEPtrAtIndex(die_idx);
                const char *name = die->GetAttributeValueAsString(dwarf2Data, cu, DW_AT_name, NULL);

                if (name)
                {
                    pubnames_set.AddDescriptor(die->GetOffset() - cu_offset, name);
                }
            }

            if (pubnames_set.NumDescriptors() > 0)
            {
                m_sets.push_back(pubnames_set);
            }
        }
    }
    return !m_sets.empty();
}

void
DWARFDebugPubnames::Dump(Log *s) const
{
    if (m_sets.empty())
        s->PutCString("< EMPTY >\n");
    else
    {
        const_iterator pos;
        const_iterator end = m_sets.end();

        for (pos = m_sets.begin(); pos != end; ++pos)
            (*pos).Dump(s);
    }
}

bool
DWARFDebugPubnames::Find(const char* name, bool ignore_case, std::vector<dw_offset_t>& die_offsets) const
{
    const_iterator pos;
    const_iterator end = m_sets.end();

    die_offsets.clear();

    for (pos = m_sets.begin(); pos != end; ++pos)
    {
        (*pos).Find(name, ignore_case, die_offsets);
    }

    return !die_offsets.empty();
}

bool
DWARFDebugPubnames::Find(const RegularExpression& regex, std::vector<dw_offset_t>& die_offsets) const
{
    const_iterator pos;
    const_iterator end = m_sets.end();

    die_offsets.clear();

    for (pos = m_sets.begin(); pos != end; ++pos)
    {
        (*pos).Find(regex, die_offsets);
    }

    return !die_offsets.empty();
}
