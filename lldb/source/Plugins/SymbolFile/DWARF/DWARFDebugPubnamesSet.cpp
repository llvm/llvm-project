//===-- DWARFDebugPubnamesSet.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugPubnamesSet.h"

#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/Log.h"

#include "SymbolFileDWARF.h"

using namespace lldb_private;

DWARFDebugPubnamesSet::DWARFDebugPubnamesSet() :
    m_offset(DW_INVALID_OFFSET),
    m_header(),
    m_descriptors(),
    m_name_to_descriptor_index()
{
}

DWARFDebugPubnamesSet::DWARFDebugPubnamesSet(dw_offset_t debug_aranges_offset, dw_offset_t cu_die_offset, dw_offset_t cu_die_length) :
    m_offset(debug_aranges_offset),
    m_header(),
    m_descriptors(),
    m_name_to_descriptor_index()
{
    m_header.length = 10;               // set the length to only include the header right for now
    m_header.version = 2;               // The DWARF version number
    m_header.die_offset = cu_die_offset;// compile unit .debug_info offset
    m_header.die_length = cu_die_length;// compile unit .debug_info length
}

void
DWARFDebugPubnamesSet::AddDescriptor(dw_offset_t cu_rel_offset, const char* name)
{
    if (name && name[0])
    {
        // Adjust our header length
        m_header.length += strlen(name) + 1 + sizeof(dw_offset_t);
        Descriptor pubnameDesc(cu_rel_offset, name);
        m_descriptors.push_back(pubnameDesc);
    }
}

void
DWARFDebugPubnamesSet::Clear()
{
    m_offset = DW_INVALID_OFFSET;
    m_header.length = 10;
    m_header.version = 2;
    m_header.die_offset = DW_INVALID_OFFSET;
    m_header.die_length = 0;
    m_descriptors.clear();
}


//----------------------------------------------------------------------
// InitNameIndexes
//----------------------------------------------------------------------
void
DWARFDebugPubnamesSet::InitNameIndexes() const
{
    // Create the name index vector to be able to quickly search by name
    const size_t count = m_descriptors.size();
    for (uint32_t idx = 0; idx < count; ++idx)
    {
        const char* name = m_descriptors[idx].name.c_str();
        if (name && name[0])
            m_name_to_descriptor_index.insert(cstr_to_index_mmap::value_type(name, idx));
    }
}


bool
DWARFDebugPubnamesSet::Extract(const DataExtractor& data, uint32_t* offset_ptr)
{
    if (data.ValidOffset(*offset_ptr))
    {
        m_descriptors.clear();
        m_offset = *offset_ptr;
        m_header.length     = data.GetU32(offset_ptr);
        m_header.version    = data.GetU16(offset_ptr);
        m_header.die_offset = data.GetU32(offset_ptr);
        m_header.die_length = data.GetU32(offset_ptr);

        Descriptor pubnameDesc;
        while (data.ValidOffset(*offset_ptr))
        {
            pubnameDesc.offset  = data.GetU32(offset_ptr);

            if (pubnameDesc.offset)
            {
                const char* name = data.GetCStr(offset_ptr);
                if (name && name[0])
                {
                    pubnameDesc.name = name;
                    m_descriptors.push_back(pubnameDesc);
                }
            }
            else
                break;  // We are done if we get a zero 4 byte offset
        }

        return !m_descriptors.empty();
    }
    return false;
}

dw_offset_t
DWARFDebugPubnamesSet::GetOffsetOfNextEntry() const
{
    return m_offset + m_header.length + 4;
}

void
DWARFDebugPubnamesSet::Dump(Log *log) const
{
    log->Printf("Pubnames Header: length = 0x%8.8x, version = 0x%4.4x, die_offset = 0x%8.8x, die_length = 0x%8.8x",
        m_header.length,
        m_header.version,
        m_header.die_offset,
        m_header.die_length);

    bool verbose = log->GetVerbose();

    DescriptorConstIter pos;
    DescriptorConstIter end = m_descriptors.end();
    for (pos = m_descriptors.begin(); pos != end; ++pos)
    {
        if (verbose)
            log->Printf("0x%8.8x + 0x%8.8x = 0x%8.8x: %s", pos->offset, m_header.die_offset, pos->offset + m_header.die_offset, pos->name.c_str());
        else
            log->Printf("0x%8.8x: %s", pos->offset + m_header.die_offset, pos->name.c_str());
    }
}


void
DWARFDebugPubnamesSet::Find(const char* name, bool ignore_case, std::vector<dw_offset_t>& die_offset_coll) const
{
    if (!m_descriptors.empty() && m_name_to_descriptor_index.empty())
        InitNameIndexes();

    std::pair<cstr_to_index_mmap::const_iterator, cstr_to_index_mmap::const_iterator> range(m_name_to_descriptor_index.equal_range(name));
    for (cstr_to_index_mmap::const_iterator pos = range.first; pos != range.second; ++pos)
        die_offset_coll.push_back(m_header.die_offset + m_descriptors[(*pos).second].offset);
}

void
DWARFDebugPubnamesSet::Find(const RegularExpression& regex, std::vector<dw_offset_t>& die_offset_coll) const
{
    DescriptorConstIter pos;
    DescriptorConstIter end = m_descriptors.end();
    for (pos = m_descriptors.begin(); pos != end; ++pos)
    {
        if ( regex.Execute(pos->name.c_str()) )
            die_offset_coll.push_back(m_header.die_offset + pos->offset);
    }
}

