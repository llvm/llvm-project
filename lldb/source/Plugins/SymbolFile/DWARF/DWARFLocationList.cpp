//===-- DWARFLocationList.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFLocationList.h"

#include "lldb/Core/Stream.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugInfo.h"
#include "DWARFLocationDescription.h"

using namespace lldb_private;

dw_offset_t
DWARFLocationList::Dump(Stream &s, const DWARFCompileUnit* cu, const DataExtractor& debug_loc_data, dw_offset_t offset)
{
    uint64_t start_addr, end_addr;
    uint32_t addr_size = DWARFCompileUnit::GetAddressByteSize(cu);
    s.SetAddressByteSize(DWARFCompileUnit::GetAddressByteSize(cu));
    dw_addr_t base_addr = cu ? cu->GetBaseAddress() : 0;
    while (debug_loc_data.ValidOffset(offset))
    {
        start_addr = debug_loc_data.GetMaxU64(&offset,addr_size);
        end_addr = debug_loc_data.GetMaxU64(&offset,addr_size);

        if (start_addr == 0 && end_addr == 0)
            break;

        s.PutCString("\n            ");
        s.Indent();
        if (cu)
            s.AddressRange (start_addr + base_addr, 
                            end_addr + base_addr, 
                            cu->GetAddressByteSize(), 
                            NULL,
                            ": ");
        uint32_t loc_length = debug_loc_data.GetU16(&offset);

        DataExtractor locationData(debug_loc_data, offset, loc_length);
    //  if ( dump_flags & DWARFDebugInfo::eDumpFlag_Verbose ) *ostrm_ptr << " ( ";
        print_dwarf_expression (s, locationData, addr_size, 4, false);
        offset += loc_length;
    }

    return offset;
}

bool
DWARFLocationList::Extract(const DataExtractor& debug_loc_data, dw_offset_t* offset_ptr, DataExtractor& location_list_data)
{
    // Initialize with no data just in case we don't find anything
    location_list_data.Clear();

    size_t loc_list_length = Size(debug_loc_data, *offset_ptr);
    if (loc_list_length > 0)
    {
        location_list_data.SetData(debug_loc_data, *offset_ptr, loc_list_length);
        *offset_ptr += loc_list_length;
        return true;
    }

    return false;
}

size_t
DWARFLocationList::Size(const DataExtractor& debug_loc_data, dw_offset_t offset)
{
    const dw_offset_t debug_loc_offset = offset;

    while (debug_loc_data.ValidOffset(offset))
    {
        dw_addr_t start_addr = debug_loc_data.GetAddress(&offset);
        dw_addr_t end_addr = debug_loc_data.GetAddress(&offset);

        if (start_addr == 0 && end_addr == 0)
            break;

        uint16_t loc_length = debug_loc_data.GetU16(&offset);
        offset += loc_length;
    }

    if (offset > debug_loc_offset)
        return offset - debug_loc_offset;
    return 0;
}



