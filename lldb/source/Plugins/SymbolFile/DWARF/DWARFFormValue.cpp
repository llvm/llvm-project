//===-- DWARFFormValue.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <assert.h>

#include "lldb/Core/dwarf.h"
#include "lldb/Core/Stream.h"

#include "DWARFFormValue.h"
#include "DWARFCompileUnit.h"

class DWARFCompileUnit;

using namespace lldb_private;


static uint8_t g_form_sizes_addr4[] = 
{
    0, // 0x00 unused
    4, // 0x01 DW_FORM_addr
    0, // 0x02 unused
    0, // 0x03 DW_FORM_block2
    0, // 0x04 DW_FORM_block4
    2, // 0x05 DW_FORM_data2
    4, // 0x06 DW_FORM_data4
    8, // 0x07 DW_FORM_data8
    0, // 0x08 DW_FORM_string
    0, // 0x09 DW_FORM_block
    0, // 0x0a DW_FORM_block1
    1, // 0x0b DW_FORM_data1
    1, // 0x0c DW_FORM_flag
    0, // 0x0d DW_FORM_sdata
    4, // 0x0e DW_FORM_strp
    0, // 0x0f DW_FORM_udata
    0, // 0x10 DW_FORM_ref_addr (addr size for DWARF2 and earlier, 4 bytes for DWARF32, 8 bytes for DWARF32 in DWARF 3 and later
    1, // 0x11 DW_FORM_ref1
    2, // 0x12 DW_FORM_ref2
    4, // 0x13 DW_FORM_ref4
    8, // 0x14 DW_FORM_ref8
    0, // 0x15 DW_FORM_ref_udata
    0, // 0x16 DW_FORM_indirect
    4, // 0x17 DW_FORM_sec_offset
    0, // 0x18 DW_FORM_exprloc
    0, // 0x19 DW_FORM_flag_present
    0, // 0x1a
    0, // 0x1b
    0, // 0x1c
    0, // 0x1d
    0, // 0x1e
    0, // 0x1f
    8, // 0x20 DW_FORM_ref_sig8

};

static uint8_t
g_form_sizes_addr8[] = 
{
    0, // 0x00 unused
    8, // 0x01 DW_FORM_addr
    0, // 0x02 unused
    0, // 0x03 DW_FORM_block2
    0, // 0x04 DW_FORM_block4
    2, // 0x05 DW_FORM_data2
    4, // 0x06 DW_FORM_data4
    8, // 0x07 DW_FORM_data8
    0, // 0x08 DW_FORM_string
    0, // 0x09 DW_FORM_block
    0, // 0x0a DW_FORM_block1
    1, // 0x0b DW_FORM_data1
    1, // 0x0c DW_FORM_flag
    0, // 0x0d DW_FORM_sdata
    4, // 0x0e DW_FORM_strp
    0, // 0x0f DW_FORM_udata
    0, // 0x10 DW_FORM_ref_addr (addr size for DWARF2 and earlier, 4 bytes for DWARF32, 8 bytes for DWARF32 in DWARF 3 and later
    1, // 0x11 DW_FORM_ref1
    2, // 0x12 DW_FORM_ref2
    4, // 0x13 DW_FORM_ref4
    8, // 0x14 DW_FORM_ref8
    0, // 0x15 DW_FORM_ref_udata
    0, // 0x16 DW_FORM_indirect
    4, // 0x17 DW_FORM_sec_offset
    0, // 0x18 DW_FORM_exprloc
    0, // 0x19 DW_FORM_flag_present
    0, // 0x1a
    0, // 0x1b
    0, // 0x1c
    0, // 0x1d
    0, // 0x1e
    0, // 0x1f
    8, // 0x20 DW_FORM_ref_sig8
};

const uint8_t * 
DWARFFormValue::GetFixedFormSizesForAddressSize (uint8_t addr_size)
{
    switch (addr_size)
    {
    case 4: return g_form_sizes_addr4;
    case 8: return g_form_sizes_addr8;
    }
    return NULL;
}

DWARFFormValue::DWARFFormValue(dw_form_t form) :
    m_form(form),
    m_value()
{
}

bool
DWARFFormValue::ExtractValue(const DataExtractor& data, lldb::offset_t* offset_ptr, const DWARFCompileUnit* cu)
{
    bool indirect = false;
    bool is_block = false;
    m_value.data = NULL;
    // Read the value for the form into value and follow and DW_FORM_indirect instances we run into
    do
    {
        indirect = false;
        switch (m_form)
        {
        case DW_FORM_addr:      m_value.value.uval = data.GetMaxU64(offset_ptr, DWARFCompileUnit::GetAddressByteSize(cu));  break;
        case DW_FORM_block2:    m_value.value.uval = data.GetU16(offset_ptr); is_block = true;          break;
        case DW_FORM_block4:    m_value.value.uval = data.GetU32(offset_ptr); is_block = true;          break;
        case DW_FORM_data2:     m_value.value.uval = data.GetU16(offset_ptr);                           break;
        case DW_FORM_data4:     m_value.value.uval = data.GetU32(offset_ptr);                           break;
        case DW_FORM_data8:     m_value.value.uval = data.GetU64(offset_ptr);                           break;
        case DW_FORM_string:    m_value.value.cstr = data.GetCStr(offset_ptr);
                                // Set the string value to also be the data for inlined cstr form values only
                                // so we can tell the differnence between DW_FORM_string and DW_FORM_strp form
                                // values;
                                m_value.data = (uint8_t*)m_value.value.cstr;                            break;
        case DW_FORM_exprloc:
        case DW_FORM_block:     m_value.value.uval = data.GetULEB128(offset_ptr); is_block = true;      break;
        case DW_FORM_block1:    m_value.value.uval = data.GetU8(offset_ptr); is_block = true;           break;
        case DW_FORM_data1:     m_value.value.uval = data.GetU8(offset_ptr);                            break;
        case DW_FORM_flag:      m_value.value.uval = data.GetU8(offset_ptr);                            break;
        case DW_FORM_sdata:     m_value.value.sval = data.GetSLEB128(offset_ptr);                       break;
        case DW_FORM_strp:      m_value.value.uval = data.GetU32(offset_ptr);                           break;
    //  case DW_FORM_APPLE_db_str:
        case DW_FORM_udata:     m_value.value.uval = data.GetULEB128(offset_ptr);                       break;
        case DW_FORM_ref_addr:
            if (cu->GetVersion() <= 2)
                m_value.value.uval = data.GetMaxU64(offset_ptr, DWARFCompileUnit::GetAddressByteSize(cu));
            else
                m_value.value.uval = data.GetU32(offset_ptr); // 4 for DWARF32, 8 for DWARF64, but we don't support DWARF64 yet
            break;
        case DW_FORM_ref1:      m_value.value.uval = data.GetU8(offset_ptr);                            break;
        case DW_FORM_ref2:      m_value.value.uval = data.GetU16(offset_ptr);                           break;
        case DW_FORM_ref4:      m_value.value.uval = data.GetU32(offset_ptr);                           break;
        case DW_FORM_ref8:      m_value.value.uval = data.GetU64(offset_ptr);                           break;
        case DW_FORM_ref_udata: m_value.value.uval = data.GetULEB128(offset_ptr);                       break;
        case DW_FORM_indirect:
            m_form = data.GetULEB128(offset_ptr);
            indirect = true;
            break;

        case DW_FORM_sec_offset:    m_value.value.uval = data.GetU32(offset_ptr);                       break;
        case DW_FORM_flag_present:  m_value.value.uval = 1;                                             break;
        case DW_FORM_ref_sig8:      m_value.value.uval = data.GetU64(offset_ptr);                       break;
        default:
            return false;
            break;
        }
    } while (indirect);

    if (is_block)
    {
        m_value.data = data.PeekData(*offset_ptr, m_value.value.uval);
        if (m_value.data != NULL)
        {
            *offset_ptr += m_value.value.uval;
        }
    }

    return true;
}

bool
DWARFFormValue::SkipValue(const DataExtractor& debug_info_data, lldb::offset_t *offset_ptr, const DWARFCompileUnit* cu) const
{
    return DWARFFormValue::SkipValue(m_form, debug_info_data, offset_ptr, cu);
}

bool
DWARFFormValue::SkipValue(dw_form_t form, const DataExtractor& debug_info_data, lldb::offset_t *offset_ptr, const DWARFCompileUnit* cu)
{
    switch (form)
    {
    // Blocks if inlined data that have a length field and the data bytes
    // inlined in the .debug_info
    case DW_FORM_exprloc:
    case DW_FORM_block:  { dw_uleb128_t size = debug_info_data.GetULEB128(offset_ptr); *offset_ptr += size; } return true;
    case DW_FORM_block1: { dw_uleb128_t size = debug_info_data.GetU8(offset_ptr);      *offset_ptr += size; } return true;
    case DW_FORM_block2: { dw_uleb128_t size = debug_info_data.GetU16(offset_ptr);     *offset_ptr += size; } return true;
    case DW_FORM_block4: { dw_uleb128_t size = debug_info_data.GetU32(offset_ptr);     *offset_ptr += size; } return true;

    // Inlined NULL terminated C-strings
    case DW_FORM_string:
        debug_info_data.GetCStr(offset_ptr);
        return true;

    // Compile unit address sized values
    case DW_FORM_addr:
        *offset_ptr += DWARFCompileUnit::GetAddressByteSize(cu);
        return true;

    case DW_FORM_ref_addr:
        if (cu->GetVersion() <= 2)
            *offset_ptr += DWARFCompileUnit::GetAddressByteSize(cu);
        else
            *offset_ptr += 4;// 4 for DWARF32, 8 for DWARF64, but we don't support DWARF64 yet
        return true;

    // 0 bytes values (implied from DW_FORM)
    case DW_FORM_flag_present:
        return true;

    // 1 byte values
    case DW_FORM_data1:
    case DW_FORM_flag:
    case DW_FORM_ref1:
        *offset_ptr += 1;
        return true;

    // 2 byte values
    case DW_FORM_data2:
    case DW_FORM_ref2:
        *offset_ptr += 2;
        return true;

    // 32 bit for DWARF 32, 64 for DWARF 64
    case DW_FORM_sec_offset:
        *offset_ptr += 4;
        return true;

    // 4 byte values
    case DW_FORM_strp:
    case DW_FORM_data4:
    case DW_FORM_ref4:
        *offset_ptr += 4;
        return true;

    // 8 byte values
    case DW_FORM_data8:
    case DW_FORM_ref8:
    case DW_FORM_ref_sig8:
        *offset_ptr += 8;
        return true;

    // signed or unsigned LEB 128 values
    case DW_FORM_sdata:
    case DW_FORM_udata:
    case DW_FORM_ref_udata:
        debug_info_data.Skip_LEB128(offset_ptr);
        return true;

    case DW_FORM_indirect:
        {
            dw_form_t indirect_form = debug_info_data.GetULEB128(offset_ptr);
            return DWARFFormValue::SkipValue (indirect_form,
                                              debug_info_data,
                                              offset_ptr,
                                              cu);
        }

    default:
        break;
    }
    return false;
}


void
DWARFFormValue::Dump(Stream &s, const DataExtractor* debug_str_data, const DWARFCompileUnit* cu) const
{
    uint64_t uvalue = Unsigned();
    bool cu_relative_offset = false;

    bool verbose = s.GetVerbose();

    switch (m_form)
    {
    case DW_FORM_addr:      s.Address(uvalue, sizeof (uint64_t)); break;
    case DW_FORM_flag:
    case DW_FORM_data1:     s.PutHex8(uvalue);     break;
    case DW_FORM_data2:     s.PutHex16(uvalue);        break;
    case DW_FORM_sec_offset:
    case DW_FORM_data4:     s.PutHex32(uvalue);        break;
    case DW_FORM_ref_sig8:
    case DW_FORM_data8:     s.PutHex64(uvalue);        break;
    case DW_FORM_string:    s.QuotedCString(AsCString(NULL));          break;
    case DW_FORM_exprloc:
    case DW_FORM_block:
    case DW_FORM_block1:
    case DW_FORM_block2:
    case DW_FORM_block4:
        if (uvalue > 0)
        {
            switch (m_form)
            {
            case DW_FORM_exprloc:
            case DW_FORM_block:  s.Printf("<0x%" PRIx64 "> ", uvalue);                break;
            case DW_FORM_block1: s.Printf("<0x%2.2x> ", (uint8_t)uvalue);      break;
            case DW_FORM_block2: s.Printf("<0x%4.4x> ", (uint16_t)uvalue);     break;
            case DW_FORM_block4: s.Printf("<0x%8.8x> ", (uint32_t)uvalue);     break;
            default:                                                            break;
            }

            const uint8_t* data_ptr = m_value.data;
            if (data_ptr)
            {
                const uint8_t* end_data_ptr = data_ptr + uvalue;    // uvalue contains size of block
                while (data_ptr < end_data_ptr)
                {
                    s.Printf("%2.2x ", *data_ptr);
                    ++data_ptr;
                }
            }
            else
                s.PutCString("NULL");
        }
        break;

    case DW_FORM_sdata:     s.PutSLEB128(uvalue); break;
    case DW_FORM_udata:     s.PutULEB128(uvalue); break;
    case DW_FORM_strp:
        if (debug_str_data)
        {
            if (verbose)
                s.Printf(" .debug_str[0x%8.8x] = ", (uint32_t)uvalue);

            const char* dbg_str = AsCString(debug_str_data);
            if (dbg_str)
                s.QuotedCString(dbg_str);
        }
        else
        {
            s.PutHex32(uvalue);
        }
        break;

    case DW_FORM_ref_addr:
    {
        if (cu->GetVersion() <= 2)
            s.Address(uvalue, sizeof (uint64_t) * 2);
        else
            s.Address(uvalue, 4 * 2);// 4 for DWARF32, 8 for DWARF64, but we don't support DWARF64 yet
        break;
    }
    case DW_FORM_ref1:      cu_relative_offset = true;  if (verbose) s.Printf("cu + 0x%2.2x", (uint8_t)uvalue); break;
    case DW_FORM_ref2:      cu_relative_offset = true;  if (verbose) s.Printf("cu + 0x%4.4x", (uint16_t)uvalue); break;
    case DW_FORM_ref4:      cu_relative_offset = true;  if (verbose) s.Printf("cu + 0x%4.4x", (uint32_t)uvalue); break;
    case DW_FORM_ref8:      cu_relative_offset = true;  if (verbose) s.Printf("cu + 0x%8.8" PRIx64, uvalue); break;
    case DW_FORM_ref_udata: cu_relative_offset = true;  if (verbose) s.Printf("cu + 0x%" PRIx64, uvalue); break;

    // All DW_FORM_indirect attributes should be resolved prior to calling this function
    case DW_FORM_indirect:  s.PutCString("DW_FORM_indirect"); break;
    case DW_FORM_flag_present: break;
    default:
        s.Printf("DW_FORM(0x%4.4x)", m_form);
        break;
    }

    if (cu_relative_offset)
    {
        if (verbose)
            s.PutCString(" => ");

        s.Printf("{0x%8.8" PRIx64 "}", (uvalue + (cu ? cu->GetOffset() : 0)));
    }
}

const char*
DWARFFormValue::AsCString(const DataExtractor* debug_str_data_ptr) const
{
    if (IsInlinedCStr())
        return m_value.value.cstr;
    else if (debug_str_data_ptr)
        return debug_str_data_ptr->PeekCStr(m_value.value.uval);
    return NULL;
}

uint64_t
DWARFFormValue::Reference(const DWARFCompileUnit* cu) const
{
    uint64_t die_offset = m_value.value.uval;
    switch (m_form)
    {
    case DW_FORM_ref1:
    case DW_FORM_ref2:
    case DW_FORM_ref4:
    case DW_FORM_ref8:
    case DW_FORM_ref_udata:
        die_offset += (cu ? cu->GetOffset() : 0);
        break;

    default:
        break;
    }

    return die_offset;
}

uint64_t
DWARFFormValue::Reference (dw_offset_t base_offset) const
{
    uint64_t die_offset = m_value.value.uval;
    switch (m_form)
    {
        case DW_FORM_ref1:
        case DW_FORM_ref2:
        case DW_FORM_ref4:
        case DW_FORM_ref8:
        case DW_FORM_ref_udata:
            die_offset += base_offset;
            break;
            
        default:
            break;
    }
    
    return die_offset;
}

//----------------------------------------------------------------------
// Resolve any compile unit specific references so that we don't need
// the compile unit at a later time in order to work with the form
// value.
//----------------------------------------------------------------------
bool
DWARFFormValue::ResolveCompileUnitReferences(const DWARFCompileUnit* cu)
{
    switch (m_form)
    {
    case DW_FORM_ref1:
    case DW_FORM_ref2:
    case DW_FORM_ref4:
    case DW_FORM_ref8:
    case DW_FORM_ref_udata:
        m_value.value.uval += cu->GetOffset();
        m_form = DW_FORM_ref_addr;
        return true;
        break;

    default:
        break;
    }

    return false;
}

const uint8_t*
DWARFFormValue::BlockData() const
{
    if (!IsInlinedCStr())
        return m_value.data;
    return NULL;
}


bool
DWARFFormValue::IsBlockForm(const dw_form_t form)
{
    switch (form)
    {
    case DW_FORM_block:
    case DW_FORM_block1:
    case DW_FORM_block2:
    case DW_FORM_block4:
        return true;
    }
    return false;
}

bool
DWARFFormValue::IsDataForm(const dw_form_t form)
{
    switch (form)
    {
    case DW_FORM_sdata:
    case DW_FORM_udata:
    case DW_FORM_data1:
    case DW_FORM_data2:
    case DW_FORM_data4:
    case DW_FORM_data8:
        return true;
    }
    return false;
}

int
DWARFFormValue::Compare (const DWARFFormValue& a_value, const DWARFFormValue& b_value, const DWARFCompileUnit* a_cu, const DWARFCompileUnit* b_cu, const DataExtractor* debug_str_data_ptr)
{
    dw_form_t a_form = a_value.Form();
    dw_form_t b_form = b_value.Form();
    if (a_form < b_form)
        return -1;
    if (a_form > b_form)
        return 1;
    switch (a_form)
    {
    case DW_FORM_addr:
    case DW_FORM_flag:
    case DW_FORM_data1:
    case DW_FORM_data2:
    case DW_FORM_data4:
    case DW_FORM_data8:
    case DW_FORM_udata:
    case DW_FORM_ref_addr:
    case DW_FORM_sec_offset:
    case DW_FORM_flag_present:
    case DW_FORM_ref_sig8:
        {
            uint64_t a = a_value.Unsigned();
            uint64_t b = b_value.Unsigned();
            if (a < b)
                return -1;
            if (a > b)
                return 1;
            return 0;
        }

    case DW_FORM_sdata:
        {
            int64_t a = a_value.Signed();
            int64_t b = b_value.Signed();
            if (a < b)
                return -1;
            if (a > b)
                return 1;
            return 0;
        }

    case DW_FORM_string:
    case DW_FORM_strp:
        {
            const char *a_string = a_value.AsCString(debug_str_data_ptr);
            const char *b_string = b_value.AsCString(debug_str_data_ptr);
            if (a_string == b_string)
                return 0;
            else if (a_string && b_string)
                return strcmp(a_string, b_string);
            else if (a_string == NULL)
                return -1;  // A string is NULL, and B is valid
            else
                return 1;   // A string valid, and B is NULL
        }


    case DW_FORM_block:
    case DW_FORM_block1:
    case DW_FORM_block2:
    case DW_FORM_block4:
    case DW_FORM_exprloc:
        {
            uint64_t a_len = a_value.Unsigned();
            uint64_t b_len = b_value.Unsigned();
            if (a_len < b_len)
                return -1;
            if (a_len > b_len)
                return 1;
            // The block lengths are the same
            return memcmp(a_value.BlockData(), b_value.BlockData(), a_value.Unsigned());
        }
        break;

    case DW_FORM_ref1:
    case DW_FORM_ref2:
    case DW_FORM_ref4:
    case DW_FORM_ref8:
    case DW_FORM_ref_udata:
        {
            uint64_t a = a_value.Reference(a_cu);
            uint64_t b = b_value.Reference(b_cu);
            if (a < b)
                return -1;
            if (a > b)
                return 1;
            return 0;
        }

    case DW_FORM_indirect:
        assert(!"This shouldn't happen after the form has been extracted...");
        break;

    default:
        assert(!"Unhandled DW_FORM");
        break;
    }
    return -1;
}

