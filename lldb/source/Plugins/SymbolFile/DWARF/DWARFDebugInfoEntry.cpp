//===-- DWARFDebugInfoEntry.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugInfoEntry.h"

#include <assert.h>

#include <algorithm>

#include "lldb/Core/Module.h"
#include "lldb/Core/Stream.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ObjectFile.h"

#include "DWARFCompileUnit.h"
#include "SymbolFileDWARF.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDeclContext.h"
#include "DWARFDIECollection.h"
#include "DWARFFormValue.h"
#include "DWARFLocationDescription.h"
#include "DWARFLocationList.h"
#include "DWARFDebugRanges.h"

using namespace lldb_private;
using namespace std;
extern int g_verbose;



DWARFDebugInfoEntry::Attributes::Attributes() :
    m_infos()
{
}

DWARFDebugInfoEntry::Attributes::~Attributes()
{
}


uint32_t
DWARFDebugInfoEntry::Attributes::FindAttributeIndex(dw_attr_t attr) const
{
    collection::const_iterator end = m_infos.end();
    collection::const_iterator beg = m_infos.begin();
    collection::const_iterator pos;
    for (pos = beg; pos != end; ++pos)
    {
        if (pos->attr == attr)
            return std::distance(beg, pos);
    }
    return UINT32_MAX;
}

void
DWARFDebugInfoEntry::Attributes::Append(const DWARFCompileUnit *cu, dw_offset_t attr_die_offset, dw_attr_t attr, dw_form_t form)
{
    Info info = { cu, attr_die_offset, attr, form };
    m_infos.push_back(info);
}

bool
DWARFDebugInfoEntry::Attributes::ContainsAttribute(dw_attr_t attr) const
{
    return FindAttributeIndex(attr) != UINT32_MAX;
}

bool
DWARFDebugInfoEntry::Attributes::RemoveAttribute(dw_attr_t attr)
{
    uint32_t attr_index = FindAttributeIndex(attr);
    if (attr_index != UINT32_MAX)
    {
        m_infos.erase(m_infos.begin() + attr_index);
        return true;
    }
    return false;
}

bool
DWARFDebugInfoEntry::Attributes::ExtractFormValueAtIndex (SymbolFileDWARF* dwarf2Data, uint32_t i, DWARFFormValue &form_value) const
{
    form_value.SetForm(FormAtIndex(i));
    lldb::offset_t offset = DIEOffsetAtIndex(i);
    return form_value.ExtractValue(dwarf2Data->get_debug_info_data(), &offset, CompileUnitAtIndex(i));
}

uint64_t
DWARFDebugInfoEntry::Attributes::FormValueAsUnsigned (SymbolFileDWARF* dwarf2Data, dw_attr_t attr, uint64_t fail_value) const
{
    const uint32_t attr_idx = FindAttributeIndex (attr);
    if (attr_idx != UINT32_MAX)
        return FormValueAsUnsignedAtIndex (dwarf2Data, attr_idx, fail_value);
    return fail_value;
}

uint64_t
DWARFDebugInfoEntry::Attributes::FormValueAsUnsignedAtIndex(SymbolFileDWARF* dwarf2Data, uint32_t i, uint64_t fail_value) const
{
    DWARFFormValue form_value;
    if (ExtractFormValueAtIndex(dwarf2Data, i, form_value))
        return form_value.Reference(CompileUnitAtIndex(i));
    return fail_value;
}



bool
DWARFDebugInfoEntry::FastExtract
(
    const DataExtractor& debug_info_data,
    const DWARFCompileUnit* cu,
    const uint8_t *fixed_form_sizes,
    lldb::offset_t *offset_ptr
)
{
    m_offset = *offset_ptr;
    m_parent_idx = 0;
    m_sibling_idx = 0;
    m_empty_children = false;
    const uint64_t abbr_idx = debug_info_data.GetULEB128 (offset_ptr);
    assert (abbr_idx < (1 << DIE_ABBR_IDX_BITSIZE));
    m_abbr_idx = abbr_idx;
    
    //assert (fixed_form_sizes);  // For best performance this should be specified!
    
    if (m_abbr_idx)
    {
        lldb::offset_t offset = *offset_ptr;

        const DWARFAbbreviationDeclaration *abbrevDecl = cu->GetAbbreviations()->GetAbbreviationDeclaration(m_abbr_idx);
        
        if (abbrevDecl == NULL)
        {
            cu->GetSymbolFileDWARF()->GetObjectFile()->GetModule()->ReportError ("{0x%8.8x}: invalid abbreviation code %u, please file a bug and attach the file at the start of this error message", 
                                                                                 m_offset, 
                                                                                 (unsigned)abbr_idx);
            // WE can't parse anymore if the DWARF is borked...
            *offset_ptr = UINT32_MAX;
            return false;
        }
        m_tag = abbrevDecl->Tag();
        m_has_children = abbrevDecl->HasChildren();
        // Skip all data in the .debug_info for the attributes
        const uint32_t numAttributes = abbrevDecl->NumAttributes();
        register uint32_t i;
        register dw_form_t form;
        for (i=0; i<numAttributes; ++i)
        {
            form = abbrevDecl->GetFormByIndexUnchecked(i);

            const uint8_t fixed_skip_size = fixed_form_sizes [form];
            if (fixed_skip_size)
                offset += fixed_skip_size;
            else
            {
                bool form_is_indirect = false;
                do
                {
                    form_is_indirect = false;
                    register uint32_t form_size = 0;
                    switch (form)
                    {
                    // Blocks if inlined data that have a length field and the data bytes
                    // inlined in the .debug_info
                    case DW_FORM_exprloc     :
                    case DW_FORM_block       : form_size = debug_info_data.GetULEB128 (&offset);      break;
                    case DW_FORM_block1      : form_size = debug_info_data.GetU8_unchecked (&offset); break;
                    case DW_FORM_block2      : form_size = debug_info_data.GetU16_unchecked (&offset);break;
                    case DW_FORM_block4      : form_size = debug_info_data.GetU32_unchecked (&offset);break;

                    // Inlined NULL terminated C-strings
                    case DW_FORM_string      :
                        debug_info_data.GetCStr (&offset);
                        break;

                    // Compile unit address sized values
                    case DW_FORM_addr        :
                        form_size = cu->GetAddressByteSize();
                        break;
                    case DW_FORM_ref_addr    :
                        if (cu->GetVersion() <= 2)
                            form_size = cu->GetAddressByteSize();
                        else
                            form_size = 4; // 4 bytes for DWARF 32, 8 bytes for DWARF 64, but we don't support DWARF64 yet
                        break;

                    // 0 sized form
                    case DW_FORM_flag_present:
                        form_size = 0;
                        break;

                    // 1 byte values
                    case DW_FORM_data1       :
                    case DW_FORM_flag        :
                    case DW_FORM_ref1        :
                        form_size = 1;
                        break;

                    // 2 byte values
                    case DW_FORM_data2       :
                    case DW_FORM_ref2        :
                        form_size = 2;
                        break;

                    // 4 byte values
                    case DW_FORM_strp        :
                    case DW_FORM_data4       :
                    case DW_FORM_ref4        :
                        form_size = 4;
                        break;

                    // 8 byte values
                    case DW_FORM_data8       :
                    case DW_FORM_ref8        :
                    case DW_FORM_ref_sig8    :
                        form_size = 8;
                        break;

                    // signed or unsigned LEB 128 values
                    case DW_FORM_sdata       :
                    case DW_FORM_udata       :
                    case DW_FORM_ref_udata   :
                        debug_info_data.Skip_LEB128 (&offset);
                        break;

                    case DW_FORM_indirect    :
                        form_is_indirect = true;
                        form = debug_info_data.GetULEB128 (&offset);
                        break;

                    case DW_FORM_sec_offset  :
                        if (cu->GetAddressByteSize () == 4)
                            debug_info_data.GetU32 (offset_ptr);
                        else
                            debug_info_data.GetU64 (offset_ptr);
                        break;

                    default:
                        *offset_ptr = m_offset;
                        return false;
                    }
                    offset += form_size;

                } while (form_is_indirect);
            }
        }
        *offset_ptr = offset;
        return true;
    }
    else
    {
        m_tag = 0;
        m_has_children = false;
        return true;    // NULL debug tag entry
    }

    return false;
}

//----------------------------------------------------------------------
// Extract
//
// Extract a debug info entry for a given compile unit from the
// .debug_info and .debug_abbrev data within the SymbolFileDWARF class
// starting at the given offset
//----------------------------------------------------------------------
bool
DWARFDebugInfoEntry::Extract
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    lldb::offset_t *offset_ptr
)
{
    const DataExtractor& debug_info_data = dwarf2Data->get_debug_info_data();
//    const DataExtractor& debug_str_data = dwarf2Data->get_debug_str_data();
    const uint32_t cu_end_offset = cu->GetNextCompileUnitOffset();
    const uint8_t cu_addr_size = cu->GetAddressByteSize();
    lldb::offset_t offset = *offset_ptr;
//  if (offset >= cu_end_offset)
//      Log::Error("DIE at offset 0x%8.8x is beyond the end of the current compile unit (0x%8.8x)", m_offset, cu_end_offset);
    if ((offset < cu_end_offset) && debug_info_data.ValidOffset(offset))
    {
        m_offset = offset;

        const uint64_t abbr_idx = debug_info_data.GetULEB128(&offset);
        assert (abbr_idx < (1 << DIE_ABBR_IDX_BITSIZE));
        m_abbr_idx = abbr_idx;
        if (abbr_idx)
        {
            const DWARFAbbreviationDeclaration *abbrevDecl = cu->GetAbbreviations()->GetAbbreviationDeclaration(abbr_idx);

            if (abbrevDecl)
            {
                m_tag = abbrevDecl->Tag();
                m_has_children = abbrevDecl->HasChildren();

                bool isCompileUnitTag = m_tag == DW_TAG_compile_unit;
                if (cu && isCompileUnitTag)
                    ((DWARFCompileUnit*)cu)->SetBaseAddress(0);

                // Skip all data in the .debug_info for the attributes
                const uint32_t numAttributes = abbrevDecl->NumAttributes();
                uint32_t i;
                dw_attr_t attr;
                dw_form_t form;
                for (i=0; i<numAttributes; ++i)
                {
                    abbrevDecl->GetAttrAndFormByIndexUnchecked(i, attr, form);

                    if (isCompileUnitTag && ((attr == DW_AT_entry_pc) || (attr == DW_AT_low_pc)))
                    {
                        DWARFFormValue form_value(form);
                        if (form_value.ExtractValue(debug_info_data, &offset, cu))
                        {
                            if (attr == DW_AT_low_pc || attr == DW_AT_entry_pc)
                                ((DWARFCompileUnit*)cu)->SetBaseAddress(form_value.Unsigned());
                        }
                    }
                    else
                    {
                        bool form_is_indirect = false;
                        do
                        {
                            form_is_indirect = false;
                            register uint32_t form_size = 0;
                            switch (form)
                            {
                            // Blocks if inlined data that have a length field and the data bytes
                            // inlined in the .debug_info
                            case DW_FORM_exprloc     :
                            case DW_FORM_block       : form_size = debug_info_data.GetULEB128(&offset);  break;
                            case DW_FORM_block1      : form_size = debug_info_data.GetU8(&offset);       break;
                            case DW_FORM_block2      : form_size = debug_info_data.GetU16(&offset);      break;
                            case DW_FORM_block4      : form_size = debug_info_data.GetU32(&offset);      break;

                            // Inlined NULL terminated C-strings
                            case DW_FORM_string      : debug_info_data.GetCStr(&offset);                 break;

                            // Compile unit address sized values
                            case DW_FORM_addr        :
                                form_size = cu_addr_size;
                                break;
                            case DW_FORM_ref_addr    :
                                if (cu->GetVersion() <= 2)
                                    form_size = cu_addr_size;
                                else
                                    form_size = 4; // 4 bytes for DWARF 32, 8 bytes for DWARF 64, but we don't support DWARF64 yet
                                break;

                            // 0 sized form
                            case DW_FORM_flag_present:
                                form_size = 0;
                                break;

                            // 1 byte values
                            case DW_FORM_data1       :
                            case DW_FORM_flag        :
                            case DW_FORM_ref1        :
                                form_size = 1;
                                break;

                            // 2 byte values
                            case DW_FORM_data2       :
                            case DW_FORM_ref2        :
                                form_size = 2;
                                break;

                            // 4 byte values
                            case DW_FORM_strp        :
                                form_size = 4;
                                break;

                            case DW_FORM_data4       :
                            case DW_FORM_ref4        :
                                form_size = 4;
                                break;

                            // 8 byte values
                            case DW_FORM_data8       :
                            case DW_FORM_ref8        :
                            case DW_FORM_ref_sig8    :
                                form_size = 8;
                                break;

                            // signed or unsigned LEB 128 values
                            case DW_FORM_sdata       :
                            case DW_FORM_udata       :
                            case DW_FORM_ref_udata   :
                                debug_info_data.Skip_LEB128(&offset);
                                break;

                            case DW_FORM_indirect    :
                                form = debug_info_data.GetULEB128(&offset);
                                form_is_indirect = true;
                                break;

                            case DW_FORM_sec_offset  :
                                if (cu->GetAddressByteSize () == 4)
                                    debug_info_data.GetU32 (offset_ptr);
                                else
                                    debug_info_data.GetU64 (offset_ptr);
                                break;

                            default:
                                *offset_ptr = offset;
                                return false;
                            }

                            offset += form_size;
                        } while (form_is_indirect);
                    }
                }
                *offset_ptr = offset;
                return true;
            }
        }
        else
        {
            m_tag = 0;
            m_has_children = false;
            *offset_ptr = offset;
            return true;    // NULL debug tag entry
        }
    }

    return false;
}

//----------------------------------------------------------------------
// DumpAncestry
//
// Dumps all of a debug information entries parents up until oldest and
// all of it's attributes to the specified stream.
//----------------------------------------------------------------------
void
DWARFDebugInfoEntry::DumpAncestry
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const DWARFDebugInfoEntry* oldest,
    Stream &s,
    uint32_t recurse_depth
) const
{
    const DWARFDebugInfoEntry* parent = GetParent();
    if (parent && parent != oldest)
        parent->DumpAncestry(dwarf2Data, cu, oldest, s, 0);
    Dump(dwarf2Data, cu, s, recurse_depth);
}

//----------------------------------------------------------------------
// Compare two DIE by comparing all their attributes values, and
// following all DW_FORM_ref attributes and comparing their contents as
// well (except for DW_AT_sibling attributes.
//
//  DWARFDebugInfoEntry::CompareState compare_state;
//  int result = DWARFDebugInfoEntry::Compare(this, 0x00017ccb, 0x0001eb2b, compare_state, false, true);
//----------------------------------------------------------------------
//int
//DWARFDebugInfoEntry::Compare
//(
//    SymbolFileDWARF* dwarf2Data,
//    dw_offset_t a_die_offset,
//    dw_offset_t b_die_offset,
//    CompareState &compare_state,
//    bool compare_siblings,
//    bool compare_children
//)
//{
//    if (a_die_offset == b_die_offset)
//        return 0;
//
//    DWARFCompileUnitSP a_cu_sp;
//    DWARFCompileUnitSP b_cu_sp;
//    const DWARFDebugInfoEntry* a_die = dwarf2Data->DebugInfo()->GetDIEPtr(a_die_offset, &a_cu_sp);
//    const DWARFDebugInfoEntry* b_die = dwarf2Data->DebugInfo()->GetDIEPtr(b_die_offset, &b_cu_sp);
//
//    return Compare(dwarf2Data, a_cu_sp.get(), a_die, b_cu_sp.get(), b_die, compare_state, compare_siblings, compare_children);
//}
//
//int
//DWARFDebugInfoEntry::Compare
//(
//    SymbolFileDWARF* dwarf2Data,
//    DWARFCompileUnit* a_cu, const DWARFDebugInfoEntry* a_die,
//    DWARFCompileUnit* b_cu, const DWARFDebugInfoEntry* b_die,
//    CompareState &compare_state,
//    bool compare_siblings,
//    bool compare_children
//)
//{
//    if (a_die == b_die)
//        return 0;
//
//    if (!compare_state.AddTypePair(a_die->GetOffset(), b_die->GetOffset()))
//    {
//        // We are already comparing both of these types, so let
//        // compares complete for the real result
//        return 0;
//    }
//
//    //printf("DWARFDebugInfoEntry::Compare(0x%8.8x, 0x%8.8x)\n", a_die->GetOffset(), b_die->GetOffset());
//
//    // Do we have two valid DIEs?
//    if (a_die && b_die)
//    {
//        // Both DIE are valid
//        int result = 0;
//
//        const dw_tag_t a_tag = a_die->Tag();
//        const dw_tag_t b_tag = b_die->Tag();
//        if (a_tag == 0 && b_tag == 0)
//            return 0;
//
//        //printf("    comparing tags: %s and %s\n", DW_TAG_value_to_name(a_tag), DW_TAG_value_to_name(b_tag));
//
//        if (a_tag < b_tag)
//            return -1;
//        else if (a_tag > b_tag)
//            return 1;
//
//        DWARFDebugInfoEntry::Attributes a_attrs;
//        DWARFDebugInfoEntry::Attributes b_attrs;
//        size_t a_attr_count = a_die->GetAttributes(dwarf2Data, a_cu, a_attrs);
//        size_t b_attr_count = b_die->GetAttributes(dwarf2Data, b_cu, b_attrs);
//        if (a_attr_count != b_attr_count)
//        {
//            a_attrs.RemoveAttribute(DW_AT_sibling);
//            b_attrs.RemoveAttribute(DW_AT_sibling);
//        }
//
//        a_attr_count = a_attrs.Size();
//        b_attr_count = b_attrs.Size();
//
//        DWARFFormValue a_form_value;
//        DWARFFormValue b_form_value;
//
//        if (a_attr_count != b_attr_count)
//        {
//            uint32_t is_decl_index = a_attrs.FindAttributeIndex(DW_AT_declaration);
//            uint32_t a_name_index = UINT32_MAX;
//            uint32_t b_name_index = UINT32_MAX;
//            if (is_decl_index != UINT32_MAX)
//            {
//                if (a_attr_count == 2)
//                {
//                    a_name_index = a_attrs.FindAttributeIndex(DW_AT_name);
//                    b_name_index = b_attrs.FindAttributeIndex(DW_AT_name);
//                }
//            }
//            else
//            {
//                is_decl_index = b_attrs.FindAttributeIndex(DW_AT_declaration);
//                if (is_decl_index != UINT32_MAX && a_attr_count == 2)
//                {
//                    a_name_index = a_attrs.FindAttributeIndex(DW_AT_name);
//                    b_name_index = b_attrs.FindAttributeIndex(DW_AT_name);
//                }
//            }
//            if (a_name_index != UINT32_MAX && b_name_index != UINT32_MAX)
//            {
//                if (a_attrs.ExtractFormValueAtIndex(dwarf2Data, a_name_index, a_form_value) &&
//                    b_attrs.ExtractFormValueAtIndex(dwarf2Data, b_name_index, b_form_value))
//                {
//                    result = DWARFFormValue::Compare (a_form_value, b_form_value, a_cu, b_cu, &dwarf2Data->get_debug_str_data());
//                    if (result == 0)
//                    {
//                        a_attr_count = b_attr_count = 0;
//                        compare_children = false;
//                    }
//                }
//            }
//        }
//
//        if (a_attr_count < b_attr_count)
//            return -1;
//        if (a_attr_count > b_attr_count)
//            return 1;
//
//
//        // The number of attributes are the same...
//        if (a_attr_count > 0)
//        {
//            const DataExtractor* debug_str_data_ptr = &dwarf2Data->get_debug_str_data();
//
//            uint32_t i;
//            for (i=0; i<a_attr_count; ++i)
//            {
//                const dw_attr_t a_attr = a_attrs.AttributeAtIndex(i);
//                const dw_attr_t b_attr = b_attrs.AttributeAtIndex(i);
//                //printf("    comparing attributes\n\t\t0x%8.8x: %s %s\t\t0x%8.8x: %s %s\n",
//                //                a_attrs.DIEOffsetAtIndex(i), DW_FORM_value_to_name(a_attrs.FormAtIndex(i)), DW_AT_value_to_name(a_attr),
//                //                b_attrs.DIEOffsetAtIndex(i), DW_FORM_value_to_name(b_attrs.FormAtIndex(i)), DW_AT_value_to_name(b_attr));
//
//                if (a_attr < b_attr)
//                    return -1;
//                else if (a_attr > b_attr)
//                    return 1;
//
//                switch (a_attr)
//                {
//                // Since we call a form of GetAttributes which inlines the
//                // attributes from DW_AT_abstract_origin and DW_AT_specification
//                // we don't care if their values mismatch...
//                case DW_AT_abstract_origin:
//                case DW_AT_specification:
//                case DW_AT_sibling:
//                case DW_AT_containing_type:
//                    //printf("        action = IGNORE\n");
//                    result = 0;
//                    break;  // ignore
//
//                default:
//                    if (a_attrs.ExtractFormValueAtIndex(dwarf2Data, i, a_form_value) &&
//                        b_attrs.ExtractFormValueAtIndex(dwarf2Data, i, b_form_value))
//                        result = DWARFFormValue::Compare (a_form_value, b_form_value, a_cu, b_cu, debug_str_data_ptr);
//                    break;
//                }
//
//                //printf("\t  result = %i\n", result);
//
//                if (result != 0)
//                {
//                    // Attributes weren't equal, lets see if we care?
//                    switch (a_attr)
//                    {
//                    case DW_AT_decl_file:
//                        // TODO: add the ability to compare files in two different compile units
//                        if (a_cu == b_cu)
//                        {
//                            //printf("        action = RETURN RESULT\n");
//                            return result;  // Only return the compare results when the compile units are the same and the decl_file attributes can be compared
//                        }
//                        else
//                        {
//                            result = 0;
//                            //printf("        action = IGNORE\n");
//                        }
//                        break;
//
//                    default:
//                        switch (a_attrs.FormAtIndex(i))
//                        {
//                        case DW_FORM_ref1:
//                        case DW_FORM_ref2:
//                        case DW_FORM_ref4:
//                        case DW_FORM_ref8:
//                        case DW_FORM_ref_udata:
//                        case DW_FORM_ref_addr:
//                            //printf("    action = COMPARE DIEs 0x%8.8x 0x%8.8x\n", (dw_offset_t)a_form_value.Reference(a_cu), (dw_offset_t)b_form_value.Reference(b_cu));
//                            // These attribute values refer to other DIEs, so lets compare those instead of their DIE offsets...
//                            result = Compare(dwarf2Data, a_form_value.Reference(a_cu), b_form_value.Reference(b_cu), compare_state, false, true);
//                            if (result != 0)
//                                return result;
//                            break;
//
//                        default:
//                            // We do care that they were different, return this result...
//                            //printf("        action = RETURN RESULT\n");
//                            return result;
//                        }
//                    }
//                }
//            }
//        }
//        //printf("    SUCCESS\n\t\t0x%8.8x: %s\n\t\t0x%8.8x: %s\n", a_die->GetOffset(), DW_TAG_value_to_name(a_tag), b_die->GetOffset(), DW_TAG_value_to_name(b_tag));
//
//        if (compare_children)
//        {
//            bool a_has_children = a_die->HasChildren();
//            bool b_has_children = b_die->HasChildren();
//            if (a_has_children == b_has_children)
//            {
//                // Both either have kids or don't
//                if (a_has_children)
//                    result = Compare(   dwarf2Data,
//                                        a_cu, a_die->GetFirstChild(),
//                                        b_cu, b_die->GetFirstChild(),
//                                        compare_state, true, compare_children);
//                else
//                    result = 0;
//            }
//            else if (!a_has_children)
//                result = -1;    // A doesn't have kids, but B does
//            else
//                result = 1; // A has kids, but B doesn't
//        }
//
//        if (compare_siblings)
//        {
//            result = Compare(   dwarf2Data,
//                                a_cu, a_die->GetSibling(),
//                                b_cu, b_die->GetSibling(),
//                                compare_state, true, compare_children);
//        }
//
//        return result;
//    }
//
//    if (a_die == NULL)
//        return -1;  // a_die is NULL, yet b_die is non-NULL
//    else
//        return 1;   // a_die is non-NULL, yet b_die is NULL
//
//}
//
//
//int
//DWARFDebugInfoEntry::Compare
//(
//  SymbolFileDWARF* dwarf2Data,
//  const DWARFCompileUnit* cu_a,
//  const DWARFDebugInfoEntry* die_a,
//  const DWARFCompileUnit* cu_a,
//  const DWARFDebugInfoEntry* die_b,
//  CompareState &compare_state
//)
//{
//}

//----------------------------------------------------------------------
// GetDIENamesAndRanges
//
// Gets the valid address ranges for a given DIE by looking for a
// DW_AT_low_pc/DW_AT_high_pc pair, DW_AT_entry_pc, or DW_AT_ranges
// attributes.
//----------------------------------------------------------------------
bool
DWARFDebugInfoEntry::GetDIENamesAndRanges
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const char * &name,
    const char * &mangled,
    DWARFDebugRanges::RangeList& ranges,
    int& decl_file,
    int& decl_line,
    int& decl_column,
    int& call_file,
    int& call_line,
    int& call_column,
    DWARFExpression *frame_base
) const
{
    if (dwarf2Data == NULL)
        return false;

    dw_addr_t lo_pc = LLDB_INVALID_ADDRESS;
    dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
    std::vector<dw_offset_t> die_offsets;
    bool set_frame_base_loclist_addr = false;
    
    lldb::offset_t offset;
    const DWARFAbbreviationDeclaration* abbrevDecl = GetAbbreviationDeclarationPtr(dwarf2Data, cu, offset);

    if (abbrevDecl)
    {
        const DataExtractor& debug_info_data = dwarf2Data->get_debug_info_data();

        if (!debug_info_data.ValidOffset(offset))
            return false;

        const uint32_t numAttributes = abbrevDecl->NumAttributes();
        uint32_t i;
        dw_attr_t attr;
        dw_form_t form;
        for (i=0; i<numAttributes; ++i)
        {
            abbrevDecl->GetAttrAndFormByIndexUnchecked(i, attr, form);
            DWARFFormValue form_value(form);
            if (form_value.ExtractValue(debug_info_data, &offset, cu))
            {
                switch (attr)
                {
                case DW_AT_low_pc:
                case DW_AT_entry_pc:
                    lo_pc = form_value.Unsigned();
                    break;

                case DW_AT_high_pc:
                    hi_pc = form_value.Unsigned();
                    break;

                case DW_AT_ranges:
                    {
                        const DWARFDebugRanges* debug_ranges = dwarf2Data->DebugRanges();
                        debug_ranges->FindRanges(form_value.Unsigned(), ranges);
                        // All DW_AT_ranges are relative to the base address of the
                        // compile unit. We add the compile unit base address to make
                        // sure all the addresses are properly fixed up.
                        ranges.Slide(cu->GetBaseAddress());
                    }
                    break;

                case DW_AT_name:
                    if (name == NULL)
                        name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
                    break;

                case DW_AT_MIPS_linkage_name:
                case DW_AT_linkage_name:
                    if (mangled == NULL)
                        mangled = form_value.AsCString(&dwarf2Data->get_debug_str_data());
                    break;

                case DW_AT_abstract_origin:
                    die_offsets.push_back(form_value.Reference(cu));
                    break;

                case DW_AT_specification:
                    die_offsets.push_back(form_value.Reference(cu));
                    break;

                case DW_AT_decl_file:
                    if (decl_file == 0)
                        decl_file = form_value.Unsigned();
                    break;

                case DW_AT_decl_line:
                    if (decl_line == 0)
                        decl_line = form_value.Unsigned();
                    break;

                case DW_AT_decl_column:
                    if (decl_column == 0)
                        decl_column = form_value.Unsigned();
                    break;

                case DW_AT_call_file:
                    if (call_file == 0)
                        call_file = form_value.Unsigned();
                    break;

                case DW_AT_call_line:
                    if (call_line == 0)
                        call_line = form_value.Unsigned();
                    break;

                case DW_AT_call_column:
                    if (call_column == 0)
                        call_column = form_value.Unsigned();
                    break;

                case DW_AT_frame_base:
                    if (frame_base)
                    {
                        if (form_value.BlockData())
                        {
                            uint32_t block_offset = form_value.BlockData() - debug_info_data.GetDataStart();
                            uint32_t block_length = form_value.Unsigned();
                            frame_base->SetOpcodeData(debug_info_data, block_offset, block_length);
                        }
                        else
                        {
                            const DataExtractor &debug_loc_data = dwarf2Data->get_debug_loc_data();
                            const dw_offset_t debug_loc_offset = form_value.Unsigned();

                            size_t loc_list_length = DWARFLocationList::Size(debug_loc_data, debug_loc_offset);
                            if (loc_list_length > 0)
                            {
                                frame_base->SetOpcodeData(debug_loc_data, debug_loc_offset, loc_list_length);
                                if (lo_pc != LLDB_INVALID_ADDRESS)
                                {
                                    assert (lo_pc >= cu->GetBaseAddress());
                                    frame_base->SetLocationListSlide(lo_pc - cu->GetBaseAddress());
                                }
                                else
                                {
                                    set_frame_base_loclist_addr = true;
                                }
                            }
                        }
                    }
                    break;

                default:
                    break;
                }
            }
        }
    }

    if (ranges.IsEmpty())
    {
        if (lo_pc != LLDB_INVALID_ADDRESS)
        {
            if (hi_pc != LLDB_INVALID_ADDRESS && hi_pc > lo_pc)
                ranges.Append(DWARFDebugRanges::Range (lo_pc, hi_pc - lo_pc));
            else
                ranges.Append(DWARFDebugRanges::Range (lo_pc, 0));
        }
    }
    
    if (set_frame_base_loclist_addr)
    {
        dw_addr_t lowest_range_pc = ranges.GetMinRangeBase(0);
        assert (lowest_range_pc >= cu->GetBaseAddress());
        frame_base->SetLocationListSlide (lowest_range_pc - cu->GetBaseAddress());
    }

    if (ranges.IsEmpty() || name == NULL || mangled == NULL)
    {
        std::vector<dw_offset_t>::const_iterator pos;
        std::vector<dw_offset_t>::const_iterator end = die_offsets.end();
        for (pos = die_offsets.begin(); pos != end; ++pos)
        {
            DWARFCompileUnitSP cu_sp_ptr;
            const DWARFDebugInfoEntry* die = NULL;
            dw_offset_t die_offset = *pos;
            if (die_offset != DW_INVALID_OFFSET)
            {
                die = dwarf2Data->DebugInfo()->GetDIEPtr(die_offset, &cu_sp_ptr);
                if (die)
                    die->GetDIENamesAndRanges(dwarf2Data, cu_sp_ptr.get(), name, mangled, ranges, decl_file, decl_line, decl_column, call_file, call_line, call_column);
            }
        }
    }
    return !ranges.IsEmpty();
}

//----------------------------------------------------------------------
// Dump
//
// Dumps a debug information entry and all of it's attributes to the
// specified stream.
//----------------------------------------------------------------------
void
DWARFDebugInfoEntry::Dump
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    Stream &s,
    uint32_t recurse_depth
) const
{
    const DataExtractor& debug_info_data = dwarf2Data->get_debug_info_data();
    lldb::offset_t offset = m_offset;

    if (debug_info_data.ValidOffset(offset))
    {
        dw_uleb128_t abbrCode = debug_info_data.GetULEB128(&offset);

        s.Printf("\n0x%8.8x: ", m_offset);
        s.Indent();
        if (abbrCode != m_abbr_idx)
        {
            s.Printf( "error: DWARF has been modified\n");
        }
        else if (abbrCode)
        {
            const DWARFAbbreviationDeclaration* abbrevDecl = cu->GetAbbreviations()->GetAbbreviationDeclaration (abbrCode);

            if (abbrevDecl)
            {
                s.PutCString(DW_TAG_value_to_name(abbrevDecl->Tag()));
                s.Printf( " [%u] %c\n", abbrCode, abbrevDecl->HasChildren() ? '*':' ');

                // Dump all data in the .debug_info for the attributes
                const uint32_t numAttributes = abbrevDecl->NumAttributes();
                uint32_t i;
                dw_attr_t attr;
                dw_form_t form;
                for (i=0; i<numAttributes; ++i)
                {
                    abbrevDecl->GetAttrAndFormByIndexUnchecked(i, attr, form);

                    DumpAttribute(dwarf2Data, cu, debug_info_data, &offset, s, attr, form);
                }

                const DWARFDebugInfoEntry* child = GetFirstChild();
                if (recurse_depth > 0 && child)
                {
                    s.IndentMore();

                    while (child)
                    {
                        child->Dump(dwarf2Data, cu, s, recurse_depth-1);
                        child = child->GetSibling();
                    }
                    s.IndentLess();
                }
            }
            else
                s.Printf( "Abbreviation code note found in 'debug_abbrev' class for code: %u\n", abbrCode);
        }
        else
        {
            s.Printf( "NULL\n");
        }
    }
}

void
DWARFDebugInfoEntry::DumpLocation
(
    SymbolFileDWARF* dwarf2Data,
    DWARFCompileUnit* cu,
    Stream &s
) const
{
    const DWARFDebugInfoEntry *cu_die = cu->GetCompileUnitDIEOnly();
    const char *cu_name = NULL;
    if (cu_die != NULL)
        cu_name = cu_die->GetName (dwarf2Data, cu);
    const char *obj_file_name = NULL;
    ObjectFile *obj_file = dwarf2Data->GetObjectFile();
    if (obj_file)
        obj_file_name = obj_file->GetFileSpec().GetFilename().AsCString();
    const char *die_name = GetName (dwarf2Data, cu);
    s.Printf ("0x%8.8x/0x%8.8x: %-30s (from %s in %s)", 
              cu->GetOffset(),
              GetOffset(),
              die_name ? die_name : "", 
              cu_name ? cu_name : "<NULL>",
              obj_file_name ? obj_file_name : "<NULL>");
}

//----------------------------------------------------------------------
// DumpAttribute
//
// Dumps a debug information entry attribute along with it's form. Any
// special display of attributes is done (disassemble location lists,
// show enumeration values for attributes, etc).
//----------------------------------------------------------------------
void
DWARFDebugInfoEntry::DumpAttribute
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const DataExtractor& debug_info_data,
    lldb::offset_t *offset_ptr,
    Stream &s,
    dw_attr_t attr,
    dw_form_t form
)
{
    bool verbose    = s.GetVerbose();
    bool show_form  = s.GetFlags().Test(DWARFDebugInfo::eDumpFlag_ShowForm);
    
    const DataExtractor* debug_str_data = dwarf2Data ? &dwarf2Data->get_debug_str_data() : NULL;
    if (verbose)
        s.Offset (*offset_ptr);
    else
        s.Printf ("            ");
    s.Indent(DW_AT_value_to_name(attr));

    if (show_form)
    {
        s.Printf( "[%s", DW_FORM_value_to_name(form));
    }

    DWARFFormValue form_value(form);

    if (!form_value.ExtractValue(debug_info_data, offset_ptr, cu))
        return;

    if (show_form)
    {
        if (form == DW_FORM_indirect)
        {
            s.Printf( " [%s]", DW_FORM_value_to_name(form_value.Form()));
        }

        s.PutCString("] ");
    }

    s.PutCString("( ");

    // Always dump form value if verbose is enabled
    if (verbose)
    {
        form_value.Dump(s, debug_str_data, cu);
    }


    // Check to see if we have any special attribute formatters
    switch (attr)
    {
    case DW_AT_stmt_list:
        if ( verbose ) s.PutCString(" ( ");
        s.Printf( "0x%8.8" PRIx64, form_value.Unsigned());
        if ( verbose ) s.PutCString(" )");
        break;

    case DW_AT_language:
        if ( verbose ) s.PutCString(" ( ");
        s.PutCString(DW_LANG_value_to_name(form_value.Unsigned()));
        if ( verbose ) s.PutCString(" )");
        break;

    case DW_AT_encoding:
        if ( verbose ) s.PutCString(" ( ");
        s.PutCString(DW_ATE_value_to_name(form_value.Unsigned()));
        if ( verbose ) s.PutCString(" )");
        break;

    case DW_AT_frame_base:
    case DW_AT_location:
    case DW_AT_data_member_location:
        {
            const uint8_t* blockData = form_value.BlockData();
            if (blockData)
            {
                if (!verbose)
                    form_value.Dump(s, debug_str_data, cu);

                // Location description is inlined in data in the form value
                DataExtractor locationData(debug_info_data, (*offset_ptr) - form_value.Unsigned(), form_value.Unsigned());
                if ( verbose ) s.PutCString(" ( ");
                print_dwarf_expression (s, locationData, DWARFCompileUnit::GetAddressByteSize(cu), 4, false);
                if ( verbose ) s.PutCString(" )");
            }
            else
            {
                // We have a location list offset as the value that is
                // the offset into the .debug_loc section that describes
                // the value over it's lifetime
                uint64_t debug_loc_offset = form_value.Unsigned();
                if (dwarf2Data)
                {
                    if ( !verbose )
                        form_value.Dump(s, debug_str_data, cu);
                    DWARFLocationList::Dump(s, cu, dwarf2Data->get_debug_loc_data(), debug_loc_offset);
                }
                else
                {
                    if ( !verbose )
                        form_value.Dump(s, NULL, cu);
                }
            }
        }
        break;

    case DW_AT_abstract_origin:
    case DW_AT_specification:
        {
            uint64_t abstract_die_offset = form_value.Reference(cu);
            form_value.Dump(s, debug_str_data, cu);
        //  *ostrm_ptr << HEX32 << abstract_die_offset << " ( ";
            if ( verbose ) s.PutCString(" ( ");
            GetName(dwarf2Data, cu, abstract_die_offset, s);
            if ( verbose ) s.PutCString(" )");
        }
        break;

    case DW_AT_type:
        {
            uint64_t type_die_offset = form_value.Reference(cu);
            if (!verbose)
                form_value.Dump(s, debug_str_data, cu);
            s.PutCString(" ( ");
            AppendTypeName(dwarf2Data, cu, type_die_offset, s);
            s.PutCString(" )");
        }
        break;

    case DW_AT_ranges:
        {
            if ( !verbose )
                form_value.Dump(s, debug_str_data, cu);
            lldb::offset_t ranges_offset = form_value.Unsigned();
            dw_addr_t base_addr = cu ? cu->GetBaseAddress() : 0;
            if (dwarf2Data)
                DWARFDebugRanges::Dump(s, dwarf2Data->get_debug_ranges_data(), &ranges_offset, base_addr);
        }
        break;

    default:
        if ( !verbose )
            form_value.Dump(s, debug_str_data, cu);
        break;
    }

    s.PutCString(" )\n");
}

//----------------------------------------------------------------------
// Get all attribute values for a given DIE, including following any
// specification or abstract origin attributes and including those in
// the results. Any duplicate attributes will have the first instance
// take precedence (this can happen for declaration attributes).
//----------------------------------------------------------------------
size_t
DWARFDebugInfoEntry::GetAttributes
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const uint8_t *fixed_form_sizes,
    DWARFDebugInfoEntry::Attributes& attributes,
    uint32_t curr_depth
) const
{
    lldb::offset_t offset;
    const DWARFAbbreviationDeclaration* abbrevDecl = GetAbbreviationDeclarationPtr(dwarf2Data, cu, offset);

    if (abbrevDecl)
    {
        const DataExtractor& debug_info_data = dwarf2Data->get_debug_info_data();

        if (fixed_form_sizes == NULL)
            fixed_form_sizes = DWARFFormValue::GetFixedFormSizesForAddressSize(cu->GetAddressByteSize());

        const uint32_t num_attributes = abbrevDecl->NumAttributes();
        uint32_t i;
        dw_attr_t attr;
        dw_form_t form;
        DWARFFormValue form_value;
        for (i=0; i<num_attributes; ++i)
        {
            abbrevDecl->GetAttrAndFormByIndexUnchecked (i, attr, form);
            
            // If we are tracking down DW_AT_specification or DW_AT_abstract_origin
            // attributes, the depth will be non-zero. We need to omit certain
            // attributes that don't make sense.
            switch (attr)
            {
            case DW_AT_sibling:
            case DW_AT_declaration:
                if (curr_depth > 0)
                {
                    // This attribute doesn't make sense when combined with
                    // the DIE that references this DIE. We know a DIE is 
                    // referencing this DIE because curr_depth is not zero
                    break;  
                }
                // Fall through...
            default:
                attributes.Append(cu, offset, attr, form);
                break;
            }

            if ((attr == DW_AT_specification) || (attr == DW_AT_abstract_origin))
            {
                form_value.SetForm(form);
                if (form_value.ExtractValue(debug_info_data, &offset, cu))
                {
                    const DWARFDebugInfoEntry* die = NULL;
                    dw_offset_t die_offset = form_value.Reference(cu);
                    if (cu->ContainsDIEOffset(die_offset))
                    {
                        die = const_cast<DWARFCompileUnit*>(cu)->GetDIEPtr(die_offset);
                        if (die)
                            die->GetAttributes(dwarf2Data, cu, fixed_form_sizes, attributes, curr_depth + 1);
                    }
                    else
                    {
                        DWARFCompileUnitSP cu_sp_ptr;
                        die = const_cast<SymbolFileDWARF*>(dwarf2Data)->DebugInfo()->GetDIEPtr(die_offset, &cu_sp_ptr);
                        if (die)
                            die->GetAttributes(dwarf2Data, cu_sp_ptr.get(), fixed_form_sizes, attributes, curr_depth + 1);
                    }
                }
            }
            else
            {
                const uint8_t fixed_skip_size = fixed_form_sizes [form];
                if (fixed_skip_size)
                    offset += fixed_skip_size;
                else
                    DWARFFormValue::SkipValue(form, debug_info_data, &offset, cu);
            }
        }
    }
    else
    {
        attributes.Clear();
    }
    return attributes.Size();

}

//----------------------------------------------------------------------
// GetAttributeValue
//
// Get the value of an attribute and return the .debug_info offset of the
// attribute if it was properly extracted into form_value, or zero
// if we fail since an offset of zero is invalid for an attribute (it
// would be a compile unit header).
//----------------------------------------------------------------------
dw_offset_t
DWARFDebugInfoEntry::GetAttributeValue
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_attr_t attr,
    DWARFFormValue& form_value,
    dw_offset_t* end_attr_offset_ptr
) const
{
    lldb::offset_t offset;
    const DWARFAbbreviationDeclaration* abbrevDecl = GetAbbreviationDeclarationPtr(dwarf2Data, cu, offset);

    if (abbrevDecl)
    {
        uint32_t attr_idx = abbrevDecl->FindAttributeIndex(attr);

        if (attr_idx != DW_INVALID_INDEX)
        {
            const DataExtractor& debug_info_data = dwarf2Data->get_debug_info_data();

            uint32_t idx=0;
            while (idx<attr_idx)
                DWARFFormValue::SkipValue(abbrevDecl->GetFormByIndex(idx++), debug_info_data, &offset, cu);

            const dw_offset_t attr_offset = offset;
            form_value.SetForm(abbrevDecl->GetFormByIndex(idx));
            if (form_value.ExtractValue(debug_info_data, &offset, cu))
            {
                if (end_attr_offset_ptr)
                    *end_attr_offset_ptr = offset;
                return attr_offset;
            }
        }
    }

    return 0;
}

//----------------------------------------------------------------------
// GetAttributeValueAsString
//
// Get the value of an attribute as a string return it. The resulting
// pointer to the string data exists within the supplied SymbolFileDWARF
// and will only be available as long as the SymbolFileDWARF is still around
// and it's content doesn't change.
//----------------------------------------------------------------------
const char*
DWARFDebugInfoEntry::GetAttributeValueAsString
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_attr_t attr,
    const char* fail_value) const
{
    DWARFFormValue form_value;
    if (GetAttributeValue(dwarf2Data, cu, attr, form_value))
        return form_value.AsCString(&dwarf2Data->get_debug_str_data());
    return fail_value;
}

//----------------------------------------------------------------------
// GetAttributeValueAsUnsigned
//
// Get the value of an attribute as unsigned and return it.
//----------------------------------------------------------------------
uint64_t
DWARFDebugInfoEntry::GetAttributeValueAsUnsigned
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_attr_t attr,
    uint64_t fail_value
) const
{
    DWARFFormValue form_value;
    if (GetAttributeValue(dwarf2Data, cu, attr, form_value))
        return form_value.Unsigned();
    return fail_value;
}

//----------------------------------------------------------------------
// GetAttributeValueAsSigned
//
// Get the value of an attribute a signed value and return it.
//----------------------------------------------------------------------
int64_t
DWARFDebugInfoEntry::GetAttributeValueAsSigned
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_attr_t attr,
    int64_t fail_value
) const
{
    DWARFFormValue form_value;
    if (GetAttributeValue(dwarf2Data, cu, attr, form_value))
        return form_value.Signed();
    return fail_value;
}

//----------------------------------------------------------------------
// GetAttributeValueAsReference
//
// Get the value of an attribute as reference and fix up and compile
// unit relative offsets as needed.
//----------------------------------------------------------------------
uint64_t
DWARFDebugInfoEntry::GetAttributeValueAsReference
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_attr_t attr,
    uint64_t fail_value
) const
{
    DWARFFormValue form_value;
    if (GetAttributeValue(dwarf2Data, cu, attr, form_value))
        return form_value.Reference(cu);
    return fail_value;
}

//----------------------------------------------------------------------
// GetAttributeValueAsLocation
//
// Get the value of an attribute as reference and fix up and compile
// unit relative offsets as needed.
//----------------------------------------------------------------------
dw_offset_t
DWARFDebugInfoEntry::GetAttributeValueAsLocation
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_attr_t attr,
    DataExtractor& location_data,
    uint32_t &block_size
) const
{
    block_size = 0;
    DWARFFormValue form_value;

    // Empty out data in case we don't find anything
    location_data.Clear();
    dw_offset_t end_addr_offset = DW_INVALID_OFFSET;
    const dw_offset_t attr_offset = GetAttributeValue(dwarf2Data, cu, attr, form_value, &end_addr_offset);
    if (attr_offset)
    {
        const uint8_t* blockData = form_value.BlockData();
        if (blockData)
        {
            // We have an inlined location list in the .debug_info section
            const DataExtractor& debug_info = dwarf2Data->get_debug_info_data();
            dw_offset_t block_offset = blockData - debug_info.GetDataStart();
            block_size = (end_addr_offset - attr_offset) - form_value.Unsigned();
            location_data.SetData(debug_info, block_offset, block_size);
        }
        else
        {
            // We have a location list offset as the value that is
            // the offset into the .debug_loc section that describes
            // the value over it's lifetime
            lldb::offset_t debug_loc_offset = form_value.Unsigned();
            if (dwarf2Data)
            {
                assert(dwarf2Data->get_debug_loc_data().GetAddressByteSize() == cu->GetAddressByteSize());
                return DWARFLocationList::Extract(dwarf2Data->get_debug_loc_data(), &debug_loc_offset, location_data);
            }
        }
    }
    return attr_offset;
}

//----------------------------------------------------------------------
// GetName
//
// Get value of the DW_AT_name attribute and return it if one exists,
// else return NULL.
//----------------------------------------------------------------------
const char*
DWARFDebugInfoEntry::GetName
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu
) const
{
    DWARFFormValue form_value;
    if (GetAttributeValue(dwarf2Data, cu, DW_AT_name, form_value))
        return form_value.AsCString(&dwarf2Data->get_debug_str_data());
    return NULL;
}


//----------------------------------------------------------------------
// GetMangledName
//
// Get value of the DW_AT_MIPS_linkage_name attribute and return it if
// one exists, else return the value of the DW_AT_name attribute
//----------------------------------------------------------------------
const char*
DWARFDebugInfoEntry::GetMangledName
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    bool substitute_name_allowed
) const
{
    const char* name = NULL;
    DWARFFormValue form_value;

    if (GetAttributeValue(dwarf2Data, cu, DW_AT_MIPS_linkage_name, form_value))
        name = form_value.AsCString(&dwarf2Data->get_debug_str_data());

    if (GetAttributeValue(dwarf2Data, cu, DW_AT_linkage_name, form_value))
        name = form_value.AsCString(&dwarf2Data->get_debug_str_data());

    if (substitute_name_allowed && name == NULL)
    {
        if (GetAttributeValue(dwarf2Data, cu, DW_AT_name, form_value))
            name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
    }
    return name;
}


//----------------------------------------------------------------------
// GetPubname
//
// Get value the name for a DIE as it should appear for a
// .debug_pubnames or .debug_pubtypes section.
//----------------------------------------------------------------------
const char*
DWARFDebugInfoEntry::GetPubname
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu
) const
{
    const char* name = NULL;
    if (!dwarf2Data)
        return name;
    
    DWARFFormValue form_value;

    if (GetAttributeValue(dwarf2Data, cu, DW_AT_MIPS_linkage_name, form_value))
        name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
    else if (GetAttributeValue(dwarf2Data, cu, DW_AT_linkage_name, form_value))
        name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
    else if (GetAttributeValue(dwarf2Data, cu, DW_AT_name, form_value))
        name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
    else if (GetAttributeValue(dwarf2Data, cu, DW_AT_specification, form_value))
    {
        // The specification DIE may be in another compile unit so we need
        // to get a die and its compile unit.
        DWARFCompileUnitSP cu_sp_ptr;
        const DWARFDebugInfoEntry* die = const_cast<SymbolFileDWARF*>(dwarf2Data)->DebugInfo()->GetDIEPtr(form_value.Reference(cu), &cu_sp_ptr);
        if (die)
            return die->GetPubname(dwarf2Data, cu_sp_ptr.get());
    }
    return name;
}


//----------------------------------------------------------------------
// GetName
//
// Get value of the DW_AT_name attribute for a debug information entry
// that exists at offset "die_offset" and place that value into the
// supplied stream object. If the DIE is a NULL object "NULL" is placed
// into the stream, and if no DW_AT_name attribute exists for the DIE
// then nothing is printed.
//----------------------------------------------------------------------
bool
DWARFDebugInfoEntry::GetName
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_offset_t die_offset,
    Stream &s
)
{
    if (dwarf2Data == NULL)
    {
        s.PutCString("NULL");
        return false;
    }
    
    DWARFDebugInfoEntry die;
    lldb::offset_t offset = die_offset;
    if (die.Extract(dwarf2Data, cu, &offset))
    {
        if (die.IsNULL())
        {
            s.PutCString("NULL");
            return true;
        }
        else
        {
            DWARFFormValue form_value;
            if (die.GetAttributeValue(dwarf2Data, cu, DW_AT_name, form_value))
            {
                const char* name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
                if (name)
                {
                    s.PutCString(name);
                    return true;
                }
            }
        }
    }
    return false;
}

//----------------------------------------------------------------------
// AppendTypeName
//
// Follows the type name definition down through all needed tags to
// end up with a fully qualified type name and dump the results to
// the supplied stream. This is used to show the name of types given
// a type identifier.
//----------------------------------------------------------------------
bool
DWARFDebugInfoEntry::AppendTypeName
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    const dw_offset_t die_offset,
    Stream &s
)
{
    if (dwarf2Data == NULL)
    {
        s.PutCString("NULL");
        return false;
    }
    
    DWARFDebugInfoEntry die;
    lldb::offset_t offset = die_offset;
    if (die.Extract(dwarf2Data, cu, &offset))
    {
        if (die.IsNULL())
        {
            s.PutCString("NULL");
            return true;
        }
        else
        {
            const char* name = die.GetPubname(dwarf2Data, cu);
        //  if (die.GetAttributeValue(dwarf2Data, cu, DW_AT_name, form_value))
        //      name = form_value.AsCString(&dwarf2Data->get_debug_str_data());
            if (name)
                s.PutCString(name);
            else
            {
                bool result = true;
                const DWARFAbbreviationDeclaration* abbrevDecl = die.GetAbbreviationDeclarationPtr(dwarf2Data, cu, offset);
                
                if (abbrevDecl == NULL)
                    return false;

                switch (abbrevDecl->Tag())
                {
                case DW_TAG_array_type:         break;  // print out a "[]" after printing the full type of the element below
                case DW_TAG_base_type:          s.PutCString("base ");         break;
                case DW_TAG_class_type:         s.PutCString("class ");            break;
                case DW_TAG_const_type:         s.PutCString("const ");            break;
                case DW_TAG_enumeration_type:   s.PutCString("enum ");         break;
                case DW_TAG_file_type:          s.PutCString("file ");         break;
                case DW_TAG_interface_type:     s.PutCString("interface ");        break;
                case DW_TAG_packed_type:        s.PutCString("packed ");       break;
                case DW_TAG_pointer_type:       break;  // print out a '*' after printing the full type below
                case DW_TAG_ptr_to_member_type: break;  // print out a '*' after printing the full type below
                case DW_TAG_reference_type:     break;  // print out a '&' after printing the full type below
                case DW_TAG_restrict_type:      s.PutCString("restrict ");     break;
                case DW_TAG_set_type:           s.PutCString("set ");          break;
                case DW_TAG_shared_type:        s.PutCString("shared ");       break;
                case DW_TAG_string_type:        s.PutCString("string ");       break;
                case DW_TAG_structure_type:     s.PutCString("struct ");       break;
                case DW_TAG_subrange_type:      s.PutCString("subrange ");     break;
                case DW_TAG_subroutine_type:    s.PutCString("function ");     break;
                case DW_TAG_thrown_type:        s.PutCString("thrown ");       break;
                case DW_TAG_union_type:         s.PutCString("union ");            break;
                case DW_TAG_unspecified_type:   s.PutCString("unspecified ");  break;
                case DW_TAG_volatile_type:      s.PutCString("volatile ");     break;
                default:
                    return false;
                }

                // Follow the DW_AT_type if possible
                DWARFFormValue form_value;
                if (die.GetAttributeValue(dwarf2Data, cu, DW_AT_type, form_value))
                {
                    uint64_t next_die_offset = form_value.Reference(cu);
                    result = AppendTypeName(dwarf2Data, cu, next_die_offset, s);
                }

                switch (abbrevDecl->Tag())
                {
                case DW_TAG_array_type:         s.PutCString("[]");    break;
                case DW_TAG_pointer_type:       s.PutChar('*');    break;
                case DW_TAG_ptr_to_member_type: s.PutChar('*');    break;
                case DW_TAG_reference_type:     s.PutChar('&');    break;
                default:
                    break;
                }
                return result;
            }
        }
    }
    return false;
}

bool
DWARFDebugInfoEntry::Contains (const DWARFDebugInfoEntry *die) const
{
    if (die)
    {
        const dw_offset_t die_offset = die->GetOffset();
        if (die_offset > GetOffset())
        {
            const DWARFDebugInfoEntry *sibling = GetSibling();
            assert (sibling); // TODO: take this out
            if (sibling)
                return die_offset < sibling->GetOffset();
        }
    }
    return false;
}

//----------------------------------------------------------------------
// BuildAddressRangeTable
//----------------------------------------------------------------------
void
DWARFDebugInfoEntry::BuildAddressRangeTable
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    DWARFDebugAranges* debug_aranges
) const
{
    if (m_tag)
    {
        if (m_tag == DW_TAG_subprogram)
        {
            dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
            dw_addr_t lo_pc = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_low_pc, LLDB_INVALID_ADDRESS);
            if (lo_pc != LLDB_INVALID_ADDRESS)
                hi_pc = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_high_pc, LLDB_INVALID_ADDRESS);
            if (hi_pc != LLDB_INVALID_ADDRESS)
            {
            /// printf("BuildAddressRangeTable() 0x%8.8x: %30s: [0x%8.8x - 0x%8.8x)\n", m_offset, DW_TAG_value_to_name(tag), lo_pc, hi_pc);
                debug_aranges->AppendRange (cu->GetOffset(), lo_pc, hi_pc);
            }
        }


        const DWARFDebugInfoEntry* child = GetFirstChild();
        while (child)
        {
            child->BuildAddressRangeTable(dwarf2Data, cu, debug_aranges);
            child = child->GetSibling();
        }
    }
}

//----------------------------------------------------------------------
// BuildFunctionAddressRangeTable
//
// This function is very similar to the BuildAddressRangeTable function
// except that the actual DIE offset for the function is placed in the
// table instead of the compile unit offset (which is the way the
// standard .debug_aranges section does it).
//----------------------------------------------------------------------
void
DWARFDebugInfoEntry::BuildFunctionAddressRangeTable
(
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    DWARFDebugAranges* debug_aranges
) const
{
    if (m_tag)
    {
        if (m_tag == DW_TAG_subprogram)
        {
            dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
            dw_addr_t lo_pc = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_low_pc, LLDB_INVALID_ADDRESS);
            if (lo_pc != LLDB_INVALID_ADDRESS)
                hi_pc = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_high_pc, LLDB_INVALID_ADDRESS);
            if (hi_pc != LLDB_INVALID_ADDRESS)
            {
            //  printf("BuildAddressRangeTable() 0x%8.8x: [0x%16.16" PRIx64 " - 0x%16.16" PRIx64 ")\n", m_offset, lo_pc, hi_pc); // DEBUG ONLY
                debug_aranges->AppendRange (GetOffset(), lo_pc, hi_pc);
            }
        }

        const DWARFDebugInfoEntry* child = GetFirstChild();
        while (child)
        {
            child->BuildFunctionAddressRangeTable(dwarf2Data, cu, debug_aranges);
            child = child->GetSibling();
        }
    }
}

void
DWARFDebugInfoEntry::GetDeclContextDIEs (SymbolFileDWARF* dwarf2Data, 
                                         DWARFCompileUnit* cu,
                                         DWARFDIECollection &decl_context_dies) const
{
    const DWARFDebugInfoEntry *parent_decl_ctx_die = GetParentDeclContextDIE (dwarf2Data, cu);
    if (parent_decl_ctx_die && parent_decl_ctx_die != this)
    {
        decl_context_dies.Append(parent_decl_ctx_die);
        parent_decl_ctx_die->GetDeclContextDIEs (dwarf2Data, cu, decl_context_dies);
    }
}

void
DWARFDebugInfoEntry::GetDWARFDeclContext (SymbolFileDWARF* dwarf2Data,
                                          DWARFCompileUnit* cu,
                                          DWARFDeclContext &dwarf_decl_ctx) const
{
    const dw_tag_t tag = Tag();
    if (tag != DW_TAG_compile_unit)
    {
        dwarf_decl_ctx.AppendDeclContext(tag, GetName(dwarf2Data, cu));
        const DWARFDebugInfoEntry *parent_decl_ctx_die = GetParentDeclContextDIE (dwarf2Data, cu);
        if (parent_decl_ctx_die && parent_decl_ctx_die != this)
        {
            if (parent_decl_ctx_die->Tag() != DW_TAG_compile_unit)
                parent_decl_ctx_die->GetDWARFDeclContext (dwarf2Data, cu, dwarf_decl_ctx);
        }
    }
}


bool
DWARFDebugInfoEntry::MatchesDWARFDeclContext (SymbolFileDWARF* dwarf2Data,
                                              DWARFCompileUnit* cu,
                                              const DWARFDeclContext &dwarf_decl_ctx) const
{
    
    DWARFDeclContext this_dwarf_decl_ctx;
    GetDWARFDeclContext (dwarf2Data, cu, this_dwarf_decl_ctx);
    return this_dwarf_decl_ctx == dwarf_decl_ctx;
}

const DWARFDebugInfoEntry *
DWARFDebugInfoEntry::GetParentDeclContextDIE (SymbolFileDWARF* dwarf2Data, 
											  DWARFCompileUnit* cu) const
{
	DWARFDebugInfoEntry::Attributes attributes;
	GetAttributes(dwarf2Data, cu, NULL, attributes);
	return GetParentDeclContextDIE (dwarf2Data, cu, attributes);
}

const DWARFDebugInfoEntry *
DWARFDebugInfoEntry::GetParentDeclContextDIE (SymbolFileDWARF* dwarf2Data, 
											  DWARFCompileUnit* cu,
											  const DWARFDebugInfoEntry::Attributes& attributes) const
{
	const DWARFDebugInfoEntry * die = this;
	
	while (die != NULL)
	{
		// If this is the original DIE that we are searching for a declaration 
		// for, then don't look in the cache as we don't want our own decl 
		// context to be our decl context...
		if (die != this)
		{            
			switch (die->Tag())
			{
				case DW_TAG_compile_unit:
				case DW_TAG_namespace:
				case DW_TAG_structure_type:
				case DW_TAG_union_type:
				case DW_TAG_class_type:
					return die;
					
				default:
					break;
			}
		}
		
		dw_offset_t die_offset;
        
		die_offset = attributes.FormValueAsUnsigned(dwarf2Data, DW_AT_specification, DW_INVALID_OFFSET);
		if (die_offset != DW_INVALID_OFFSET)
		{
			const DWARFDebugInfoEntry *spec_die = cu->GetDIEPtr (die_offset);
			if (spec_die)
			{
				const DWARFDebugInfoEntry *spec_die_decl_ctx_die = spec_die->GetParentDeclContextDIE (dwarf2Data, cu);
				if (spec_die_decl_ctx_die)
					return spec_die_decl_ctx_die;
			}
		}
		
        die_offset = attributes.FormValueAsUnsigned(dwarf2Data, DW_AT_abstract_origin, DW_INVALID_OFFSET);
		if (die_offset != DW_INVALID_OFFSET)
		{
			const DWARFDebugInfoEntry *abs_die = cu->GetDIEPtr (die_offset);
			if (abs_die)
			{
				const DWARFDebugInfoEntry *abs_die_decl_ctx_die = abs_die->GetParentDeclContextDIE (dwarf2Data, cu);
				if (abs_die_decl_ctx_die)
					return abs_die_decl_ctx_die;
			}
		}
		
		die = die->GetParent();
	}
    return NULL;
}


const char *
DWARFDebugInfoEntry::GetQualifiedName (SymbolFileDWARF* dwarf2Data, 
									   DWARFCompileUnit* cu,
									   std::string &storage) const
{
	DWARFDebugInfoEntry::Attributes attributes;
	GetAttributes(dwarf2Data, cu, NULL, attributes);
	return GetQualifiedName (dwarf2Data, cu, attributes, storage);
}

const char*
DWARFDebugInfoEntry::GetQualifiedName (SymbolFileDWARF* dwarf2Data, 
									   DWARFCompileUnit* cu,
									   const DWARFDebugInfoEntry::Attributes& attributes,
									   std::string &storage) const
{
	
	const char *name = GetName (dwarf2Data, cu);
	
	if (name)
	{
		const DWARFDebugInfoEntry *parent_decl_ctx_die = GetParentDeclContextDIE (dwarf2Data, cu);
		storage.clear();
		// TODO: change this to get the correct decl context parent....
		while (parent_decl_ctx_die)
		{
			const dw_tag_t parent_tag = parent_decl_ctx_die->Tag();
			switch (parent_tag)
			{
                case DW_TAG_namespace:
				{
					const char *namespace_name = parent_decl_ctx_die->GetName (dwarf2Data, cu);
					if (namespace_name)
					{
						storage.insert (0, "::");
						storage.insert (0, namespace_name);
					}
					else
					{
						storage.insert (0, "(anonymous namespace)::");
					}
					parent_decl_ctx_die = parent_decl_ctx_die->GetParentDeclContextDIE(dwarf2Data, cu);
				}
                    break;
					
                case DW_TAG_class_type:
                case DW_TAG_structure_type:
                case DW_TAG_union_type:
				{
					const char *class_union_struct_name = parent_decl_ctx_die->GetName (dwarf2Data, cu);
                    
					if (class_union_struct_name)
					{
						storage.insert (0, "::");
						storage.insert (0, class_union_struct_name);
					}
					parent_decl_ctx_die = parent_decl_ctx_die->GetParentDeclContextDIE(dwarf2Data, cu);
				}
                    break;
                    
                default:
                    parent_decl_ctx_die = NULL;
                    break;
			}
		}
		
		if (storage.empty())
			storage.append ("::");
        
		storage.append (name);
	}
	if (storage.empty())
		return NULL;
	return storage.c_str();
}


//----------------------------------------------------------------------
// LookupAddress
//----------------------------------------------------------------------
bool
DWARFDebugInfoEntry::LookupAddress
(
    const dw_addr_t address,
    SymbolFileDWARF* dwarf2Data,
    const DWARFCompileUnit* cu,
    DWARFDebugInfoEntry** function_die,
    DWARFDebugInfoEntry** block_die
)
{
    bool found_address = false;
    if (m_tag)
    {
        bool check_children = false;
        bool match_addr_range = false;
    //  printf("0x%8.8x: %30s: address = 0x%8.8x - ", m_offset, DW_TAG_value_to_name(tag), address);
        switch (m_tag)
        {
        case DW_TAG_array_type                 : break;
        case DW_TAG_class_type                 : check_children = true; break;
        case DW_TAG_entry_point                : break;
        case DW_TAG_enumeration_type           : break;
        case DW_TAG_formal_parameter           : break;
        case DW_TAG_imported_declaration       : break;
        case DW_TAG_label                      : break;
        case DW_TAG_lexical_block              : check_children = true; match_addr_range = true; break;
        case DW_TAG_member                     : break;
        case DW_TAG_pointer_type               : break;
        case DW_TAG_reference_type             : break;
        case DW_TAG_compile_unit               : match_addr_range = true; break;
        case DW_TAG_string_type                : break;
        case DW_TAG_structure_type             : check_children = true; break;
        case DW_TAG_subroutine_type            : break;
        case DW_TAG_typedef                    : break;
        case DW_TAG_union_type                 : break;
        case DW_TAG_unspecified_parameters     : break;
        case DW_TAG_variant                    : break;
        case DW_TAG_common_block               : check_children = true; break;
        case DW_TAG_common_inclusion           : break;
        case DW_TAG_inheritance                : break;
        case DW_TAG_inlined_subroutine         : check_children = true; match_addr_range = true; break;
        case DW_TAG_module                     : match_addr_range = true; break;
        case DW_TAG_ptr_to_member_type         : break;
        case DW_TAG_set_type                   : break;
        case DW_TAG_subrange_type              : break;
        case DW_TAG_with_stmt                  : break;
        case DW_TAG_access_declaration         : break;
        case DW_TAG_base_type                  : break;
        case DW_TAG_catch_block                : match_addr_range = true; break;
        case DW_TAG_const_type                 : break;
        case DW_TAG_constant                   : break;
        case DW_TAG_enumerator                 : break;
        case DW_TAG_file_type                  : break;
        case DW_TAG_friend                     : break;
        case DW_TAG_namelist                   : break;
        case DW_TAG_namelist_item              : break;
        case DW_TAG_packed_type                : break;
        case DW_TAG_subprogram                 : match_addr_range = true; break;
        case DW_TAG_template_type_parameter    : break;
        case DW_TAG_template_value_parameter   : break;
        case DW_TAG_thrown_type                : break;
        case DW_TAG_try_block                  : match_addr_range = true; break;
        case DW_TAG_variant_part               : break;
        case DW_TAG_variable                   : break;
        case DW_TAG_volatile_type              : break;
        case DW_TAG_dwarf_procedure            : break;
        case DW_TAG_restrict_type              : break;
        case DW_TAG_interface_type             : break;
        case DW_TAG_namespace                  : check_children = true; break;
        case DW_TAG_imported_module            : break;
        case DW_TAG_unspecified_type           : break;
        case DW_TAG_partial_unit               : break;
        case DW_TAG_imported_unit              : break;
        case DW_TAG_shared_type                : break;
        default: break;
        }

        if (match_addr_range)
        {
            dw_addr_t lo_pc = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_low_pc, LLDB_INVALID_ADDRESS);
            if (lo_pc != LLDB_INVALID_ADDRESS)
            {
                dw_addr_t hi_pc = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_high_pc, LLDB_INVALID_ADDRESS);
                if (hi_pc != LLDB_INVALID_ADDRESS)
                {
                    //  printf("\n0x%8.8x: %30s: address = 0x%8.8x  [0x%8.8x - 0x%8.8x) ", m_offset, DW_TAG_value_to_name(tag), address, lo_pc, hi_pc);
                    if ((lo_pc <= address) && (address < hi_pc))
                    {
                        found_address = true;
                    //  puts("***MATCH***");
                        switch (m_tag)
                        {
                        case DW_TAG_compile_unit:       // File
                            check_children = ((function_die != NULL) || (block_die != NULL));
                            break;

                        case DW_TAG_subprogram:         // Function
                            if (function_die)
                                *function_die = this;
                            check_children = (block_die != NULL);
                            break;

                        case DW_TAG_inlined_subroutine: // Inlined Function
                        case DW_TAG_lexical_block:      // Block { } in code
                            if (block_die)
                            {
                                *block_die = this;
                                check_children = true;
                            }
                            break;

                        default:
                            check_children = true;
                            break;
                        }
                    }
                }
                else
                {   // compile units may not have a valid high/low pc when there
                    // are address gaps in subroutines so we must always search
                    // if there is no valid high and low PC
                    check_children = (m_tag == DW_TAG_compile_unit) && ((function_die != NULL) || (block_die != NULL));
                }
            }
            else
            {
                dw_offset_t debug_ranges_offset = GetAttributeValueAsUnsigned(dwarf2Data, cu, DW_AT_ranges, DW_INVALID_OFFSET);
                if (debug_ranges_offset != DW_INVALID_OFFSET)
                {
                    DWARFDebugRanges::RangeList ranges;
                    DWARFDebugRanges* debug_ranges = dwarf2Data->DebugRanges();
                    debug_ranges->FindRanges(debug_ranges_offset, ranges);
                    // All DW_AT_ranges are relative to the base address of the
                    // compile unit. We add the compile unit base address to make
                    // sure all the addresses are properly fixed up.
                    ranges.Slide (cu->GetBaseAddress());
                    if (ranges.FindEntryThatContains(address))
                    {
                        found_address = true;
                    //  puts("***MATCH***");
                        switch (m_tag)
                        {
                        case DW_TAG_compile_unit:       // File
                            check_children = ((function_die != NULL) || (block_die != NULL));
                            break;

                        case DW_TAG_subprogram:         // Function
                            if (function_die)
                                *function_die = this;
                            check_children = (block_die != NULL);
                            break;

                        case DW_TAG_inlined_subroutine: // Inlined Function
                        case DW_TAG_lexical_block:      // Block { } in code
                            if (block_die)
                            {
                                *block_die = this;
                                check_children = true;
                            }
                            break;

                        default:
                            check_children = true;
                            break;
                        }
                    }
                    else
                    {
                        check_children = false;
                    }
                }
            }
        }


        if (check_children)
        {
        //  printf("checking children\n");
            DWARFDebugInfoEntry* child = GetFirstChild();
            while (child)
            {
                if (child->LookupAddress(address, dwarf2Data, cu, function_die, block_die))
                    return true;
                child = child->GetSibling();
            }
        }
    }
    return found_address;
}

const DWARFAbbreviationDeclaration* 
DWARFDebugInfoEntry::GetAbbreviationDeclarationPtr (SymbolFileDWARF* dwarf2Data,
                                                    const DWARFCompileUnit *cu,
                                                    lldb::offset_t &offset) const
{
    if (dwarf2Data)
    {
        offset = GetOffset();
        
        const DWARFAbbreviationDeclaration* abbrev_decl = cu->GetAbbreviations()->GetAbbreviationDeclaration (m_abbr_idx);
        if (abbrev_decl)
        {
            // Make sure the abbreviation code still matches. If it doesn't and
            // the DWARF data was mmap'ed, the backing file might have been modified
            // which is bad news.
            const uint64_t abbrev_code = dwarf2Data->get_debug_info_data().GetULEB128 (&offset);
        
            if (abbrev_decl->Code() == abbrev_code)
                return abbrev_decl;
            
            dwarf2Data->GetObjectFile()->GetModule()->ReportErrorIfModifyDetected ("0x%8.8x: the DWARF debug information has been modified (abbrev code was %u, and is now %u)", 
                                                                                   GetOffset(),
                                                                                   (uint32_t)abbrev_decl->Code(),
                                                                                   (uint32_t)abbrev_code);
        }
    }
    offset = DW_INVALID_OFFSET;
    return NULL;
}


bool
DWARFDebugInfoEntry::OffsetLessThan (const DWARFDebugInfoEntry& a, const DWARFDebugInfoEntry& b)
{
    return a.GetOffset() < b.GetOffset();
}

void
DWARFDebugInfoEntry::DumpDIECollection (Stream &strm, DWARFDebugInfoEntry::collection &die_collection)
{
    DWARFDebugInfoEntry::const_iterator pos;
    DWARFDebugInfoEntry::const_iterator end = die_collection.end();
    strm.PutCString("\noffset    parent   sibling  child\n");
    strm.PutCString("--------  -------- -------- --------\n");
    for (pos = die_collection.begin(); pos != end; ++pos)
    {
        const DWARFDebugInfoEntry& die_ref = *pos;
        const DWARFDebugInfoEntry* p = die_ref.GetParent();
        const DWARFDebugInfoEntry* s = die_ref.GetSibling();
        const DWARFDebugInfoEntry* c = die_ref.GetFirstChild();
        strm.Printf("%.8x: %.8x %.8x %.8x 0x%4.4x %s%s\n", 
                    die_ref.GetOffset(),
                    p ? p->GetOffset() : 0,
                    s ? s->GetOffset() : 0,
                    c ? c->GetOffset() : 0,
                    die_ref.Tag(), 
                    DW_TAG_value_to_name(die_ref.Tag()),
                    die_ref.HasChildren() ? " *" : "");
    }
}


