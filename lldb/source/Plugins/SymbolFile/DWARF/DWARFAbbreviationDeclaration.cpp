//===-- DWARFAbbreviationDeclaration.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFAbbreviationDeclaration.h"

#include "lldb/Core/dwarf.h"

#include "DWARFFormValue.h"

using namespace lldb_private;

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration() :
    m_code  (InvalidCode),
    m_tag   (0),
    m_has_children (0),
    m_attributes()
{
}

DWARFAbbreviationDeclaration::DWARFAbbreviationDeclaration(dw_tag_t tag, uint8_t has_children) :
    m_code  (InvalidCode),
    m_tag   (tag),
    m_has_children (has_children),
    m_attributes()
{
}

bool
DWARFAbbreviationDeclaration::Extract(const DataExtractor& data, uint32_t* offset_ptr)
{
    return Extract(data, offset_ptr, data.GetULEB128(offset_ptr));
}

bool
DWARFAbbreviationDeclaration::Extract(const DataExtractor& data, uint32_t* offset_ptr, dw_uleb128_t code)
{
    m_code = code;
    m_attributes.clear();
    if (m_code)
    {
        m_tag = data.GetULEB128(offset_ptr);
        m_has_children = data.GetU8(offset_ptr);

        while (data.ValidOffset(*offset_ptr))
        {
            dw_attr_t attr = data.GetULEB128(offset_ptr);
            dw_form_t form = data.GetULEB128(offset_ptr);

            if (attr && form)
                m_attributes.push_back(DWARFAttribute(attr, form));
            else
                break;
        }

        return m_tag != 0;
    }
    else
    {
        m_tag = 0;
        m_has_children = 0;
    }

    return false;
}


void
DWARFAbbreviationDeclaration::Dump(Stream *s)  const
{
//  *ostrm_ptr << std::setfill(' ') << std::dec << '[' << std::setw(3) << std::right << m_code << ']' << ' ' << std::setw(30) << std::left << DW_TAG_value_to_name(m_tag) << DW_CHILDREN_value_to_name(m_has_children) << std::endl;
//
//  DWARFAttribute::const_iterator pos;
//
//  for (pos = m_attributes.begin(); pos != m_attributes.end(); ++pos)
//      *ostrm_ptr << "      " << std::setw(29) << std::left << DW_AT_value_to_name(pos->attr()) << ' ' << DW_FORM_value_to_name(pos->form()) << std::endl;
//
//  *ostrm_ptr << std::endl;
}



bool
DWARFAbbreviationDeclaration::IsValid()
{
    return m_code != 0 && m_tag != 0;
}


void
DWARFAbbreviationDeclaration::CopyExcludingAddressAttributes(const DWARFAbbreviationDeclaration& abbr_decl, const uint32_t idx)
{
    m_code = abbr_decl.Code();  // Invalidate the code since that can't be copied safely.
    m_tag = abbr_decl.Tag();
    m_has_children = abbr_decl.HasChildren();

    const DWARFAttribute::collection& attributes = abbr_decl.Attributes();
    const uint32_t num_abbr_decl_attributes = attributes.size();

    dw_attr_t attr;
    dw_form_t form;
    uint32_t i;

    for (i = 0; i < num_abbr_decl_attributes; ++i)
    {
        attributes[i].get(attr, form);
        switch (attr)
        {
        case DW_AT_location:
        case DW_AT_frame_base:
            // Only add these if they are location expressions (have a single
            // value) and not location lists (have a lists of location
            // expressions which are only valid over specific address ranges)
            if (DWARFFormValue::IsBlockForm(form))
                m_attributes.push_back(DWARFAttribute(attr, form));
            break;

        case DW_AT_low_pc:
        case DW_AT_high_pc:
        case DW_AT_ranges:
        case DW_AT_entry_pc:
            // Don't add these attributes
            if (i >= idx)
                break;
            // Fall through and add attribute
        default:
            // Add anything that isn't address related
            m_attributes.push_back(DWARFAttribute(attr, form));
            break;
        }
    }
}

void
DWARFAbbreviationDeclaration::CopyChangingStringToStrp(
    const DWARFAbbreviationDeclaration& abbr_decl,
    const DataExtractor& debug_info_data,
    dw_offset_t debug_info_offset,
    const DWARFCompileUnit* cu,
    const uint32_t strp_min_len
)
{
    m_code = InvalidCode;
    m_tag = abbr_decl.Tag();
    m_has_children = abbr_decl.HasChildren();

    const DWARFAttribute::collection& attributes = abbr_decl.Attributes();
    const uint32_t num_abbr_decl_attributes = attributes.size();

    dw_attr_t attr;
    dw_form_t form;
    uint32_t i;
    dw_offset_t offset = debug_info_offset;

    for (i = 0; i < num_abbr_decl_attributes; ++i)
    {
        attributes[i].get(attr, form);
        dw_offset_t attr_offset = offset;
        DWARFFormValue::SkipValue(form, debug_info_data, &offset, cu);

        if (form == DW_FORM_string && ((offset - attr_offset) >= strp_min_len))
            m_attributes.push_back(DWARFAttribute(attr, DW_FORM_strp));
        else
            m_attributes.push_back(DWARFAttribute(attr, form));
    }
}


uint32_t
DWARFAbbreviationDeclaration::FindAttributeIndex(dw_attr_t attr) const
{
    uint32_t i;
    const uint32_t kNumAttributes = m_attributes.size();
    for (i = 0; i < kNumAttributes; ++i)
    {
        if (m_attributes[i].get_attr() == attr)
            return i;
    }
    return DW_INVALID_INDEX;
}


bool
DWARFAbbreviationDeclaration::operator == (const DWARFAbbreviationDeclaration& rhs) const
{
    return Tag()            == rhs.Tag()
        && HasChildren()    == rhs.HasChildren()
        && Attributes()     == rhs.Attributes();
}

#if 0
DWARFAbbreviationDeclaration::Append(BinaryStreamBuf& out_buff) const
{
    out_buff.Append32_as_ULEB128(Code());
    out_buff.Append32_as_ULEB128(Tag());
    out_buff.Append8(HasChildren());
    const uint32_t kNumAttributes = m_attributes.size();
    for (uint32_t i = 0; i < kNumAttributes; ++i)
    {
        out_buff.Append32_as_ULEB128(m_attributes[i].attr());
        out_buff.Append32_as_ULEB128(m_attributes[i].form());
    }
    out_buff.Append8(0);    // Output a zero for attr (faster than using Append32_as_ULEB128)
    out_buff.Append8(0);    // Output a zero for attr (faster than using Append32_as_ULEB128)
}
#endif  // 0
