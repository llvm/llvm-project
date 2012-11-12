//===-- DWARFAbbreviationDeclaration.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DWARFAbbreviationDeclaration_h_
#define liblldb_DWARFAbbreviationDeclaration_h_

#include "SymbolFileDWARF.h"
#include "DWARFAttribute.h"

class DWARFCompileUnit;

class DWARFAbbreviationDeclaration
{
public:
    enum { InvalidCode = 0 };
                    DWARFAbbreviationDeclaration();

                    // For hand crafting an abbreviation declaration
                    DWARFAbbreviationDeclaration(dw_tag_t tag, uint8_t has_children);
    void            AddAttribute(const DWARFAttribute& attr)
                    {
                        m_attributes.push_back(attr);
                    }

    dw_uleb128_t    Code() const { return m_code; }
    void            SetCode(dw_uleb128_t code) { m_code = code; }
    dw_tag_t        Tag() const { return m_tag; }
    bool            HasChildren() const { return m_has_children; }
    uint32_t        NumAttributes() const { return m_attributes.size(); }
    dw_attr_t       GetAttrByIndex(uint32_t idx) const { return m_attributes.size() > idx ? m_attributes[idx].get_attr() : 0; }
    dw_form_t       GetFormByIndex(uint32_t idx) const { return m_attributes.size() > idx ? m_attributes[idx].get_form() : 0; }
    bool            GetAttrAndFormByIndex(uint32_t idx, dw_attr_t& attr, dw_form_t& form) const
                    {
                        if (m_attributes.size() > idx)
                        {
                            m_attributes[idx].get(attr, form);
                            return true;
                        }
                        attr = form = 0;
                        return false;
                    }

                    // idx is assumed to be valid when calling GetAttrAndFormByIndexUnchecked()
    void            GetAttrAndFormByIndexUnchecked(uint32_t idx, dw_attr_t& attr, dw_form_t& form) const
                    {
                        m_attributes[idx].get(attr, form);
                    }
    dw_form_t       GetFormByIndexUnchecked (uint32_t idx) const
                    {
                        return m_attributes[idx].get_form();
                    }
    void            CopyExcludingAddressAttributes(const DWARFAbbreviationDeclaration& abbr_decl, const uint32_t idx);
    void            CopyChangingStringToStrp(
                        const DWARFAbbreviationDeclaration& abbr_decl,
                        const lldb_private::DataExtractor& debug_info_data,
                        dw_offset_t debug_info_offset,
                        const DWARFCompileUnit* cu,
                        const uint32_t strp_min_len);
    uint32_t        FindAttributeIndex(dw_attr_t attr) const;
    bool            Extract(const lldb_private::DataExtractor& data, uint32_t* offset_ptr);
    bool            Extract(const lldb_private::DataExtractor& data, uint32_t* offset_ptr, dw_uleb128_t code);
//  void            Append(BinaryStreamBuf& out_buff) const;
    bool            IsValid();
    void            Dump(lldb_private::Stream *s) const;
    bool            operator == (const DWARFAbbreviationDeclaration& rhs) const;
//  DWARFAttribute::collection& Attributes() { return m_attributes; }
    const DWARFAttribute::collection& Attributes() const { return m_attributes; }
protected:
    dw_uleb128_t        m_code;
    dw_tag_t            m_tag;
    uint8_t             m_has_children;
    DWARFAttribute::collection m_attributes;
};

#endif  // liblldb_DWARFAbbreviationDeclaration_h_
