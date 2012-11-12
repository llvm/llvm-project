//===-- DWARFAttribute.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFAttribute_h_
#define SymbolFileDWARF_DWARFAttribute_h_

#include "DWARFDefines.h"
#include <vector>

class DWARFAttribute
{
public:
    DWARFAttribute(dw_attr_t attr, dw_form_t form) :
        m_attr_form ( attr << 16 | form )
    {
    }

    void        set(dw_attr_t attr, dw_form_t form) { m_attr_form = (attr << 16) | form; }
    void        set_attr(dw_attr_t attr) { m_attr_form = (m_attr_form & 0x0000ffffu) | (attr << 16); }
    void        set_form(dw_form_t form) { m_attr_form = (m_attr_form & 0xffff0000u) | form; }
    dw_attr_t   get_attr() const { return m_attr_form >> 16; }
    dw_form_t   get_form() const { return m_attr_form; }
    void        get(dw_attr_t& attr, dw_form_t& form)  const
    {
        register uint32_t attr_form = m_attr_form;
        attr = attr_form >> 16;
        form = attr_form;
    }
    bool        operator == (const DWARFAttribute& rhs) const { return m_attr_form == rhs.m_attr_form; }
    typedef std::vector<DWARFAttribute> collection;
    typedef collection::iterator iterator;
    typedef collection::const_iterator const_iterator;

protected:
    uint32_t    m_attr_form;    // Upper 16 bits is attribute, lower 16 bits is form
};


#endif  // SymbolFileDWARF_DWARFAttribute_h_
