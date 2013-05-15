//===-- DynamicRegisterInfo.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-python.h"

#include "DynamicRegisterInfo.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Interpreter/Args.h"

#ifndef LLDB_DISABLE_PYTHON
#include "lldb/Interpreter/PythonDataObjects.h"
#endif

using namespace lldb;
using namespace lldb_private;

DynamicRegisterInfo::DynamicRegisterInfo () :
    m_regs (),
    m_sets (),
    m_set_reg_nums (),
    m_set_names (),
    m_reg_data_byte_size (0)
{
}

DynamicRegisterInfo::DynamicRegisterInfo (const lldb_private::PythonDictionary &dict) :
    m_regs (),
    m_sets (),
    m_set_reg_nums (),
    m_set_names (),
    m_reg_data_byte_size (0)
{
    SetRegisterInfo (dict);
}

DynamicRegisterInfo::~DynamicRegisterInfo ()
{
}


size_t
DynamicRegisterInfo::SetRegisterInfo (const lldb_private::PythonDictionary &dict)
{
#ifndef LLDB_DISABLE_PYTHON
    PythonList sets (dict.GetItemForKey("sets"));
    if (sets)
    {
        const uint32_t num_sets = sets.GetSize();
        for (uint32_t i=0; i<num_sets; ++i)
        {
            PythonString py_set_name(sets.GetItemAtIndex(i));
            ConstString set_name;
            if (py_set_name)
                set_name.SetCString(py_set_name.GetString());
            if (set_name)
            {
                RegisterSet new_set = { set_name.AsCString(), NULL, 0, NULL };
                m_sets.push_back (new_set);
            }
            else
            {
                Clear();
                return 0;
            }
        }
        m_set_reg_nums.resize(m_sets.size());
    }
    PythonList regs (dict.GetItemForKey("registers"));
    if (regs)
    {
        const uint32_t num_regs = regs.GetSize();
        PythonString name_pystr("name");
        PythonString altname_pystr("alt-name");
        PythonString bitsize_pystr("bitsize");
        PythonString offset_pystr("offset");
        PythonString encoding_pystr("encoding");
        PythonString format_pystr("format");
        PythonString set_pystr("set");
        PythonString gcc_pystr("gcc");
        PythonString dwarf_pystr("dwarf");
        PythonString generic_pystr("generic");
        for (uint32_t i=0; i<num_regs; ++i)
        {
            PythonDictionary reg_info_dict(regs.GetItemAtIndex(i));
            if (reg_info_dict)
            {
                // { 'name':'rcx'       , 'bitsize' :  64, 'offset' :  16, 'encoding':'uint'  , 'format':'hex'         , 'set': 0, 'gcc' : 2, 'dwarf' : 2, 'generic':'arg4', 'alt-name':'arg4', },
                RegisterInfo reg_info;
                bzero (&reg_info, sizeof(reg_info));
                
                reg_info.name = ConstString (reg_info_dict.GetItemForKeyAsString(name_pystr)).GetCString();
                if (reg_info.name == NULL)
                {
                    Clear();
                    return 0;
                }
                    
                reg_info.alt_name = ConstString (reg_info_dict.GetItemForKeyAsString(altname_pystr)).GetCString();
                
                reg_info.byte_offset = reg_info_dict.GetItemForKeyAsInteger(offset_pystr, UINT32_MAX);

                if (reg_info.byte_offset == UINT32_MAX)
                {
                    Clear();
                    return 0;
                }
                reg_info.byte_size = reg_info_dict.GetItemForKeyAsInteger(bitsize_pystr, 0) / 8;
                
                if (reg_info.byte_size == 0)
                {
                    Clear();
                    return 0;
                }
                
                const char *format_cstr = reg_info_dict.GetItemForKeyAsString(format_pystr);
                if (format_cstr)
                {
                    if (Args::StringToFormat(format_cstr, reg_info.format, NULL).Fail())
                    {
                        Clear();
                        return 0;
                    }
                }
                else
                    reg_info.format = eFormatHex;
                    
                const char *encoding_cstr = reg_info_dict.GetItemForKeyAsString(encoding_pystr);
                if (encoding_cstr)
                    reg_info.encoding = Args::StringToEncoding (encoding_cstr, eEncodingUint);
                else
                    reg_info.encoding = eEncodingUint;

                const int64_t set = reg_info_dict.GetItemForKeyAsInteger(set_pystr, -1);
                if (set >= m_sets.size())
                {
                    Clear();
                    return 0;
                }

                reg_info.kinds[lldb::eRegisterKindLLDB]    = i;
                reg_info.kinds[lldb::eRegisterKindGDB]     = i;
                reg_info.kinds[lldb::eRegisterKindGCC]     = reg_info_dict.GetItemForKeyAsInteger(gcc_pystr, LLDB_INVALID_REGNUM);
                reg_info.kinds[lldb::eRegisterKindDWARF]   = reg_info_dict.GetItemForKeyAsInteger(dwarf_pystr, LLDB_INVALID_REGNUM);
                reg_info.kinds[lldb::eRegisterKindGeneric] = Args::StringToGenericRegister (reg_info_dict.GetItemForKeyAsString(generic_pystr));
                const size_t end_reg_offset = reg_info.byte_offset + reg_info.byte_size;
                if (m_reg_data_byte_size < end_reg_offset)
                    m_reg_data_byte_size = end_reg_offset;

                m_regs.push_back (reg_info);
                m_set_reg_nums[set].push_back(i);

            }
            else
            {
                Clear();
                return 0;
            }
        }
        Finalize ();
    }
#endif
    return 0;
}


void
DynamicRegisterInfo::AddRegister (RegisterInfo &reg_info,
                                  ConstString &reg_name, 
                                  ConstString &reg_alt_name, 
                                  ConstString &set_name)
{
    const uint32_t reg_num = m_regs.size();
    reg_info.name = reg_name.AsCString();
    assert (reg_info.name);
    reg_info.alt_name = reg_alt_name.AsCString(NULL);
    m_regs.push_back (reg_info);
    uint32_t set = GetRegisterSetIndexByName (set_name, true);
    assert (set < m_sets.size());
    assert (set < m_set_reg_nums.size());
    assert (set < m_set_names.size());
    m_set_reg_nums[set].push_back(reg_num);
    size_t end_reg_offset = reg_info.byte_offset + reg_info.byte_size;
    if (m_reg_data_byte_size < end_reg_offset)
        m_reg_data_byte_size = end_reg_offset;
}

void
DynamicRegisterInfo::Finalize ()
{
    for (uint32_t set = 0; set < m_sets.size(); ++set)
    {
        assert (m_sets.size() == m_set_reg_nums.size());
        m_sets[set].num_registers = m_set_reg_nums[set].size();
        m_sets[set].registers = &m_set_reg_nums[set][0];
    }
}

size_t
DynamicRegisterInfo::GetNumRegisters() const
{
    return m_regs.size();
}

size_t
DynamicRegisterInfo::GetNumRegisterSets() const
{
    return m_sets.size();
}

size_t
DynamicRegisterInfo::GetRegisterDataByteSize() const
{
    return m_reg_data_byte_size;
}

const RegisterInfo *
DynamicRegisterInfo::GetRegisterInfoAtIndex (uint32_t i) const
{
    if (i < m_regs.size())
        return &m_regs[i];
    return NULL;
}

const RegisterSet *
DynamicRegisterInfo::GetRegisterSet (uint32_t i) const
{
    if (i < m_sets.size())
        return &m_sets[i];
    return NULL;
}

uint32_t
DynamicRegisterInfo::GetRegisterSetIndexByName (ConstString &set_name, bool can_create)
{
    name_collection::iterator pos, end = m_set_names.end();
    for (pos = m_set_names.begin(); pos != end; ++pos)
    {
        if (*pos == set_name)
            return std::distance (m_set_names.begin(), pos);
    }
    
    m_set_names.push_back(set_name);
    m_set_reg_nums.resize(m_set_reg_nums.size()+1);
    RegisterSet new_set = { set_name.AsCString(), NULL, 0, NULL };
    m_sets.push_back (new_set);
    return m_sets.size() - 1;
}

uint32_t
DynamicRegisterInfo::ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num) const
{
    reg_collection::const_iterator pos, end = m_regs.end();
    for (pos = m_regs.begin(); pos != end; ++pos)
    {
        if (pos->kinds[kind] == num)
            return std::distance (m_regs.begin(), pos);
    }
    
    return LLDB_INVALID_REGNUM;
}

void
DynamicRegisterInfo::Clear()
{
    m_regs.clear();
    m_sets.clear();
    m_set_reg_nums.clear();
    m_set_names.clear();
}
