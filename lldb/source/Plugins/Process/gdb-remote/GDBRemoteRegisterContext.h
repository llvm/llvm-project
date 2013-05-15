//===-- GDBRemoteRegisterContext.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_GDBRemoteRegisterContext_h_
#define lldb_GDBRemoteRegisterContext_h_

// C Includes
// C++ Includes
#include <vector>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/RegisterContext.h"
#include "GDBRemoteCommunicationClient.h"

class ThreadGDBRemote;
class ProcessGDBRemote;
class StringExtractor;

class GDBRemoteDynamicRegisterInfo
{
public:
    GDBRemoteDynamicRegisterInfo () :
        m_regs (),
        m_sets (),
        m_set_reg_nums (),
        m_reg_names (),
        m_reg_alt_names (),
        m_set_names (),
        m_reg_data_byte_size (0)
    {
    }

    ~GDBRemoteDynamicRegisterInfo ()
    {
    }

    void
    AddRegister (lldb_private::RegisterInfo reg_info, 
                 lldb_private::ConstString &reg_name, 
                 lldb_private::ConstString &reg_alt_name, 
                 lldb_private::ConstString &set_name)
    {
        const uint32_t reg_num = (uint32_t)m_regs.size();
        m_reg_names.push_back (reg_name);
        m_reg_alt_names.push_back (reg_alt_name);
        reg_info.name = reg_name.AsCString();
        assert (reg_info.name);
        reg_info.alt_name = reg_alt_name.AsCString(NULL);
        uint32_t i;
        if (reg_info.value_regs)
        {
            for (i=0; reg_info.value_regs[i] != LLDB_INVALID_REGNUM; ++i)
                m_value_regs_map[reg_num].push_back(reg_info.value_regs[i]);
            m_value_regs_map[reg_num].push_back(LLDB_INVALID_REGNUM);
            reg_info.value_regs = m_value_regs_map[reg_num].data();
        }
        if (reg_info.invalidate_regs)
        {
            for (i=0; reg_info.invalidate_regs[i] != LLDB_INVALID_REGNUM; ++i)
                m_invalidate_regs_map[reg_num].push_back(reg_info.invalidate_regs[i]);
            m_invalidate_regs_map[reg_num].push_back(LLDB_INVALID_REGNUM);
            reg_info.invalidate_regs = m_invalidate_regs_map[reg_num].data();
        }
        m_regs.push_back (reg_info);
        uint32_t set = GetRegisterSetIndexByName (set_name);
        assert (set < m_sets.size());
        assert (set < m_set_reg_nums.size());
        assert (set < m_set_names.size());
        m_set_reg_nums[set].push_back(reg_num);
        size_t end_reg_offset = reg_info.byte_offset + reg_info.byte_size;
        if (m_reg_data_byte_size < end_reg_offset)
            m_reg_data_byte_size = end_reg_offset;
    }

    void
    Finalize ()
    {
        for (uint32_t set = 0; set < m_sets.size(); ++set)
        {
            assert (m_sets.size() == m_set_reg_nums.size());
            m_sets[set].num_registers = m_set_reg_nums[set].size();
            m_sets[set].registers = &m_set_reg_nums[set][0];
        }
    }

    size_t
    GetNumRegisters() const
    {
        return m_regs.size();
    }

    size_t
    GetNumRegisterSets() const
    {
        return m_sets.size();
    }

    size_t
    GetRegisterDataByteSize() const
    {
        return m_reg_data_byte_size;
    }

    const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex (uint32_t i) const
    {
        if (i < m_regs.size())
            return &m_regs[i];
        return NULL;
    }

    const lldb_private::RegisterSet *
    GetRegisterSet (uint32_t i) const
    {
        if (i < m_sets.size())
            return &m_sets[i];
        return NULL;
    }

    uint32_t
    GetRegisterSetIndexByName (lldb_private::ConstString &set_name)
    {
        name_collection::iterator pos, end = m_set_names.end();
        for (pos = m_set_names.begin(); pos != end; ++pos)
        {
            if (*pos == set_name)
                return static_cast<uint32_t>(std::distance (m_set_names.begin(), pos));
        }

        m_set_names.push_back(set_name);
        m_set_reg_nums.resize(m_set_reg_nums.size()+1);
        lldb_private::RegisterSet new_set = { set_name.AsCString(), NULL, 0, NULL };
        m_sets.push_back (new_set);
        return static_cast<uint32_t>(m_sets.size() - 1);
    }

    uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num) const
    {
        reg_collection::const_iterator pos, end = m_regs.end();
        for (pos = m_regs.begin(); pos != end; ++pos)
        {
            if (pos->kinds[kind] == num)
                return static_cast<uint32_t>(std::distance (m_regs.begin(), pos));
        }

        return LLDB_INVALID_REGNUM;
    }
    void
    Clear()
    {
        m_regs.clear();
        m_sets.clear();
        m_set_reg_nums.clear();
        m_reg_names.clear();
        m_reg_alt_names.clear();
        m_set_names.clear();
    }

    void
    HardcodeARMRegisters(bool from_scratch);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from GDBRemoteRegisterContext can see and modify these
    //------------------------------------------------------------------
    typedef std::vector <lldb_private::RegisterInfo> reg_collection;
    typedef std::vector <lldb_private::RegisterSet> set_collection;
    typedef std::vector <uint32_t> reg_num_collection;
    typedef std::vector <reg_num_collection> set_reg_num_collection;
    typedef std::vector <lldb_private::ConstString> name_collection;
    typedef std::map<uint32_t, reg_num_collection> reg_to_regs_map;

    reg_collection m_regs;
    set_collection m_sets;
    set_reg_num_collection m_set_reg_nums;
    name_collection m_reg_names;
    name_collection m_reg_alt_names;
    name_collection m_set_names;
    reg_to_regs_map m_value_regs_map;
    reg_to_regs_map m_invalidate_regs_map;
    size_t m_reg_data_byte_size;   // The number of bytes required to store all registers
};

class GDBRemoteRegisterContext : public lldb_private::RegisterContext
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    GDBRemoteRegisterContext (ThreadGDBRemote &thread,
                              uint32_t concrete_frame_idx,
                              GDBRemoteDynamicRegisterInfo &reg_info,
                              bool read_all_at_once);

    virtual
    ~GDBRemoteRegisterContext ();

    //------------------------------------------------------------------
    // Subclasses must override these functions
    //------------------------------------------------------------------
    virtual void
    InvalidateAllRegisters ();

    virtual size_t
    GetRegisterCount ();

    virtual const lldb_private::RegisterInfo *
    GetRegisterInfoAtIndex (size_t reg);

    virtual size_t
    GetRegisterSetCount ();

    virtual const lldb_private::RegisterSet *
    GetRegisterSet (size_t reg_set);

    virtual bool
    ReadRegister (const lldb_private::RegisterInfo *reg_info, lldb_private::RegisterValue &value);

    virtual bool
    WriteRegister (const lldb_private::RegisterInfo *reg_info, const lldb_private::RegisterValue &value);
    
    virtual bool
    ReadAllRegisterValues (lldb::DataBufferSP &data_sp);

    virtual bool
    WriteAllRegisterValues (const lldb::DataBufferSP &data_sp);

    virtual uint32_t
    ConvertRegisterKindToRegisterNumber (uint32_t kind, uint32_t num);

protected:
    friend class ThreadGDBRemote;

    bool
    ReadRegisterBytes (const lldb_private::RegisterInfo *reg_info,
                       lldb_private::DataExtractor &data);

    bool
    WriteRegisterBytes (const lldb_private::RegisterInfo *reg_info,
                        lldb_private::DataExtractor &data, 
                        uint32_t data_offset);

    bool
    PrivateSetRegisterValue (uint32_t reg, StringExtractor &response);
    
    void
    SetAllRegisterValid (bool b);

    bool
    GetRegisterIsValid (uint32_t reg) const
    {
#if defined (LLDB_CONFIGURATION_DEBUG)
        assert (reg < m_reg_valid.size());
#endif
        if (reg < m_reg_valid.size())
            return m_reg_valid[reg];
        return false;
    }

    void
    SetRegisterIsValid (const lldb_private::RegisterInfo *reg_info, bool valid)
    {
        if (reg_info)
            return SetRegisterIsValid (reg_info->kinds[lldb::eRegisterKindLLDB], valid);
    }

    void
    SetRegisterIsValid (uint32_t reg, bool valid)
    {
#if defined (LLDB_CONFIGURATION_DEBUG)
        assert (reg < m_reg_valid.size());
#endif
        if (reg < m_reg_valid.size())
            m_reg_valid[reg] = valid;
    }

    void
    SyncThreadState(lldb_private::Process *process);  // Assumes the sequence mutex has already been acquired.
    
    GDBRemoteDynamicRegisterInfo &m_reg_info;
    std::vector<bool> m_reg_valid;
    lldb_private::DataExtractor m_reg_data;
    bool m_read_all_at_once;

private:
    // Helper function for ReadRegisterBytes().
    bool GetPrimordialRegister(const lldb_private::RegisterInfo *reg_info,
                               GDBRemoteCommunicationClient &gdb_comm);
    // Helper function for WriteRegisterBytes().
    bool SetPrimordialRegister(const lldb_private::RegisterInfo *reg_info,
                               GDBRemoteCommunicationClient &gdb_comm);

    //------------------------------------------------------------------
    // For GDBRemoteRegisterContext only
    //------------------------------------------------------------------
    DISALLOW_COPY_AND_ASSIGN (GDBRemoteRegisterContext);
};

#endif  // lldb_GDBRemoteRegisterContext_h_
