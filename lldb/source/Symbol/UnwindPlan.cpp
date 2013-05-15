//===-- UnwindPlan.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/UnwindPlan.h"

#include "lldb/Core/ConstString.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

bool
UnwindPlan::Row::RegisterLocation::operator == (const UnwindPlan::Row::RegisterLocation& rhs) const
{
    if (m_type == rhs.m_type)
    {
        switch (m_type)
        {
            case unspecified:
            case undefined:
            case same:
                return true;
                
            case atCFAPlusOffset:
            case isCFAPlusOffset:
                return m_location.offset == rhs.m_location.offset;

            case inOtherRegister:
                return m_location.reg_num == rhs.m_location.reg_num;
            
            case atDWARFExpression:
            case isDWARFExpression:
                if (m_location.expr.length == rhs.m_location.expr.length)
                    return !memcmp (m_location.expr.opcodes, rhs.m_location.expr.opcodes, m_location.expr.length);
                break;
        }
    }
    return false;
}

// This function doesn't copy the dwarf expression bytes; they must remain in allocated
// memory for the lifespan of this UnwindPlan object.
void
UnwindPlan::Row::RegisterLocation::SetAtDWARFExpression (const uint8_t *opcodes, uint32_t len)
{
    m_type = atDWARFExpression;
    m_location.expr.opcodes = opcodes;
    m_location.expr.length = len;
}

// This function doesn't copy the dwarf expression bytes; they must remain in allocated
// memory for the lifespan of this UnwindPlan object.
void
UnwindPlan::Row::RegisterLocation::SetIsDWARFExpression (const uint8_t *opcodes, uint32_t len)
{
    m_type = isDWARFExpression;
    m_location.expr.opcodes = opcodes;
    m_location.expr.length = len;
}

void
UnwindPlan::Row::RegisterLocation::Dump (Stream &s, const UnwindPlan* unwind_plan, const UnwindPlan::Row* row, Thread* thread, bool verbose) const
{
    switch (m_type)
    {
        case unspecified: 
            if (verbose)
                s.PutCString ("=<unspec>"); 
            else
                s.PutCString ("=!"); 
            break;
        case undefined: 
            if (verbose)
                s.PutCString ("=<undef>"); 
            else
                s.PutCString ("=?"); 
            break;
        case same: 
            s.PutCString ("= <same>"); 
            break;

        case atCFAPlusOffset: 
        case isCFAPlusOffset: 
            {
                s.PutChar('=');
                if (m_type == atCFAPlusOffset)
                    s.PutChar('[');
                if (verbose)
                    s.Printf ("CFA%+d", m_location.offset);

                if (unwind_plan && row)
                {
                    const uint32_t cfa_reg = row->GetCFARegister();
                    const RegisterInfo *cfa_reg_info = unwind_plan->GetRegisterInfo (thread, cfa_reg);
                    const int32_t offset = row->GetCFAOffset() + m_location.offset;
                    if (verbose)
                    {                        
                        if (cfa_reg_info)
                            s.Printf (" (%s%+d)",  cfa_reg_info->name, offset); 
                        else
                            s.Printf (" (reg(%u)%+d)",  cfa_reg, offset); 
                    }
                    else
                    {
                        if (cfa_reg_info)
                            s.Printf ("%s",  cfa_reg_info->name); 
                        else
                            s.Printf ("reg(%u)",  cfa_reg); 
                        if (offset != 0)
                            s.Printf ("%+d", offset);
                    }
                }
                if (m_type == atCFAPlusOffset)
                    s.PutChar(']');
            }
            break;

        case inOtherRegister: 
            {
                const RegisterInfo *other_reg_info = NULL;
                if (unwind_plan)
                    other_reg_info = unwind_plan->GetRegisterInfo (thread, m_location.reg_num);
                if (other_reg_info)
                    s.Printf ("=%s", other_reg_info->name); 
                else
                    s.Printf ("=reg(%u)", m_location.reg_num); 
            }
            break;

        case atDWARFExpression: 
        case isDWARFExpression: 
            {
                s.PutChar('=');
                if (m_type == atDWARFExpression)
                    s.PutCString("[dwarf-expr]");
                else
                    s.PutCString("dwarf-expr");
            }
            break;
        
    }
}

void
UnwindPlan::Row::Clear ()
{
    m_offset = 0;
    m_cfa_reg_num = LLDB_INVALID_REGNUM;
    m_cfa_offset = 0;
    m_register_locations.clear();
}

void
UnwindPlan::Row::Dump (Stream& s, const UnwindPlan* unwind_plan, Thread* thread, addr_t base_addr) const
{
    const RegisterInfo *reg_info = unwind_plan->GetRegisterInfo (thread, GetCFARegister());

    if (base_addr != LLDB_INVALID_ADDRESS)
        s.Printf ("0x%16.16" PRIx64 ": CFA=", base_addr + GetOffset());
    else
        s.Printf ("0x%8.8" PRIx64 ": CFA=", GetOffset());
            
    if (reg_info)
        s.Printf ("%s", reg_info->name);
    else
        s.Printf ("reg(%u)", GetCFARegister());
    s.Printf ("%+3d => ", GetCFAOffset ());
    for (collection::const_iterator idx = m_register_locations.begin (); idx != m_register_locations.end (); ++idx)
    {
        reg_info = unwind_plan->GetRegisterInfo (thread, idx->first);
        if (reg_info)
            s.Printf ("%s", reg_info->name);
        else
            s.Printf ("reg(%u)", idx->first);
        const bool verbose = false;
        idx->second.Dump(s, unwind_plan, this, thread, verbose);
        s.PutChar (' ');
    }
    s.EOL();
}

UnwindPlan::Row::Row() :
    m_offset(0),
    m_cfa_reg_num(LLDB_INVALID_REGNUM),
    m_cfa_offset(0),
    m_register_locations()
{
}

bool
UnwindPlan::Row::GetRegisterInfo (uint32_t reg_num, UnwindPlan::Row::RegisterLocation& register_location) const
{
    collection::const_iterator pos = m_register_locations.find(reg_num);
    if (pos != m_register_locations.end())
    {
        register_location = pos->second;
        return true;
    }
    return false;
}

void
UnwindPlan::Row::SetRegisterInfo (uint32_t reg_num, const UnwindPlan::Row::RegisterLocation register_location)
{
    m_register_locations[reg_num] = register_location;
}

bool
UnwindPlan::Row::SetRegisterLocationToAtCFAPlusOffset (uint32_t reg_num, int32_t offset, bool can_replace)
{
    if (!can_replace && m_register_locations.find(reg_num) != m_register_locations.end())
        return false;
    RegisterLocation reg_loc;
    reg_loc.SetAtCFAPlusOffset(offset);
    m_register_locations[reg_num] = reg_loc;
    return true;
}

bool
UnwindPlan::Row::SetRegisterLocationToIsCFAPlusOffset (uint32_t reg_num, int32_t offset, bool can_replace)
{
    if (!can_replace && m_register_locations.find(reg_num) != m_register_locations.end())
        return false;
    RegisterLocation reg_loc;
    reg_loc.SetIsCFAPlusOffset(offset);
    m_register_locations[reg_num] = reg_loc;
    return true;
}

bool
UnwindPlan::Row::SetRegisterLocationToUndefined (uint32_t reg_num, bool can_replace, bool can_replace_only_if_unspecified)
{
    collection::iterator pos = m_register_locations.find(reg_num);
    collection::iterator end = m_register_locations.end();
    
    if (pos != end)
    {
        if (!can_replace)
            return false;
        if (can_replace_only_if_unspecified && !pos->second.IsUnspecified())
            return false;
    }
    RegisterLocation reg_loc;
    reg_loc.SetUndefined();
    m_register_locations[reg_num] = reg_loc;
    return true;
}

bool
UnwindPlan::Row::SetRegisterLocationToUnspecified (uint32_t reg_num, bool can_replace)
{
    if (!can_replace && m_register_locations.find(reg_num) != m_register_locations.end())
        return false;
    RegisterLocation reg_loc;
    reg_loc.SetUnspecified();
    m_register_locations[reg_num] = reg_loc;
    return true;
}

bool
UnwindPlan::Row::SetRegisterLocationToRegister (uint32_t reg_num, 
                                                uint32_t other_reg_num,
                                                bool can_replace)
{
    if (!can_replace && m_register_locations.find(reg_num) != m_register_locations.end())
        return false;
    RegisterLocation reg_loc;
    reg_loc.SetInRegister(other_reg_num);
    m_register_locations[reg_num] = reg_loc;
    return true;
}

bool
UnwindPlan::Row::SetRegisterLocationToSame (uint32_t reg_num, bool must_replace)
{
    if (must_replace && m_register_locations.find(reg_num) == m_register_locations.end())
        return false;
    RegisterLocation reg_loc;
    reg_loc.SetSame();
    m_register_locations[reg_num] = reg_loc;
    return true;
}

void
UnwindPlan::Row::SetCFARegister (uint32_t reg_num)
{
    m_cfa_reg_num = reg_num;
}

bool
UnwindPlan::Row::operator == (const UnwindPlan::Row& rhs) const
{
    if (m_offset != rhs.m_offset || m_cfa_reg_num != rhs.m_cfa_reg_num || m_cfa_offset != rhs.m_cfa_offset)
        return false;
    return m_register_locations == rhs.m_register_locations;
}

void
UnwindPlan::AppendRow (const UnwindPlan::RowSP &row_sp)
{
    if (m_row_list.empty() || m_row_list.back()->GetOffset() != row_sp->GetOffset())
        m_row_list.push_back(row_sp);
    else
        m_row_list.back() = row_sp;
}

UnwindPlan::RowSP
UnwindPlan::GetRowForFunctionOffset (int offset) const
{
    RowSP row;
    if (!m_row_list.empty())
    {
        if (offset == -1)
            row = m_row_list.back();
        else
        {
            collection::const_iterator pos, end = m_row_list.end();
            for (pos = m_row_list.begin(); pos != end; ++pos)
            {
                if ((*pos)->GetOffset() <= offset)
                    row = *pos;
                else
                    break;
            }
        }
    }
    return row;
}

bool
UnwindPlan::IsValidRowIndex (uint32_t idx) const
{
    return idx < m_row_list.size();
}

const UnwindPlan::RowSP
UnwindPlan::GetRowAtIndex (uint32_t idx) const
{
    // You must call IsValidRowIndex(idx) first before calling this!!!
    assert (idx < m_row_list.size());
    return m_row_list[idx];
}

const UnwindPlan::RowSP
UnwindPlan::GetLastRow () const
{
    // You must call GetRowCount() first to make sure there is at least one row
    assert (!m_row_list.empty());
    return m_row_list.back();
}

int
UnwindPlan::GetRowCount () const
{
    return m_row_list.size ();
}

void
UnwindPlan::SetPlanValidAddressRange (const AddressRange& range)
{
   if (range.GetBaseAddress().IsValid() && range.GetByteSize() != 0)
       m_plan_valid_address_range = range;
}

bool
UnwindPlan::PlanValidAtAddress (Address addr)
{
    if (!m_plan_valid_address_range.GetBaseAddress().IsValid() || m_plan_valid_address_range.GetByteSize() == 0)
        return true;

    if (!addr.IsValid())
        return true;

    if (m_plan_valid_address_range.ContainsFileAddress (addr))
        return true;

    return false;
}

void
UnwindPlan::Dump (Stream& s, Thread *thread, lldb::addr_t base_addr) const
{
    if (!m_source_name.IsEmpty())
    {
        s.Printf ("This UnwindPlan originally sourced from %s\n", m_source_name.GetCString());
    }
    if (m_plan_valid_address_range.GetBaseAddress().IsValid() && m_plan_valid_address_range.GetByteSize() > 0)
    {
        s.PutCString ("Address range of this UnwindPlan: ");
        TargetSP target_sp(thread->CalculateTarget());
        m_plan_valid_address_range.Dump (&s, target_sp.get(), Address::DumpStyleSectionNameOffset);
        s.EOL();
    }
    collection::const_iterator pos, begin = m_row_list.begin(), end = m_row_list.end();
    for (pos = begin; pos != end; ++pos)
    {
        s.Printf ("row[%u]: ", (uint32_t)std::distance (begin, pos));
        (*pos)->Dump(s, this, thread, base_addr);
    }
}

void
UnwindPlan::SetSourceName (const char *source)
{
    m_source_name = ConstString (source);
}

ConstString
UnwindPlan::GetSourceName () const
{
    return m_source_name;
}

const RegisterInfo *
UnwindPlan::GetRegisterInfo (Thread* thread, uint32_t unwind_reg) const
{
    if (thread)
    {
        RegisterContext *reg_ctx = thread->GetRegisterContext().get();
        if (reg_ctx)
        {
            uint32_t reg;
            if (m_register_kind == eRegisterKindLLDB)
                reg = unwind_reg;
            else
                reg = reg_ctx->ConvertRegisterKindToRegisterNumber (m_register_kind, unwind_reg);
            if (reg != LLDB_INVALID_REGNUM)
                return reg_ctx->GetRegisterInfoAtIndex (reg);
        }
    }
    return NULL;
}
    
