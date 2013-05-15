//===-- DWARFCallFrameInfo.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// C Includes
// C++ Includes
#include <list>

#include "lldb/Core/Log.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/ArchSpec.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/Section.h"
#include "lldb/Core/Timer.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/DWARFCallFrameInfo.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

DWARFCallFrameInfo::DWARFCallFrameInfo(ObjectFile& objfile, SectionSP& section_sp, lldb::RegisterKind reg_kind, bool is_eh_frame) :
    m_objfile (objfile),
    m_section_sp (section_sp),
    m_reg_kind (reg_kind),  // The flavor of registers that the CFI data uses (enum RegisterKind)
    m_flags (),
    m_cie_map (),
    m_cfi_data (),
    m_cfi_data_initialized (false),
    m_fde_index (),
    m_fde_index_initialized (false),
    m_is_eh_frame (is_eh_frame)
{
}

DWARFCallFrameInfo::~DWARFCallFrameInfo()
{
}


bool
DWARFCallFrameInfo::GetUnwindPlan (Address addr, UnwindPlan& unwind_plan)
{
    FDEEntryMap::Entry fde_entry;

    // Make sure that the Address we're searching for is the same object file
    // as this DWARFCallFrameInfo, we only store File offsets in m_fde_index.
    ModuleSP module_sp = addr.GetModule();
    if (module_sp.get() == NULL || module_sp->GetObjectFile() == NULL || module_sp->GetObjectFile() != &m_objfile)
        return false;

    if (GetFDEEntryByFileAddress (addr.GetFileAddress(), fde_entry) == false)
        return false;
    return FDEToUnwindPlan (fde_entry.data, addr, unwind_plan);
}

bool
DWARFCallFrameInfo::GetAddressRange (Address addr, AddressRange &range)
{

    // Make sure that the Address we're searching for is the same object file
    // as this DWARFCallFrameInfo, we only store File offsets in m_fde_index.
    ModuleSP module_sp = addr.GetModule();
    if (module_sp.get() == NULL || module_sp->GetObjectFile() == NULL || module_sp->GetObjectFile() != &m_objfile)
        return false;

    if (m_section_sp.get() == NULL || m_section_sp->IsEncrypted())
        return false;
    GetFDEIndex();
    FDEEntryMap::Entry *fde_entry = m_fde_index.FindEntryThatContains (addr.GetFileAddress());
    if (!fde_entry)
        return false;

    range = AddressRange(fde_entry->base, fde_entry->size, m_objfile.GetSectionList());
    return true;
}

bool
DWARFCallFrameInfo::GetFDEEntryByFileAddress (addr_t file_addr, FDEEntryMap::Entry &fde_entry)
{
    if (m_section_sp.get() == NULL || m_section_sp->IsEncrypted())
        return false;

    GetFDEIndex();

    if (m_fde_index.IsEmpty())
        return false;

    FDEEntryMap::Entry *fde = m_fde_index.FindEntryThatContains (file_addr);

    if (fde == NULL)
        return false;

    fde_entry = *fde;
    return true;
}

void
DWARFCallFrameInfo::GetFunctionAddressAndSizeVector (FunctionAddressAndSizeVector &function_info)
{
    GetFDEIndex();
    const size_t count = m_fde_index.GetSize();
    function_info.Clear();
    if (count > 0)
        function_info.Reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        const FDEEntryMap::Entry *func_offset_data_entry = m_fde_index.GetEntryAtIndex (i);
        if (func_offset_data_entry)
        {
            FunctionAddressAndSizeVector::Entry function_offset_entry (func_offset_data_entry->base, func_offset_data_entry->size);
            function_info.Append (function_offset_entry);
        }
    }
}

const DWARFCallFrameInfo::CIE*
DWARFCallFrameInfo::GetCIE(dw_offset_t cie_offset)
{
    cie_map_t::iterator pos = m_cie_map.find(cie_offset);

    if (pos != m_cie_map.end())
    {
        // Parse and cache the CIE
        if (pos->second.get() == NULL)
            pos->second = ParseCIE (cie_offset);

        return pos->second.get();
    }
    return NULL;
}

DWARFCallFrameInfo::CIESP
DWARFCallFrameInfo::ParseCIE (const dw_offset_t cie_offset)
{
    CIESP cie_sp(new CIE(cie_offset));
    lldb::offset_t offset = cie_offset;
    if (m_cfi_data_initialized == false)
        GetCFIData();
    const uint32_t length = m_cfi_data.GetU32(&offset);
    const dw_offset_t cie_id = m_cfi_data.GetU32(&offset);
    const dw_offset_t end_offset = cie_offset + length + 4;
    if (length > 0 && ((!m_is_eh_frame && cie_id == UINT32_MAX) || (m_is_eh_frame && cie_id == 0ul)))
    {
        size_t i;
        //    cie.offset = cie_offset;
        //    cie.length = length;
        //    cie.cieID = cieID;
        cie_sp->ptr_encoding = DW_EH_PE_absptr;
        cie_sp->version = m_cfi_data.GetU8(&offset);

        for (i=0; i<CFI_AUG_MAX_SIZE; ++i)
        {
            cie_sp->augmentation[i] = m_cfi_data.GetU8(&offset);
            if (cie_sp->augmentation[i] == '\0')
            {
                // Zero out remaining bytes in augmentation string
                for (size_t j = i+1; j<CFI_AUG_MAX_SIZE; ++j)
                    cie_sp->augmentation[j] = '\0';

                break;
            }
        }

        if (i == CFI_AUG_MAX_SIZE && cie_sp->augmentation[CFI_AUG_MAX_SIZE-1] != '\0')
        {
            Host::SystemLog (Host::eSystemLogError, "CIE parse error: CIE augmentation string was too large for the fixed sized buffer of %d bytes.\n", CFI_AUG_MAX_SIZE);
            return cie_sp;
        }
        cie_sp->code_align = (uint32_t)m_cfi_data.GetULEB128(&offset);
        cie_sp->data_align = (int32_t)m_cfi_data.GetSLEB128(&offset);
        cie_sp->return_addr_reg_num = m_cfi_data.GetU8(&offset);

        if (cie_sp->augmentation[0])
        {
            // Get the length of the eh_frame augmentation data
            // which starts with a ULEB128 length in bytes
            const size_t aug_data_len = (size_t)m_cfi_data.GetULEB128(&offset);
            const size_t aug_data_end = offset + aug_data_len;
            const size_t aug_str_len = strlen(cie_sp->augmentation);
            // A 'z' may be present as the first character of the string.
            // If present, the Augmentation Data field shall be present.
            // The contents of the Augmentation Data shall be intepreted
            // according to other characters in the Augmentation String.
            if (cie_sp->augmentation[0] == 'z')
            {
                // Extract the Augmentation Data
                size_t aug_str_idx = 0;
                for (aug_str_idx = 1; aug_str_idx < aug_str_len; aug_str_idx++)
                {
                    char aug = cie_sp->augmentation[aug_str_idx];
                    switch (aug)
                    {
                        case 'L':
                            // Indicates the presence of one argument in the
                            // Augmentation Data of the CIE, and a corresponding
                            // argument in the Augmentation Data of the FDE. The
                            // argument in the Augmentation Data of the CIE is
                            // 1-byte and represents the pointer encoding used
                            // for the argument in the Augmentation Data of the
                            // FDE, which is the address of a language-specific
                            // data area (LSDA). The size of the LSDA pointer is
                            // specified by the pointer encoding used.
                            m_cfi_data.GetU8(&offset);
                            break;

                        case 'P':
                            // Indicates the presence of two arguments in the
                            // Augmentation Data of the cie_sp-> The first argument
                            // is 1-byte and represents the pointer encoding
                            // used for the second argument, which is the
                            // address of a personality routine handler. The
                            // size of the personality routine pointer is
                            // specified by the pointer encoding used.
                        {
                            uint8_t arg_ptr_encoding = m_cfi_data.GetU8(&offset);
                            m_cfi_data.GetGNUEHPointer(&offset, arg_ptr_encoding, LLDB_INVALID_ADDRESS, LLDB_INVALID_ADDRESS, LLDB_INVALID_ADDRESS);
                        }
                            break;

                        case 'R':
                            // A 'R' may be present at any position after the
                            // first character of the string. The Augmentation
                            // Data shall include a 1 byte argument that
                            // represents the pointer encoding for the address
                            // pointers used in the FDE.
                            cie_sp->ptr_encoding = m_cfi_data.GetU8(&offset);
                            break;
                    }
                }
            }
            else if (strcmp(cie_sp->augmentation, "eh") == 0)
            {
                // If the Augmentation string has the value "eh", then
                // the EH Data field shall be present
            }

            // Set the offset to be the end of the augmentation data just in case
            // we didn't understand any of the data.
            offset = (uint32_t)aug_data_end;
        }

        if (end_offset > offset)
        {
            cie_sp->inst_offset = offset;
            cie_sp->inst_length = end_offset - offset;
        }
        while (offset < end_offset)
        {
            uint8_t inst = m_cfi_data.GetU8(&offset);
            uint8_t primary_opcode  = inst & 0xC0;
            uint8_t extended_opcode = inst & 0x3F;

            if (extended_opcode == DW_CFA_def_cfa)
            {
                // Takes two unsigned LEB128 operands representing a register
                // number and a (non-factored) offset. The required action
                // is to define the current CFA rule to use the provided
                // register and offset.
                uint32_t reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                int op_offset = (int32_t)m_cfi_data.GetULEB128(&offset);
                cie_sp->initial_row.SetCFARegister (reg_num);
                cie_sp->initial_row.SetCFAOffset (op_offset);
                continue;
            }
            if (primary_opcode == DW_CFA_offset)
            {   
                // 0x80 - high 2 bits are 0x2, lower 6 bits are register.
                // Takes two arguments: an unsigned LEB128 constant representing a
                // factored offset and a register number. The required action is to
                // change the rule for the register indicated by the register number
                // to be an offset(N) rule with a value of
                // (N = factored offset * data_align).
                uint32_t reg_num = extended_opcode;
                int op_offset = (int32_t)m_cfi_data.GetULEB128(&offset) * cie_sp->data_align;
                UnwindPlan::Row::RegisterLocation reg_location;
                reg_location.SetAtCFAPlusOffset(op_offset);
                cie_sp->initial_row.SetRegisterInfo (reg_num, reg_location);
                continue;
            }
            if (extended_opcode == DW_CFA_nop)
            {
                continue;
            }
            break;  // Stop if we hit an unrecognized opcode
        }
    }

    return cie_sp;
}

void
DWARFCallFrameInfo::GetCFIData()
{
    if (m_cfi_data_initialized == false)
    {
        Log *log(GetLogIfAllCategoriesSet (LIBLLDB_LOG_UNWIND));
        if (log)
            m_objfile.GetModule()->LogMessage(log, "Reading EH frame info");
        m_objfile.ReadSectionData (m_section_sp.get(), m_cfi_data);
        m_cfi_data_initialized = true;
    }
}
// Scan through the eh_frame or debug_frame section looking for FDEs and noting the start/end addresses
// of the functions and a pointer back to the function's FDE for later expansion.
// Internalize CIEs as we come across them.

void
DWARFCallFrameInfo::GetFDEIndex ()
{
    if (m_section_sp.get() == NULL || m_section_sp->IsEncrypted())
        return;
    
    if (m_fde_index_initialized)
        return;
    
    Mutex::Locker locker(m_fde_index_mutex);
    
    if (m_fde_index_initialized) // if two threads hit the locker
        return;

    Timer scoped_timer (__PRETTY_FUNCTION__, "%s - %s", __PRETTY_FUNCTION__, m_objfile.GetFileSpec().GetFilename().AsCString(""));

    lldb::offset_t offset = 0;
    if (m_cfi_data_initialized == false)
        GetCFIData();
    while (m_cfi_data.ValidOffsetForDataOfSize (offset, 8))
    {
        const dw_offset_t current_entry = offset;
        uint32_t len = m_cfi_data.GetU32 (&offset);
        dw_offset_t next_entry = current_entry + len + 4;
        dw_offset_t cie_id = m_cfi_data.GetU32 (&offset);

        if (cie_id == 0 || cie_id == UINT32_MAX)
        {
            m_cie_map[current_entry] = ParseCIE (current_entry);
            offset = next_entry;
            continue;
        }

        const dw_offset_t cie_offset = current_entry + 4 - cie_id;
        const CIE *cie = GetCIE (cie_offset);
        if (cie)
        {
            const lldb::addr_t pc_rel_addr = m_section_sp->GetFileAddress();
            const lldb::addr_t text_addr = LLDB_INVALID_ADDRESS;
            const lldb::addr_t data_addr = LLDB_INVALID_ADDRESS;

            lldb::addr_t addr = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding, pc_rel_addr, text_addr, data_addr);
            lldb::addr_t length = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding & DW_EH_PE_MASK_ENCODING, pc_rel_addr, text_addr, data_addr);
            FDEEntryMap::Entry fde (addr, length, current_entry);
            m_fde_index.Append(fde);
        }
        else
        {
            Host::SystemLog (Host::eSystemLogError, 
                             "error: unable to find CIE at 0x%8.8x for cie_id = 0x%8.8x for entry at 0x%8.8x.\n", 
                             cie_offset,
                             cie_id,
                             current_entry);
        }
        offset = next_entry;
    }
    m_fde_index.Sort();
    m_fde_index_initialized = true;
}

bool
DWARFCallFrameInfo::FDEToUnwindPlan (dw_offset_t dwarf_offset, Address startaddr, UnwindPlan& unwind_plan)
{
    lldb::offset_t offset = dwarf_offset;
    lldb::offset_t current_entry = offset;

    if (m_section_sp.get() == NULL || m_section_sp->IsEncrypted())
        return false;

    if (m_cfi_data_initialized == false)
        GetCFIData();

    uint32_t length = m_cfi_data.GetU32 (&offset);
    dw_offset_t cie_offset = m_cfi_data.GetU32 (&offset);

    assert (cie_offset != 0 && cie_offset != UINT32_MAX);

    // Translate the CIE_id from the eh_frame format, which
    // is relative to the FDE offset, into a __eh_frame section
    // offset
    if (m_is_eh_frame)
    {
        unwind_plan.SetSourceName ("eh_frame CFI");
        cie_offset = current_entry + 4 - cie_offset;
        unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    }
    else
    {
        unwind_plan.SetSourceName ("DWARF CFI");
        // In theory the debug_frame info should be valid at all call sites
        // ("asynchronous unwind info" as it is sometimes called) but in practice
        // gcc et al all emit call frame info for the prologue and call sites, but
        // not for the epilogue or all the other locations during the function reliably.
        unwind_plan.SetUnwindPlanValidAtAllInstructions (eLazyBoolNo);
    }
    unwind_plan.SetSourcedFromCompiler (eLazyBoolYes);

    const CIE *cie = GetCIE (cie_offset);
    assert (cie != NULL);

    const dw_offset_t end_offset = current_entry + length + 4;

    const lldb::addr_t pc_rel_addr = m_section_sp->GetFileAddress();
    const lldb::addr_t text_addr = LLDB_INVALID_ADDRESS;
    const lldb::addr_t data_addr = LLDB_INVALID_ADDRESS;
    lldb::addr_t range_base = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding, pc_rel_addr, text_addr, data_addr);
    lldb::addr_t range_len = m_cfi_data.GetGNUEHPointer(&offset, cie->ptr_encoding & DW_EH_PE_MASK_ENCODING, pc_rel_addr, text_addr, data_addr);
    AddressRange range (range_base, m_objfile.GetAddressByteSize(), m_objfile.GetSectionList());
    range.SetByteSize (range_len);

    if (cie->augmentation[0] == 'z')
    {
        uint32_t aug_data_len = (uint32_t)m_cfi_data.GetULEB128(&offset);
        offset += aug_data_len;
    }

    uint32_t reg_num = 0;
    int32_t op_offset = 0;
    uint32_t code_align = cie->code_align;
    int32_t data_align = cie->data_align;

    unwind_plan.SetPlanValidAddressRange (range);
    UnwindPlan::Row *cie_initial_row = new UnwindPlan::Row;
    *cie_initial_row = cie->initial_row;
    UnwindPlan::RowSP row(cie_initial_row);

    unwind_plan.SetRegisterKind (m_reg_kind);
    unwind_plan.SetReturnAddressRegister (cie->return_addr_reg_num);

    UnwindPlan::Row::RegisterLocation reg_location;
    while (m_cfi_data.ValidOffset(offset) && offset < end_offset)
    {
        uint8_t inst = m_cfi_data.GetU8(&offset);
        uint8_t primary_opcode  = inst & 0xC0;
        uint8_t extended_opcode = inst & 0x3F;

        if (primary_opcode)
        {
            switch (primary_opcode)
            {
                case DW_CFA_advance_loc :   // (Row Creation Instruction)
                    {   // 0x40 - high 2 bits are 0x1, lower 6 bits are delta
                        // takes a single argument that represents a constant delta. The
                        // required action is to create a new table row with a location
                        // value that is computed by taking the current entry's location
                        // value and adding (delta * code_align). All other
                        // values in the new row are initially identical to the current row.
                        unwind_plan.AppendRow(row);
                        UnwindPlan::Row *newrow = new UnwindPlan::Row;
                        *newrow = *row.get();
                        row.reset (newrow);
                        row->SlideOffset(extended_opcode * code_align);
                    }
                    break;

                case DW_CFA_offset      :
                    {   // 0x80 - high 2 bits are 0x2, lower 6 bits are register
                        // takes two arguments: an unsigned LEB128 constant representing a
                        // factored offset and a register number. The required action is to
                        // change the rule for the register indicated by the register number
                        // to be an offset(N) rule with a value of
                        // (N = factored offset * data_align).
                        reg_num = extended_opcode;
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset) * data_align;
                        reg_location.SetAtCFAPlusOffset(op_offset);
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_restore     :
                    {   // 0xC0 - high 2 bits are 0x3, lower 6 bits are register
                        // takes a single argument that represents a register number. The
                        // required action is to change the rule for the indicated register
                        // to the rule assigned it by the initial_instructions in the CIE.
                        reg_num = extended_opcode;
                        // We only keep enough register locations around to
                        // unwind what is in our thread, and these are organized
                        // by the register index in that state, so we need to convert our
                        // GCC register number from the EH frame info, to a register index

                        if (unwind_plan.IsValidRowIndex(0) && unwind_plan.GetRowAtIndex(0)->GetRegisterInfo(reg_num, reg_location))
                            row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;
            }
        }
        else
        {
            switch (extended_opcode)
            {
                case DW_CFA_nop                 : // 0x0
                    break;

                case DW_CFA_set_loc             : // 0x1 (Row Creation Instruction)
                    {
                        // DW_CFA_set_loc takes a single argument that represents an address.
                        // The required action is to create a new table row using the
                        // specified address as the location. All other values in the new row
                        // are initially identical to the current row. The new location value
                        // should always be greater than the current one.
                        unwind_plan.AppendRow(row);
                        UnwindPlan::Row *newrow = new UnwindPlan::Row;
                        *newrow = *row.get();
                        row.reset (newrow);
                        row->SetOffset(m_cfi_data.GetPointer(&offset) - startaddr.GetFileAddress());
                    }
                    break;

                case DW_CFA_advance_loc1        : // 0x2 (Row Creation Instruction)
                    {
                        // takes a single uword argument that represents a constant delta.
                        // This instruction is identical to DW_CFA_advance_loc except for the
                        // encoding and size of the delta argument.
                        unwind_plan.AppendRow(row);
                        UnwindPlan::Row *newrow = new UnwindPlan::Row;
                        *newrow = *row.get();
                        row.reset (newrow);
                        row->SlideOffset (m_cfi_data.GetU8(&offset) * code_align);
                    }
                    break;

                case DW_CFA_advance_loc2        : // 0x3 (Row Creation Instruction)
                    {
                        // takes a single uword argument that represents a constant delta.
                        // This instruction is identical to DW_CFA_advance_loc except for the
                        // encoding and size of the delta argument.
                        unwind_plan.AppendRow(row);
                        UnwindPlan::Row *newrow = new UnwindPlan::Row;
                        *newrow = *row.get();
                        row.reset (newrow);
                        row->SlideOffset (m_cfi_data.GetU16(&offset) * code_align);
                    }
                    break;

                case DW_CFA_advance_loc4        : // 0x4 (Row Creation Instruction)
                    {
                        // takes a single uword argument that represents a constant delta.
                        // This instruction is identical to DW_CFA_advance_loc except for the
                        // encoding and size of the delta argument.
                        unwind_plan.AppendRow(row);
                        UnwindPlan::Row *newrow = new UnwindPlan::Row;
                        *newrow = *row.get();
                        row.reset (newrow);
                        row->SlideOffset (m_cfi_data.GetU32(&offset) * code_align);
                    }
                    break;

                case DW_CFA_offset_extended     : // 0x5
                    {
                        // takes two unsigned LEB128 arguments representing a register number
                        // and a factored offset. This instruction is identical to DW_CFA_offset
                        // except for the encoding and size of the register argument.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset) * data_align;
                        reg_location.SetAtCFAPlusOffset(op_offset);
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_restore_extended    : // 0x6
                    {
                        // takes a single unsigned LEB128 argument that represents a register
                        // number. This instruction is identical to DW_CFA_restore except for
                        // the encoding and size of the register argument.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        if (unwind_plan.IsValidRowIndex(0) && unwind_plan.GetRowAtIndex(0)->GetRegisterInfo(reg_num, reg_location))
                            row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_undefined           : // 0x7
                    {
                        // takes a single unsigned LEB128 argument that represents a register
                        // number. The required action is to set the rule for the specified
                        // register to undefined.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        reg_location.SetUndefined();
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_same_value          : // 0x8
                    {
                        // takes a single unsigned LEB128 argument that represents a register
                        // number. The required action is to set the rule for the specified
                        // register to same value.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        reg_location.SetSame();
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_register            : // 0x9
                    {
                        // takes two unsigned LEB128 arguments representing register numbers.
                        // The required action is to set the rule for the first register to be
                        // the second register.

                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        uint32_t other_reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        reg_location.SetInRegister(other_reg_num);
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_remember_state      : // 0xA
                    {
                        // These instructions define a stack of information. Encountering the
                        // DW_CFA_remember_state instruction means to save the rules for every
                        // register on the current row on the stack. Encountering the
                        // DW_CFA_restore_state instruction means to pop the set of rules off
                        // the stack and place them in the current row. (This operation is
                        // useful for compilers that move epilogue code into the body of a
                        // function.)
                        unwind_plan.AppendRow (row);
                        UnwindPlan::Row *newrow = new UnwindPlan::Row;
                        *newrow = *row.get();
                        row.reset (newrow);
                    }
                    break;

                case DW_CFA_restore_state       : // 0xB
                    // These instructions define a stack of information. Encountering the
                    // DW_CFA_remember_state instruction means to save the rules for every
                    // register on the current row on the stack. Encountering the
                    // DW_CFA_restore_state instruction means to pop the set of rules off
                    // the stack and place them in the current row. (This operation is
                    // useful for compilers that move epilogue code into the body of a
                    // function.)
                    {
                        row = unwind_plan.GetRowAtIndex(unwind_plan.GetRowCount() - 1);
                    }
                    break;

                case DW_CFA_def_cfa             : // 0xC    (CFA Definition Instruction)
                    {
                        // Takes two unsigned LEB128 operands representing a register
                        // number and a (non-factored) offset. The required action
                        // is to define the current CFA rule to use the provided
                        // register and offset.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset);
                        row->SetCFARegister (reg_num);
                        row->SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_def_cfa_register    : // 0xD    (CFA Definition Instruction)
                    {
                        // takes a single unsigned LEB128 argument representing a register
                        // number. The required action is to define the current CFA rule to
                        // use the provided register (but to keep the old offset).
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        row->SetCFARegister (reg_num);
                    }
                    break;

                case DW_CFA_def_cfa_offset      : // 0xE    (CFA Definition Instruction)
                    {
                        // Takes a single unsigned LEB128 operand representing a
                        // (non-factored) offset. The required action is to define
                        // the current CFA rule to use the provided offset (but
                        // to keep the old register).
                        op_offset = (int32_t)m_cfi_data.GetULEB128(&offset);
                        row->SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_def_cfa_expression  : // 0xF    (CFA Definition Instruction)
                    {
                        size_t block_len = (size_t)m_cfi_data.GetULEB128(&offset);
                        offset += (uint32_t)block_len;
                    }
                    break;

                case DW_CFA_expression          : // 0x10
                    {
                        // Takes two operands: an unsigned LEB128 value representing
                        // a register number, and a DW_FORM_block value representing a DWARF
                        // expression. The required action is to change the rule for the
                        // register indicated by the register number to be an expression(E)
                        // rule where E is the DWARF expression. That is, the DWARF
                        // expression computes the address. The value of the CFA is
                        // pushed on the DWARF evaluation stack prior to execution of
                        // the DWARF expression.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        uint32_t block_len = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        const uint8_t *block_data = (uint8_t *)m_cfi_data.GetData(&offset, block_len);

                        reg_location.SetAtDWARFExpression(block_data, block_len);
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_offset_extended_sf  : // 0x11
                    {
                        // takes two operands: an unsigned LEB128 value representing a
                        // register number and a signed LEB128 factored offset. This
                        // instruction is identical to DW_CFA_offset_extended except
                        //that the second operand is signed and factored.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetSLEB128(&offset) * data_align;
                        reg_location.SetAtCFAPlusOffset(op_offset);
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_def_cfa_sf          : // 0x12   (CFA Definition Instruction)
                    {
                        // Takes two operands: an unsigned LEB128 value representing
                        // a register number and a signed LEB128 factored offset.
                        // This instruction is identical to DW_CFA_def_cfa except
                        // that the second operand is signed and factored.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        op_offset = (int32_t)m_cfi_data.GetSLEB128(&offset) * data_align;
                        row->SetCFARegister (reg_num);
                        row->SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_def_cfa_offset_sf   : // 0x13   (CFA Definition Instruction)
                    {
                        // takes a signed LEB128 operand representing a factored
                        // offset. This instruction is identical to  DW_CFA_def_cfa_offset
                        // except that the operand is signed and factored.
                        op_offset = (int32_t)m_cfi_data.GetSLEB128(&offset) * data_align;
                        row->SetCFAOffset (op_offset);
                    }
                    break;

                case DW_CFA_val_expression      :   // 0x16
                    {
                        // takes two operands: an unsigned LEB128 value representing a register
                        // number, and a DW_FORM_block value representing a DWARF expression.
                        // The required action is to change the rule for the register indicated
                        // by the register number to be a val_expression(E) rule where E is the
                        // DWARF expression. That is, the DWARF expression computes the value of
                        // the given register. The value of the CFA is pushed on the DWARF
                        // evaluation stack prior to execution of the DWARF expression.
                        reg_num = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        uint32_t block_len = (uint32_t)m_cfi_data.GetULEB128(&offset);
                        const uint8_t* block_data = (uint8_t*)m_cfi_data.GetData(&offset, block_len);
//#if defined(__i386__) || defined(__x86_64__)
//                      // The EH frame info for EIP and RIP contains code that looks for traps to
//                      // be a specific type and increments the PC.
//                      // For i386:
//                      // DW_CFA_val_expression where:
//                      // eip = DW_OP_breg6(+28), DW_OP_deref, DW_OP_dup, DW_OP_plus_uconst(0x34),
//                      //       DW_OP_deref, DW_OP_swap, DW_OP_plus_uconst(0), DW_OP_deref,
//                      //       DW_OP_dup, DW_OP_lit3, DW_OP_ne, DW_OP_swap, DW_OP_lit4, DW_OP_ne,
//                      //       DW_OP_and, DW_OP_plus
//                      // This basically does a:
//                      // eip = ucontenxt.mcontext32->gpr.eip;
//                      // if (ucontenxt.mcontext32->exc.trapno != 3 && ucontenxt.mcontext32->exc.trapno != 4)
//                      //   eip++;
//                      //
//                      // For x86_64:
//                      // DW_CFA_val_expression where:
//                      // rip =  DW_OP_breg3(+48), DW_OP_deref, DW_OP_dup, DW_OP_plus_uconst(0x90), DW_OP_deref,
//                      //          DW_OP_swap, DW_OP_plus_uconst(0), DW_OP_deref_size(4), DW_OP_dup, DW_OP_lit3,
//                      //          DW_OP_ne, DW_OP_swap, DW_OP_lit4, DW_OP_ne, DW_OP_and, DW_OP_plus
//                      // This basically does a:
//                      // rip = ucontenxt.mcontext64->gpr.rip;
//                      // if (ucontenxt.mcontext64->exc.trapno != 3 && ucontenxt.mcontext64->exc.trapno != 4)
//                      //   rip++;
//                      // The trap comparisons and increments are not needed as it hoses up the unwound PC which
//                      // is expected to point at least past the instruction that causes the fault/trap. So we
//                      // take it out by trimming the expression right at the first "DW_OP_swap" opcodes
//                      if (block_data != NULL && thread->GetPCRegNum(Thread::GCC) == reg_num)
//                      {
//                          if (thread->Is64Bit())
//                          {
//                              if (block_len > 9 && block_data[8] == DW_OP_swap && block_data[9] == DW_OP_plus_uconst)
//                                  block_len = 8;
//                          }
//                          else
//                          {
//                              if (block_len > 8 && block_data[7] == DW_OP_swap && block_data[8] == DW_OP_plus_uconst)
//                                  block_len = 7;
//                          }
//                      }
//#endif
                        reg_location.SetIsDWARFExpression(block_data, block_len);
                        row->SetRegisterInfo (reg_num, reg_location);
                    }
                    break;

                case DW_CFA_val_offset          :   // 0x14
                case DW_CFA_val_offset_sf       :   // 0x15
                default:
                    break;
            }
        }
    }
    unwind_plan.AppendRow(row);

    return true;
}
