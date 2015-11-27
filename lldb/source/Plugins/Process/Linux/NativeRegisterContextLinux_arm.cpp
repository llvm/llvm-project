//===-- NativeRegisterContextLinux_arm.cpp --------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#if defined(__arm__)

#include "NativeRegisterContextLinux_arm.h"

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Core/Log.h"
#include "lldb/Core/RegisterValue.h"

#include "Plugins/Process/Utility/RegisterContextLinux_arm.h"

#define REG_CONTEXT_SIZE (GetGPRSize() + sizeof (m_fpr))

#ifndef PTRACE_GETVFPREGS
  #define PTRACE_GETVFPREGS 27
  #define PTRACE_SETVFPREGS 28
#endif
#ifndef PTRACE_GETHBPREGS
  #define PTRACE_GETHBPREGS 29
  #define PTRACE_SETHBPREGS 30
#endif
#if !defined(PTRACE_TYPE_ARG3)
  #define PTRACE_TYPE_ARG3 void *
#endif
#if !defined(PTRACE_TYPE_ARG4)
  #define PTRACE_TYPE_ARG4 void *
#endif

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::process_linux;

// arm general purpose registers.
static const uint32_t g_gpr_regnums_arm[] =
{
    gpr_r0_arm,
    gpr_r1_arm,
    gpr_r2_arm,
    gpr_r3_arm,
    gpr_r4_arm,
    gpr_r5_arm,
    gpr_r6_arm,
    gpr_r7_arm,
    gpr_r8_arm,
    gpr_r9_arm,
    gpr_r10_arm,
    gpr_r11_arm,
    gpr_r12_arm,
    gpr_sp_arm,
    gpr_lr_arm,
    gpr_pc_arm,
    gpr_cpsr_arm,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_gpr_regnums_arm / sizeof g_gpr_regnums_arm[0]) - 1) == k_num_gpr_registers_arm, \
              "g_gpr_regnums_arm has wrong number of register infos");

// arm floating point registers.
static const uint32_t g_fpu_regnums_arm[] =
{
    fpu_s0_arm,
    fpu_s1_arm,
    fpu_s2_arm,
    fpu_s3_arm,
    fpu_s4_arm,
    fpu_s5_arm,
    fpu_s6_arm,
    fpu_s7_arm,
    fpu_s8_arm,
    fpu_s9_arm,
    fpu_s10_arm,
    fpu_s11_arm,
    fpu_s12_arm,
    fpu_s13_arm,
    fpu_s14_arm,
    fpu_s15_arm,
    fpu_s16_arm,
    fpu_s17_arm,
    fpu_s18_arm,
    fpu_s19_arm,
    fpu_s20_arm,
    fpu_s21_arm,
    fpu_s22_arm,
    fpu_s23_arm,
    fpu_s24_arm,
    fpu_s25_arm,
    fpu_s26_arm,
    fpu_s27_arm,
    fpu_s28_arm,
    fpu_s29_arm,
    fpu_s30_arm,
    fpu_s31_arm,
    fpu_fpscr_arm,
    fpu_d0_arm,
    fpu_d1_arm,
    fpu_d2_arm,
    fpu_d3_arm,
    fpu_d4_arm,
    fpu_d5_arm,
    fpu_d6_arm,
    fpu_d7_arm,
    fpu_d8_arm,
    fpu_d9_arm,
    fpu_d10_arm,
    fpu_d11_arm,
    fpu_d12_arm,
    fpu_d13_arm,
    fpu_d14_arm,
    fpu_d15_arm,
    fpu_d16_arm,
    fpu_d17_arm,
    fpu_d18_arm,
    fpu_d19_arm,
    fpu_d20_arm,
    fpu_d21_arm,
    fpu_d22_arm,
    fpu_d23_arm,
    fpu_d24_arm,
    fpu_d25_arm,
    fpu_d26_arm,
    fpu_d27_arm,
    fpu_d28_arm,
    fpu_d29_arm,
    fpu_d30_arm,
    fpu_d31_arm,
    fpu_q0_arm,
    fpu_q1_arm,
    fpu_q2_arm,
    fpu_q3_arm,
    fpu_q4_arm,
    fpu_q5_arm,
    fpu_q6_arm,
    fpu_q7_arm,
    fpu_q8_arm,
    fpu_q9_arm,
    fpu_q10_arm,
    fpu_q11_arm,
    fpu_q12_arm,
    fpu_q13_arm,
    fpu_q14_arm,
    fpu_q15_arm,
    LLDB_INVALID_REGNUM // register sets need to end with this flag
};
static_assert(((sizeof g_fpu_regnums_arm / sizeof g_fpu_regnums_arm[0]) - 1) == k_num_fpr_registers_arm, \
              "g_fpu_regnums_arm has wrong number of register infos");

namespace {
    // Number of register sets provided by this context.
    enum
    {
        k_num_register_sets = 2
    };
}

// Register sets for arm.
static const RegisterSet
g_reg_sets_arm[k_num_register_sets] =
{
    { "General Purpose Registers",  "gpr", k_num_gpr_registers_arm, g_gpr_regnums_arm },
    { "Floating Point Registers",   "fpu", k_num_fpr_registers_arm, g_fpu_regnums_arm }
};

NativeRegisterContextLinux*
NativeRegisterContextLinux::CreateHostNativeRegisterContextLinux(const ArchSpec& target_arch,
                                                                 NativeThreadProtocol &native_thread,
                                                                 uint32_t concrete_frame_idx)
{
    return new NativeRegisterContextLinux_arm(target_arch, native_thread, concrete_frame_idx);
}

NativeRegisterContextLinux_arm::NativeRegisterContextLinux_arm (const ArchSpec& target_arch,
                                                                NativeThreadProtocol &native_thread,
                                                                uint32_t concrete_frame_idx) :
    NativeRegisterContextLinux (native_thread, concrete_frame_idx, new RegisterContextLinux_arm(target_arch))
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::arm:
            m_reg_info.num_registers     = k_num_registers_arm;
            m_reg_info.num_gpr_registers = k_num_gpr_registers_arm;
            m_reg_info.num_fpr_registers = k_num_fpr_registers_arm;
            m_reg_info.last_gpr          = k_last_gpr_arm;
            m_reg_info.first_fpr         = k_first_fpr_arm;
            m_reg_info.last_fpr          = k_last_fpr_arm;
            m_reg_info.first_fpr_v       = fpu_s0_arm;
            m_reg_info.last_fpr_v        = fpu_s31_arm;
            m_reg_info.gpr_flags         = gpr_cpsr_arm;
            break;
        default:
            assert(false && "Unhandled target architecture.");
            break;
    }

    ::memset(&m_fpr, 0, sizeof (m_fpr));
    ::memset(&m_gpr_arm, 0, sizeof (m_gpr_arm));
    ::memset(&m_hwp_regs, 0, sizeof (m_hwp_regs));

    // 16 is just a maximum value, query hardware for actual watchpoint count
    m_max_hwp_supported = 16;
    m_max_hbp_supported = 16;
    m_refresh_hwdebug_info = true;
}

uint32_t
NativeRegisterContextLinux_arm::GetRegisterSetCount () const
{
    return k_num_register_sets;
}

uint32_t
NativeRegisterContextLinux_arm::GetUserRegisterCount() const
{
    uint32_t count = 0;
    for (uint32_t set_index = 0; set_index < k_num_register_sets; ++set_index)
        count += g_reg_sets_arm[set_index].num_registers;
    return count;
}

const RegisterSet *
NativeRegisterContextLinux_arm::GetRegisterSet (uint32_t set_index) const
{
    if (set_index < k_num_register_sets)
        return &g_reg_sets_arm[set_index];

    return nullptr;
}

Error
NativeRegisterContextLinux_arm::ReadRegister (const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    Error error;

    if (!reg_info)
    {
        error.SetErrorString ("reg_info NULL");
        return error;
    }

    const uint32_t reg = reg_info->kinds[lldb::eRegisterKindLLDB];

    if (IsFPR(reg))
    {
        error = ReadFPR();
        if (error.Fail())
            return error;
    }
    else
    {
        uint32_t full_reg = reg;
        bool is_subreg = reg_info->invalidate_regs && (reg_info->invalidate_regs[0] != LLDB_INVALID_REGNUM);

        if (is_subreg)
        {
            // Read the full aligned 64-bit register.
            full_reg = reg_info->invalidate_regs[0];
        }

        error = ReadRegisterRaw(full_reg, reg_value);

        if (error.Success ())
        {
            // If our read was not aligned (for ah,bh,ch,dh), shift our returned value one byte to the right.
            if (is_subreg && (reg_info->byte_offset & 0x1))
                reg_value.SetUInt64(reg_value.GetAsUInt64() >> 8);

            // If our return byte size was greater than the return value reg size, then
            // use the type specified by reg_info rather than the uint64_t default
            if (reg_value.GetByteSize() > reg_info->byte_size)
                reg_value.SetType(reg_info);
        }
        return error;
    }

    // Get pointer to m_fpr variable and set the data from it.
    uint32_t fpr_offset = CalculateFprOffset(reg_info);
    assert (fpr_offset < sizeof m_fpr);
    uint8_t *src = (uint8_t *)&m_fpr + fpr_offset;
    switch (reg_info->byte_size)
    {
        case 2:
            reg_value.SetUInt16(*(uint16_t *)src);
            break;
        case 4:
            reg_value.SetUInt32(*(uint32_t *)src);
            break;
        case 8:
            reg_value.SetUInt64(*(uint64_t *)src);
            break;
        case 16:
            reg_value.SetBytes(src, 16, GetByteOrder());
            break;
        default:
            assert(false && "Unhandled data size.");
            error.SetErrorStringWithFormat ("unhandled byte size: %" PRIu32, reg_info->byte_size);
            break;
    }

    return error;
}

Error
NativeRegisterContextLinux_arm::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    if (!reg_info)
        return Error ("reg_info NULL");

    const uint32_t reg_index = reg_info->kinds[lldb::eRegisterKindLLDB];
    if (reg_index == LLDB_INVALID_REGNUM)
        return Error ("no lldb regnum for %s", reg_info && reg_info->name ? reg_info->name : "<unknown register>");

    if (IsGPR(reg_index))
        return WriteRegisterRaw(reg_index, reg_value);

    if (IsFPR(reg_index))
    {
        // Get pointer to m_fpr variable and set the data to it.
        uint32_t fpr_offset = CalculateFprOffset(reg_info);
        assert (fpr_offset < sizeof m_fpr);
        uint8_t *dst = (uint8_t *)&m_fpr + fpr_offset;
        switch (reg_info->byte_size)
        {
            case 2:
                *(uint16_t *)dst = reg_value.GetAsUInt16();
                break;
            case 4:
                *(uint32_t *)dst = reg_value.GetAsUInt32();
                break;
            case 8:
                *(uint64_t *)dst = reg_value.GetAsUInt64();
                break;
            default:
                assert(false && "Unhandled data size.");
                return Error ("unhandled register data size %" PRIu32, reg_info->byte_size);
        }

        Error error = WriteFPR();
        if (error.Fail())
            return error;

        return Error ();
    }

    return Error ("failed - register wasn't recognized to be a GPR or an FPR, write strategy unknown");
}

Error
NativeRegisterContextLinux_arm::ReadAllRegisterValues (lldb::DataBufferSP &data_sp)
{
    Error error;

    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (!data_sp)
        return Error ("failed to allocate DataBufferHeap instance of size %" PRIu64, (uint64_t)REG_CONTEXT_SIZE);

    error = ReadGPR();
    if (error.Fail())
        return error;

    error = ReadFPR();
    if (error.Fail())
        return error;

    uint8_t *dst = data_sp->GetBytes ();
    if (dst == nullptr)
    {
        error.SetErrorStringWithFormat ("DataBufferHeap instance of size %" PRIu64 " returned a null pointer", (uint64_t)REG_CONTEXT_SIZE);
        return error;
    }

    ::memcpy (dst, &m_gpr_arm, GetGPRSize());
    dst += GetGPRSize();
    ::memcpy (dst, &m_fpr, sizeof(m_fpr));

    return error;
}

Error
NativeRegisterContextLinux_arm::WriteAllRegisterValues (const lldb::DataBufferSP &data_sp)
{
    Error error;

    if (!data_sp)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s invalid data_sp provided", __FUNCTION__);
        return error;
    }

    if (data_sp->GetByteSize () != REG_CONTEXT_SIZE)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s data_sp contained mismatched data size, expected %" PRIu64 ", actual %" PRIu64, __FUNCTION__, (uint64_t)REG_CONTEXT_SIZE, data_sp->GetByteSize ());
        return error;
    }


    uint8_t *src = data_sp->GetBytes ();
    if (src == nullptr)
    {
        error.SetErrorStringWithFormat ("NativeRegisterContextLinux_x86_64::%s DataBuffer::GetBytes() returned a null pointer", __FUNCTION__);
        return error;
    }
    ::memcpy (&m_gpr_arm, src, GetRegisterInfoInterface ().GetGPRSize ());

    error = WriteGPR();
    if (error.Fail())
        return error;

    src += GetRegisterInfoInterface ().GetGPRSize ();
    ::memcpy (&m_fpr, src, sizeof(m_fpr));

    error = WriteFPR();
    if (error.Fail())
        return error;

    return error;
}

bool
NativeRegisterContextLinux_arm::IsGPR(unsigned reg) const
{
    return reg <= m_reg_info.last_gpr;   // GPR's come first.
}

bool
NativeRegisterContextLinux_arm::IsFPR(unsigned reg) const
{
    return (m_reg_info.first_fpr <= reg && reg <= m_reg_info.last_fpr);
}

uint32_t
NativeRegisterContextLinux_arm::SetHardwareBreakpoint (lldb::addr_t addr, size_t size)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    Error error;

    // Read hardware breakpoint and watchpoint information.
    error = ReadHardwareDebugInfo ();

    if (error.Fail())
        return LLDB_INVALID_INDEX32;

    uint32_t control_value = 0, bp_index = 0;

    // Check if size has a valid hardware breakpoint length.
    // Thumb instructions are 2-bytes but we have no way here to determine
    // if target address is a thumb or arm instruction.
    // TODO: Add support for setting thumb mode hardware breakpoints
    if (size != 4 && size != 2)
        return LLDB_INVALID_INDEX32;

    // Setup control value
    // Make the byte_mask into a valid Byte Address Select mask
    control_value = 0xfu << 5;

    // Enable this breakpoint and make it stop in privileged or user mode;
    control_value |= 7;

    // Make sure bits 1:0 are clear in our address
    // This should be different once we support thumb here.
    addr &= ~((lldb::addr_t)3);

    // Iterate over stored hardware breakpoints
    // Find a free bp_index or update reference count if duplicate.
    bp_index = LLDB_INVALID_INDEX32;

    for (uint32_t i = 0; i < m_max_hbp_supported; i++)
    {
        if ((m_hbr_regs[i].control & 1) == 0)
        {
            bp_index = i;  // Mark last free slot
        }
        else if (m_hbr_regs[i].address == addr && m_hbr_regs[i].control == control_value)
        {
            bp_index = i;  // Mark duplicate index
            break;  // Stop searching here
        }
    }

     if (bp_index == LLDB_INVALID_INDEX32)
         return LLDB_INVALID_INDEX32;

    // Add new or update existing breakpoint
    if ((m_hbr_regs[bp_index].control & 1) == 0)
    {
        m_hbr_regs[bp_index].address = addr;
        m_hbr_regs[bp_index].control = control_value;
        m_hbr_regs[bp_index].refcount = 1;

        // PTRACE call to set corresponding hardware breakpoint register.
        error = WriteHardwareDebugRegs(eDREGTypeBREAK, bp_index);

        if (error.Fail())
        {
            m_hbr_regs[bp_index].address = 0;
            m_hbr_regs[bp_index].control &= ~1;
            m_hbr_regs[bp_index].refcount = 0;

            return LLDB_INVALID_INDEX32;
        }
    }
    else
        m_hbr_regs[bp_index].refcount++;

    return bp_index;
}

bool
NativeRegisterContextLinux_arm::ClearHardwareBreakpoint (uint32_t hw_idx)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    Error error;

    // Read hardware breakpoint and watchpoint information.
    error = ReadHardwareDebugInfo ();

    if (error.Fail())
        return false;

    if (hw_idx >= m_max_hbp_supported)
        return false;

    // Update reference count if multiple references.
    if (m_hbr_regs[hw_idx].refcount > 1)
    {
        m_hbr_regs[hw_idx].refcount--;
        return true;
    }
    else if (m_hbr_regs[hw_idx].refcount == 1)
    {
        // Create a backup we can revert to in case of failure.
        lldb::addr_t tempAddr = m_hbr_regs[hw_idx].address;
        uint32_t tempControl = m_hbr_regs[hw_idx].control;
        uint32_t tempRefCount = m_hbr_regs[hw_idx].refcount;

        m_hbr_regs[hw_idx].control &= ~1;
        m_hbr_regs[hw_idx].address = 0;
        m_hbr_regs[hw_idx].refcount = 0;

        // PTRACE call to clear corresponding hardware breakpoint register.
        WriteHardwareDebugRegs(eDREGTypeBREAK, hw_idx);

        if (error.Fail())
        {
            m_hbr_regs[hw_idx].control = tempControl;
            m_hbr_regs[hw_idx].address = tempAddr;
            m_hbr_regs[hw_idx].refcount = tempRefCount;

            return false;
        }

        return true;
    }

    return false;
}

uint32_t
NativeRegisterContextLinux_arm::NumSupportedHardwareWatchpoints ()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    Error error;

    // Read hardware breakpoint and watchpoint information.
    error = ReadHardwareDebugInfo ();

    if (error.Fail())
        return LLDB_INVALID_INDEX32;

    return m_max_hwp_supported;
}

uint32_t
NativeRegisterContextLinux_arm::SetHardwareWatchpoint (lldb::addr_t addr, size_t size, uint32_t watch_flags)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);
    
    Error error;

    // Read hardware breakpoint and watchpoint information.
    error = ReadHardwareDebugInfo ();

    if (error.Fail())
        return LLDB_INVALID_INDEX32;
		
    uint32_t control_value = 0, wp_index = 0, addr_word_offset = 0, byte_mask = 0;

    // Check if we are setting watchpoint other than read/write/access
    // Also update watchpoint flag to match Arm write-read bit configuration.
    switch (watch_flags)
    {
        case 1:
            watch_flags = 2;
            break;
        case 2:
            watch_flags = 1;
            break;
        case 3:
            break;
        default:
            return LLDB_INVALID_INDEX32;
    }

    // Can't watch zero bytes
    // Can't watch more than 4 bytes per WVR/WCR pair

    if (size == 0 || size > 4)
        return LLDB_INVALID_INDEX32;

    // We can only watch up to four bytes that follow a 4 byte aligned address
    // per watchpoint register pair, so make sure we can properly encode this.
    addr_word_offset = addr % 4;
    byte_mask = ((1u << size) - 1u) << addr_word_offset;

    // Check if we need multiple watchpoint register
    if (byte_mask > 0xfu)
        return LLDB_INVALID_INDEX32;

    // Setup control value
    // Make the byte_mask into a valid Byte Address Select mask
    control_value = byte_mask << 5;

    //Turn on appropriate watchpoint flags read or write
    control_value |= (watch_flags << 3);

    // Enable this watchpoint and make it stop in privileged or user mode;
    control_value |= 7;

    // Make sure bits 1:0 are clear in our address
    addr &= ~((lldb::addr_t)3);

    // Iterate over stored watchpoints
    // Find a free wp_index or update reference count if duplicate.
    wp_index = LLDB_INVALID_INDEX32;
    for (uint32_t i = 0; i < m_max_hwp_supported; i++)
    {
        if ((m_hwp_regs[i].control & 1) == 0)
        {
            wp_index = i; // Mark last free slot
        }
        else if (m_hwp_regs[i].address == addr && m_hwp_regs[i].control == control_value)
        {
            wp_index = i; // Mark duplicate index
            break; // Stop searching here
        }
    }

     if (wp_index == LLDB_INVALID_INDEX32)
        return LLDB_INVALID_INDEX32;

    // Add new or update existing watchpoint
    if ((m_hwp_regs[wp_index].control & 1) == 0)
    {
        // Update watchpoint in local cache
        m_hwp_regs[wp_index].address = addr;
        m_hwp_regs[wp_index].control = control_value;
        m_hwp_regs[wp_index].refcount = 1;

        // PTRACE call to set corresponding watchpoint register.
        error = WriteHardwareDebugRegs(eDREGTypeWATCH, wp_index);

        if (error.Fail())
        {
            m_hwp_regs[wp_index].address = 0;
            m_hwp_regs[wp_index].control &= ~1;
            m_hwp_regs[wp_index].refcount = 0;

            return LLDB_INVALID_INDEX32;
        }
    }
    else
        m_hwp_regs[wp_index].refcount++;

    return wp_index;
}

bool
NativeRegisterContextLinux_arm::ClearHardwareWatchpoint (uint32_t wp_index)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    Error error;

    // Read hardware breakpoint and watchpoint information.
    error = ReadHardwareDebugInfo ();

    if (error.Fail())
        return false;

    if (wp_index >= m_max_hwp_supported)
        return false;

    // Update reference count if multiple references.
    if (m_hwp_regs[wp_index].refcount > 1)
    {
        m_hwp_regs[wp_index].refcount--;
        return true;
    }
    else if (m_hwp_regs[wp_index].refcount == 1)
    {
        // Create a backup we can revert to in case of failure.
        lldb::addr_t tempAddr = m_hwp_regs[wp_index].address;
        uint32_t tempControl = m_hwp_regs[wp_index].control;
        uint32_t tempRefCount = m_hwp_regs[wp_index].refcount;

        // Update watchpoint in local cache
        m_hwp_regs[wp_index].control &= ~1;
        m_hwp_regs[wp_index].address = 0;
        m_hwp_regs[wp_index].refcount = 0;

        // Ptrace call to update hardware debug registers
        error = WriteHardwareDebugRegs(eDREGTypeWATCH, wp_index);

        if (error.Fail())
        {
            m_hwp_regs[wp_index].control = tempControl;
            m_hwp_regs[wp_index].address = tempAddr;
            m_hwp_regs[wp_index].refcount = tempRefCount;

            return false;
        }

        return true;
    }

    return false;
}

Error
NativeRegisterContextLinux_arm::ClearAllHardwareWatchpoints ()
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    Error error;

    // Read hardware breakpoint and watchpoint information.
    error = ReadHardwareDebugInfo ();

    if (error.Fail())
        return error;

    lldb::addr_t tempAddr = 0;
    uint32_t tempControl = 0, tempRefCount = 0;

    for (uint32_t i = 0; i < m_max_hwp_supported; i++)
    {
        if (m_hwp_regs[i].control & 0x01)
        {
            // Create a backup we can revert to in case of failure.
            tempAddr = m_hwp_regs[i].address;
            tempControl = m_hwp_regs[i].control;
            tempRefCount = m_hwp_regs[i].refcount;

            // Clear watchpoints in local cache
            m_hwp_regs[i].control &= ~1;
            m_hwp_regs[i].address = 0;
            m_hwp_regs[i].refcount = 0;

            // Ptrace call to update hardware debug registers
            error = WriteHardwareDebugRegs(eDREGTypeWATCH, i);

            if (error.Fail())
            {
                m_hwp_regs[i].control = tempControl;
                m_hwp_regs[i].address = tempAddr;
                m_hwp_regs[i].refcount = tempRefCount;

                return error;
            }
        }
    }

    return Error();
}

uint32_t
NativeRegisterContextLinux_arm::GetWatchpointSize(uint32_t wp_index)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    switch ((m_hwp_regs[wp_index].control >> 5) & 0x0f)
    {
        case 0x01:
            return 1;
        case 0x03:
            return 2;
        case 0x07:
            return 3;
        case 0x0f:
            return 4;
        default:
            return 0;
    }
}
bool
NativeRegisterContextLinux_arm::WatchpointIsEnabled(uint32_t wp_index)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    if ((m_hwp_regs[wp_index].control & 0x1) == 0x1)
        return true;
    else
        return false;
}

Error
NativeRegisterContextLinux_arm::GetWatchpointHitIndex(uint32_t &wp_index, lldb::addr_t trap_addr)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    uint32_t watch_size;
    lldb::addr_t watch_addr;

    for (wp_index = 0; wp_index < m_max_hwp_supported; ++wp_index)
    {
        watch_size = GetWatchpointSize (wp_index);
        watch_addr = m_hwp_regs[wp_index].address;

        if (m_hwp_regs[wp_index].refcount >= 1 && WatchpointIsEnabled(wp_index)
            && trap_addr >= watch_addr && trap_addr < watch_addr + watch_size)
        {
            return Error();
        }
    }

    wp_index = LLDB_INVALID_INDEX32;
    return Error();
}

lldb::addr_t
NativeRegisterContextLinux_arm::GetWatchpointAddress (uint32_t wp_index)
{
    Log *log(lldb_private::GetLogIfAllCategoriesSet (LIBLLDB_LOG_WATCHPOINTS));

    if (log)
        log->Printf ("NativeRegisterContextLinux_arm::%s()", __FUNCTION__);

    if (wp_index >= m_max_hwp_supported)
        return LLDB_INVALID_ADDRESS;

    if (WatchpointIsEnabled(wp_index))
        return m_hwp_regs[wp_index].address;
    else
        return LLDB_INVALID_ADDRESS;
}

Error
NativeRegisterContextLinux_arm::ReadHardwareDebugInfo()
{
    Error error;

    if (!m_refresh_hwdebug_info)
    {
        return Error();
    }

    unsigned int cap_val;

    error = NativeProcessLinux::PtraceWrapper(PTRACE_GETHBPREGS, m_thread.GetID(), nullptr, &cap_val, sizeof(unsigned int));

    if (error.Fail())
        return error;

    m_max_hwp_supported = (cap_val >> 8) & 0xff;
    m_max_hbp_supported = cap_val & 0xff;
    m_refresh_hwdebug_info = false;

    return error;
}

Error
NativeRegisterContextLinux_arm::WriteHardwareDebugRegs(int hwbType, int hwb_index)
{
    Error error;

    lldb::addr_t *addr_buf;
    uint32_t *ctrl_buf;

    if (hwbType == eDREGTypeWATCH)
    {
        addr_buf = &m_hwp_regs[hwb_index].address;
        ctrl_buf = &m_hwp_regs[hwb_index].control;

        error = NativeProcessLinux::PtraceWrapper(PTRACE_SETHBPREGS,
                m_thread.GetID(), (PTRACE_TYPE_ARG3) -((hwb_index << 1) + 1),
                addr_buf, sizeof(unsigned int));

        if (error.Fail())
            return error;

        error = NativeProcessLinux::PtraceWrapper(PTRACE_SETHBPREGS,
                m_thread.GetID(), (PTRACE_TYPE_ARG3) -((hwb_index << 1) + 2),
                ctrl_buf, sizeof(unsigned int));
    }
    else
    {
        addr_buf = &m_hwp_regs[hwb_index].address;
        ctrl_buf = &m_hwp_regs[hwb_index].control;

        error = NativeProcessLinux::PtraceWrapper(PTRACE_SETHBPREGS,
                m_thread.GetID(), (PTRACE_TYPE_ARG3) ((hwb_index << 1) + 1),
                addr_buf, sizeof(unsigned int));

        if (error.Fail())
            return error;

        error = NativeProcessLinux::PtraceWrapper(PTRACE_SETHBPREGS,
                m_thread.GetID(), (PTRACE_TYPE_ARG3) ((hwb_index << 1) + 2),
                ctrl_buf, sizeof(unsigned int));

    }

    return error;
}

uint32_t
NativeRegisterContextLinux_arm::CalculateFprOffset(const RegisterInfo* reg_info) const
{
    return reg_info->byte_offset - GetRegisterInfoAtIndex(m_reg_info.first_fpr)->byte_offset;
}

Error
NativeRegisterContextLinux_arm::DoWriteRegisterValue(uint32_t offset,
                                                     const char* reg_name,
                                                     const RegisterValue &value)
{
    // PTRACE_POKEUSER don't work in the aarch64 liux kernel used on android devices (always return
    // "Bad address"). To avoid using PTRACE_POKEUSER we read out the full GPR register set, modify
    // the requested register and write it back. This approach is about 4 times slower but the
    // performance overhead is negligible in comparision to processing time in lldb-server.
    assert(offset % 4 == 0 && "Try to write a register with unaligned offset");
    if (offset + sizeof(uint32_t) > sizeof(m_gpr_arm))
        return Error("Register isn't fit into the size of the GPR area");

    Error error = DoReadGPR(m_gpr_arm, sizeof(m_gpr_arm));
    if (error.Fail())
        return error;

    m_gpr_arm[offset / sizeof(uint32_t)] = value.GetAsUInt32();
    return DoWriteGPR(m_gpr_arm, sizeof(m_gpr_arm));
}

Error
NativeRegisterContextLinux_arm::DoReadFPR(void *buf, size_t buf_size)
{
    return NativeProcessLinux::PtraceWrapper(PTRACE_GETVFPREGS,
                                             m_thread.GetID(),
                                             nullptr,
                                             buf,
                                             buf_size);
}

Error
NativeRegisterContextLinux_arm::DoWriteFPR(void *buf, size_t buf_size)
{
    return NativeProcessLinux::PtraceWrapper(PTRACE_SETVFPREGS,
                                             m_thread.GetID(),
                                             nullptr,
                                             buf,
                                             buf_size);
}

#endif // defined(__arm__)
