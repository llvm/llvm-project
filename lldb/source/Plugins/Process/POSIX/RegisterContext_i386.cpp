//===-- RegisterContextPOSIX_i386.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Core/DataExtractor.h"
#include "lldb/Target/Thread.h"
#include "lldb/Host/Endian.h"

#include "ProcessPOSIX.h"
#include "ProcessPOSIXLog.h"
#include "ProcessMonitor.h"
#include "RegisterContext_i386.h"
#include "RegisterContext_x86.h"

using namespace lldb_private;
using namespace lldb;

enum
{
    k_first_gpr,
    gpr_eax = k_first_gpr,
    gpr_ebx,
    gpr_ecx,
    gpr_edx,
    gpr_edi,
    gpr_esi,
    gpr_ebp,
    gpr_esp,
    gpr_ss,
    gpr_eflags,
#ifdef __FreeBSD__
    gpr_orig_ax,
#endif
    gpr_eip,
    gpr_cs,
    gpr_ds,
    gpr_es,
    gpr_fs,
    gpr_gs,
    k_last_gpr = gpr_gs,

    k_first_fpr,
    fpu_fcw = k_first_fpr,
    fpu_fsw,
    fpu_ftw,
    fpu_fop,
    fpu_ip,
    fpu_cs,
    fpu_foo,
    fpu_fos,
    fpu_mxcsr,
    fpu_stmm0,
    fpu_stmm1,
    fpu_stmm2,
    fpu_stmm3,
    fpu_stmm4,
    fpu_stmm5,
    fpu_stmm6,
    fpu_stmm7,
    fpu_xmm0,
    fpu_xmm1,
    fpu_xmm2,
    fpu_xmm3,
    fpu_xmm4,
    fpu_xmm5,
    fpu_xmm6,
    fpu_xmm7,
    k_last_fpr = fpu_xmm7,

    k_num_registers,
    k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
    k_num_fpu_registers = k_last_fpr - k_first_fpr + 1
};

// Number of register sets provided by this context.
enum
{
    k_num_register_sets = 2
};

static const
uint32_t g_gpr_regnums[k_num_gpr_registers] =
{
    gpr_eax,
    gpr_ebx,
    gpr_ecx,
    gpr_edx,
    gpr_edi,
    gpr_esi,
    gpr_ebp,
    gpr_esp,
    gpr_ss,
    gpr_eflags,
#ifdef __FreeBSD__
    gpr_orig_ax,
#endif
    gpr_eip,
    gpr_cs,
    gpr_ds,
    gpr_es,
    gpr_fs,
    gpr_gs,
};

static const uint32_t
g_fpu_regnums[k_num_fpu_registers] =
{
    fpu_fcw,
    fpu_fsw,
    fpu_ftw,
    fpu_fop,
    fpu_ip,
    fpu_cs,
    fpu_foo,
    fpu_fos,
    fpu_mxcsr,
    fpu_stmm0,
    fpu_stmm1,
    fpu_stmm2,
    fpu_stmm3,
    fpu_stmm4,
    fpu_stmm5,
    fpu_stmm6,
    fpu_stmm7,
    fpu_xmm0,
    fpu_xmm1,
    fpu_xmm2,
    fpu_xmm3,
    fpu_xmm4,
    fpu_xmm5,
    fpu_xmm6,
    fpu_xmm7,
};

static const RegisterSet
g_reg_sets[k_num_register_sets] =
{
    { "General Purpose Registers", "gpr", k_num_gpr_registers, g_gpr_regnums },
    { "Floating Point Registers",  "fpu", k_num_fpu_registers, g_fpu_regnums }
};

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) \
    (offsetof(RegisterContext_i386::UserArea, regs) + \
     offsetof(RegisterContext_i386::GPR, regname))

// Computes the offset of the given FPR in the user data area.
#define FPR_OFFSET(regname) \
    (offsetof(RegisterContext_i386::UserArea, i387) + \
     offsetof(RegisterContext_i386::FPU, regname))

// Number of bytes needed to represent a GPR.
#define GPR_SIZE(reg) sizeof(((RegisterContext_i386::GPR*)NULL)->reg)

// Number of bytes needed to represent a FPR.
#define FPR_SIZE(reg) sizeof(((RegisterContext_i386::FPU*)NULL)->reg)

// Number of bytes needed to represent the i'th FP register.
#define FP_SIZE sizeof(((RegisterContext_i386::MMSReg*)NULL)->bytes)

// Number of bytes needed to represent an XMM register.
#define XMM_SIZE sizeof(RegisterContext_i386::XMMReg)

#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)        \
    { #reg, alt, GPR_SIZE(reg), GPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg }, NULL, NULL }

#define DEFINE_FPR(reg, kind1, kind2, kind3, kind4)              \
    { #reg, NULL, FPR_SIZE(reg), FPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, fpu_##reg }, NULL, NULL }

#define DEFINE_FP(reg, i)                                          \
    { #reg#i, NULL, FP_SIZE, FPR_OFFSET(reg[i]), eEncodingVector,  \
      eFormatVectorOfUInt8,                                        \
      { dwarf_##reg##i, dwarf_##reg##i,                            \
        LLDB_INVALID_REGNUM, gdb_##reg##i, fpu_##reg##i }, NULL, NULL }

#define DEFINE_XMM(reg, i)                                         \
    { #reg#i, NULL, XMM_SIZE, FPR_OFFSET(reg[i]), eEncodingVector, \
      eFormatVectorOfUInt8,                                        \
      { dwarf_##reg##i, dwarf_##reg##i,                            \
        LLDB_INVALID_REGNUM, gdb_##reg##i, fpu_##reg##i }, NULL, NULL }

static RegisterInfo
g_register_infos[k_num_registers] =
{
    // General purpose registers.
    DEFINE_GPR(eax,    NULL,    gcc_eax,    dwarf_eax,    LLDB_INVALID_REGNUM,    gdb_eax),
    DEFINE_GPR(ebx,    NULL,    gcc_ebx,    dwarf_ebx,    LLDB_INVALID_REGNUM,    gdb_ebx),
    DEFINE_GPR(ecx,    NULL,    gcc_ecx,    dwarf_ecx,    LLDB_INVALID_REGNUM,    gdb_ecx),
    DEFINE_GPR(edx,    NULL,    gcc_edx,    dwarf_edx,    LLDB_INVALID_REGNUM,    gdb_edx),
    DEFINE_GPR(edi,    NULL,    gcc_edi,    dwarf_edi,    LLDB_INVALID_REGNUM,    gdb_edi),
    DEFINE_GPR(esi,    NULL,    gcc_esi,    dwarf_esi,    LLDB_INVALID_REGNUM,    gdb_esi),
    DEFINE_GPR(ebp,    "fp",    gcc_ebp,    dwarf_ebp,    LLDB_INVALID_REGNUM,    gdb_ebp),
    DEFINE_GPR(esp,    "sp",    gcc_esp,    dwarf_esp,    LLDB_INVALID_REGNUM,    gdb_esp),
    DEFINE_GPR(ss,     NULL,    LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,    gdb_ss),
    DEFINE_GPR(eflags, "flags", gcc_eflags, dwarf_eflags, LLDB_INVALID_REGNUM,    gdb_eflags),
    DEFINE_GPR(eip,    "pc",    gcc_eip,    dwarf_eip,    LLDB_INVALID_REGNUM,    gdb_eip),
    DEFINE_GPR(cs,     NULL,    LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,    gdb_cs),
    DEFINE_GPR(ds,     NULL,    LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,    gdb_ds),
    DEFINE_GPR(es,     NULL,    LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,    gdb_es),
    DEFINE_GPR(fs,     NULL,    LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,    gdb_fs),
    DEFINE_GPR(gs,     NULL,    LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,     LLDB_INVALID_REGNUM,    gdb_gs),

    // Floating point registers.
    DEFINE_FPR(fcw,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fcw),
    DEFINE_FPR(fsw,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fsw),
    DEFINE_FPR(ftw,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_ftw),
    DEFINE_FPR(fop,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fop),
    DEFINE_FPR(ip,        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_ip),
    DEFINE_FPR(cs,        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_cs),
    DEFINE_FPR(foo,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_dp),
    DEFINE_FPR(fos,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_ds),
    DEFINE_FPR(mxcsr,     LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_mxcsr),

    DEFINE_FP(stmm, 0),
    DEFINE_FP(stmm, 1),
    DEFINE_FP(stmm, 2),
    DEFINE_FP(stmm, 3),
    DEFINE_FP(stmm, 4),
    DEFINE_FP(stmm, 5),
    DEFINE_FP(stmm, 6),
    DEFINE_FP(stmm, 7),

    // XMM registers
    DEFINE_XMM(xmm, 0),
    DEFINE_XMM(xmm, 1),
    DEFINE_XMM(xmm, 2),
    DEFINE_XMM(xmm, 3),
    DEFINE_XMM(xmm, 4),
    DEFINE_XMM(xmm, 5),
    DEFINE_XMM(xmm, 6),
    DEFINE_XMM(xmm, 7),

};

#ifndef NDEBUG
static size_t k_num_register_infos = (sizeof(g_register_infos)/sizeof(RegisterInfo));
#endif

static unsigned GetRegOffset(unsigned reg)
{
    assert(reg < k_num_registers && "Invalid register number.");
    return g_register_infos[reg].byte_offset;
}

static unsigned GetRegSize(unsigned reg)
{
    assert(reg < k_num_registers && "Invalid register number.");
    return g_register_infos[reg].byte_size;
}

RegisterContext_i386::RegisterContext_i386(Thread &thread,
                                                     uint32_t concrete_frame_idx)
    : RegisterContextPOSIX(thread, concrete_frame_idx)
{
}

RegisterContext_i386::~RegisterContext_i386()
{
}

ProcessMonitor &
RegisterContext_i386::GetMonitor()
{
    ProcessSP base = CalculateProcess();
    ProcessPOSIX *process = static_cast<ProcessPOSIX*>(base.get());
    return process->GetMonitor();
}

void
RegisterContext_i386::Invalidate()
{
}

void
RegisterContext_i386::InvalidateAllRegisters()
{
}

size_t
RegisterContext_i386::GetRegisterCount()
{
    assert(k_num_register_infos == k_num_registers);
    return k_num_registers;
}

const RegisterInfo *
RegisterContext_i386::GetRegisterInfoAtIndex(size_t reg)
{
    assert(k_num_register_infos == k_num_registers);
    if (reg < k_num_registers)
        return &g_register_infos[reg];
    else
        return NULL;
}

size_t
RegisterContext_i386::GetRegisterSetCount()
{
    return k_num_register_sets;
}

const RegisterSet *
RegisterContext_i386::GetRegisterSet(size_t set)
{
    if (set < k_num_register_sets)
        return &g_reg_sets[set];
    else
        return NULL;
}

unsigned
RegisterContext_i386::GetRegisterIndexFromOffset(unsigned offset)
{
    unsigned reg;
    for (reg = 0; reg < k_num_registers; reg++)
    {
        if (g_register_infos[reg].byte_offset == offset)
            break;
    }
    assert(reg < k_num_registers && "Invalid register offset.");
    return reg;
}

const char *
RegisterContext_i386::GetRegisterName(unsigned reg)
{
    assert(reg < k_num_registers && "Invalid register offset.");
    return g_register_infos[reg].name;
}

bool
RegisterContext_i386::ReadRegister(const RegisterInfo *reg_info,
                                        RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadRegisterValue(m_thread.GetID(), GetRegOffset(reg),
                                     GetRegisterName(reg), GetRegSize(reg), value);
}

bool
RegisterContext_i386::ReadAllRegisterValues(DataBufferSP &data_sp)
{
    return false;
}

bool RegisterContext_i386::WriteRegister(const RegisterInfo *reg_info,
                                         const RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteRegisterValue(m_thread.GetID(), GetRegOffset(reg),
                                      GetRegisterName(reg), value);
}

bool
RegisterContext_i386::WriteAllRegisterValues(const DataBufferSP &data)
{
    return false;
}

bool
RegisterContext_i386::UpdateAfterBreakpoint()
{
    // PC points one byte past the int3 responsible for the breakpoint.
    lldb::addr_t pc;

    if ((pc = GetPC()) == LLDB_INVALID_ADDRESS)
        return false;

    SetPC(pc - 1);
    return true;
}

uint32_t
RegisterContext_i386::ConvertRegisterKindToRegisterNumber(uint32_t kind,
                                                               uint32_t num)
{
    if (kind == eRegisterKindGeneric)
    {
        switch (num)
        {
        case LLDB_REGNUM_GENERIC_PC:    return gpr_eip;
        case LLDB_REGNUM_GENERIC_SP:    return gpr_esp;
        case LLDB_REGNUM_GENERIC_FP:    return gpr_ebp;
        case LLDB_REGNUM_GENERIC_FLAGS: return gpr_eflags;
        case LLDB_REGNUM_GENERIC_RA:
        default:
            return LLDB_INVALID_REGNUM;
        }
    }

    if (kind == eRegisterKindGCC || kind == eRegisterKindDWARF)
    {
        switch (num)
        {
        case dwarf_eax:  return gpr_eax;
        case dwarf_edx:  return gpr_edx;
        case dwarf_ecx:  return gpr_ecx;
        case dwarf_ebx:  return gpr_ebx;
        case dwarf_esi:  return gpr_esi;
        case dwarf_edi:  return gpr_edi;
        case dwarf_ebp:  return gpr_ebp;
        case dwarf_esp:  return gpr_esp;
        case dwarf_eip:  return gpr_eip;
        case dwarf_xmm0: return fpu_xmm0;
        case dwarf_xmm1: return fpu_xmm1;
        case dwarf_xmm2: return fpu_xmm2;
        case dwarf_xmm3: return fpu_xmm3;
        case dwarf_xmm4: return fpu_xmm4;
        case dwarf_xmm5: return fpu_xmm5;
        case dwarf_xmm6: return fpu_xmm6;
        case dwarf_xmm7: return fpu_xmm7;
        case dwarf_stmm0: return fpu_stmm0;
        case dwarf_stmm1: return fpu_stmm1;
        case dwarf_stmm2: return fpu_stmm2;
        case dwarf_stmm3: return fpu_stmm3;
        case dwarf_stmm4: return fpu_stmm4;
        case dwarf_stmm5: return fpu_stmm5;
        case dwarf_stmm6: return fpu_stmm6;
        case dwarf_stmm7: return fpu_stmm7;
        default:
            return LLDB_INVALID_REGNUM;
        }
    }

    if (kind == eRegisterKindGDB)
    {
        switch (num)
        {
        case gdb_eax     : return gpr_eax;
        case gdb_ebx     : return gpr_ebx;
        case gdb_ecx     : return gpr_ecx;
        case gdb_edx     : return gpr_edx;
        case gdb_esi     : return gpr_esi;
        case gdb_edi     : return gpr_edi;
        case gdb_ebp     : return gpr_ebp;
        case gdb_esp     : return gpr_esp;
        case gdb_eip     : return gpr_eip;
        case gdb_eflags  : return gpr_eflags;
        case gdb_cs      : return gpr_cs;
        case gdb_ss      : return gpr_ss;
        case gdb_ds      : return gpr_ds;
        case gdb_es      : return gpr_es;
        case gdb_fs      : return gpr_fs;
        case gdb_gs      : return gpr_gs;
        case gdb_stmm0   : return fpu_stmm0;
        case gdb_stmm1   : return fpu_stmm1;
        case gdb_stmm2   : return fpu_stmm2;
        case gdb_stmm3   : return fpu_stmm3;
        case gdb_stmm4   : return fpu_stmm4;
        case gdb_stmm5   : return fpu_stmm5;
        case gdb_stmm6   : return fpu_stmm6;
        case gdb_stmm7   : return fpu_stmm7;
        case gdb_fcw     : return fpu_fcw;
        case gdb_fsw     : return fpu_fsw;
        case gdb_ftw     : return fpu_ftw;
        case gdb_fpu_cs  : return fpu_cs;
        case gdb_ip      : return fpu_ip;
        case gdb_fpu_ds  : return fpu_fos;
        case gdb_dp      : return fpu_foo;
        case gdb_fop     : return fpu_fop;
        case gdb_xmm0    : return fpu_xmm0;
        case gdb_xmm1    : return fpu_xmm1;
        case gdb_xmm2    : return fpu_xmm2;
        case gdb_xmm3    : return fpu_xmm3;
        case gdb_xmm4    : return fpu_xmm4;
        case gdb_xmm5    : return fpu_xmm5;
        case gdb_xmm6    : return fpu_xmm6;
        case gdb_xmm7    : return fpu_xmm7;
        case gdb_mxcsr   : return fpu_mxcsr;
        default:
            return LLDB_INVALID_REGNUM;
        }
    }
    else if (kind == eRegisterKindLLDB)
    {
        return num;
    }

    return LLDB_INVALID_REGNUM;
}

bool
RegisterContext_i386::HardwareSingleStep(bool enable)
{
    enum { TRACE_BIT = 0x100 };
    uint64_t eflags;

    if ((eflags = ReadRegisterAsUnsigned(gpr_eflags, -1UL)) == -1UL)
        return false;

    if (enable)
    {
        if (eflags & TRACE_BIT)
            return true;

        eflags |= TRACE_BIT;
    }
    else
    {
        if (!(eflags & TRACE_BIT))
            return false;

        eflags &= ~TRACE_BIT;
    }

    return WriteRegisterFromUnsigned(gpr_eflags, eflags);
}

void
RegisterContext_i386::LogGPR(const char *title)
{
    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_REGISTERS));
    if (log)
    {
        if (title)
            log->Printf ("%s", title);
        for (uint32_t i=0; i<k_num_gpr_registers; i++)
        {
            uint32_t reg = gpr_eax + i;
            log->Printf("%12s = 0x%8.8" PRIx64, g_register_infos[reg].name, ((uint64_t*)&user.regs)[reg]);
        }
    }
}

bool
RegisterContext_i386::ReadGPR()
{
    bool result;

    ProcessMonitor &monitor = GetMonitor();
    result = monitor.ReadGPR(m_thread.GetID(), &user.regs, sizeof(user.regs));
    LogGPR("RegisterContext_i386::ReadGPR()");
    return result;
}

bool
RegisterContext_i386::ReadFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadFPR(m_thread.GetID(), &user.i387, sizeof(user.i387));
}
