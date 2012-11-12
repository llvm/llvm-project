//===-- RegisterContext_x86_64.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <cstring>
#include <errno.h>
#include <stdint.h>

#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/RegisterValue.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Target/Thread.h"
#include "lldb/Host/Endian.h"

#include "ProcessPOSIX.h"
#include "ProcessMonitor.h"
#include "RegisterContext_x86_64.h"

using namespace lldb_private;
using namespace lldb;

// Internal codes for all x86_64 registers.
enum
{
    k_first_gpr,
    gpr_rax = k_first_gpr,
    gpr_rbx,
    gpr_rcx,
    gpr_rdx,
    gpr_rdi,
    gpr_rsi,
    gpr_rbp,
    gpr_rsp,
    gpr_r8,
    gpr_r9,
    gpr_r10,
    gpr_r11,
    gpr_r12,
    gpr_r13,
    gpr_r14,
    gpr_r15,
    gpr_rip,
    gpr_rflags,
    gpr_cs,
    gpr_fs,
    gpr_gs,
    gpr_ss,
    gpr_ds,
    gpr_es,
    k_last_gpr = gpr_es,

    k_first_fpr,
    fpu_fcw = k_first_fpr,
    fpu_fsw,
    fpu_ftw,
    fpu_fop,
    fpu_ip,
    fpu_cs,
    fpu_dp,
    fpu_ds,
    fpu_mxcsr,
    fpu_mxcsrmask,
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
    fpu_xmm8,
    fpu_xmm9,
    fpu_xmm10,
    fpu_xmm11,
    fpu_xmm12,
    fpu_xmm13,
    fpu_xmm14,
    fpu_xmm15,
    k_last_fpr = fpu_xmm15,

    k_num_registers,
    k_num_gpr_registers = k_last_gpr - k_first_gpr + 1,
    k_num_fpu_registers = k_last_fpr - k_first_fpr + 1
};

// Number of register sets provided by this context.
enum
{
    k_num_register_sets = 2
};

enum gcc_dwarf_regnums
{
    gcc_dwarf_gpr_rax = 0,
    gcc_dwarf_gpr_rdx,
    gcc_dwarf_gpr_rcx,
    gcc_dwarf_gpr_rbx,
    gcc_dwarf_gpr_rsi,
    gcc_dwarf_gpr_rdi,
    gcc_dwarf_gpr_rbp,
    gcc_dwarf_gpr_rsp,
    gcc_dwarf_gpr_r8,
    gcc_dwarf_gpr_r9,
    gcc_dwarf_gpr_r10,
    gcc_dwarf_gpr_r11,
    gcc_dwarf_gpr_r12,
    gcc_dwarf_gpr_r13,
    gcc_dwarf_gpr_r14,
    gcc_dwarf_gpr_r15,
    gcc_dwarf_gpr_rip,
    gcc_dwarf_fpu_xmm0,
    gcc_dwarf_fpu_xmm1,
    gcc_dwarf_fpu_xmm2,
    gcc_dwarf_fpu_xmm3,
    gcc_dwarf_fpu_xmm4,
    gcc_dwarf_fpu_xmm5,
    gcc_dwarf_fpu_xmm6,
    gcc_dwarf_fpu_xmm7,
    gcc_dwarf_fpu_xmm8,
    gcc_dwarf_fpu_xmm9,
    gcc_dwarf_fpu_xmm10,
    gcc_dwarf_fpu_xmm11,
    gcc_dwarf_fpu_xmm12,
    gcc_dwarf_fpu_xmm13,
    gcc_dwarf_fpu_xmm14,
    gcc_dwarf_fpu_xmm15,
    gcc_dwarf_fpu_stmm0,
    gcc_dwarf_fpu_stmm1,
    gcc_dwarf_fpu_stmm2,
    gcc_dwarf_fpu_stmm3,
    gcc_dwarf_fpu_stmm4,
    gcc_dwarf_fpu_stmm5,
    gcc_dwarf_fpu_stmm6,
    gcc_dwarf_fpu_stmm7
};

enum gdb_regnums
{
    gdb_gpr_rax     =   0,
    gdb_gpr_rbx     =   1,
    gdb_gpr_rcx     =   2,
    gdb_gpr_rdx     =   3,
    gdb_gpr_rsi     =   4,
    gdb_gpr_rdi     =   5,
    gdb_gpr_rbp     =   6,
    gdb_gpr_rsp     =   7,
    gdb_gpr_r8      =   8,
    gdb_gpr_r9      =   9,
    gdb_gpr_r10     =  10,
    gdb_gpr_r11     =  11,
    gdb_gpr_r12     =  12,
    gdb_gpr_r13     =  13,
    gdb_gpr_r14     =  14,
    gdb_gpr_r15     =  15,
    gdb_gpr_rip     =  16,
    gdb_gpr_rflags  =  17,
    gdb_gpr_cs      =  18,
    gdb_gpr_ss      =  19,
    gdb_gpr_ds      =  20,
    gdb_gpr_es      =  21,
    gdb_gpr_fs      =  22,
    gdb_gpr_gs      =  23,
    gdb_fpu_stmm0   =  24,
    gdb_fpu_stmm1   =  25,
    gdb_fpu_stmm2   =  26,
    gdb_fpu_stmm3   =  27,
    gdb_fpu_stmm4   =  28,
    gdb_fpu_stmm5   =  29,
    gdb_fpu_stmm6   =  30,
    gdb_fpu_stmm7   =  31,
    gdb_fpu_fcw     =  32,
    gdb_fpu_fsw     =  33,
    gdb_fpu_ftw     =  34,
    gdb_fpu_cs      =  35,
    gdb_fpu_ip      =  36,
    gdb_fpu_ds      =  37,
    gdb_fpu_dp      =  38,
    gdb_fpu_fop     =  39,
    gdb_fpu_xmm0    =  40,
    gdb_fpu_xmm1    =  41,
    gdb_fpu_xmm2    =  42,
    gdb_fpu_xmm3    =  43,
    gdb_fpu_xmm4    =  44,
    gdb_fpu_xmm5    =  45,
    gdb_fpu_xmm6    =  46,
    gdb_fpu_xmm7    =  47,
    gdb_fpu_xmm8    =  48,
    gdb_fpu_xmm9    =  49,
    gdb_fpu_xmm10   =  50,
    gdb_fpu_xmm11   =  51,
    gdb_fpu_xmm12   =  52,
    gdb_fpu_xmm13   =  53,
    gdb_fpu_xmm14   =  54,
    gdb_fpu_xmm15   =  55,
    gdb_fpu_mxcsr   =  56
};

static const
uint32_t g_gpr_regnums[k_num_gpr_registers] =
{
    gpr_rax,
    gpr_rbx,
    gpr_rcx,
    gpr_rdx,
    gpr_rdi,
    gpr_rsi,
    gpr_rbp,
    gpr_rsp,
    gpr_r8,
    gpr_r9,
    gpr_r10,
    gpr_r11,
    gpr_r12,
    gpr_r13,
    gpr_r14,
    gpr_r15,
    gpr_rip,
    gpr_rflags,
    gpr_cs,
    gpr_fs,
    gpr_gs,
    gpr_ss,
    gpr_ds,
    gpr_es
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
    fpu_dp,
    fpu_ds,
    fpu_mxcsr,
    fpu_mxcsrmask,
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
    fpu_xmm8,
    fpu_xmm9,
    fpu_xmm10,
    fpu_xmm11,
    fpu_xmm12,
    fpu_xmm13,
    fpu_xmm14,
    fpu_xmm15
};

static const RegisterSet
g_reg_sets[k_num_register_sets] =
{
    { "General Purpose Registers", "gpr", k_num_gpr_registers, g_gpr_regnums },
    { "Floating Point Registers",  "fpu", k_num_fpu_registers, g_fpu_regnums }
};

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname) \
    (offsetof(RegisterContext_x86_64::UserArea, regs) + \
     offsetof(GPR, regname))

// Computes the offset of the given FPR in the user data area.
#define FPR_OFFSET(regname) \
    (offsetof(RegisterContext_x86_64::UserArea, i387) + \
     offsetof(RegisterContext_x86_64::FPU, regname))

// Number of bytes needed to represent a GPR.
#define GPR_SIZE(reg) sizeof(((GPR*)NULL)->reg)

// Number of bytes needed to represent a FPR.
#define FPR_SIZE(reg) sizeof(((RegisterContext_x86_64::FPU*)NULL)->reg)

// Number of bytes needed to represent the i'th FP register.
#define FP_SIZE sizeof(((RegisterContext_x86_64::MMSReg*)NULL)->bytes)

// Number of bytes needed to represent an XMM register.
#define XMM_SIZE sizeof(RegisterContext_x86_64::XMMReg)

#define DEFINE_GPR(reg, alt, kind1, kind2, kind3, kind4)        \
    { #reg, alt, GPR_SIZE(reg), GPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, gpr_##reg }, NULL, NULL }

#define DEFINE_FPR(reg, kind1, kind2, kind3, kind4)              \
    { #reg, NULL, FPR_SIZE(reg), FPR_OFFSET(reg), eEncodingUint, \
      eFormatHex, { kind1, kind2, kind3, kind4, fpu_##reg }, NULL, NULL }

#define DEFINE_FP(reg, i)                                          \
    { #reg#i, NULL, FP_SIZE, FPR_OFFSET(reg[i]), eEncodingVector,  \
      eFormatVectorOfUInt8,                                        \
      { gcc_dwarf_fpu_##reg##i, gcc_dwarf_fpu_##reg##i,            \
        LLDB_INVALID_REGNUM, gdb_fpu_##reg##i, fpu_##reg##i }, NULL, NULL }

#define DEFINE_XMM(reg, i)                                         \
    { #reg#i, NULL, XMM_SIZE, FPR_OFFSET(reg[i]), eEncodingVector, \
      eFormatVectorOfUInt8,                                        \
      { gcc_dwarf_fpu_##reg##i, gcc_dwarf_fpu_##reg##i,            \
        LLDB_INVALID_REGNUM, gdb_fpu_##reg##i, fpu_##reg##i }, NULL, NULL }

#define REG_CONTEXT_SIZE (sizeof(GPR) + sizeof(RegisterContext_x86_64::FPU))

static RegisterInfo
g_register_infos[k_num_registers] =
{
    // General purpose registers.
    DEFINE_GPR(rax,    NULL,    gcc_dwarf_gpr_rax,   gcc_dwarf_gpr_rax,   LLDB_INVALID_REGNUM,       gdb_gpr_rax),
    DEFINE_GPR(rbx,    NULL,    gcc_dwarf_gpr_rbx,   gcc_dwarf_gpr_rbx,   LLDB_INVALID_REGNUM,       gdb_gpr_rbx),
    DEFINE_GPR(rcx,    NULL,    gcc_dwarf_gpr_rcx,   gcc_dwarf_gpr_rcx,   LLDB_INVALID_REGNUM,       gdb_gpr_rcx),
    DEFINE_GPR(rdx,    NULL,    gcc_dwarf_gpr_rdx,   gcc_dwarf_gpr_rdx,   LLDB_INVALID_REGNUM,       gdb_gpr_rdx),
    DEFINE_GPR(rdi,    NULL,    gcc_dwarf_gpr_rdi,   gcc_dwarf_gpr_rdi,   LLDB_INVALID_REGNUM,       gdb_gpr_rdi),
    DEFINE_GPR(rsi,    NULL,    gcc_dwarf_gpr_rsi,   gcc_dwarf_gpr_rsi,   LLDB_INVALID_REGNUM,       gdb_gpr_rsi),
    DEFINE_GPR(rbp,    "fp",    gcc_dwarf_gpr_rbp,   gcc_dwarf_gpr_rbp,   LLDB_REGNUM_GENERIC_FP,    gdb_gpr_rbp),
    DEFINE_GPR(rsp,    "sp",    gcc_dwarf_gpr_rsp,   gcc_dwarf_gpr_rsp,   LLDB_REGNUM_GENERIC_SP,    gdb_gpr_rsp),
    DEFINE_GPR(r8,     NULL,    gcc_dwarf_gpr_r8,    gcc_dwarf_gpr_r8,    LLDB_INVALID_REGNUM,       gdb_gpr_r8),
    DEFINE_GPR(r9,     NULL,    gcc_dwarf_gpr_r9,    gcc_dwarf_gpr_r9,    LLDB_INVALID_REGNUM,       gdb_gpr_r9),
    DEFINE_GPR(r10,    NULL,    gcc_dwarf_gpr_r10,   gcc_dwarf_gpr_r10,   LLDB_INVALID_REGNUM,       gdb_gpr_r10),
    DEFINE_GPR(r11,    NULL,    gcc_dwarf_gpr_r11,   gcc_dwarf_gpr_r11,   LLDB_INVALID_REGNUM,       gdb_gpr_r11),
    DEFINE_GPR(r12,    NULL,    gcc_dwarf_gpr_r12,   gcc_dwarf_gpr_r12,   LLDB_INVALID_REGNUM,       gdb_gpr_r12),
    DEFINE_GPR(r13,    NULL,    gcc_dwarf_gpr_r13,   gcc_dwarf_gpr_r13,   LLDB_INVALID_REGNUM,       gdb_gpr_r13),
    DEFINE_GPR(r14,    NULL,    gcc_dwarf_gpr_r14,   gcc_dwarf_gpr_r14,   LLDB_INVALID_REGNUM,       gdb_gpr_r14),
    DEFINE_GPR(r15,    NULL,    gcc_dwarf_gpr_r15,   gcc_dwarf_gpr_r15,   LLDB_INVALID_REGNUM,       gdb_gpr_r15),
    DEFINE_GPR(rip,    "pc",    gcc_dwarf_gpr_rip,   gcc_dwarf_gpr_rip,   LLDB_REGNUM_GENERIC_PC,    gdb_gpr_rip),
    DEFINE_GPR(rflags, "flags", LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_REGNUM_GENERIC_FLAGS, gdb_gpr_rflags),
    DEFINE_GPR(cs,     NULL,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       gdb_gpr_cs),
    DEFINE_GPR(fs,     NULL,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       gdb_gpr_fs),
    DEFINE_GPR(gs,     NULL,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       gdb_gpr_gs),
    DEFINE_GPR(ss,     NULL,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       gdb_gpr_ss),
    DEFINE_GPR(ds,     NULL,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       gdb_gpr_ds),
    DEFINE_GPR(es,     NULL,    LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM,       gdb_gpr_es),

    // i387 Floating point registers.
    DEFINE_FPR(fcw,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_fcw),
    DEFINE_FPR(fsw,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_fsw),
    DEFINE_FPR(ftw,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_ftw),
    DEFINE_FPR(fop,       LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_fop),
    DEFINE_FPR(ip,        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_ip),
    // FIXME: Extract segment from ip.
    DEFINE_FPR(ip,        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_cs),
    DEFINE_FPR(dp,        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_dp),
    // FIXME: Extract segment from dp.
    DEFINE_FPR(dp,        LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_ds),
    DEFINE_FPR(mxcsr,     LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, gdb_fpu_mxcsr),
    DEFINE_FPR(mxcsrmask, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM, LLDB_INVALID_REGNUM),

    // FP registers.
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
    DEFINE_XMM(xmm, 8),
    DEFINE_XMM(xmm, 9),
    DEFINE_XMM(xmm, 10),
    DEFINE_XMM(xmm, 11),
    DEFINE_XMM(xmm, 12),
    DEFINE_XMM(xmm, 13),
    DEFINE_XMM(xmm, 14),
    DEFINE_XMM(xmm, 15)
};

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

static bool IsGPR(unsigned reg)
{
    return reg <= k_last_gpr;   // GPR's come first.
}

static bool IsFPR(unsigned reg)
{
    return (k_first_fpr <= reg && reg <= k_last_fpr);
}

RegisterContext_x86_64::RegisterContext_x86_64(Thread &thread,
                                                         uint32_t concrete_frame_idx)
    : RegisterContextPOSIX(thread, concrete_frame_idx)
{
}

RegisterContext_x86_64::~RegisterContext_x86_64()
{
}

ProcessMonitor &
RegisterContext_x86_64::GetMonitor()
{
    ProcessSP base = CalculateProcess();
    ProcessPOSIX *process = static_cast<ProcessPOSIX*>(base.get());
    return process->GetMonitor();
}

void
RegisterContext_x86_64::Invalidate()
{
}

void
RegisterContext_x86_64::InvalidateAllRegisters()
{
}

size_t
RegisterContext_x86_64::GetRegisterCount()
{
    return k_num_registers;
}

const RegisterInfo *
RegisterContext_x86_64::GetRegisterInfoAtIndex(uint32_t reg)
{
    if (reg < k_num_registers)
        return &g_register_infos[reg];
    else
        return NULL;
}

size_t
RegisterContext_x86_64::GetRegisterSetCount()
{
    return k_num_register_sets;
}

const RegisterSet *
RegisterContext_x86_64::GetRegisterSet(uint32_t set)
{
    if (set < k_num_register_sets)
        return &g_reg_sets[set];
    else
        return NULL;
}

unsigned
RegisterContext_x86_64::GetRegisterIndexFromOffset(unsigned offset)
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
RegisterContext_x86_64::GetRegisterName(unsigned reg)
{
    assert(reg < k_num_registers && "Invalid register offset.");
    return g_register_infos[reg].name;
}

bool
RegisterContext_x86_64::ReadRegister(const RegisterInfo *reg_info, RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];

    if (reg >= k_first_fpr && reg <= k_last_fpr) {
        if (!ReadFPR())
            return false;
    }
    else {
        ProcessMonitor &monitor = GetMonitor();
        return monitor.ReadRegisterValue(GetRegOffset(reg),GetRegSize(reg), value);
    }

    if (reg_info->encoding == eEncodingVector) {
        // Get the target process whose privileged thread was used for the register read.
        Process *process = CalculateProcess().get();
        if (process) {
            if (reg >= fpu_stmm0 && reg <= fpu_stmm7) {
               value.SetBytes(user.i387.stmm[reg - fpu_stmm0].bytes, reg_info->byte_size, process->GetByteOrder());
               return value.GetType() == RegisterValue::eTypeBytes;
            }
            if (reg >= fpu_xmm0 && reg <= fpu_xmm15) {
                value.SetBytes(user.i387.xmm[reg - fpu_xmm0].bytes, reg_info->byte_size, process->GetByteOrder());
                return value.GetType() == RegisterValue::eTypeBytes;
            }
        }
        return false;
    }

    // Note that lldb uses slightly different naming conventions from sys/user.h
    switch (reg)
    {
    default:
        return false;
    case fpu_dp:
        value = user.i387.dp;
        break;
    case fpu_fcw:
        value = user.i387.fcw;
        break;
    case fpu_fsw:
        value = user.i387.fsw;
        break;
    case fpu_ip:
        value = user.i387.ip;
        break;
    case fpu_fop:
        value = user.i387.fop;
        break;
    case fpu_ftw:
        value = user.i387.ftw;
        break;
    case fpu_mxcsr:
        value = user.i387.mxcsr;
        break;
    case fpu_mxcsrmask:
        value = user.i387.mxcsrmask;
        break;
    }
    return true;
}

bool
RegisterContext_x86_64::ReadAllRegisterValues(DataBufferSP &data_sp)
{
    data_sp.reset (new DataBufferHeap (REG_CONTEXT_SIZE, 0));
    if (data_sp && ReadGPR () && ReadFPR ())
    {
        uint8_t *dst = data_sp->GetBytes();
        ::memcpy (dst, &user.regs, sizeof(user.regs));
        dst += sizeof(user.regs);

        ::memcpy (dst, &user.i387, sizeof(user.i387));
        return true;
    }
    return false;
}

bool
RegisterContext_x86_64::WriteRegister(const lldb_private::RegisterInfo *reg_info,
                                           const lldb_private::RegisterValue &value)
{
    const uint32_t reg = reg_info->kinds[eRegisterKindLLDB];
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteRegisterValue(GetRegOffset(reg), value);
}

bool
RegisterContext_x86_64::WriteAllRegisterValues(const DataBufferSP &data_sp)
{
    if (data_sp && data_sp->GetByteSize() == REG_CONTEXT_SIZE)
    {
        const uint8_t *src = data_sp->GetBytes();
        ::memcpy (&user.regs, src, sizeof(user.regs));
        src += sizeof(user.regs);

        ::memcpy (&user.i387, src, sizeof(user.i387));
        return WriteGPR() & WriteFPR();
    }
    return false;
}

bool
RegisterContext_x86_64::UpdateAfterBreakpoint()
{
    // PC points one byte past the int3 responsible for the breakpoint.
    lldb::addr_t pc;

    if ((pc = GetPC()) == LLDB_INVALID_ADDRESS)
        return false;

    SetPC(pc - 1);
    return true;
}

uint32_t
RegisterContext_x86_64::ConvertRegisterKindToRegisterNumber(uint32_t kind,
                                                                 uint32_t num)
{
    if (kind == eRegisterKindGeneric)
    {
        switch (num)
        {
        case LLDB_REGNUM_GENERIC_PC:    return gpr_rip;
        case LLDB_REGNUM_GENERIC_SP:    return gpr_rsp;
        case LLDB_REGNUM_GENERIC_FP:    return gpr_rbp;
        case LLDB_REGNUM_GENERIC_FLAGS: return gpr_rflags;
        case LLDB_REGNUM_GENERIC_RA:
        default:
            return LLDB_INVALID_REGNUM;
        }
    }

    if (kind == eRegisterKindGCC || kind == eRegisterKindDWARF)
    {
        switch (num)
        {
        case gcc_dwarf_gpr_rax:  return gpr_rax;
        case gcc_dwarf_gpr_rdx:  return gpr_rdx;
        case gcc_dwarf_gpr_rcx:  return gpr_rcx;
        case gcc_dwarf_gpr_rbx:  return gpr_rbx;
        case gcc_dwarf_gpr_rsi:  return gpr_rsi;
        case gcc_dwarf_gpr_rdi:  return gpr_rdi;
        case gcc_dwarf_gpr_rbp:  return gpr_rbp;
        case gcc_dwarf_gpr_rsp:  return gpr_rsp;
        case gcc_dwarf_gpr_r8:   return gpr_r8;
        case gcc_dwarf_gpr_r9:   return gpr_r9;
        case gcc_dwarf_gpr_r10:  return gpr_r10;
        case gcc_dwarf_gpr_r11:  return gpr_r11;
        case gcc_dwarf_gpr_r12:  return gpr_r12;
        case gcc_dwarf_gpr_r13:  return gpr_r13;
        case gcc_dwarf_gpr_r14:  return gpr_r14;
        case gcc_dwarf_gpr_r15:  return gpr_r15;
        case gcc_dwarf_gpr_rip:  return gpr_rip;
        case gcc_dwarf_fpu_xmm0: return fpu_xmm0;
        case gcc_dwarf_fpu_xmm1: return fpu_xmm1;
        case gcc_dwarf_fpu_xmm2: return fpu_xmm2;
        case gcc_dwarf_fpu_xmm3: return fpu_xmm3;
        case gcc_dwarf_fpu_xmm4: return fpu_xmm4;
        case gcc_dwarf_fpu_xmm5: return fpu_xmm5;
        case gcc_dwarf_fpu_xmm6: return fpu_xmm6;
        case gcc_dwarf_fpu_xmm7: return fpu_xmm7;
        case gcc_dwarf_fpu_xmm8: return fpu_xmm8;
        case gcc_dwarf_fpu_xmm9: return fpu_xmm9;
        case gcc_dwarf_fpu_xmm10: return fpu_xmm10;
        case gcc_dwarf_fpu_xmm11: return fpu_xmm11;
        case gcc_dwarf_fpu_xmm12: return fpu_xmm12;
        case gcc_dwarf_fpu_xmm13: return fpu_xmm13;
        case gcc_dwarf_fpu_xmm14: return fpu_xmm14;
        case gcc_dwarf_fpu_xmm15: return fpu_xmm15;
        case gcc_dwarf_fpu_stmm0: return fpu_stmm0;
        case gcc_dwarf_fpu_stmm1: return fpu_stmm1;
        case gcc_dwarf_fpu_stmm2: return fpu_stmm2;
        case gcc_dwarf_fpu_stmm3: return fpu_stmm3;
        case gcc_dwarf_fpu_stmm4: return fpu_stmm4;
        case gcc_dwarf_fpu_stmm5: return fpu_stmm5;
        case gcc_dwarf_fpu_stmm6: return fpu_stmm6;
        case gcc_dwarf_fpu_stmm7: return fpu_stmm7;
        default:
            return LLDB_INVALID_REGNUM;
        }
    }

    if (kind == eRegisterKindGDB)
    {
        switch (num)
        {
        case gdb_gpr_rax     : return gpr_rax;
        case gdb_gpr_rbx     : return gpr_rbx;
        case gdb_gpr_rcx     : return gpr_rcx;
        case gdb_gpr_rdx     : return gpr_rdx;
        case gdb_gpr_rsi     : return gpr_rsi;
        case gdb_gpr_rdi     : return gpr_rdi;
        case gdb_gpr_rbp     : return gpr_rbp;
        case gdb_gpr_rsp     : return gpr_rsp;
        case gdb_gpr_r8      : return gpr_r8;
        case gdb_gpr_r9      : return gpr_r9;
        case gdb_gpr_r10     : return gpr_r10;
        case gdb_gpr_r11     : return gpr_r11;
        case gdb_gpr_r12     : return gpr_r12;
        case gdb_gpr_r13     : return gpr_r13;
        case gdb_gpr_r14     : return gpr_r14;
        case gdb_gpr_r15     : return gpr_r15;
        case gdb_gpr_rip     : return gpr_rip;
        case gdb_gpr_rflags  : return gpr_rflags;
        case gdb_gpr_cs      : return gpr_cs;
        case gdb_gpr_ss      : return gpr_ss;
        case gdb_gpr_ds      : return gpr_ds;
        case gdb_gpr_es      : return gpr_es;
        case gdb_gpr_fs      : return gpr_fs;
        case gdb_gpr_gs      : return gpr_gs;
        case gdb_fpu_stmm0   : return fpu_stmm0;
        case gdb_fpu_stmm1   : return fpu_stmm1;
        case gdb_fpu_stmm2   : return fpu_stmm2;
        case gdb_fpu_stmm3   : return fpu_stmm3;
        case gdb_fpu_stmm4   : return fpu_stmm4;
        case gdb_fpu_stmm5   : return fpu_stmm5;
        case gdb_fpu_stmm6   : return fpu_stmm6;
        case gdb_fpu_stmm7   : return fpu_stmm7;
        case gdb_fpu_fcw     : return fpu_fcw;
        case gdb_fpu_fsw     : return fpu_fsw;
        case gdb_fpu_ftw     : return fpu_ftw;
        case gdb_fpu_cs      : return fpu_cs;
        case gdb_fpu_ip      : return fpu_ip;
        case gdb_fpu_ds      : return fpu_ds;
        case gdb_fpu_dp      : return fpu_dp;
        case gdb_fpu_fop     : return fpu_fop;
        case gdb_fpu_xmm0    : return fpu_xmm0;
        case gdb_fpu_xmm1    : return fpu_xmm1;
        case gdb_fpu_xmm2    : return fpu_xmm2;
        case gdb_fpu_xmm3    : return fpu_xmm3;
        case gdb_fpu_xmm4    : return fpu_xmm4;
        case gdb_fpu_xmm5    : return fpu_xmm5;
        case gdb_fpu_xmm6    : return fpu_xmm6;
        case gdb_fpu_xmm7    : return fpu_xmm7;
        case gdb_fpu_xmm8    : return fpu_xmm8;
        case gdb_fpu_xmm9    : return fpu_xmm9;
        case gdb_fpu_xmm10   : return fpu_xmm10;
        case gdb_fpu_xmm11   : return fpu_xmm11;
        case gdb_fpu_xmm12   : return fpu_xmm12;
        case gdb_fpu_xmm13   : return fpu_xmm13;
        case gdb_fpu_xmm14   : return fpu_xmm14;
        case gdb_fpu_xmm15   : return fpu_xmm15;
        case gdb_fpu_mxcsr   : return fpu_mxcsr;
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
RegisterContext_x86_64::HardwareSingleStep(bool enable)
{
    enum { TRACE_BIT = 0x100 };
    uint64_t rflags;

    if ((rflags = ReadRegisterAsUnsigned(gpr_rflags, -1UL)) == -1UL)
        return false;
    
    if (enable)
    {
        if (rflags & TRACE_BIT)
            return true;

        rflags |= TRACE_BIT;
    }
    else
    {
        if (!(rflags & TRACE_BIT))
            return false;

        rflags &= ~TRACE_BIT;
    }

    return WriteRegisterFromUnsigned(gpr_rflags, rflags);
}

bool
RegisterContext_x86_64::ReadGPR()
{
     ProcessMonitor &monitor = GetMonitor();
     return monitor.ReadGPR(&user.regs);
}

bool
RegisterContext_x86_64::ReadFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.ReadFPR(&user.i387);
}

bool
RegisterContext_x86_64::WriteGPR()
{
     ProcessMonitor &monitor = GetMonitor();
     return monitor.WriteGPR(&user.regs);
}

bool
RegisterContext_x86_64::WriteFPR()
{
    ProcessMonitor &monitor = GetMonitor();
    return monitor.WriteFPR(&user.i387);
}
