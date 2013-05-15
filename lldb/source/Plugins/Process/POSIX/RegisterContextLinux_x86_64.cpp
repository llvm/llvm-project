//===-- RegisterContextLinux_x86_64.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "RegisterContextLinux_x86_64.h"

using namespace lldb_private;

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname)                                                 \
    (offsetof(GPR, regname))

// Update the Linux specific information (offset and size).
#define UPDATE_GPR_INFO(reg)                                                \
do {                                                                        \
    m_register_infos[gpr_##reg].byte_size = sizeof(GPR::reg);               \
    m_register_infos[gpr_##reg].byte_offset = GPR_OFFSET(reg);              \
} while(false);

#define UPDATE_I386_GPR_INFO(i386_reg, reg)                                 \
do {                                                                        \
    m_register_infos[gpr_##i386_reg].byte_offset = GPR_OFFSET(reg);         \
} while(false);

#define DR_OFFSET(reg_index)                                                \
    (offsetof(UserArea, u_debugreg[reg_index]))

#define UPDATE_DR_INFO(reg_index)                                                \
do {                                                                             \
    m_register_infos[dr##reg_index].byte_size = sizeof(UserArea::u_debugreg[0]); \
    m_register_infos[dr##reg_index].byte_offset = DR_OFFSET(reg_index);          \
} while(false);

typedef struct _GPR
{
    uint64_t r15;
    uint64_t r14;
    uint64_t r13;
    uint64_t r12;
    uint64_t rbp;
    uint64_t rbx;
    uint64_t r11;
    uint64_t r10;
    uint64_t r9;
    uint64_t r8;
    uint64_t rax;
    uint64_t rcx;
    uint64_t rdx;
    uint64_t rsi;
    uint64_t rdi;
    uint64_t orig_ax;
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
    uint64_t fs_base;
    uint64_t gs_base;
    uint64_t ds;
    uint64_t es;
    uint64_t fs;
    uint64_t gs;
} GPR;

typedef RegisterContext_x86_64::FXSAVE FXSAVE;

struct UserArea
{
    GPR      gpr;           // General purpose registers.
    int32_t  fpvalid;       // True if FPU is being used.
    int32_t  pad0;
    FXSAVE   i387;          // General purpose floating point registers (see FPR for extended register sets).
    uint64_t tsize;         // Text segment size.
    uint64_t dsize;         // Data segment size.
    uint64_t ssize;         // Stack segment size.
    uint64_t start_code;    // VM address of text.
    uint64_t start_stack;   // VM address of stack bottom (top in rsp).
    int64_t  signal;        // Signal causing core dump.
    int32_t  reserved;      // Unused.
    int32_t  pad1;
    uint64_t ar0;           // Location of GPR's.
    FXSAVE*  fpstate;       // Location of FPR's.
    uint64_t magic;         // Identifier for core dumps.
    char     u_comm[32];    // Command causing core dump.
    uint64_t u_debugreg[8]; // Debug registers (DR0 - DR7).
    uint64_t error_code;    // CPU error code.
    uint64_t fault_address; // Control register CR3.
};

RegisterInfo *RegisterContextLinux_x86_64::m_register_infos = nullptr;

RegisterContextLinux_x86_64::RegisterContextLinux_x86_64(Thread &thread, uint32_t concrete_frame_idx):
    RegisterContext_x86_64(thread, concrete_frame_idx)
{
}

RegisterContextLinux_x86_64::~RegisterContextLinux_x86_64()
{
    if (m_register_infos)
        delete m_register_infos;
    m_register_infos = nullptr;
}

size_t
RegisterContextLinux_x86_64::GetGPRSize()
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextLinux_x86_64::GetRegisterInfo()
{
    // Allocate RegisterInfo only once
    if (!m_register_infos)
    {
        m_register_infos = new RegisterInfo[k_num_registers];
        // Copy the register information from base class
        if (m_register_infos)
        {
            memcpy(m_register_infos, RegisterContext_x86_64::GetRegisterInfo(),
                   sizeof(RegisterInfo) * k_num_registers);
            // Update the Linux specific register information (offset and size).
            UpdateRegisterInfo();
        }
    }
    return m_register_infos;
}

void
RegisterContextLinux_x86_64::UpdateRegisterInfo()
{
    UPDATE_GPR_INFO(rax);
    UPDATE_GPR_INFO(rbx);
    UPDATE_GPR_INFO(rcx);
    UPDATE_GPR_INFO(rdx);
    UPDATE_GPR_INFO(rdi);
    UPDATE_GPR_INFO(rsi);
    UPDATE_GPR_INFO(rbp);
    UPDATE_GPR_INFO(rsp);
    UPDATE_GPR_INFO(r8);
    UPDATE_GPR_INFO(r9);
    UPDATE_GPR_INFO(r10);
    UPDATE_GPR_INFO(r11);
    UPDATE_GPR_INFO(r12);
    UPDATE_GPR_INFO(r13);
    UPDATE_GPR_INFO(r14);
    UPDATE_GPR_INFO(r15);
    UPDATE_GPR_INFO(rip);
    UPDATE_GPR_INFO(rflags);
    UPDATE_GPR_INFO(cs);
    UPDATE_GPR_INFO(fs);
    UPDATE_GPR_INFO(gs);
    UPDATE_GPR_INFO(ss);
    UPDATE_GPR_INFO(ds);
    UPDATE_GPR_INFO(es);

    UPDATE_I386_GPR_INFO(eax, rax);
    UPDATE_I386_GPR_INFO(ebx, rbx);
    UPDATE_I386_GPR_INFO(ecx, rcx);
    UPDATE_I386_GPR_INFO(edx, rdx);
    UPDATE_I386_GPR_INFO(edi, rdi);
    UPDATE_I386_GPR_INFO(esi, rsi);
    UPDATE_I386_GPR_INFO(ebp, rbp);
    UPDATE_I386_GPR_INFO(esp, rsp);
    UPDATE_I386_GPR_INFO(eip, rip);
    UPDATE_I386_GPR_INFO(eflags, rflags);

    UPDATE_DR_INFO(0);
    UPDATE_DR_INFO(1);
    UPDATE_DR_INFO(2);
    UPDATE_DR_INFO(3);
    UPDATE_DR_INFO(4);
    UPDATE_DR_INFO(5);
    UPDATE_DR_INFO(6);
    UPDATE_DR_INFO(7);
}

