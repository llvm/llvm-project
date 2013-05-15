//===-- RegisterContextFreeBSD_x86_64.h ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "RegisterContextFreeBSD_x86_64.h"

using namespace lldb_private;

// Computes the offset of the given GPR in the user data area.
#define GPR_OFFSET(regname)                                                 \
    (offsetof(GPR, regname))

// Update the FreeBSD specific information (offset and size).
#define UPDATE_GPR_INFO(reg)                                                \
do {                                                                        \
    m_register_infos[gpr_##reg].byte_size = sizeof(GPR::reg);               \
    m_register_infos[gpr_##reg].byte_offset = GPR_OFFSET(reg);              \
} while(false);

#define UPDATE_I386_GPR_INFO(i386_reg, reg)                                 \
do {                                                                        \
    m_register_infos[gpr_##i386_reg].byte_offset = GPR_OFFSET(reg);         \
} while(false);

typedef struct _GPR
{
    uint64_t r15;
    uint64_t r14;
    uint64_t r13;
    uint64_t r12;
    uint64_t r11;
    uint64_t r10;
    uint64_t r9;
    uint64_t r8;
    uint64_t rdi;
    uint64_t rsi;
    uint64_t rbp;
    uint64_t rbx;
    uint64_t rdx;
    uint64_t rcx;
    uint64_t rax;
    uint32_t trapno;
    uint16_t fs;
    uint16_t gs;
    uint32_t err;
    uint16_t es;
    uint16_t ds;
    uint64_t rip;
    uint64_t cs;
    uint64_t rflags;
    uint64_t rsp;
    uint64_t ss;
} GPR;

RegisterInfo *RegisterContextFreeBSD_x86_64::m_register_infos = nullptr;

RegisterContextFreeBSD_x86_64::RegisterContextFreeBSD_x86_64(Thread &thread, uint32_t concrete_frame_idx):
    RegisterContext_x86_64(thread, concrete_frame_idx)
{
}

RegisterContextFreeBSD_x86_64::~RegisterContextFreeBSD_x86_64()
{
    if (m_register_infos)
        delete m_register_infos;
    m_register_infos = nullptr;
}

size_t
RegisterContextFreeBSD_x86_64::GetGPRSize()
{
    return sizeof(GPR);
}

const RegisterInfo *
RegisterContextFreeBSD_x86_64::GetRegisterInfo()
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
RegisterContextFreeBSD_x86_64::UpdateRegisterInfo()
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
}

