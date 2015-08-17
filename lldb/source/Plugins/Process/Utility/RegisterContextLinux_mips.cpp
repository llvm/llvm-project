//===-- RegisterContextLinux_mips.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include <vector>
#include <stddef.h>

// For GDB, GCC and DWARF Register numbers
#include "RegisterContextLinux_mips.h"

// Internal codes for mips registers
#include "lldb-mips-linux-register-enums.h"

// For GP and FP buffers
#include "RegisterContext_mips.h"

using namespace lldb_private;
using namespace lldb;

//---------------------------------------------------------------------------
// Include RegisterInfos_mips to declare our g_register_infos_mips structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS_STRUCT
#include "RegisterInfos_mips.h"
#undef DECLARE_REGISTER_INFOS_MIPS_STRUCT

RegisterContextLinux_mips::RegisterContextLinux_mips(const ArchSpec &target_arch) :
    RegisterInfoInterface(target_arch)
{
}

size_t
RegisterContextLinux_mips::GetGPRSize() const
{
    return sizeof(GPR_linux_mips);
}

const RegisterInfo *
RegisterContextLinux_mips::GetRegisterInfo() const
{
    switch (m_target_arch.GetMachine())
    {
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return g_register_infos_mips;
        default:
            assert(false && "Unhandled target architecture.");
            return NULL;
    }
}

uint32_t
RegisterContextLinux_mips::GetRegisterCount () const
{
    return static_cast<uint32_t> (sizeof (g_register_infos_mips) / sizeof (g_register_infos_mips [0]));
}

uint32_t
RegisterContextLinux_mips::GetUserRegisterCount () const
{
    return static_cast<uint32_t> (k_num_user_registers_mips);
}
