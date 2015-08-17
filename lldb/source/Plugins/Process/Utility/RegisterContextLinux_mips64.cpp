//===-- RegisterContextLinux_mips64.cpp ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#if defined (__mips__)

#include <vector>
#include <stddef.h>

// For GDB, GCC and DWARF Register numbers
#include "RegisterContextLinux_mips64.h"

// For GP and FP buffers
#include "RegisterContext_mips.h"

// Internal codes for all mips32 and mips64 registers
#include "lldb-mips-linux-register-enums.h"

using namespace lldb;
using namespace lldb_private;

//---------------------------------------------------------------------------
// Include RegisterInfos_mips64 to declare our g_register_infos_mips64 structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS64_STRUCT
#define LINUX_MIPS64
#include "RegisterInfos_mips64.h"
#undef LINUX_MIPS64
#undef DECLARE_REGISTER_INFOS_MIPS64_STRUCT

//---------------------------------------------------------------------------
// Include RegisterInfos_mips to declare our g_register_infos_mips structure.
//---------------------------------------------------------------------------
#define DECLARE_REGISTER_INFOS_MIPS_STRUCT
#include "RegisterInfos_mips.h"
#undef DECLARE_REGISTER_INFOS_MIPS_STRUCT

static const RegisterInfo *
GetRegisterInfoPtr (const ArchSpec &target_arch)
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            return g_register_infos_mips64;
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return g_register_infos_mips;
        default:
            assert(false && "Unhandled target architecture.");
            return nullptr;
    }
}

static uint32_t
GetRegisterInfoCount (const ArchSpec &target_arch)
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::mips64:
        case llvm::Triple::mips64el:
            return static_cast<uint32_t> (sizeof (g_register_infos_mips64) / sizeof (g_register_infos_mips64 [0]));
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return static_cast<uint32_t> (sizeof (g_register_infos_mips) / sizeof (g_register_infos_mips [0]));
        default:
            assert(false && "Unhandled target architecture.");
            return 0;
    }
}

uint32_t
GetUserRegisterInfoCount (const ArchSpec &target_arch)
{
    switch (target_arch.GetMachine())
    {
        case llvm::Triple::mips:
        case llvm::Triple::mipsel:
            return static_cast<uint32_t> (k_num_user_registers_mips);
        case llvm::Triple::mips64el:
        case llvm::Triple::mips64:
            return static_cast<uint32_t> (k_num_user_registers_mips64);
        default:
            assert(false && "Unhandled target architecture.");
            return 0;
    }
}

RegisterContextLinux_mips64::RegisterContextLinux_mips64(const ArchSpec &target_arch) :
    lldb_private::RegisterInfoInterface(target_arch),
    m_register_info_p (GetRegisterInfoPtr (target_arch)),
    m_register_info_count (GetRegisterInfoCount (target_arch)),
    m_user_register_count (GetUserRegisterInfoCount (target_arch))
{
}

size_t
RegisterContextLinux_mips64::GetGPRSize() const
{
    return sizeof(GPR_linux_mips);
}

const RegisterInfo *
RegisterContextLinux_mips64::GetRegisterInfo() const
{
    return m_register_info_p;
}

uint32_t
RegisterContextLinux_mips64::GetRegisterCount () const
{
    return m_register_info_count;
}

uint32_t
RegisterContextLinux_mips64::GetUserRegisterCount () const
{
    return m_user_register_count;
}

#endif
