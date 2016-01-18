//===-- RegisterContextWindowsMiniDump.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-types.h"
#include "lldb/Core/DataBufferHeap.h"
#include "lldb/Core/Error.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "RegisterContextWindowsMiniDump.h"

#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

// This is a do-nothing stub implementation for now.

RegisterContextWindowsMiniDump::RegisterContextWindowsMiniDump(Thread &thread, uint32_t concrete_frame_idx)
    : RegisterContextWindows(thread, concrete_frame_idx)
{
}

RegisterContextWindowsMiniDump::~RegisterContextWindowsMiniDump()
{
}

void
RegisterContextWindowsMiniDump::InvalidateAllRegisters()
{
}

size_t
RegisterContextWindowsMiniDump::GetRegisterCount()
{
    return 0;
}

const RegisterInfo *
RegisterContextWindowsMiniDump::GetRegisterInfoAtIndex(size_t reg)
{
    return nullptr;
}

size_t
RegisterContextWindowsMiniDump::GetRegisterSetCount()
{
    return 0;
}

const RegisterSet *
RegisterContextWindowsMiniDump::GetRegisterSet(size_t reg_set)
{
    return nullptr;
}

bool
RegisterContextWindowsMiniDump::ReadRegister(const RegisterInfo *reg_info, RegisterValue &reg_value)
{
    return false;
}

bool
RegisterContextWindowsMiniDump::WriteRegister (const RegisterInfo *reg_info, const RegisterValue &reg_value)
{
    return false;
}

bool
RegisterContextWindowsMiniDump::ReadAllRegisterValues(lldb::DataBufferSP &data_sp)
{
    return false;
}

bool
RegisterContextWindowsMiniDump::WriteAllRegisterValues(const lldb::DataBufferSP &data_sp)
{
    return false;
}

uint32_t
RegisterContextWindowsMiniDump::ConvertRegisterKindToRegisterNumber(lldb::RegisterKind kind, uint32_t num)
{
    const uint32_t num_regs = GetRegisterCount();

    assert(kind < kNumRegisterKinds);
    for (uint32_t reg_idx = 0; reg_idx < num_regs; ++reg_idx)
    {
        const RegisterInfo *reg_info = GetRegisterInfoAtIndex(reg_idx);

        if (reg_info->kinds[kind] == num)
            return reg_idx;
    }

    return LLDB_INVALID_REGNUM;
}

uint32_t
RegisterContextWindowsMiniDump::NumSupportedHardwareBreakpoints()
{
    // Support for hardware breakpoints not yet implemented.
    return 0;
}

uint32_t
RegisterContextWindowsMiniDump::SetHardwareBreakpoint(lldb::addr_t addr, size_t size)
{
    return 0;
}

bool
RegisterContextWindowsMiniDump::ClearHardwareBreakpoint(uint32_t hw_idx)
{
    return false;
}

uint32_t
RegisterContextWindowsMiniDump::NumSupportedHardwareWatchpoints()
{
    // Support for hardware watchpoints not yet implemented.
    return 0;
}

uint32_t
RegisterContextWindowsMiniDump::SetHardwareWatchpoint(lldb::addr_t addr, size_t size, bool read, bool write)
{
    return 0;
}

bool
RegisterContextWindowsMiniDump::ClearHardwareWatchpoint(uint32_t hw_index)
{
    return false;
}

bool
RegisterContextWindowsMiniDump::HardwareSingleStep(bool enable)
{
    return false;
}
