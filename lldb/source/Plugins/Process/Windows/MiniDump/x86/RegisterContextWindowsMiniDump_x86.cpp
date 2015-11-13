//===-- RegisterContextWindowsMiniDump_x86.cpp ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/lldb-private-types.h"
#include "lldb/Host/windows/windows.h"

#include "RegisterContextWindowsMiniDump_x86.h"

using namespace lldb;

namespace lldb_private
{

RegisterContextWindowsMiniDump_x86::RegisterContextWindowsMiniDump_x86(Thread &thread, uint32_t concrete_frame_idx, const CONTEXT *context)
    : RegisterContextWindows_x86(thread, concrete_frame_idx)
{
    if (context)
    {
        m_context = *context;
        m_context_stale = false;
    }
}

RegisterContextWindowsMiniDump_x86::~RegisterContextWindowsMiniDump_x86()
{
}

bool
RegisterContextWindowsMiniDump_x86::WriteRegister(const RegisterInfo * /* reg_info */, const RegisterValue & /* reg_value */)
{
    return false;
}

bool
RegisterContextWindowsMiniDump_x86::CacheAllRegisterValues()
{
    // Since this is post-mortem debugging, we either have the context or we don't.
    return !m_context_stale;
}

}  // namespace lldb_private
