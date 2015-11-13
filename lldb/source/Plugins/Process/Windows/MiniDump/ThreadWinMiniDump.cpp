//===-- ThreadWinMiniDump.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThreadWinMiniDump.h"

#include "lldb/Host/HostInfo.h"
#include "lldb/Host/windows/windows.h"
#include <DbgHelp.h>

#include "ProcessWinMiniDump.h"
#if defined(_WIN64)
#include "x64/RegisterContextWindowsMiniDump_x64.h"
#else
#include "x86/RegisterContextWindowsMiniDump_x86.h"
#endif

using namespace lldb;
using namespace lldb_private;

// This is a minimal implementation in order to get something running.  It will
// be fleshed out as more mini-dump functionality is added.

class ThreadWinMiniDump::Data {
  public:
    Data() : m_context(nullptr) {}
    const CONTEXT *m_context;
};

ThreadWinMiniDump::ThreadWinMiniDump(lldb_private::Process &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_data(new Data)
{
}

ThreadWinMiniDump::~ThreadWinMiniDump()
{
}

void
ThreadWinMiniDump::RefreshStateAfterStop()
{
}

lldb::RegisterContextSP
ThreadWinMiniDump::GetRegisterContext()
{
    if (m_reg_context_sp.get() == NULL) {
        m_reg_context_sp = CreateRegisterContextForFrame (NULL);
    }
    return m_reg_context_sp;
}

lldb::RegisterContextSP
ThreadWinMiniDump::CreateRegisterContextForFrame(lldb_private::StackFrame *frame)
{
    const uint32_t concrete_frame_idx = (frame) ? frame->GetConcreteFrameIndex() : 0;
    RegisterContextSP reg_ctx_sp;
    ArchSpec arch = HostInfo::GetArchitecture();
    switch (arch.GetMachine())
    {
        case llvm::Triple::x86:
#if defined(_WIN64)
            // FIXME: This is a Wow64 process, create a RegisterContextWindows_Wow64
#else
            reg_ctx_sp.reset(new RegisterContextWindowsMiniDump_x86(*this, concrete_frame_idx, m_data->m_context));
#endif
            break;
        case llvm::Triple::x86_64:
#if defined(_WIN64)
            reg_ctx_sp.reset(new RegisterContextWindowsMiniDump_x64(*this, concrete_frame_idx, m_data->m_context));
#else
            // LLDB is 32-bit, but the target process is 64-bit.  We probably can't debug this.
#endif
        default:
            break;
    }
    return reg_ctx_sp;
}

void
ThreadWinMiniDump::ClearStackFrames()
{
}

void
ThreadWinMiniDump::SetContext(const void *context)
{
    if (m_data)
    {
        m_data->m_context = static_cast<const CONTEXT *>(context);
    }
}

bool
ThreadWinMiniDump::CalculateStopInfo()
{
    return false;
}
