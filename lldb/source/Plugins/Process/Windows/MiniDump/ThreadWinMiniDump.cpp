//===-- ThreadWinMiniDump.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ThreadWinMiniDump.h"

// Windows includes
#include "lldb/Host/windows/windows.h"
#include <DbgHelp.h>

#include "ProcessWinMiniDump.h"
#include "RegisterContextWindowsMiniDump.h"

using namespace lldb;
using namespace lldb_private;

// This is a minimal implementation in order to get something running.  It will
// be fleshed out as more mini-dump functionality is added.

ThreadWinMiniDump::ThreadWinMiniDump(lldb_private::Process &process, lldb::tid_t tid) :
    Thread(process, tid),
    m_thread_name()
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
    RegisterContextSP reg_ctx_sp(new RegisterContextWindowsMiniDump(*this, concrete_frame_idx));
    return reg_ctx_sp;
}

void
ThreadWinMiniDump::ClearStackFrames()
{
}

const char *
ThreadWinMiniDump::GetName()
{
    return m_thread_name.empty() ? nullptr : m_thread_name.c_str();
}

void
ThreadWinMiniDump::SetName(const char *name)
{
    if (name && name[0])
        m_thread_name.assign(name);
    else
        m_thread_name.clear();
}

bool ThreadWinMiniDump::CalculateStopInfo()
{
    return false;
}
