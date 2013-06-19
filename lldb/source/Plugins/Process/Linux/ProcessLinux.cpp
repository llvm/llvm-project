//===-- ProcessLinux.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// C Includes
#include <errno.h>

// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/State.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Target.h"

#include "ProcessLinux.h"
#include "ProcessPOSIXLog.h"
#include "Plugins/Process/Utility/InferiorCallPOSIX.h"
#include "ProcessMonitor.h"
#include "POSIXThread.h"

using namespace lldb;
using namespace lldb_private;

//------------------------------------------------------------------------------
// Static functions.

ProcessSP
ProcessLinux::CreateInstance(Target &target, Listener &listener, const FileSpec *)
{
    return ProcessSP(new ProcessLinux(target, listener));
}

void
ProcessLinux::Initialize()
{
    static bool g_initialized = false;

    if (!g_initialized)
    {
        g_initialized = true;
        PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                      GetPluginDescriptionStatic(),
                                      CreateInstance);

        Log::Callbacks log_callbacks = {
            ProcessPOSIXLog::DisableLog,
            ProcessPOSIXLog::EnableLog,
            ProcessPOSIXLog::ListLogCategories
        };
        
        Log::RegisterLogChannel (ProcessLinux::GetPluginNameStatic(), log_callbacks);
        ProcessPOSIXLog::RegisterPluginName(GetPluginNameStatic());
    }
}

//------------------------------------------------------------------------------
// Constructors and destructors.

ProcessLinux::ProcessLinux(Target& target, Listener &listener)
    : ProcessPOSIX(target, listener), m_stopping_threads(false)
{
#if 0
    // FIXME: Putting this code in the ctor and saving the byte order in a
    // member variable is a hack to avoid const qual issues in GetByteOrder.
    ObjectFile *obj_file = GetTarget().GetExecutableModule()->GetObjectFile();
    m_byte_order = obj_file->GetByteOrder();
#else
    // XXX: Will work only for local processes.
    m_byte_order = lldb::endian::InlHostByteOrder();
#endif
}

void
ProcessLinux::Terminate()
{
}
lldb_private::ConstString
ProcessLinux::GetPluginNameStatic()
{
    static ConstString g_name("linux");
    return g_name;
}

const char *
ProcessLinux::GetPluginDescriptionStatic()
{
    return "Process plugin for Linux";
}


bool
ProcessLinux::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    new_thread_list = old_thread_list;
    return new_thread_list.GetSize(false) > 0;
}


//------------------------------------------------------------------------------
// ProcessInterface protocol.

lldb_private::ConstString
ProcessLinux::GetPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
ProcessLinux::GetPluginVersion()
{
    return 1;
}

void
ProcessLinux::GetPluginCommandHelp(const char *command, Stream *strm)
{
}

Error
ProcessLinux::ExecutePluginCommand(Args &command, Stream *strm)
{
    return Error(1, eErrorTypeGeneric);
}

Log *
ProcessLinux::EnablePluginLogging(Stream *strm, Args &command)
{
    return NULL;
}

// ProcessPOSIX override
void
ProcessLinux::StopAllThreads(lldb::tid_t stop_tid)
{
    // If a breakpoint occurs while we're stopping threads, we'll get back
    // here, but we don't want to do it again.  Only the MonitorChildProcess
    // thread calls this function, so we don't need to protect this flag.
    if (m_stopping_threads)
      return;
    m_stopping_threads = true;

    Log *log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_PROCESS));
    if (log)
        log->Printf ("ProcessLinux::%s() stopping all threads", __FUNCTION__);

    // Walk the thread list and stop the other threads.  The thread that caused
    // the stop should already be marked as stopped before we get here.
    Mutex::Locker thread_list_lock(m_thread_list.GetMutex());

    uint32_t thread_count = m_thread_list.GetSize(false);
    for (uint32_t i = 0; i < thread_count; ++i)
    {
        POSIXThread *thread = static_cast<POSIXThread*>(
            m_thread_list.GetThreadAtIndex(i, false).get());
        assert(thread);
        lldb::tid_t tid = thread->GetID();
        if (!StateIsStoppedState(thread->GetState(), false))
            m_monitor->StopThread(tid);
    }

    m_stopping_threads = false;

    if (log)
        log->Printf ("ProcessLinux::%s() finished", __FUNCTION__);
}
