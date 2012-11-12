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
    : ProcessPOSIX(target, listener)
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
const char *
ProcessLinux::GetPluginNameStatic()
{
    return "linux";
}

const char *
ProcessLinux::GetPluginDescriptionStatic()
{
    return "Process plugin for Linux";
}


bool
ProcessLinux::UpdateThreadList(ThreadList &old_thread_list, ThreadList &new_thread_list)
{
    LogSP log (ProcessPOSIXLog::GetLogIfAllCategoriesSet (POSIX_LOG_THREAD));
    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessLinux::%s() (pid = %i)", __FUNCTION__, GetID());

    // Update the process thread list with this new thread.
    // FIXME: We should be using tid, not pid.
    assert(m_monitor);
    ThreadSP thread_sp (old_thread_list.FindThreadByID (GetID(), false));
    if (!thread_sp) {
        thread_sp.reset(new POSIXThread(*this, GetID()));
    }

    if (log && log->GetMask().Test(POSIX_LOG_VERBOSE))
        log->Printf ("ProcessLinux::%s() updated pid = %i", __FUNCTION__, GetID());
    new_thread_list.AddThread(thread_sp);

    return new_thread_list.GetSize(false) > 0;
}


//------------------------------------------------------------------------------
// ProcessInterface protocol.

const char *
ProcessLinux::GetPluginName()
{
    return "process.linux";
}

const char *
ProcessLinux::GetShortPluginName()
{
    return "process.linux";
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
