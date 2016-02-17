//===-- IDebugDelegate.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_Plugins_Process_Windows_IDebugDelegate_H_
#define liblldb_Plugins_Process_Windows_IDebugDelegate_H_

#include "ForwardDecl.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"
#include <string>

namespace lldb_private
{
class Error;
class HostThread;

//----------------------------------------------------------------------
// IDebugDelegate
//
// IDebugDelegate defines an interface which allows implementors to receive
// notification of events that happen in a debugged process.
//----------------------------------------------------------------------
class IDebugDelegate
{
  public:
    virtual ~IDebugDelegate() {}

    virtual void OnExitProcess(uint32_t exit_code) = 0;
    virtual void OnDebuggerConnected(lldb::addr_t image_base) = 0;
    virtual ExceptionResult OnDebugException(bool first_chance, const ExceptionRecord &record) = 0;
    virtual void OnCreateThread(const HostThread &thread) = 0;
    virtual void OnExitThread(lldb::tid_t thread_id, uint32_t exit_code) = 0;
    virtual void OnLoadDll(const ModuleSpec &module_spec, lldb::addr_t module_addr) = 0;
    virtual void OnUnloadDll(lldb::addr_t module_addr) = 0;
    virtual void OnDebugString(const std::string &string) = 0;
    virtual void OnDebuggerError(const Error &error, uint32_t type) = 0;
};
}

#endif
