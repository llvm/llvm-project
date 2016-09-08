//===-- LocalDebugDelegate.cpp ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "LocalDebugDelegate.h"
#include "ProcessWindowsLive.h"

using namespace lldb;
using namespace lldb_private;

LocalDebugDelegate::LocalDebugDelegate(ProcessWP process)
    : m_process(process) {}

void LocalDebugDelegate::OnExitProcess(uint32_t exit_code) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnExitProcess(exit_code);
}

void LocalDebugDelegate::OnDebuggerConnected(lldb::addr_t image_base) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnDebuggerConnected(image_base);
}

ExceptionResult
LocalDebugDelegate::OnDebugException(bool first_chance,
                                     const ExceptionRecord &record) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    return process->OnDebugException(first_chance, record);
  else
    return ExceptionResult::MaskException;
}

void LocalDebugDelegate::OnCreateThread(const HostThread &thread) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnCreateThread(thread);
}

void LocalDebugDelegate::OnExitThread(lldb::tid_t thread_id,
                                      uint32_t exit_code) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnExitThread(thread_id, exit_code);
}

void LocalDebugDelegate::OnLoadDll(const lldb_private::ModuleSpec &module_spec,
                                   lldb::addr_t module_addr) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnLoadDll(module_spec, module_addr);
}

void LocalDebugDelegate::OnUnloadDll(lldb::addr_t module_addr) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnUnloadDll(module_addr);
}

void LocalDebugDelegate::OnDebugString(const std::string &string) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnDebugString(string);
}

void LocalDebugDelegate::OnDebuggerError(const Error &error, uint32_t type) {
  if (ProcessWindowsLiveSP process = GetProcessPointer())
    process->OnDebuggerError(error, type);
}

ProcessWindowsLiveSP LocalDebugDelegate::GetProcessPointer() {
  ProcessSP process = m_process.lock();
  return std::static_pointer_cast<ProcessWindowsLive>(process);
}
