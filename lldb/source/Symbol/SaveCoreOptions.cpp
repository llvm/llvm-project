//===-- SaveCoreOptions.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SaveCoreOptions.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"

using namespace lldb;
using namespace lldb_private;

Status SaveCoreOptions::SetPluginName(const char *name) {
  Status error;
  if (!name || !name[0]) {
    m_plugin_name = std::nullopt;
    return error;
  }

  if (!PluginManager::IsRegisteredObjectFilePluginName(name)) {
    error.SetErrorStringWithFormat(
        "plugin name '%s' is not a valid ObjectFile plugin name", name);
    return error;
  }

  m_plugin_name = name;
  return error;
}

void SaveCoreOptions::SetStyle(lldb::SaveCoreStyle style) { m_style = style; }

void SaveCoreOptions::SetOutputFile(FileSpec file) { m_file = file; }

std::optional<std::string> SaveCoreOptions::GetPluginName() const {
  return m_plugin_name;
}

lldb::SaveCoreStyle SaveCoreOptions::GetStyle() const {
  return m_style.value_or(lldb::eSaveCoreUnspecified);
}

const std::optional<lldb_private::FileSpec>
SaveCoreOptions::GetOutputFile() const {
  return m_file;
}

Status SaveCoreOptions::SetProcess(lldb::ProcessSP process_sp) {
  Status error;
  if (!process_sp) {
    ClearProcessSpecificData();
    m_process_sp.reset();
    return error;
  }

  if (!process_sp->IsValid()) {
    error.SetErrorString("Cannot assign an invalid process.");
    return error;
  }

  if (m_process_sp == process_sp)
    return error;
    
  ClearProcessSpecificData();
  m_process_sp = process_sp;
  return error;
}

Status SaveCoreOptions::AddThread(lldb::ThreadSP thread_sp) {
  Status error;
  if (!thread_sp) {
    error.SetErrorString("invalid thread");
    return error;
  }

  if (m_process_sp) {
    if (m_process_sp != thread_sp->GetProcess()) {
      error.SetErrorString("Cannot add a thread from a different process.");
      return error;
    }
  } else {
    m_process_sp = thread_sp->GetProcess();
  }

  m_threads_to_save[thread_sp->GetID()];
  return error;
}

bool SaveCoreOptions::RemoveThread(lldb::ThreadSP thread_sp) {
  return thread_sp && m_threads_to_save.erase(thread_sp->GetID()) > 0;
}

bool SaveCoreOptions::ShouldThreadBeSaved(lldb::tid_t tid) const {
  // If the user specified no threads to save, then we save all threads.
  if (m_threads_to_save.empty())
    return true;
  return m_threads_to_save.count(tid) > 0;
}

Status SaveCoreOptions::EnsureValidConfiguration(
    lldb::ProcessSP process_to_save) const {
  Status error;
  std::string error_str;
  if (!m_threads_to_save.empty() && GetStyle() == lldb::eSaveCoreFull)
    error_str += "Cannot save a full core with a subset of threads\n";

  if (m_process_sp && m_process_sp != process_to_save)
    error_str += "Cannot save core for process using supplied core options. "
                 "Options were constructed targeting a different process. \n";

  if (!error_str.empty())
    error.SetErrorString(error_str);

  return error;
}

void SaveCoreOptions::ClearProcessSpecificData() { 
  m_threads_to_save.clear(); 
}

void SaveCoreOptions::Clear() {
  m_file = std::nullopt;
  m_plugin_name = std::nullopt;
  m_style = std::nullopt;
  m_threads_to_save.clear();
  m_process_sp.reset();
}
