//===-- SaveCoreOptions.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SaveCoreOptions.h"
#include "lldb/Core/PluginManager.h"

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

void SaveCoreOptions::AddThread(lldb::tid_t tid) {
  if (m_threads_to_save.count(tid) == 0)
    m_threads_to_save.emplace(tid);
}

bool SaveCoreOptions::RemoveThread(lldb::tid_t tid) {
  if (m_threads_to_save.count(tid) == 0) {
    m_threads_to_save.erase(tid);
    return true;
  }

  return false;
}

size_t SaveCoreOptions::GetNumThreads() const {
  return m_threads_to_save.size();
}

int64_t SaveCoreOptions::GetThreadAtIndex(size_t index) const {
  auto iter = m_threads_to_save.begin();
  while (index >= 0 && iter != m_threads_to_save.end()) {
    if (index == 0)
      return *iter;
    index--;
    iter++;
  }

  return -1;
}

bool SaveCoreOptions::ShouldSaveThread(lldb::tid_t tid) const {
  // If the user specified no threads to save, then we save all threads.
  if (m_threads_to_save.empty())
    return true;
  return m_threads_to_save.count(tid) > 0;
}

Status SaveCoreOptions::EnsureValidConfiguration() const {
  Status error;
  std::string error_str;
  if (!m_threads_to_save.empty() && GetStyle() == lldb::eSaveCoreFull) {
    error_str += "Cannot save a full core with a subset of threads\n";
  }

  if (!error_str.empty())
    error.SetErrorString(error_str);

  return error;
}

void SaveCoreOptions::Clear() {
  m_file = std::nullopt;
  m_plugin_name = std::nullopt;
  m_style = std::nullopt;
  m_threads_to_save.clear();
}
