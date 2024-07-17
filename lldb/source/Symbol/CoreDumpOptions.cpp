//===-- CoreDumpOptions.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/CoreDumpOptions.h"
#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

Status CoreDumpOptions::SetPluginName(const char *name) {
  Status error;
  if (!name || !name[0]) {
    m_plugin_name = std::nullopt;
  }

  if (!PluginManager::IsRegisteredObjectFilePluginName(name)) {
    error.SetErrorStringWithFormat(
        "plugin name '%s' is not a valid ObjectFile plugin name", name);
    return error;
  }

  m_plugin_name = name;
  return error;
}

void CoreDumpOptions::SetStyle(lldb::SaveCoreStyle style) { m_style = style; }

void CoreDumpOptions::SetOutputFile(FileSpec file) { m_file = file; }

std::optional<std::string> CoreDumpOptions::GetPluginName() const {
  return m_plugin_name;
}

lldb::SaveCoreStyle CoreDumpOptions::GetStyle() const {
  return m_style.value_or(lldb::eSaveCoreUnspecified);
}

const std::optional<lldb_private::FileSpec>
CoreDumpOptions::GetOutputFile() const {
  return m_file;
}

void CoreDumpOptions::Clear() {
  m_file = std::nullopt;
  m_plugin_name = std::nullopt;
  m_style = std::nullopt;
}
