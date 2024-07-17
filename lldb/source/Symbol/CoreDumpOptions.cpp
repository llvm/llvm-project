//===-- CoreDumpOptions.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/CoreDumpOptions.h"

using namespace lldb;
using namespace lldb_private;

void CoreDumpOptions::SetCoreDumpPluginName(const char * name) {
  m_plugin_name = name;
}

void CoreDumpOptions::SetCoreDumpStyle(lldb::SaveCoreStyle style) {
  m_style = style;
}

void CoreDumpOptions::SetOutputFile(FileSpec file) {
  m_file = file;
}

std::optional<std::string> CoreDumpOptions::GetCoreDumpPluginName() const {
  return m_plugin_name;
}

lldb::SaveCoreStyle CoreDumpOptions::GetCoreDumpStyle() const {
  // If unspecified, default to stack only
  if (m_style == lldb::eSaveCoreUnspecified)
    return lldb::eSaveCoreStackOnly;
  return m_style.value();
}

const std::optional<lldb_private::FileSpec> CoreDumpOptions::GetOutputFile() const {
  return m_file;
}

void CoreDumpOptions::Clear() {
  m_file = std::nullopt;
  m_plugin_name = std::nullopt;
  m_style = std::nullopt;
}
