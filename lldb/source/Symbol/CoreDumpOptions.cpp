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

void CoreDumpOptions::SetCoreDumpPluginName(const llvm::StringRef name) {
  m_core_dump_plugin_name = name.data();
}

std::optional<llvm::StringRef> CoreDumpOptions::GetCoreDumpPluginName() const { 
  if (!m_core_dump_plugin_name)
    return std::nullopt;
  return m_core_dump_plugin_name->data();
}

void CoreDumpOptions::SetCoreDumpStyle(lldb::SaveCoreStyle style) {
  m_core_dump_style = style;
}

lldb::SaveCoreStyle CoreDumpOptions::GetCoreDumpStyle() const { 
  // If unspecified, default to stack only
  if (m_core_dump_style == lldb::eSaveCoreUnspecified)
    return lldb::eSaveCoreStackOnly;
  return m_core_dump_style;
}

const lldb_private::FileSpec& CoreDumpOptions::GetOutputFile() const { return m_core_dump_file; }
