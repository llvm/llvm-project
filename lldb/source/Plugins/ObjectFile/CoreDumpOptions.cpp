//===-- CoreDumpOptions.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/CoreDumpOptions.h"

CoreDumpOptions::SetCoreDumpPluginName(const char *name) {
  m_core_dump_plugin_name = name;
}

CoreDumpOptions::GetCoreDumpPluginName() { return m_core_dump_plugin_name; }

CoreDumpOptions::SetCoreDumpStyle(lldb::SaveCoreStyle style) {
  m_core_dump_style = style;
}

CoreDumpOptions::GetCoreDumpStyle() { return m_core_dump_style; }

CoreDumpOptions::SetCoreDumpFile(const char *file) { m_core_dump_file = file; }

CoreDumpOptions::GetCoreDumpFile() { return m_core_dump_file; }
