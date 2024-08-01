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

Status SaveCoreOptions::AddMemoryRegionToSave(const lldb_private::MemoryRegionInfo &region) {
  Status error;
  m_regions_to_save.insert(region.GetRange().GetRangeBase());
  return error;
}

bool SaveCoreOptions::ShouldSaveRegion(const lldb_private::MemoryRegionInfo &region) const {
  return m_regions_to_save.empty() || m_regions_to_save.count(region.GetRange().GetRangeBase()) > 0;
}

void SaveCoreOptions::Clear() {
  m_file = std::nullopt;
  m_plugin_name = std::nullopt;
  m_style = std::nullopt;
}
