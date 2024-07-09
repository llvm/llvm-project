//===-- CoreDumpOptions.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H
#define LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H

#include <string>

using namespace lldb;

class CoreDumpOptions {
  public:
    CoreDumpOptions() {};
    ~CoreDumpOptions() = default;


  void SetCoreDumpPluginName(const char* name);
  const char* GetCoreDumpPluginName() const;

  void SetCoreDumpStyle(lldb::SaveCoreStyle style);
  lldb::SaveCoreStyle GetCoreDumpStyle() const;

  void SetOutputFilePath(const char* path);
  const char* GetOutputFilePath() const;

private:
  std::string m_core_dump_plugin_name;
  std::string m_output_file_path;
  lldb::SaveCoreStyle m_core_dump_style = lldb::eSaveCoreStyleNone;
};

#endif // LLDB_SOURCE_PLUGINS_OBJECTFILE_COREDUMPOPTIONS_H
