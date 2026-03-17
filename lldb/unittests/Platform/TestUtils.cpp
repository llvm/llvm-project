//===-- TestUtils.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestUtils.h"

#include "lldb/Core/PluginManager.h"

using namespace lldb;
using namespace lldb_private;

MockScriptInterpreterPython::MockScriptInterpreterPython(Debugger &debugger)
    : ScriptInterpreter(debugger, lldb::ScriptLanguage::eScriptLanguagePython) {
}

void MockScriptInterpreterPython::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                GetPluginDescriptionStatic(),
                                lldb::eScriptLanguagePython, CreateInstance);
}

void MockScriptInterpreterPython::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

LLDB_PLUGIN_DEFINE(MockScriptInterpreterPython)

std::string lldb_private::CreateFile(llvm::StringRef filename,
                                     llvm::SmallString<128> parent_dir) {
  llvm::SmallString<128> path(parent_dir);
  llvm::sys::path::append(path, filename);
  int fd;
  std::error_code ret = llvm::sys::fs::openFileForWrite(path, fd);
  assert(!ret && "Failed to create test file.");
  ::close(fd);

  return path.c_str();
}
