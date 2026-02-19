//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorScripted.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/Interfaces/ScriptedSymbolLocatorInterface.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorScripted)

SymbolLocatorScripted::SymbolLocatorScripted() : SymbolLocator() {}

void SymbolLocatorScripted::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      nullptr, nullptr, nullptr, nullptr, LocateSourceFile);
}

void SymbolLocatorScripted::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolLocatorScripted::GetPluginDescriptionStatic() {
  return "Scripted symbol locator plug-in.";
}

SymbolLocator *SymbolLocatorScripted::CreateInstance() {
  return new SymbolLocatorScripted();
}

std::optional<FileSpec>
SymbolLocatorScripted::LocateSourceFile(const lldb::TargetSP &target_sp,
                                        const lldb::ModuleSP &module_sp,
                                        const FileSpec &original_source_file) {
  if (!module_sp || !target_sp)
    return {};

  auto interface_sp = target_sp->GetScriptedSymbolLocatorInterface();
  if (!interface_sp)
    return {};

  // Build cache key from module UUID and source file path.
  std::string cache_key =
      module_sp->GetUUID().GetAsString() + ":" + original_source_file.GetPath();

  // Check the per-target cache first.
  std::optional<FileSpec> cached;
  if (target_sp->LookupScriptedSourceFileCache(cache_key, cached))
    return cached;

  Status error;
  auto located =
      interface_sp->LocateSourceFile(module_sp, original_source_file, error);

  if (!error.Success()) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOG(log, "SymbolLocatorScripted: locate_source_file failed: {0}",
             error);
  }

  target_sp->InsertScriptedSourceFileCache(cache_key, located);

  return located;
}
