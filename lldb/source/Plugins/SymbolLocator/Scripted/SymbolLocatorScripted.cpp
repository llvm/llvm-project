//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorScripted.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/Interfaces/ScriptedSymbolLocatorInterface.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorScripted)

SymbolLocatorScripted::SymbolLocatorScripted() : SymbolLocator() {}

void SymbolLocatorScripted::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      LocateExecutableObjectFile, LocateExecutableSymbolFile,
      DownloadObjectAndSymbolFile, nullptr, LocateSourceFile,
      DebuggerInitialize);
}

void SymbolLocatorScripted::Terminate() {
  PluginManager::ClearScriptedSymbolLocators();
  PluginManager::UnregisterPlugin(CreateInstance);
}

llvm::StringRef SymbolLocatorScripted::GetPluginDescriptionStatic() {
  return "Scripted symbol locator plug-in.";
}

SymbolLocator *SymbolLocatorScripted::CreateInstance() {
  return new SymbolLocatorScripted();
}

void SymbolLocatorScripted::DebuggerInitialize(Debugger &debugger) {
  // Nothing to initialize per-debugger for now.
}

std::optional<FileSpec>
SymbolLocatorScripted::LocateSourceFile(const lldb::ModuleSP &module_sp,
                                        const FileSpec &original_source_file) {
  auto &instances = PluginManager::GetScriptedSymbolLocatorInstances();
  if (instances.empty() || !module_sp)
    return {};

  std::string cache_key =
      module_sp->GetUUID().GetAsString() + ":" + original_source_file.GetPath();

  for (auto &inst : instances) {
    // Check the per-instance cache first.
    auto it = inst.source_file_cache.find(cache_key);
    if (it != inst.source_file_cache.end()) {
      if (it->second)
        return it->second;
      continue; // This instance cached a miss; try the next one.
    }

    Status error;
    auto located = inst.interface_sp->LocateSourceFile(
        module_sp, original_source_file, error);

    if (!error.Success()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log, "SymbolLocatorScripted: locate_source_file failed: {0}",
               error);
    }

    inst.source_file_cache[cache_key] = located;

    if (located)
      return located;
  }

  return {};
}

std::optional<ModuleSpec> SymbolLocatorScripted::LocateExecutableObjectFile(
    const ModuleSpec &module_spec) {
  auto &instances = PluginManager::GetScriptedSymbolLocatorInstances();
  if (instances.empty())
    return {};

  for (auto &inst : instances) {
    Status error;
    auto located =
        inst.interface_sp->LocateExecutableObjectFile(module_spec, error);

    if (!error.Success()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log,
               "SymbolLocatorScripted: locate_executable_object_file failed: "
               "{0}",
               error);
    }

    if (located)
      return located;
  }

  return {};
}

std::optional<FileSpec> SymbolLocatorScripted::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  auto &instances = PluginManager::GetScriptedSymbolLocatorInstances();
  if (instances.empty())
    return {};

  for (auto &inst : instances) {
    Status error;
    auto located =
        inst.interface_sp->LocateExecutableSymbolFile(module_spec, error);

    if (!error.Success()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log,
               "SymbolLocatorScripted: locate_executable_symbol_file failed: "
               "{0}",
               error);
    }

    if (located)
      return located;
  }

  return {};
}

bool SymbolLocatorScripted::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                                        Status &error,
                                                        bool force_lookup,
                                                        bool copy_executable) {
  auto &instances = PluginManager::GetScriptedSymbolLocatorInstances();
  if (instances.empty())
    return false;

  for (auto &inst : instances) {
    Status inst_error;
    bool downloaded =
        inst.interface_sp->DownloadObjectAndSymbolFile(module_spec, inst_error);

    if (!inst_error.Success()) {
      Log *log = GetLog(LLDBLog::Symbols);
      LLDB_LOG(log,
               "SymbolLocatorScripted: download_object_and_symbol_file "
               "failed: {0}",
               inst_error);
    }

    if (downloaded)
      return true;
  }

  return false;
}
