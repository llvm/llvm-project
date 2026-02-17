//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorScripted.h"

#include "lldb/Core/Debugger.h"
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
      LocateExecutableObjectFile, LocateExecutableSymbolFile,
      DownloadObjectAndSymbolFile, nullptr, LocateSourceFile);
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

/// Iterate all debuggers and their targets, calling \p callback for each
/// target that has a scripted symbol locator registered. The callback
/// receives the target and its interface. If the callback returns true,
/// iteration stops early.
template <typename Callback>
static void ForEachScriptedTarget(Callback &&callback) {
  for (size_t di = 0; di < Debugger::GetNumDebuggers(); di++) {
    DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(di);
    if (!debugger_sp)
      continue;
    TargetList &target_list = debugger_sp->GetTargetList();
    for (size_t ti = 0; ti < target_list.GetNumTargets(); ti++) {
      TargetSP target_sp = target_list.GetTargetAtIndex(ti);
      if (!target_sp)
        continue;
      auto interface_sp = target_sp->GetScriptedSymbolLocatorInterface();
      if (!interface_sp)
        continue;
      if (callback(*target_sp, interface_sp))
        return;
    }
  }
}

std::optional<ModuleSpec> SymbolLocatorScripted::LocateExecutableObjectFile(
    const ModuleSpec &module_spec) {
  std::optional<ModuleSpec> result;
  ForEachScriptedTarget(
      [&](Target &target,
          ScriptedSymbolLocatorInterfaceSP &interface_sp) -> bool {
        Status error;
        auto located =
            interface_sp->LocateExecutableObjectFile(module_spec, error);
        if (!error.Success()) {
          Log *log = GetLog(LLDBLog::Symbols);
          LLDB_LOG(log,
                   "SymbolLocatorScripted: locate_executable_object_file "
                   "failed: {0}",
                   error);
        }
        if (located) {
          result = located;
          return true; // Stop iterating.
        }
        return false;
      });
  return result;
}

std::optional<FileSpec> SymbolLocatorScripted::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  std::optional<FileSpec> result;
  ForEachScriptedTarget(
      [&](Target &target,
          ScriptedSymbolLocatorInterfaceSP &interface_sp) -> bool {
        Status error;
        auto located = interface_sp->LocateExecutableSymbolFile(
            module_spec, default_search_paths, error);
        if (!error.Success()) {
          Log *log = GetLog(LLDBLog::Symbols);
          LLDB_LOG(log,
                   "SymbolLocatorScripted: locate_executable_symbol_file "
                   "failed: {0}",
                   error);
        }
        if (located) {
          result = located;
          return true;
        }
        return false;
      });
  return result;
}

bool SymbolLocatorScripted::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                                        Status &error,
                                                        bool force_lookup,
                                                        bool copy_executable) {
  bool result = false;
  ForEachScriptedTarget(
      [&](Target &target,
          ScriptedSymbolLocatorInterfaceSP &interface_sp) -> bool {
        bool success = interface_sp->DownloadObjectAndSymbolFile(
            module_spec, error, force_lookup, copy_executable);
        if (success) {
          result = true;
          return true;
        }
        return false;
      });
  return result;
}

std::optional<FileSpec>
SymbolLocatorScripted::LocateSourceFile(const lldb::ModuleSP &module_sp,
                                        const FileSpec &original_source_file) {
  if (!module_sp)
    return {};

  // Find the target that owns this module.
  Target *owning_target = nullptr;
  for (size_t di = 0; di < Debugger::GetNumDebuggers(); di++) {
    DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(di);
    if (!debugger_sp)
      continue;
    TargetList &target_list = debugger_sp->GetTargetList();
    for (size_t ti = 0; ti < target_list.GetNumTargets(); ti++) {
      TargetSP target_sp = target_list.GetTargetAtIndex(ti);
      if (!target_sp)
        continue;
      ModuleSP found_module =
          target_sp->GetImages().FindModule(module_sp.get());
      if (found_module) {
        owning_target = target_sp.get();
        break;
      }
    }
    if (owning_target)
      break;
  }

  if (!owning_target)
    return {};

  auto interface_sp = owning_target->GetScriptedSymbolLocatorInterface();
  if (!interface_sp)
    return {};

  // Cache resolved source files to avoid repeated Python calls for the same
  // (module, source_file) pair.
  std::string cache_key =
      module_sp->GetUUID().GetAsString() + ":" + original_source_file.GetPath();

  std::optional<FileSpec> cached;
  if (owning_target->LookupScriptedSourceFileCache(cache_key, cached))
    return cached;

  Status error;
  auto located =
      interface_sp->LocateSourceFile(module_sp, original_source_file, error);

  Log *log = GetLog(LLDBLog::Symbols);
  if (!error.Success()) {
    LLDB_LOG(log, "SymbolLocatorScripted: locate_source_file failed: {0}",
             error);
  }

  owning_target->InsertScriptedSourceFileCache(cache_key, located);

  if (located) {
    LLDB_LOGF(log,
              "SymbolLocatorScripted::%s: resolved source file '%s' to '%s'",
              __FUNCTION__, original_source_file.GetPath().c_str(),
              located->GetPath().c_str());
  }
  return located;
}
