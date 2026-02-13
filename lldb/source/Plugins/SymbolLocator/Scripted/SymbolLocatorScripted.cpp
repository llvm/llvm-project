//===-- SymbolLocatorScripted.cpp ------------------------------------------===//
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
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Interpreter/ScriptInterpreter.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"

#include <unordered_map>

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorScripted)

namespace {

#define LLDB_PROPERTIES_symbollocatorscripted
#include "SymbolLocatorScriptedProperties.inc"

enum {
#define LLDB_PROPERTIES_symbollocatorscripted
#include "SymbolLocatorScriptedPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  static llvm::StringRef GetSettingName() {
    return SymbolLocatorScripted::GetPluginNameStatic();
  }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_symbollocatorscripted_properties);

    m_collection_sp->SetValueChangedCallback(
        ePropertyScriptClass, [this] { ScriptClassChangedCallback(); });
  }

  llvm::StringRef GetScriptClassName() const {
    const OptionValueString *s =
        m_collection_sp->GetPropertyAtIndexAsOptionValueString(
            ePropertyScriptClass);
    if (s)
      return s->GetCurrentValueAsRef();
    return {};
  }

  ScriptedSymbolLocatorInterfaceSP GetInterface() const {
    return m_interface_sp;
  }

  void SetInterface(ScriptedSymbolLocatorInterfaceSP interface_sp) {
    m_interface_sp = interface_sp;
  }

  /// Look up a previously cached source file resolution result.
  /// Returns true if a cached entry exists (even if the result is nullopt).
  bool LookupSourceFileCache(const std::string &key,
                             std::optional<FileSpec> &result) {
    auto it = m_source_file_cache.find(key);
    if (it != m_source_file_cache.end()) {
      result = it->second;
      return true;
    }
    return false;
  }

  void InsertSourceFileCache(const std::string &key,
                             const std::optional<FileSpec> &result) {
    m_source_file_cache[key] = result;
  }

private:
  void ScriptClassChangedCallback() {
    // Invalidate the cached interface and source file cache when the user
    // changes the script class.
    m_interface_sp.reset();
    m_source_file_cache.clear();
  }

  ScriptedSymbolLocatorInterfaceSP m_interface_sp;
  std::unordered_map<std::string, std::optional<FileSpec>> m_source_file_cache;
};

} // namespace

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

static ScriptedSymbolLocatorInterfaceSP GetOrCreateInterface() {
  PluginProperties &props = GetGlobalPluginProperties();

  llvm::StringRef class_name = props.GetScriptClassName();
  if (class_name.empty())
    return {};

  // Return the cached interface if available.
  auto interface_sp = props.GetInterface();
  if (interface_sp)
    return interface_sp;

  // Find a debugger with a script interpreter.
  DebuggerSP debugger_sp = Debugger::GetDebuggerAtIndex(0);
  if (!debugger_sp)
    return {};

  ScriptInterpreter *interpreter = debugger_sp->GetScriptInterpreter();
  if (!interpreter)
    return {};

  interface_sp = interpreter->CreateScriptedSymbolLocatorInterface();
  if (!interface_sp)
    return {};

  // Create the Python script object from the user's class.
  ExecutionContext exe_ctx;
  StructuredData::DictionarySP args_sp;
  auto obj_or_err =
      interface_sp->CreatePluginObject(class_name, exe_ctx, args_sp);

  if (!obj_or_err) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOG_ERROR(log, obj_or_err.takeError(),
                   "SymbolLocatorScripted: failed to create Python object for "
                   "class '{0}': {1}",
                   class_name);
    return {};
  }

  props.SetInterface(interface_sp);
  return interface_sp;
}

SymbolLocatorScripted::SymbolLocatorScripted() : SymbolLocator() {}

void SymbolLocatorScripted::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      LocateExecutableObjectFile, LocateExecutableSymbolFile,
      DownloadObjectAndSymbolFile, nullptr, LocateSourceFile,
      SymbolLocatorScripted::DebuggerInitialize);
}

void SymbolLocatorScripted::Terminate() {
  GetGlobalPluginProperties().SetInterface(nullptr);
  PluginManager::UnregisterPlugin(CreateInstance);
}

void SymbolLocatorScripted::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForSymbolLocatorPlugin(
          debugger, PluginProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForSymbolLocatorPlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        "Properties for the Scripted Symbol Locator plug-in.",
        is_global_setting);
  }
}

llvm::StringRef SymbolLocatorScripted::GetPluginDescriptionStatic() {
  return "Scripted symbol locator plug-in.";
}

SymbolLocator *SymbolLocatorScripted::CreateInstance() {
  return new SymbolLocatorScripted();
}

std::optional<ModuleSpec> SymbolLocatorScripted::LocateExecutableObjectFile(
    const ModuleSpec &module_spec) {
  auto interface_sp = GetOrCreateInterface();
  if (!interface_sp)
    return {};
  Status error;
  auto result = interface_sp->LocateExecutableObjectFile(module_spec, error);
  if (!error.Success()) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOG(log, "SymbolLocatorScripted: locate_executable_object_file "
                  "failed: {0}",
             error);
  }
  return result;
}

std::optional<FileSpec> SymbolLocatorScripted::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  auto interface_sp = GetOrCreateInterface();
  if (!interface_sp)
    return {};
  Status error;
  auto result = interface_sp->LocateExecutableSymbolFile(
      module_spec, default_search_paths, error);
  if (!error.Success()) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOG(log, "SymbolLocatorScripted: locate_executable_symbol_file "
                  "failed: {0}",
             error);
  }
  return result;
}

bool SymbolLocatorScripted::DownloadObjectAndSymbolFile(
    ModuleSpec &module_spec, Status &error, bool force_lookup,
    bool copy_executable) {
  auto interface_sp = GetOrCreateInterface();
  if (!interface_sp)
    return false;
  return interface_sp->DownloadObjectAndSymbolFile(module_spec, error,
                                                    force_lookup,
                                                    copy_executable);
}

std::optional<FileSpec> SymbolLocatorScripted::LocateSourceFile(
    const lldb::ModuleSP &module_sp, const FileSpec &original_source_file) {
  if (!module_sp)
    return {};

  PluginProperties &props = GetGlobalPluginProperties();

  // Cache resolved source files to avoid repeated Python calls for the same
  // (module, source_file) pair.
  std::string cache_key =
      module_sp->GetUUID().GetAsString() + ":" +
      original_source_file.GetPath();

  std::optional<FileSpec> cached;
  if (props.LookupSourceFileCache(cache_key, cached))
    return cached;

  auto interface_sp = GetOrCreateInterface();
  if (!interface_sp) {
    props.InsertSourceFileCache(cache_key, std::nullopt);
    return {};
  }

  Status error;
  auto located =
      interface_sp->LocateSourceFile(module_sp, original_source_file, error);

  if (!error.Success()) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOG(log, "SymbolLocatorScripted: locate_source_file failed: {0}",
             error);
  }

  props.InsertSourceFileCache(cache_key, located);

  if (located) {
    Log *log = GetLog(LLDBLog::Symbols);
    LLDB_LOGF(log,
              "SymbolLocatorScripted::%s: resolved source file '%s' to '%s'",
              __FUNCTION__, original_source_file.GetPath().c_str(),
              located->GetPath().c_str());
  }
  return located;
}
