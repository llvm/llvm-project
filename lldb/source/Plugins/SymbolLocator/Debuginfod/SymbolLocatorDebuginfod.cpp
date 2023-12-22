//===-- SymbolLocatorDebuginfod.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorDebuginfod.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Utility/Args.h"

#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Debuginfod/HTTPClient.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE(SymbolLocatorDebuginfod)

namespace {

#define LLDB_PROPERTIES_symbollocatordebuginfod
#include "SymbolLocatorDebuginfodProperties.inc"

enum {
#define LLDB_PROPERTIES_symbollocatordebuginfod
#include "SymbolLocatorDebuginfodPropertiesEnum.inc"
};

class PluginProperties : public Properties {
public:
  static llvm::StringRef GetSettingName() {
    return SymbolLocatorDebuginfod::GetPluginNameStatic();
  }

  PluginProperties() {
    m_collection_sp = std::make_shared<OptionValueProperties>(GetSettingName());
    m_collection_sp->Initialize(g_symbollocatordebuginfod_properties);

    // We need to read the default value first to read the environment variable.
    llvm::SmallVector<llvm::StringRef> urls = llvm::getDefaultDebuginfodUrls();
    Args arg_urls{urls};
    m_collection_sp->SetPropertyAtIndexFromArgs(ePropertyServerURLs, arg_urls);

    m_collection_sp->SetValueChangedCallback(
        ePropertyServerURLs, [this] { ServerURLsChangedCallback(); });
  }

  Args GetDebugInfoDURLs() const {
    Args urls;
    m_collection_sp->GetPropertyAtIndexAsArgs(ePropertyServerURLs, urls);
    return urls;
  }

private:
  void ServerURLsChangedCallback() {
    m_server_urls = GetDebugInfoDURLs();
    llvm::SmallVector<llvm::StringRef> dbginfod_urls;
    llvm::for_each(m_server_urls, [&](const auto &obj) {
      dbginfod_urls.push_back(obj.ref());
    });
    llvm::setDefaultDebuginfodUrls(dbginfod_urls);
  }
  // Storage for the StringRef's used within the Debuginfod library.
  Args m_server_urls;
};

} // namespace

static PluginProperties &GetGlobalPluginProperties() {
  static PluginProperties g_settings;
  return g_settings;
}

SymbolLocatorDebuginfod::SymbolLocatorDebuginfod() : SymbolLocator() {}

void SymbolLocatorDebuginfod::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, []() {
    PluginManager::RegisterPlugin(
        GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
        LocateExecutableObjectFile, LocateExecutableSymbolFile, nullptr,
        nullptr, SymbolLocatorDebuginfod::DebuggerInitialize);
    llvm::HTTPClient::initialize();
  });
}

void SymbolLocatorDebuginfod::DebuggerInitialize(Debugger &debugger) {
  if (!PluginManager::GetSettingForSymbolLocatorPlugin(
          debugger, PluginProperties::GetSettingName())) {
    const bool is_global_setting = true;
    PluginManager::CreateSettingForSymbolLocatorPlugin(
        debugger, GetGlobalPluginProperties().GetValueProperties(),
        "Properties for the Debuginfod Symbol Locator plug-in.",
        is_global_setting);
  }
}

void SymbolLocatorDebuginfod::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
  llvm::HTTPClient::cleanup();
}

llvm::StringRef SymbolLocatorDebuginfod::GetPluginDescriptionStatic() {
  return "Debuginfod symbol locator.";
}

SymbolLocator *SymbolLocatorDebuginfod::CreateInstance() {
  return new SymbolLocatorDebuginfod();
}

static std::optional<FileSpec> GetFileForModule(
    const ModuleSpec &module_spec,
    std::function<llvm::Expected<std::string>(llvm::object::BuildIDRef)>
        PullFromServer) {
  if (!ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup())
    return {};
  const UUID &module_uuid = module_spec.GetUUID();
  if (module_uuid.IsValid() && llvm::canUseDebuginfod()) {
    llvm::object::BuildID build_id(module_uuid.GetBytes());
    llvm::Expected<std::string> result = PullFromServer(build_id);
    if (result)
      return FileSpec(*result);
    // An error here should be logged as a failure in the Debuginfod library,
    // so just consume it here
    consumeError(result.takeError());
  }
  return {};
}

std::optional<ModuleSpec> SymbolLocatorDebuginfod::LocateExecutableObjectFile(
    const ModuleSpec &module_spec) {
  return GetFileForModule(module_spec, llvm::getCachedOrDownloadExecutable);
}

std::optional<FileSpec> SymbolLocatorDebuginfod::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  return GetFileForModule(module_spec, llvm::getCachedOrDownloadDebuginfo);
}
