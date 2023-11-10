//===-- SymbolLocatorDebuginfod.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SymbolLocatorDebuginfod.h"

#include <cstring>
#include <optional>

#include "Plugins/ObjectFile/wasm/ObjectFileWasm.h"
#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Progress.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/ArchSpec.h"
#include "lldb/Utility/DataBuffer.h"
#include "lldb/Utility/DataExtractor.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/Timer.h"
#include "lldb/Utility/UUID.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Debuginfod/HTTPClient.h"
#include "llvm/Debuginfod/Debuginfod.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ThreadPool.h"

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
    m_collection_sp->SetValueChangedCallback(
        ePropertyURLs, [this] { URLsChangedCallback(); }
    );
  }

  Args GetDebugInfoDURLs() const {
    Args urls;
    m_collection_sp->GetPropertyAtIndexAsArgs(ePropertyURLs, urls);
    return urls;
  }

private:
  void URLsChangedCallback() {
    Args urls = GetDebugInfoDURLs();
    llvm::SmallVector<llvm::StringRef> dbginfod_urls;
    llvm::transform(urls, dbginfod_urls.end(),
                    [](const auto &obj) { return obj.ref(); });
    llvm::setDefaultDebuginfodUrls(dbginfod_urls);
  }
};

} // namespace


SymbolLocatorDebuginfod::SymbolLocatorDebuginfod() : SymbolLocator() {}

void SymbolLocatorDebuginfod::Initialize() {
  PluginManager::RegisterPlugin(
      GetPluginNameStatic(), GetPluginDescriptionStatic(), CreateInstance,
      LocateExecutableObjectFile, LocateExecutableSymbolFile,
      DownloadObjectAndSymbolFile);
  // There's a "safety" concern on this:
  // Does plugin initialization occur while things are still single threaded?
  llvm::HTTPClient::initialize();
}

void SymbolLocatorDebuginfod::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
  // There's a "safety" concern on this:
  // Does plugin termination occur while things are still single threaded?
  llvm::HTTPClient::cleanup();
}

llvm::StringRef SymbolLocatorDebuginfod::GetPluginDescriptionStatic() {
  return "Debuginfod symbol locator.";
}

SymbolLocator *SymbolLocatorDebuginfod::CreateInstance() {
  return new SymbolLocatorDebuginfod();
}

std::optional<ModuleSpec> SymbolLocatorDebuginfod::LocateExecutableObjectFile(
    const ModuleSpec &module_spec) {
  const UUID &module_uuid = module_spec.GetUUID();
  if (module_uuid.IsValid() && llvm::canUseDebuginfod()) {
    llvm::object::BuildID build_id(module_uuid.GetBytes());
    llvm::Expected<std::string> result =
        llvm::getCachedOrDownloadExecutable(build_id);
    if (result)
      return FileSpec(*result);
    // An error here should be logged as a failure in the Debuginfod library,
    // so just consume it here
    consumeError(result.takeError());
  }
  return {};
}

std::optional<FileSpec> SymbolLocatorDebuginfod::LocateExecutableSymbolFile(
    const ModuleSpec &module_spec, const FileSpecList &default_search_paths) {
  const UUID &module_uuid = module_spec.GetUUID();
  if (module_uuid.IsValid() && llvm::canUseDebuginfod()) {
    llvm::object::BuildID build_id(module_uuid.GetBytes());
    llvm::Expected<std::string> result =
        llvm::getCachedOrDownloadDebuginfo(build_id);
    if (result)
      return FileSpec(*result);
    // An error here should be logged as a failure in the Debuginfod library,
    // so just consume it here
    consumeError(result.takeError());
  }
  return {};
}

bool SymbolLocatorDebuginfod::DownloadObjectAndSymbolFile(ModuleSpec &module_spec,
                                                       Status &error,
                                                       bool force_lookup,
                                                       bool copy_executable) {
  // TODO: Continue to add more Debuginfod capabilities
  return false;
}
