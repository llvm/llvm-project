//
// Created by da-viper on 26/05/25.
//

#include "SourceLocatorDebuginfod.h"

#include "lldb/Core/PluginManager.h"
#include "lldb/Interpreter/OptionValueString.h"
#include "lldb/Utility/Args.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include <lldb/Core/PluginManager.h>
#include <llvm/Debuginfod/Debuginfod.h>
#include <llvm/Debuginfod/HTTPClient.h>

LLDB_PLUGIN_DEFINE(SourceLocatorDebuginfod)

namespace lldb_private {
void SourceLocatorDebuginfod::Initialize() {
  static llvm::once_flag g_once_flag;

  llvm::call_once(g_once_flag, [] {
    PluginManager::RegisterPlugin(GetPluginNameStatic(),
                                  GetPluginDescriptionStatic(), CreateInstance,
                                  LocateSourceFile);
    llvm::HTTPClient::initialize();
  });
}
void SourceLocatorDebuginfod::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
  llvm::HTTPClient::cleanup();
}

SourceLocator *SourceLocatorDebuginfod::CreateInstance() {
  return new SourceLocatorDebuginfod();
}
std::optional<FileSpec>
SourceLocatorDebuginfod::LocateSourceFile(const ModuleSpec &module_spec,
                                          const FileSpec &file_spec) {

  const UUID &module_uuid = module_spec.GetUUID();
  const std::string file_path = file_spec.GetPath();
  // Don't bother if we don't have a path or valid UUID, Debuginfod isn't
  // available, or if the 'symbols.enable-external-lookup' setting is false.
  if (file_path.empty() || !module_uuid.IsValid() ||
      !llvm::canUseDebuginfod() ||
      !ModuleList::GetGlobalModuleListProperties().GetEnableExternalLookup())
    return {};

  llvm::SmallVector<llvm::StringRef> debuginfod_urls =
      llvm::getDefaultDebuginfodUrls();

  llvm::object::BuildID build_id(module_uuid.GetBytes());

  llvm::Expected<std::string> result =
      llvm::getCachedOrDownloadSource(build_id, file_path);
  if (result)
    return FileSpec(*result);

  Log *log = GetLog(LLDBLog::Source);
  auto err_message = llvm::toString(result.takeError());
  LLDB_LOGV(log, "Debuginfod failed to download source file {0} with error {1}",
            file_path, err_message);
  return std::nullopt;
}
} // namespace lldb_private