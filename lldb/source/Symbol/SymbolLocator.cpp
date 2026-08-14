//===-- symbolLocator.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Symbol/SymbolLocator.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Platform.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ThreadPool.h"

using namespace lldb;
using namespace lldb_private;

char SymbolLocator::NotFound::ID = 0;

void SymbolLocator::NotFound::log(llvm::raw_ostream &os) const {
  os << "binary not found";
}

std::error_code SymbolLocator::NotFound::convertToErrorCode() const {
  return std::make_error_code(std::errc::no_such_file_or_directory);
}

llvm::Expected<SymbolLocator::Result>
SymbolLocator::Locate(const Request &request,
                      const FileSpecList &search_paths) {
  FileSystem &fs = FileSystem::Instance();
  Result result;
  ModuleSpec &module_spec = result.module_spec;
  module_spec = request.module_spec;

  // The locator plugins have no Platform to consult, so ask it here.
  if (request.platform) {
    if (std::optional<ModuleSpec> found = request.platform->FindModuleFiles(
            module_spec, search_paths, result.statistics)) {
      result.module_spec = *found;
      return result;
    }
  }

  // Can lldb's symbol and executable location schemes find them locally?
  module_spec.GetSymbolFileSpec() = PluginManager::LocateExecutableSymbolFile(
      module_spec, search_paths, result.statistics);
  module_spec.GetFileSpec() =
      PluginManager::LocateExecutableObjectFile(module_spec, result.statistics)
          .GetFileSpec();

  // If we're still missing either file, see if an external symbol server can
  // provide it.
  if (!fs.Exists(module_spec.GetFileSpec()) ||
      !fs.Exists(module_spec.GetSymbolFileSpec())) {
    Status error;
    PluginManager::DownloadObjectAndSymbolFile(module_spec, error,
                                               request.external_lookup);
    if (error.Fail() && request.external_lookup)
      result.symbol_error.emplace(error.takeError());
  }

  // Not finding the binary is the one hard failure. Whatever a symbol server
  // had to say is the best explanation available for it. Without one there is
  // nothing to add beyond the fact of the miss.
  if (!fs.Exists(module_spec.GetFileSpec())) {
    if (result.symbol_error)
      return std::move(*result.symbol_error);
    return llvm::make_error<NotFound>();
  }

  return result;
}

void SymbolLocator::DownloadSymbolFileAsync(const UUID &uuid) {
  static llvm::SmallSet<UUID, 8> g_seen_uuids;
  static std::mutex g_mutex;

  auto lookup = [=]() {
    {
      std::lock_guard<std::mutex> guard(g_mutex);
      if (g_seen_uuids.count(uuid))
        return;
      g_seen_uuids.insert(uuid);
    }

    Status error;
    ModuleSpec module_spec;
    module_spec.GetUUID() = uuid;
    if (!PluginManager::DownloadObjectAndSymbolFile(module_spec, error,
                                                    /*force_lookup=*/true,
                                                    /*copy_executable=*/true))
      return;

    if (error.Fail())
      return;

    Debugger::ReportSymbolChange(module_spec);
  };

  switch (ModuleList::GetGlobalModuleListProperties().GetSymbolAutoDownload()) {
  case eSymbolDownloadOff:
    break;
  case eSymbolDownloadBackground:
    Debugger::GetThreadPool().async(lookup);
    break;
  case eSymbolDownloadForeground:
    lookup();
    break;
  };
}
