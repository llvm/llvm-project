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
#include "lldb/Core/Progress.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Target/Platform.h"

#include "llvm/ADT/STLExtras.h"
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
SymbolLocator::LocateWithPlugins(const Request &request,
                                 const FileSpecList &search_paths) {
  FileSystem &fs = FileSystem::Instance();
  Result result;
  ModuleSpec &module_spec = result.module_spec;
  module_spec = request.module_spec;

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
    const bool downloaded = PluginManager::DownloadObjectAndSymbolFile(
        module_spec, error, request.external_lookup);
    // The plugins share one Status, so a failure recorded along the way says
    // nothing about a search that ended in a download.
    if (!downloaded && error.Fail() && request.external_lookup)
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

static std::optional<SymbolLocator::Result>
AskPlatform(const SymbolLocator::Request &request,
            const FileSpecList &search_paths) {
  if (!request.platform)
    return std::nullopt;

  SymbolLocator::Result result;
  std::optional<ModuleSpec> found = request.platform->FindModuleFiles(
      request.module_spec, search_paths, result.statistics);
  if (!found)
    return std::nullopt;

  result.module_spec = *found;
  return result;
}

llvm::Expected<SymbolLocator::Result>
SymbolLocator::Locate(const Request &request,
                      const FileSpecList &search_paths) {
  if (std::optional<Result> answer = AskPlatform(request, search_paths))
    return std::move(*answer);
  return LocateWithPlugins(request, search_paths);
}

std::vector<llvm::Expected<SymbolLocator::Result>>
SymbolLocator::Locate(llvm::ArrayRef<Request> requests,
                      const FileSpecList &search_paths, bool parallel) {
  // One slot per request, so concurrent searches never contend.
  std::vector<std::optional<llvm::Expected<Result>>> slots(requests.size());
  std::vector<size_t> remaining;

  if (!requests.empty()) {
    // Throttled because every search reports through this from its own thread.
    Progress progress("Locating binaries", "", requests.size(),
                      /*debugger=*/nullptr,
                      Progress::kDefaultHighFrequencyReportTime);

    for (auto [i, request] : llvm::enumerate(requests)) {
      if (std::optional<Result> answer = AskPlatform(request, search_paths)) {
        slots[i] = std::move(*answer);
        progress.Increment(1, request.description);
      } else {
        remaining.push_back(i);
      }
    }

    auto locate = [&](size_t i) {
      slots[i] = LocateWithPlugins(requests[i], search_paths);
      progress.Increment(1, requests[i].description);
    };

    // One search has nothing to overlap with.
    if (parallel && remaining.size() > 1) {
      llvm::ThreadPoolTaskGroup task_group(Debugger::GetThreadPool());
      for (size_t i : remaining)
        task_group.async(locate, i);
      task_group.wait();
    } else {
      for (size_t i : remaining)
        locate(i);
    }
  }

  std::vector<llvm::Expected<Result>> results;
  results.reserve(slots.size());
  for (std::optional<llvm::Expected<Result>> &slot : slots) {
    assert(slot && "every request has a result");
    results.emplace_back(std::move(*slot));
  }
  return results;
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
