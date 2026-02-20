//===--------- CoverageIngestionManager.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CoverageIngestionManager.h"
#include "CoverageProcessor.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

using namespace llvm;

namespace llvm::advisor {

CoverageIngestionManager::CoverageIngestionManager() = default;

CoverageIngestionManager::~CoverageIngestionManager() { stopWatching(); }

void CoverageIngestionManager::registerSites(
    ArrayRef<CoverageProfileSite> Sites) {
  std::scoped_lock Guard(Mutex);
  for (const auto &Site : Sites) {
    if (Site.rawProfile.empty())
      continue;
    bool Exists = llvm::any_of(Watches, [&](const WatchEntry &Entry) {
      return Entry.Site.rawProfile == Site.rawProfile;
    });
    if (Exists)
      continue;
    Watches.push_back({Site, std::nullopt});
  }
}

void CoverageIngestionManager::processOnce() {
  std::scoped_lock Guard(Mutex);
  for (auto &Entry : Watches)
    processSite(Entry);
}

void CoverageIngestionManager::startWatching() {
  std::scoped_lock Guard(Mutex);
  if (Watching || Watches.empty())
    return;
  StopRequested = false;
  Watching = true;
  Worker = std::thread([this]() { runWatcher(); });
}

void CoverageIngestionManager::stopWatching() {
  if (!Watching)
    return;
  StopRequested = true;
  if (Worker.joinable())
    Worker.join();
  Watching = false;
}

void CoverageIngestionManager::runWatcher() {
  while (!StopRequested.load()) {
    {
      std::scoped_lock Guard(Mutex);
      for (auto &Entry : Watches)
        processSite(Entry);
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

static std::optional<std::chrono::system_clock::time_point>
getWriteTime(StringRef Path) {
  sys::fs::file_status Status;
  if (sys::fs::status(Path, Status))
    return std::nullopt;
  return Status.getLastModificationTime();
}

bool CoverageIngestionManager::processSite(WatchEntry &Entry) {
  auto WriteTime = getWriteTime(Entry.Site.rawProfile);
  if (!WriteTime)
    return false;
  if (Entry.LastProcessed && *Entry.LastProcessed >= *WriteTime)
    return false;

  if (auto Err = CoverageProcessor::mergeRawProfile(
          Entry.Site.rawProfile, Entry.Site.indexedProfile)) {
    errs() << "llvm-advisor: failed to ingest profile '"
           << Entry.Site.rawProfile << "': " << toString(std::move(Err))
           << "\n";
    return false;
  }

  if (!Entry.Site.instrumentedBinary.empty() &&
      sys::fs::exists(Entry.Site.instrumentedBinary)) {
    if (auto Err = CoverageProcessor::exportCoverageReport(
            Entry.Site.instrumentedBinary, Entry.Site.indexedProfile,
            Entry.Site.reportPath)) {
      errs() << "llvm-advisor: failed to export coverage for '"
             << Entry.Site.instrumentedBinary << "': "
             << toString(std::move(Err)) << "\n";
    }
  }

  Entry.LastProcessed = WriteTime;
  return true;
}

} // namespace llvm::advisor
