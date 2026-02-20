//===------------ CoverageIngestionManager.h - LLVM Advisor --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COVERAGEINGESTIONMANAGER_H
#define LLVM_TOOLS_LLVM_ADVISOR_SRC_CORE_COVERAGEINGESTIONMANAGER_H

#include "BuildContext.h"
#include "llvm/ADT/ArrayRef.h"
#include <atomic>
#include <chrono>
#include <optional>
#include <thread>
#include <vector>

namespace llvm::advisor {

class CoverageIngestionManager {
public:
  CoverageIngestionManager();
  ~CoverageIngestionManager();

  void registerSites(llvm::ArrayRef<CoverageProfileSite> Sites);
  void processOnce();
  void startWatching();
  void stopWatching();

private:
  struct WatchEntry {
    CoverageProfileSite Site;
    std::optional<std::chrono::system_clock::time_point> LastProcessed;
  };

  void runWatcher();
  bool processSite(WatchEntry &Entry);

  std::mutex Mutex;
  std::vector<WatchEntry> Watches;
  std::atomic<bool> StopRequested{false};
  bool Watching = false;
  std::thread Worker;
};

} // namespace llvm::advisor

#endif
