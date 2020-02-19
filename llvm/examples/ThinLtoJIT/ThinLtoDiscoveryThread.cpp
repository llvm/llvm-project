#include "ThinLtoDiscoveryThread.h"

#include "llvm/IR/GlobalValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#include "ThinLtoInstrumentationLayer.h"
#include "ThinLtoModuleIndex.h"

#include <thread>

#define DEBUG_TYPE "thinltojit"

namespace llvm {
namespace orc {

void ThinLtoDiscoveryThread::operator()() {
  while (KeepRunning.load()) {
    std::vector<unsigned> Indexes = Layer.takeFlagsThatFired();

    if (!Indexes.empty()) {
      LLVM_DEBUG(dbgs() << Indexes.size() << " new flags raised\n");
      auto ReachedFunctions = Layer.takeFlagOwners(std::move(Indexes));

      for (GlobalValue::GUID F : ReachedFunctions) {
        if (GlobalValueSummary *S = GlobalIndex.getSummary(F)) {
          assert(isa<FunctionSummary>(S) && "Reached symbols are functions");
          GlobalIndex.discoverCalleeModulePaths(cast<FunctionSummary>(S),
                                                LookaheadLevels);
        } else {
          LLVM_DEBUG(dbgs() << "No summary for GUID: " << F << "\n");
        }
      }

      if (GlobalIndex.getNumDiscoveredModules() > 0)
        spawnLookupForHighRankModules();
    }
  }
}

void ThinLtoDiscoveryThread::spawnLookupForHighRankModules() {
  std::vector<std::string> Paths = GlobalIndex.selectNextPaths();
  GlobalIndex.scheduleModuleParsing(Paths);

  // In order to add modules we need exclusive access to the execution session.
  std::thread([this, Paths = std::move(Paths)]() {
    ES.runSessionLocked([this, Paths = std::move(Paths)]() mutable {
      for (const std::string &Path : Paths) {
        ThreadSafeModule TSM = GlobalIndex.takeModule(Path);
        if (!TSM)
          // In the meantime the module was added synchronously.
          continue;

        if (Error LoadErr = AddModule(std::move(TSM)))
          // Failed to add the module to the session.
          ES.reportError(std::move(LoadErr));

        ++NumModulesSubmitted;
      }
    });
  }).detach();
}

} // namespace orc
} // namespace llvm
