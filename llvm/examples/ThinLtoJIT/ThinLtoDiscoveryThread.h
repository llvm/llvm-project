#ifndef LLVM_EXAMPLES_THINLTOJIT_DISCOVERYTHREAD_H
#define LLVM_EXAMPLES_THINLTOJIT_DISCOVERYTHREAD_H

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ModuleSummaryIndex.h"

#include "ThinLtoJIT.h"

#include <atomic>
#include <vector>

namespace llvm {
namespace orc {

class ExecutionSession;
class ThinLtoModuleIndex;
class ThinLtoInstrumentationLayer;

class ThinLtoDiscoveryThread {
public:
  ThinLtoDiscoveryThread(std::atomic<bool> &RunningFlag, ExecutionSession &ES,
                         JITDylib *MainJD, ThinLtoInstrumentationLayer &L,
                         ThinLtoModuleIndex &GlobalIndex,
                         ThinLtoJIT::AddModuleFunction AddModule,
                         unsigned LookaheadLevels, bool PrintStats)
      : KeepRunning(RunningFlag), ES(ES), Layer(L), GlobalIndex(GlobalIndex),
        AddModule(std::move(AddModule)), LookaheadLevels(LookaheadLevels),
        PrintStats(PrintStats) {}

  ~ThinLtoDiscoveryThread() {
    if (PrintStats)
      dump(errs());
  }

  void operator()();

  void dump(raw_ostream &OS) {
    OS << format("Modules submitted asynchronously: %d\n", NumModulesSubmitted);
  }

private:
  std::atomic<bool> &KeepRunning;
  ExecutionSession &ES;
  ThinLtoInstrumentationLayer &Layer;
  ThinLtoModuleIndex &GlobalIndex;
  ThinLtoJIT::AddModuleFunction AddModule;
  unsigned LookaheadLevels;
  bool PrintStats;
  unsigned NumModulesSubmitted{0};

  void spawnLookupForHighRankModules();
};

} // namespace orc
} // namespace llvm

#endif
