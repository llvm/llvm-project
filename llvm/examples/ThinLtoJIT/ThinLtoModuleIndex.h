#ifndef LLVM_EXAMPLES_THINLTOJIT_THINLTOJITMODULEINDEX_H
#define LLVM_EXAMPLES_THINLTOJIT_THINLTOJITMODULEINDEX_H

#include "llvm/ADT/Optional.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"

#include <cstdint>
#include <future>
#include <mutex>
#include <set>
#include <vector>

namespace llvm {
namespace orc {

class SymbolStringPtr;

class ThinLtoModuleIndex {
  static constexpr bool HaveGVs = false;

public:
  ThinLtoModuleIndex(ExecutionSession &ES, unsigned ParseModuleThreads)
      : ES(ES), CombinedSummaryIndex(HaveGVs),
        ParseModuleWorkers(llvm::hardware_concurrency(ParseModuleThreads)),
        NumParseModuleThreads(ParseModuleThreads) {}

  Error add(StringRef InputPath);
  GlobalValueSummary *getSummary(GlobalValue::GUID Function) const;
  std::vector<StringRef> getAllModulePaths() const;
  Optional<StringRef> getModulePathForSymbol(StringRef Name) const;

  template <typename RangeT> void scheduleModuleParsing(const RangeT &Paths);
  ThreadSafeModule takeModule(StringRef Path);

  // Blocking module parsing, returns a Null-module on error.
  // Only used for the main module.
  ThreadSafeModule parseModuleFromFile(StringRef Path);

  std::vector<std::string> selectNextPaths();
  unsigned getNumDiscoveredModules() const;
  void discoverCalleeModulePaths(FunctionSummary *S, unsigned LookaheadLevels);

  VModuleKey getModuleId(StringRef Path) const {
    return CombinedSummaryIndex.getModuleId(Path);
  }

private:
  ExecutionSession &ES;
  ModuleSummaryIndex CombinedSummaryIndex;
  uint64_t NextModuleId{0};

  struct PathRankEntry {
    uint32_t Count{0};
    uint32_t MinDist{100};
  };
  StringMap<PathRankEntry> PathRank;

  ThreadPool ParseModuleWorkers;
  unsigned NumParseModuleThreads;

  std::mutex ScheduledModulesLock;
  StringMap<std::shared_future<void>> ScheduledModules;

  std::mutex ParsedModulesLock;
  StringMap<ThreadSafeModule> ParsedModules;

  void updatePathRank(StringRef Path, unsigned Distance);
  void addToWorklist(std::vector<FunctionSummary *> &List,
                     ArrayRef<FunctionSummary::EdgeTy> Calls);

  std::vector<StringRef> selectAllPaths();
  std::vector<StringRef> selectHotPaths(unsigned Count);

  void scheduleModuleParsingPrelocked(StringRef Path);
  Expected<ThreadSafeModule> doParseModule(StringRef Path);
};

template <typename RangeT>
inline void ThinLtoModuleIndex::scheduleModuleParsing(const RangeT &Paths) {
  std::lock_guard<std::mutex> Lock(ScheduledModulesLock);
  for (const auto &Path : Paths) {
    scheduleModuleParsingPrelocked(Path);
  }
}

} // namespace orc
} // namespace llvm

#endif
