#include "ThinLtoModuleIndex.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/Orc/SymbolStringPool.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

#define DEBUG_TYPE "thinltojit"

namespace llvm {
namespace orc {

Error ThinLtoModuleIndex::add(StringRef InputPath) {
  auto Buffer = errorOrToExpected(MemoryBuffer::getFile(InputPath));
  if (!Buffer)
    return Buffer.takeError();

  Error ParseErr = readModuleSummaryIndex((*Buffer)->getMemBufferRef(),
                                          CombinedSummaryIndex, NextModuleId);
  if (ParseErr)
    return ParseErr;

#ifndef NDEBUG
  auto Paths = getAllModulePaths();
  unsigned TotalPaths = Paths.size();
  std::sort(Paths.begin(), Paths.end());
  Paths.erase(std::unique(Paths.begin(), Paths.end()), Paths.end());
  assert(TotalPaths == Paths.size() && "Module paths must be unique");
#endif

  ++NextModuleId;
  return Error::success();
}

std::vector<StringRef> ThinLtoModuleIndex::getAllModulePaths() const {
  auto ModuleTable = CombinedSummaryIndex.modulePaths();

  std::vector<StringRef> Paths;
  Paths.resize(ModuleTable.size());

  for (const auto &KV : ModuleTable) {
    assert(Paths[KV.second.first].empty() && "IDs are unique and continuous");
    Paths[KV.second.first] = KV.first();
  }

  return Paths;
}

GlobalValueSummary *
ThinLtoModuleIndex::getSummary(GlobalValue::GUID Function) const {
  ValueInfo VI = CombinedSummaryIndex.getValueInfo(Function);
  if (!VI || VI.getSummaryList().empty())
    return nullptr;

  // There can be more than one symbol with the same GUID, in the case of same-
  // named locals in different but same-named source files that were compiled in
  // their respective directories (so the source file name and resulting GUID is
  // the same). We avoid this by checking that module paths are unique upon
  // add().
  //
  // TODO: We can still get duplicates on symbols declared with
  // attribute((weak)), a GNU extension supported by gcc and clang.
  // We should support it by looking for a symbol in the current module
  // or in the same module as the caller.
  assert(VI.getSummaryList().size() == 1 && "Weak symbols not yet supported");

  return VI.getSummaryList().front().get()->getBaseObject();
}

Optional<StringRef>
ThinLtoModuleIndex::getModulePathForSymbol(StringRef Name) const {
  if (GlobalValueSummary *S = getSummary(GlobalValue::getGUID(Name)))
    return S->modulePath();
  return None; // We don't know the symbol.
}

void ThinLtoModuleIndex::scheduleModuleParsingPrelocked(StringRef Path) {
  // Once the module was scheduled, we can call takeModule().
  auto ScheduledIt = ScheduledModules.find(Path);
  if (ScheduledIt != ScheduledModules.end())
    return;

  auto Worker = [this](std::string Path) {
    if (auto TSM = doParseModule(Path)) {
      std::lock_guard<std::mutex> Lock(ParsedModulesLock);
      ParsedModules[Path] = std::move(*TSM);

      LLVM_DEBUG(dbgs() << "Finished parsing module: " << Path << "\n");
    } else {
      ES.reportError(TSM.takeError());
    }
  };

  LLVM_DEBUG(dbgs() << "Schedule module for parsing: " << Path << "\n");
  ScheduledModules[Path] = ParseModuleWorkers.async(Worker, Path.str());
}

ThreadSafeModule ThinLtoModuleIndex::takeModule(StringRef Path) {
  std::unique_lock<std::mutex> ParseLock(ParsedModulesLock);

  auto ParsedIt = ParsedModules.find(Path);
  if (ParsedIt == ParsedModules.end()) {
    ParseLock.unlock();

    // The module is not ready, wait for the future we stored.
    std::unique_lock<std::mutex> ScheduleLock(ScheduledModulesLock);
    auto ScheduledIt = ScheduledModules.find(Path);
    assert(ScheduledIt != ScheduledModules.end() &&
           "Don't call for unscheduled modules");
    std::shared_future<void> Future = ScheduledIt->getValue();
    ScheduleLock.unlock();
    Future.get();

    ParseLock.lock();
    ParsedIt = ParsedModules.find(Path);
    assert(ParsedIt != ParsedModules.end() && "Must be ready now");
  }

  // We only add each module once. If it's not here anymore, we can skip it.
  ThreadSafeModule TSM = std::move(ParsedIt->getValue());
  ParsedIt->getValue() = ThreadSafeModule();
  return TSM;
}

ThreadSafeModule ThinLtoModuleIndex::parseModuleFromFile(StringRef Path) {
  {
    std::lock_guard<std::mutex> ScheduleLock(ScheduledModulesLock);
    scheduleModuleParsingPrelocked(Path);
  }
  return takeModule(Path);
}

Expected<ThreadSafeModule> ThinLtoModuleIndex::doParseModule(StringRef Path) {
  // TODO: make a SMDiagnosticError class for this
  SMDiagnostic Err;
  auto Ctx = std::make_unique<LLVMContext>();
  auto M = parseIRFile(Path, Err, *Ctx);
  if (!M) {
    std::string ErrDescription;
    {
      raw_string_ostream S(ErrDescription);
      Err.print("ThinLtoJIT", S);
    }
    return createStringError(inconvertibleErrorCode(),
                             "Failed to load module from file '%s' (%s)",
                             Path.data(), ErrDescription.c_str());
  }

  return ThreadSafeModule(std::move(M), std::move(Ctx));
}

// We don't filter visited functions. Discovery will often be retriggered
// from the middle of already visited functions and it aims to reach a little
// further each time.
void ThinLtoModuleIndex::discoverCalleeModulePaths(FunctionSummary *S,
                                                   unsigned LookaheadLevels) {
  // Populate initial worklist
  std::vector<FunctionSummary *> Worklist;
  addToWorklist(Worklist, S->calls());
  unsigned Distance = 0;

  while (++Distance < LookaheadLevels) {
    // Process current worklist and populate a new one.
    std::vector<FunctionSummary *> NextWorklist;
    for (FunctionSummary *F : Worklist) {
      updatePathRank(F->modulePath(), Distance);
      addToWorklist(NextWorklist, F->calls());
    }
    Worklist = std::move(NextWorklist);
  }

  // Process the last worklist without filling a new one
  for (FunctionSummary *F : Worklist) {
    updatePathRank(F->modulePath(), Distance);
  }

  // Reset counts for known paths (includes both, scheduled and parsed modules).
  std::lock_guard<std::mutex> Lock(ScheduledModulesLock);
  for (const auto &KV : ScheduledModules) {
    PathRank[KV.first()].Count = 0;
  }
}

void ThinLtoModuleIndex::addToWorklist(
    std::vector<FunctionSummary *> &List,
    ArrayRef<FunctionSummary::EdgeTy> Calls) {
  for (const auto &Edge : Calls) {
    const auto &SummaryList = Edge.first.getSummaryList();
    if (!SummaryList.empty()) {
      GlobalValueSummary *S = SummaryList.front().get()->getBaseObject();
      assert(isa<FunctionSummary>(S) && "Callees must be functions");
      List.push_back(cast<FunctionSummary>(S));
    }
  }
}

// PathRank is global and continuous.
void ThinLtoModuleIndex::updatePathRank(StringRef Path, unsigned Distance) {
  auto &Entry = PathRank[Path];
  Entry.Count += 1;
  Entry.MinDist = std::min(Entry.MinDist, Distance);
  assert(Entry.MinDist > 0 && "We want it as a divisor");
}

// TODO: The size of a ThreadPool's task queue is not accessible. It would
// be great to know in order to estimate how many modules we schedule. The
// more we schedule, the less precise is the ranking. The less we schedule,
// the higher the risk for downtime.
std::vector<std::string> ThinLtoModuleIndex::selectNextPaths() {
  struct ScorePath {
    float Score;
    unsigned MinDist;
    StringRef Path;
  };

  std::vector<ScorePath> Candidates;
  Candidates.reserve(PathRank.size());
  for (const auto &KV : PathRank) {
    float Score = static_cast<float>(KV.second.Count) / KV.second.MinDist;
    if (Score > .0f) {
      Candidates.push_back({Score, KV.second.MinDist, KV.first()});
    }
  }

  // Sort candidates by descending score.
  std::sort(Candidates.begin(), Candidates.end(),
            [](const ScorePath &LHS, const ScorePath &RHS) {
              return LHS.Score > RHS.Score;
            });

  // Sort highest score candidates by ascending minimal distance.
  size_t Selected =
      std::min(std::max<size_t>(NumParseModuleThreads, Candidates.size() / 2),
               Candidates.size());
  std::sort(Candidates.begin(), Candidates.begin() + Selected,
            [](const ScorePath &LHS, const ScorePath &RHS) {
              return LHS.MinDist < RHS.MinDist;
            });

  std::vector<std::string> Paths;
  Paths.reserve(Selected);
  for (unsigned i = 0; i < Selected; i++) {
    Paths.push_back(Candidates[i].Path.str());
  }

  LLVM_DEBUG(dbgs() << "ModuleIndex: select " << Paths.size() << " out of "
                    << Candidates.size() << " discovered paths\n");

  return Paths;
}

unsigned ThinLtoModuleIndex::getNumDiscoveredModules() const {
  // TODO: It would probably be more efficient to track the number of
  // unscheduled modules.
  unsigned NonNullItems = 0;
  for (const auto &KV : PathRank)
    if (KV.second.Count > 0)
      ++NonNullItems;
  return NonNullItems;
}

} // namespace orc
} // namespace llvm
