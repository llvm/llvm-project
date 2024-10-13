#include "llvm/CompilerAssistedFuzzing/FuzzInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"

#include <cassert>
#include <cmath>

#define DEBUG_TYPE "fuzz"

using namespace llvm;

std::string FuzzComponents;
static cl::opt<std::string, true>
    FuzzingComponents("fuzz-components",
                      cl::desc("Compiler components to fuzz."),
                      cl::location(FuzzComponents), cl::init(""));

int64_t FuzzSeed;
static cl::opt<int64_t, true> FuzzingSeed("fuzz-seed",
                                          cl::desc("Compiler fuzzing seed."),
                                          cl::location(FuzzSeed), cl::init(0));

using fuzz::Component;

Component fuzz::Scheduler("scheduler");
Component fuzz::MBBPlacement("mbb-placement");
Component fuzz::BPU("bpu");
Component fuzz::RegAlloc("regalloc");
Component fuzz::ISel("isel");
Component fuzz::Alloca("alloca");
std::array<std::reference_wrapper<Component>, 6> Components{
    fuzz::Scheduler, fuzz::MBBPlacement, fuzz::RegAlloc, fuzz::ISel,
    fuzz::Alloca, fuzz::BPU};

inline bool isCompUsed(const Component &Comp) {
  if (FuzzComponents.find("all") != std::string::npos)
    return true;

  return FuzzComponents.find(Comp.GetName()) != std::string::npos;
}

const Component &Component::operator+=(llvm::TrackingStatistic &Stat) {
  if (llvm::find_if(Stats, [&Stat](const fuzz::StatRefWrapper &St) -> bool {
        return St.get().getName() == Stat.getName();
      }) == Stats.end()) {
    Stats.emplace_back(Stat);
  }
  Stat++;
  return *this;
}

bool fuzz::isFuzzed(Component &Comp, llvm::TrackingStatistic &Stat) {
  if (isCompUsed(Comp)) {
    Comp += Stat;
    return true;
  }

  return false;
}

int64_t fuzz::fuzzedIntRange(Component &Comp, llvm::TrackingStatistic &Stat,
                             int64_t Start, int64_t End, int64_t Default) {
  if (!isFuzzed(Comp, Stat)) {
    return Default;
  }

  int64_t Length = End - Start;
  assert(Length > 0 && "Length must be non negative");

  return (std::abs(FuzzSeed) % Length) + Start;
}

bool fuzz::CheckStat(const llvm::TrackingStatistic &Stat) {
  for (const auto &C : Components) {
    for (const auto &St : C.get().GetStats()) {
      if (Stat.getName() == St.get().getName())
        return true;
    }
  }

  return false;
}
