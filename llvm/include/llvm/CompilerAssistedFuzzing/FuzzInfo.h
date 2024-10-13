#ifndef LLVM_FUZZINFO_H
#define LLVM_FUZZINFO_H

#include "llvm/ADT/Statistic.h"
#include <functional>
#include <string>
#include <vector>

namespace fuzz {
using StatRefWrapper = std::reference_wrapper<llvm::TrackingStatistic>;

class Component final {
  std::string Name;
  std::vector<StatRefWrapper> Stats;

public:
  Component(std::string &&n) noexcept : Name(std::move(n)) {}

  Component &operator=(const Component &) = delete;
  Component(const Component &) = delete;

  Component &operator=(Component &&) = delete;
  Component(Component &&) = delete;

  const Component &operator+=(llvm::TrackingStatistic &Stat);
  const std::string &GetName() const { return Name; }
  const std::vector<StatRefWrapper> &GetStats() const { return Stats; }
};

extern Component Scheduler;
extern Component MBBPlacement;
extern Component RegAlloc;
extern Component ISel;
extern Component Alloca;
extern Component BPU;

bool isFuzzed(fuzz::Component &Comp, llvm::TrackingStatistic &Stat);
int64_t fuzzedIntRange(fuzz::Component &Comp, llvm::TrackingStatistic &Stat,
                       int64_t Start, int64_t End, int64_t Default);

bool CheckStat(const llvm::TrackingStatistic &Stat);
} // namespace fuzz

#endif // LLVM_FUZZINFO_H
