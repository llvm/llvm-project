//===-- TimeProfiler.cpp - Hierarchical Time Profiler ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hierarchical time profiler.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TimeProfiler.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Threading.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono;
using namespace llvm;

namespace {
std::mutex Mu;
// List of all instances
std::vector<TimeTraceProfiler *>
    ThreadTimeTraceProfilerInstances; // guarded by Mu
// Per Thread instance
LLVM_THREAD_LOCAL TimeTraceProfiler *TimeTraceProfilerInstance = nullptr;
} // namespace

namespace llvm {

TimeTraceProfiler *getTimeTraceProfilerInstance() {
  return TimeTraceProfilerInstance;
}

typedef duration<steady_clock::rep, steady_clock::period> DurationType;
typedef time_point<steady_clock> TimePointType;
typedef std::pair<size_t, DurationType> CountAndDurationType;
typedef std::pair<std::string, CountAndDurationType>
    NameAndCountAndDurationType;

struct Entry {
  const TimePointType Start;
  TimePointType End;
  const std::string Name;
  const std::string Detail;

  Entry(TimePointType &&S, TimePointType &&E, std::string &&N, std::string &&Dt)
      : Start(std::move(S)), End(std::move(E)), Name(std::move(N)),
        Detail(std::move(Dt)) {}

  // Calculate timings for FlameGraph. Cast time points to microsecond precision
  // rather than casting duration. This avoid truncation issues causing inner
  // scopes overruning outer scopes.
  steady_clock::rep getFlameGraphStartUs(TimePointType StartTime) const {
    return (time_point_cast<microseconds>(Start) -
            time_point_cast<microseconds>(StartTime))
        .count();
  }

  steady_clock::rep getFlameGraphDurUs() const {
    return (time_point_cast<microseconds>(End) -
            time_point_cast<microseconds>(Start))
        .count();
  }
};

struct TimeTraceProfiler {
  TimeTraceProfiler(unsigned TimeTraceGranularity = 0, StringRef ProcName = "")
      : StartTime(steady_clock::now()), ProcName(ProcName),
        Tid(llvm::get_threadid()), TimeTraceGranularity(TimeTraceGranularity) {}

  void begin(std::string Name, llvm::function_ref<std::string()> Detail) {
    Stack.emplace_back(steady_clock::now(), TimePointType(), std::move(Name),
                       Detail());
  }

  void end() {
    assert(!Stack.empty() && "Must call begin() first");
    auto &E = Stack.back();
    E.End = steady_clock::now();

    // Check that end times monotonically increase.
    assert((Entries.empty() ||
            (E.getFlameGraphStartUs(StartTime) + E.getFlameGraphDurUs() >=
             Entries.back().getFlameGraphStartUs(StartTime) +
                 Entries.back().getFlameGraphDurUs())) &&
           "TimeProfiler scope ended earlier than previous scope");

    // Calculate duration at full precision for overall counts.
    DurationType Duration = E.End - E.Start;

    // Only include sections longer or equal to TimeTraceGranularity msec.
    if (duration_cast<microseconds>(Duration).count() >= TimeTraceGranularity)
      Entries.emplace_back(E);

    // Track total time taken by each "name", but only the topmost levels of
    // them; e.g. if there's a template instantiation that instantiates other
    // templates from within, we only want to add the topmost one. "topmost"
    // happens to be the ones that don't have any currently open entries above
    // itself.
    if (std::find_if(++Stack.rbegin(), Stack.rend(), [&](const Entry &Val) {
          return Val.Name == E.Name;
        }) == Stack.rend()) {
      auto &CountAndTotal = CountAndTotalPerName[E.Name];
      CountAndTotal.first++;
      CountAndTotal.second += Duration;
    }

    Stack.pop_back();
  }

  // Write events from this TimeTraceProfilerInstance and
  // ThreadTimeTraceProfilerInstances.
  void Write(raw_pwrite_stream &OS) {
    // Acquire Mutex as reading ThreadTimeTraceProfilerInstances.
    std::lock_guard<std::mutex> Lock(Mu);
    assert(Stack.empty() &&
           "All profiler sections should be ended when calling Write");
    assert(std::all_of(ThreadTimeTraceProfilerInstances.begin(),
                       ThreadTimeTraceProfilerInstances.end(),
                       [](const auto &TTP) { return TTP->Stack.empty(); }) &&
           "All profiler sections should be ended when calling Write");

    json::OStream J(OS);
    J.objectBegin();
    J.attributeBegin("traceEvents");
    J.arrayBegin();

    // Emit all events for the main flame graph.
    auto writeEvent = [&](const auto &E, uint64_t Tid) {
      auto StartUs = E.getFlameGraphStartUs(StartTime);
      auto DurUs = E.getFlameGraphDurUs();

      J.object([&]{
        J.attribute("pid", 1);
        J.attribute("tid", int64_t(Tid));
        J.attribute("ph", "X");
        J.attribute("ts", StartUs);
        J.attribute("dur", DurUs);
        J.attribute("name", E.Name);
        if (!E.Detail.empty()) {
          J.attributeObject("args", [&] { J.attribute("detail", E.Detail); });
        }
      });
    };
    for (const auto &E : Entries) {
      writeEvent(E, this->Tid);
    }
    for (const auto &TTP : ThreadTimeTraceProfilerInstances) {
      for (const auto &E : TTP->Entries) {
        writeEvent(E, TTP->Tid);
      }
    }

    // Emit totals by section name as additional "thread" events, sorted from
    // longest one.
    // Find highest used thread id.
    uint64_t MaxTid = this->Tid;
    for (const auto &TTP : ThreadTimeTraceProfilerInstances) {
      MaxTid = std::max(MaxTid, TTP->Tid);
    }

    // Combine all CountAndTotalPerName from threads into one.
    StringMap<CountAndDurationType> AllCountAndTotalPerName;
    auto combineStat = [&](const auto &Stat) {
      StringRef Key = Stat.getKey();
      auto Value = Stat.getValue();
      auto &CountAndTotal = AllCountAndTotalPerName[Key];
      CountAndTotal.first += Value.first;
      CountAndTotal.second += Value.second;
    };
    for (const auto &Stat : CountAndTotalPerName) {
      combineStat(Stat);
    }
    for (const auto &TTP : ThreadTimeTraceProfilerInstances) {
      for (const auto &Stat : TTP->CountAndTotalPerName) {
        combineStat(Stat);
      }
    }

    std::vector<NameAndCountAndDurationType> SortedTotals;
    SortedTotals.reserve(AllCountAndTotalPerName.size());
    for (const auto &Total : AllCountAndTotalPerName)
      SortedTotals.emplace_back(std::string(Total.getKey()), Total.getValue());

    llvm::sort(SortedTotals.begin(), SortedTotals.end(),
               [](const NameAndCountAndDurationType &A,
                  const NameAndCountAndDurationType &B) {
                 return A.second.second > B.second.second;
               });

    // Report totals on separate threads of tracing file.
    uint64_t TotalTid = MaxTid + 1;
    for (const auto &Total : SortedTotals) {
      auto DurUs = duration_cast<microseconds>(Total.second.second).count();
      auto Count = AllCountAndTotalPerName[Total.first].first;

      J.object([&]{
        J.attribute("pid", 1);
        J.attribute("tid", int64_t(TotalTid));
        J.attribute("ph", "X");
        J.attribute("ts", 0);
        J.attribute("dur", DurUs);
        J.attribute("name", "Total " + Total.first);
        J.attributeObject("args", [&] {
          J.attribute("count", int64_t(Count));
          J.attribute("avg ms", int64_t(DurUs / Count / 1000));
        });
      });

      ++TotalTid;
    }

    // Emit metadata event with process name.
    J.object([&] {
      J.attribute("cat", "");
      J.attribute("pid", 1);
      J.attribute("tid", 0);
      J.attribute("ts", 0);
      J.attribute("ph", "M");
      J.attribute("name", "process_name");
      J.attributeObject("args", [&] { J.attribute("name", ProcName); });
    });

    J.arrayEnd();
    J.attributeEnd();
    J.objectEnd();
  }

  SmallVector<Entry, 16> Stack;
  SmallVector<Entry, 128> Entries;
  StringMap<CountAndDurationType> CountAndTotalPerName;
  const TimePointType StartTime;
  const std::string ProcName;
  const uint64_t Tid;

  // Minimum time granularity (in microseconds)
  const unsigned TimeTraceGranularity;
};

void timeTraceProfilerInitialize(unsigned TimeTraceGranularity,
                                 StringRef ProcName) {
  assert(TimeTraceProfilerInstance == nullptr &&
         "Profiler should not be initialized");
  TimeTraceProfilerInstance = new TimeTraceProfiler(
      TimeTraceGranularity, llvm::sys::path::filename(ProcName));
}

// Removes all TimeTraceProfilerInstances.
// Called from main thread.
void timeTraceProfilerCleanup() {
  delete TimeTraceProfilerInstance;
  std::lock_guard<std::mutex> Lock(Mu);
  for (auto TTP : ThreadTimeTraceProfilerInstances)
    delete TTP;
  ThreadTimeTraceProfilerInstances.clear();
}

// Finish TimeTraceProfilerInstance on a worker thread.
// This doesn't remove the instance, just moves the pointer to global vector.
void timeTraceProfilerFinishThread() {
  std::lock_guard<std::mutex> Lock(Mu);
  ThreadTimeTraceProfilerInstances.push_back(TimeTraceProfilerInstance);
  TimeTraceProfilerInstance = nullptr;
}

void timeTraceProfilerWrite(raw_pwrite_stream &OS) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");
  TimeTraceProfilerInstance->Write(OS);
}

Error timeTraceProfilerWrite(StringRef PreferredFileName,
                             StringRef FallbackFileName) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");

  std::string Path = PreferredFileName.str();
  if (Path.empty()) {
    Path = FallbackFileName == "-" ? "out" : FallbackFileName.str();
    Path += ".time-trace";
  }

  std::error_code EC;
  raw_fd_ostream OS(Path, EC, sys::fs::OF_Text);
  if (EC)
    return createStringError(EC, "Could not open " + Path);

  timeTraceProfilerWrite(OS);
  return Error::success();
}

void timeTraceProfilerBegin(StringRef Name, StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->begin(std::string(Name),
                                     [&]() { return std::string(Detail); });
}

void timeTraceProfilerBegin(StringRef Name,
                            llvm::function_ref<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->begin(std::string(Name), Detail);
}

void timeTraceProfilerEnd() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end();
}

} // namespace llvm
