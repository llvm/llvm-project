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
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include <cassert>
#include <chrono>
#include <string>
#include <vector>

using namespace std::chrono;

namespace llvm {

static cl::opt<unsigned> TimeTraceGranularity(
    "time-trace-granularity",
    cl::desc(
        "Minimum time granularity (in microseconds) traced by time profiler"),
    cl::init(500));

TimeTraceProfiler *TimeTraceProfilerInstance = nullptr;

typedef duration<steady_clock::rep, steady_clock::period> DurationType;
typedef std::pair<size_t, DurationType> CountAndDurationType;
typedef std::pair<std::string, CountAndDurationType>
    NameAndCountAndDurationType;

struct Entry {
  time_point<steady_clock> Start;
  DurationType Duration;
  std::string Name;
  std::string Detail;

  Entry(time_point<steady_clock> &&S, DurationType &&D, std::string &&N,
        std::string &&Dt)
      : Start(std::move(S)), Duration(std::move(D)), Name(std::move(N)),
        Detail(std::move(Dt)){};
};

struct TimeTraceProfiler {
  TimeTraceProfiler() {
    StartTime = steady_clock::now();
  }

  void begin(std::string Name, llvm::function_ref<std::string()> Detail) {
    Stack.emplace_back(steady_clock::now(), DurationType{}, std::move(Name),
                       Detail());
  }

  void end() {
    assert(!Stack.empty() && "Must call begin() first");
    auto &E = Stack.back();
    E.Duration = steady_clock::now() - E.Start;

    // Only include sections longer than TimeTraceGranularity msec.
    if (duration_cast<microseconds>(E.Duration).count() > TimeTraceGranularity)
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
      CountAndTotal.second += E.Duration;
    }

    Stack.pop_back();
  }

  void Write(raw_pwrite_stream &OS) {
    assert(Stack.empty() &&
           "All profiler sections should be ended when calling Write");
    json::OStream J(OS);
    J.objectBegin();
    J.attributeBegin("traceEvents");
    J.arrayBegin();

    // Emit all events for the main flame graph.
    for (const auto &E : Entries) {
      auto StartUs = duration_cast<microseconds>(E.Start - StartTime).count();
      auto DurUs = duration_cast<microseconds>(E.Duration).count();

      J.object([&]{
        J.attribute("pid", 1);
        J.attribute("tid", 0);
        J.attribute("ph", "X");
        J.attribute("ts", StartUs);
        J.attribute("dur", DurUs);
        J.attribute("name", E.Name);
        J.attributeObject("args", [&] { J.attribute("detail", E.Detail); });
      });
    }

    // Emit totals by section name as additional "thread" events, sorted from
    // longest one.
    int Tid = 1;
    std::vector<NameAndCountAndDurationType> SortedTotals;
    SortedTotals.reserve(CountAndTotalPerName.size());
    for (const auto &E : CountAndTotalPerName)
      SortedTotals.emplace_back(E.getKey(), E.getValue());

    llvm::sort(SortedTotals.begin(), SortedTotals.end(),
               [](const NameAndCountAndDurationType &A,
                  const NameAndCountAndDurationType &B) {
                 return A.second.second > B.second.second;
               });
    for (const auto &E : SortedTotals) {
      auto DurUs = duration_cast<microseconds>(E.second.second).count();
      auto Count = CountAndTotalPerName[E.first].first;

      J.object([&]{
        J.attribute("pid", 1);
        J.attribute("tid", Tid);
        J.attribute("ph", "X");
        J.attribute("ts", 0);
        J.attribute("dur", DurUs);
        J.attribute("name", "Total " + E.first);
        J.attributeObject("args", [&] {
          J.attribute("count", int64_t(Count));
          J.attribute("avg ms", int64_t(DurUs / Count / 1000));
        });
      });

      ++Tid;
    }

    // Emit metadata event with process name.
    J.object([&] {
      J.attribute("cat", "");
      J.attribute("pid", 1);
      J.attribute("tid", 0);
      J.attribute("ts", 0);
      J.attribute("ph", "M");
      J.attribute("name", "process_name");
      J.attributeObject("args", [&] { J.attribute("name", "clang"); });
    });

    J.arrayEnd();
    J.attributeEnd();
    J.objectEnd();
  }

  SmallVector<Entry, 16> Stack;
  SmallVector<Entry, 128> Entries;
  StringMap<CountAndDurationType> CountAndTotalPerName;
  time_point<steady_clock> StartTime;
};

void timeTraceProfilerInitialize() {
  assert(TimeTraceProfilerInstance == nullptr &&
         "Profiler should not be initialized");
  TimeTraceProfilerInstance = new TimeTraceProfiler();
}

void timeTraceProfilerCleanup() {
  delete TimeTraceProfilerInstance;
  TimeTraceProfilerInstance = nullptr;
}

void timeTraceProfilerWrite(raw_pwrite_stream &OS) {
  assert(TimeTraceProfilerInstance != nullptr &&
         "Profiler object can't be null");
  TimeTraceProfilerInstance->Write(OS);
}

void timeTraceProfilerBegin(StringRef Name, StringRef Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->begin(Name, [&]() { return Detail; });
}

void timeTraceProfilerBegin(StringRef Name,
                            llvm::function_ref<std::string()> Detail) {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->begin(Name, Detail);
}

void timeTraceProfilerEnd() {
  if (TimeTraceProfilerInstance != nullptr)
    TimeTraceProfilerInstance->end();
}

} // namespace llvm
