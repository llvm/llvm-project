//===-- TimeProfiler.cpp - Hierarchical Time Profiler ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file Hierarchical time profiler implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/TimeProfiler.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/FileSystem.h"
#include <cassert>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std::chrono;

namespace llvm {

TimeTraceProfiler *TimeTraceProfilerInstance = nullptr;

static std::string escapeString(StringRef Src) {
  std::string OS;
  for (const unsigned char &C : Src) {
    switch (C) {
    case '"':
    case '/':
    case '\\':
    case '\b':
    case '\f':
    case '\n':
    case '\r':
    case '\t':
      OS += '\\';
      OS += C;
      break;
    default:
      if (isPrint(C)) {
        OS += C;
      }
    }
  }
  return OS;
}

typedef duration<steady_clock::rep, steady_clock::period> DurationType;
typedef std::pair<size_t, DurationType> CountAndDurationType;
typedef std::pair<std::string, CountAndDurationType>
    NameAndCountAndDurationType;

struct Entry {
  time_point<steady_clock> Start;
  DurationType Duration;
  std::string Name;
  std::string Detail;
};

struct TimeTraceProfiler {
  TimeTraceProfiler() {
    Stack.reserve(8);
    Entries.reserve(128);
    StartTime = steady_clock::now();
  }

  void begin(std::string Name, llvm::function_ref<std::string()> Detail) {
    Entry E = {steady_clock::now(), {}, Name, Detail()};
    Stack.push_back(std::move(E));
  }

  void end() {
    assert(!Stack.empty() && "Must call begin() first");
    auto &E = Stack.back();
    E.Duration = steady_clock::now() - E.Start;

    // Only include sections longer than 500us.
    if (duration_cast<microseconds>(E.Duration).count() > 500)
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

  void Write(std::unique_ptr<raw_pwrite_stream> &OS) {
    assert(Stack.empty() &&
           "All profiler sections should be ended when calling Write");

    *OS << "{ \"traceEvents\": [\n";

    // Emit all events for the main flame graph.
    for (const auto &E : Entries) {
      auto StartUs = duration_cast<microseconds>(E.Start - StartTime).count();
      auto DurUs = duration_cast<microseconds>(E.Duration).count();
      *OS << "{ \"pid\":1, \"tid\":0, \"ph\":\"X\", \"ts\":" << StartUs
          << ", \"dur\":" << DurUs << ", \"name\":\"" << escapeString(E.Name)
          << "\", \"args\":{ \"detail\":\"" << escapeString(E.Detail)
          << "\"} },\n";
    }

    // Emit totals by section name as additional "thread" events, sorted from
    // longest one.
    int Tid = 1;
    std::vector<NameAndCountAndDurationType> SortedTotals;
    SortedTotals.reserve(CountAndTotalPerName.size());
    for (const auto &E : CountAndTotalPerName) {
      SortedTotals.emplace_back(E.getKey(), E.getValue());
    }
    std::sort(SortedTotals.begin(), SortedTotals.end(),
              [](const NameAndCountAndDurationType &A,
                 const NameAndCountAndDurationType &B) {
                return A.second.second > B.second.second;
              });
    for (const auto &E : SortedTotals) {
      auto DurUs = duration_cast<microseconds>(E.second.second).count();
      auto Count = CountAndTotalPerName[E.first].first;
      *OS << "{ \"pid\":1, \"tid\":" << Tid << ", \"ph\":\"X\", \"ts\":" << 0
          << ", \"dur\":" << DurUs << ", \"name\":\"Total "
          << escapeString(E.first) << "\", \"args\":{ \"count\":" << Count
          << ", \"avg ms\":" << (DurUs / Count / 1000) << "} },\n";
      ++Tid;
    }

    // Emit metadata event with process name.
    *OS << "{ \"cat\":\"\", \"pid\":1, \"tid\":0, \"ts\":0, \"ph\":\"M\", "
           "\"name\":\"process_name\", \"args\":{ \"name\":\"clang\" } }\n";
    *OS << "] }\n";
  }

  std::vector<Entry> Stack;
  std::vector<Entry> Entries;
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

void timeTraceProfilerWrite(std::unique_ptr<raw_pwrite_stream> &OS) {
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
