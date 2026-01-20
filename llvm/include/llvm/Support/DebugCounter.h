//===- llvm/Support/DebugCounter.h - Debug counter support ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides an implementation of debug counters.  Debug
/// counters are a tool that let you narrow down a miscompilation to a specific
/// thing happening.
///
/// To give a use case: Imagine you have a file, very large, and you
/// are trying to understand the minimal transformation that breaks it. Bugpoint
/// and bisection is often helpful here in narrowing it down to a specific pass,
/// but it's still a very large file, and a very complicated pass to try to
/// debug.  That is where debug counting steps in.  You can instrument the pass
/// with a debug counter before it does a certain thing, and depending on the
/// counts, it will either execute that thing or not.  The debug counter itself
/// consists of a list of chunks (inclusive numeric intervals). `shouldExecute`
/// returns true iff the list is empty or the current count is in one of the
/// chunks.
///
/// Note that a counter set to a negative number will always execute. For a
/// concrete example, during predicateinfo creation, the renaming pass replaces
/// each use with a renamed use.
////
/// If I use DEBUG_COUNTER to create a counter called "predicateinfo", and
/// variable name RenameCounter, and then instrument this renaming with a debug
/// counter, like so:
///
/// if (!DebugCounter::shouldExecute(RenameCounter)
/// <continue or return or whatever not executing looks like>
///
/// Now I can, from the command line, make it rename or not rename certain uses
/// by setting the chunk list.
/// So for example
/// bin/opt -debug-counter=predicateinfo=47
/// will skip renaming the first 47 uses, then rename one, then skip the rest.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUGCOUNTER_H
#define LLVM_SUPPORT_DEBUGCOUNTER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/IntegerInclusiveInterval.h"
#include <string>

namespace llvm {

class raw_ostream;

class DebugCounter {
public:
  /// Struct to store counter info.
  class CounterInfo {
    friend class DebugCounter;

    /// Whether counting should be enabled, either due to -debug-counter or
    /// -print-debug-counter.
    bool Active = false;
    /// Whether chunks for the counter are set (differs from Active in that
    /// -print-debug-counter uses Active=true, IsSet=false).
    bool IsSet = false;

    int64_t Count = 0;
    uint64_t CurrChunkIdx = 0;
    StringRef Name;
    StringRef Desc;
    IntegerInclusiveIntervalUtils::IntervalList Chunks;

  public:
    CounterInfo(StringRef Name, StringRef Desc) : Name(Name), Desc(Desc) {
      DebugCounter::registerCounter(this);
    }
  };

  LLVM_ABI static void
  printChunks(raw_ostream &OS, ArrayRef<IntegerInclusiveInterval> Intervals);

  /// Return true on parsing error and print the error message on the
  /// llvm::errs()
  LLVM_ABI static bool
  parseChunks(StringRef Str, IntegerInclusiveIntervalUtils::IntervalList &Res);

  /// Returns a reference to the singleton instance.
  LLVM_ABI static DebugCounter &instance();

  // Used by the command line option parser to push a new value it parsed.
  LLVM_ABI void push_back(const std::string &);

  // Register a counter with the specified counter information.
  //
  // FIXME: Currently, counter registration is required to happen before command
  // line option parsing. The main reason to register counters is to produce a
  // nice list of them on the command line, but i'm not sure this is worth it.
  static void registerCounter(CounterInfo *Info) {
    instance().addCounter(Info);
  }
  LLVM_ABI static bool shouldExecuteImpl(CounterInfo &Counter);

  inline static bool shouldExecute(CounterInfo &Counter) {
    if (!Counter.Active)
      return true;
    return shouldExecuteImpl(Counter);
  }

  // Return true if a given counter had values set (either programatically or on
  // the command line).  This will return true even if those values are
  // currently in a state where the counter will always execute.
  static bool isCounterSet(CounterInfo &Info) { return Info.IsSet; }

  struct CounterState {
    int64_t Count;
    uint64_t ChunkIdx;
  };

  // Return the state of a counter. This only works for set counters.
  static CounterState getCounterState(CounterInfo &Info) {
    return {Info.Count, Info.CurrChunkIdx};
  }

  // Set a registered counter to a given state.
  static void setCounterState(CounterInfo &Info, CounterState State) {
    Info.Count = State.Count;
    Info.CurrChunkIdx = State.ChunkIdx;
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  // Dump or print the current counter set into llvm::dbgs().
  LLVM_DUMP_METHOD void dump() const;
#endif

  LLVM_ABI void print(raw_ostream &OS) const;

  // Get the counter info for a given named counter,
  // or return null if none is found.
  CounterInfo *getCounterInfo(StringRef Name) const {
    return Counters.lookup(Name);
  }

  // Return the number of registered counters.
  unsigned int getNumCounters() const { return Counters.size(); }

  // Return the name and description of the counter with the given info.
  std::pair<StringRef, StringRef> getCounterDesc(CounterInfo *Info) const {
    return {Info->Name, Info->Desc};
  }

  // Iterate through the registered counters
  MapVector<StringRef, CounterInfo *>::const_iterator begin() const {
    return Counters.begin();
  }
  MapVector<StringRef, CounterInfo *>::const_iterator end() const {
    return Counters.end();
  }

  void activateAllCounters() {
    for (auto &[_, Counter] : Counters)
      Counter->Active = true;
  }

protected:
  void addCounter(CounterInfo *Info) { Counters[Info->Name] = Info; }
  bool handleCounterIncrement(CounterInfo &Info);

  MapVector<StringRef, CounterInfo *> Counters;

  bool ShouldPrintCounter = false;

  bool ShouldPrintCounterQueries = false;

  bool BreakOnLast = false;
};

#define DEBUG_COUNTER(VARNAME, COUNTERNAME, DESC)                              \
  static DebugCounter::CounterInfo VARNAME(COUNTERNAME, DESC)

} // namespace llvm
#endif
