//===- llvm/Support/DebugCounter.h - Debug counter support ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file provides an implementation of debug counters.  Debug
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
/// consists of a skip and a count.  Skip is the number of times shouldExecute
/// needs to be called before it returns true.  Count is the number of times to
/// return true once Skip is 0.  So a skip=47, count=2 ,would skip the first 47
/// executions by returning false from shouldExecute, then execute twice, and
/// then return false again.
/// Note that a counter set to a negative number will always execute.
/// For a concrete example, during predicateinfo creation, the renaming pass
/// replaces each use with a renamed use.
////
/// If I use DEBUG_COUNTER to create a counter called "predicateinfo", and
/// variable name RenameCounter, and then instrument this renaming with a debug
/// counter, like so:
///
/// if (!DebugCounter::shouldExecute(RenameCounter)
/// <continue or return or whatever not executing looks like>
///
/// Now I can, from the command line, make it rename or not rename certain uses
/// by setting the skip and count.
/// So for example
/// bin/opt -debug-counter=predicateinfo-skip=47,predicateinfo-count=1
/// will skip renaming the first 47 uses, then rename one, then skip the rest.
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DEBUGCOUNTER_H
#define LLVM_SUPPORT_DEBUGCOUNTER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/UniqueVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace llvm {

class DebugCounter {
public:
  /// \brief Returns a reference to the singleton instance.
  static DebugCounter &instance();

  // Used by the command line option parser to push a new value it parsed.
  void push_back(const std::string &);

  // Register a counter with the specified name.
  //
  // FIXME: Currently, counter registration is required to happen before command
  // line option parsing. The main reason to register counters is to produce a
  // nice list of them on the command line, but i'm not sure this is worth it.
  static unsigned registerCounter(StringRef Name, StringRef Desc) {
    return instance().addCounter(Name, Desc);
  }
  inline static bool shouldExecute(unsigned CounterName) {
// Compile to nothing when debugging is off
#ifdef NDEBUG
    return true;
#else
    auto &Us = instance();
    auto Result = Us.Counters.find(CounterName);
    if (Result != Us.Counters.end()) {
      auto &CounterPair = Result->second;
      // We only execute while the skip (first) is zero and the count (second)
      // is non-zero.
      // Negative counters always execute.
      if (CounterPair.first < 0)
        return true;
      if (CounterPair.first != 0) {
        --CounterPair.first;
        return false;
      }
      if (CounterPair.second < 0)
        return true;
      if (CounterPair.second != 0) {
        --CounterPair.second;
        return true;
      }
      return false;
    }
    // Didn't find the counter, should we warn?
    return true;
#endif // NDEBUG
  }

  // Return true if a given counter had values set (either programatically or on
  // the command line).  This will return true even if those values are
  // currently in a state where the counter will always execute.
  static bool isCounterSet(unsigned ID) {
    return instance().Counters.count(ID);
  }

  // Return the skip and count for a counter. This only works for set counters.
  static std::pair<int, int> getCounterValue(unsigned ID) {
    auto &Us = instance();
    auto Result = Us.Counters.find(ID);
    assert(Result != Us.Counters.end() && "Asking about a non-set counter");
    return Result->second;
  }

  // Set a registered counter to a given value.
  static void setCounterValue(unsigned ID, const std::pair<int, int> &Val) {
    auto &Us = instance();
    Us.Counters[ID] = Val;
  }

  // Dump or print the current counter set into llvm::dbgs().
  LLVM_DUMP_METHOD void dump() const;

  void print(raw_ostream &OS) const;

  // Get the counter ID for a given named counter, or return 0 if none is found.
  unsigned getCounterId(const std::string &Name) const {
    return RegisteredCounters.idFor(Name);
  }

  // Return the number of registered counters.
  unsigned int getNumCounters() const { return RegisteredCounters.size(); }

  // Return the name and description of the counter with the given ID.
  std::pair<std::string, std::string> getCounterInfo(unsigned ID) const {
    return std::make_pair(RegisteredCounters[ID], CounterDesc.lookup(ID));
  }

  // Iterate through the registered counters
  typedef UniqueVector<std::string> CounterVector;
  CounterVector::const_iterator begin() const {
    return RegisteredCounters.begin();
  }
  CounterVector::const_iterator end() const { return RegisteredCounters.end(); }

private:
  unsigned addCounter(const std::string &Name, const std::string &Desc) {
    unsigned Result = RegisteredCounters.insert(Name);
    CounterDesc[Result] = Desc;
    return Result;
  }
  DenseMap<unsigned, std::pair<long, long>> Counters;
  DenseMap<unsigned, std::string> CounterDesc;
  CounterVector RegisteredCounters;
};

#define DEBUG_COUNTER(VARNAME, COUNTERNAME, DESC)                              \
  static const unsigned VARNAME =                                              \
      DebugCounter::registerCounter(COUNTERNAME, DESC);

} // namespace llvm
#endif
