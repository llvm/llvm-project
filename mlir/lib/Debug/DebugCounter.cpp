//===- DebugCounter.cpp - Debug Counter Facilities ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Debug/Counter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"

using namespace mlir;
using namespace mlir::tracing;

//===----------------------------------------------------------------------===//
// DebugCounter CommandLine Options
//===----------------------------------------------------------------------===//

namespace {
/// This struct contains command line options that can be used to initialize
/// various bits of a DebugCounter. This uses a struct wrapper to avoid the need
/// for global command line options.
struct DebugCounterOptions {
  llvm::cl::list<std::string> counters{
      "mlir-debug-counter",
      llvm::cl::desc(
          "Comma separated list of debug counter skip and count arguments"),
      llvm::cl::CommaSeparated};

  llvm::cl::opt<bool> printCounterInfo{
      "mlir-print-debug-counter", llvm::cl::init(false), llvm::cl::Optional,
      llvm::cl::desc("Print out debug counter information after all counters "
                     "have been accumulated")};
};
} // namespace

static llvm::ManagedStatic<DebugCounterOptions> clOptions;

//===----------------------------------------------------------------------===//
// DebugCounter
//===----------------------------------------------------------------------===//

DebugCounter::DebugCounter() { applyCLOptions(); }

DebugCounter::~DebugCounter() {
  // Print information when destroyed, iff command line option is specified.
  if (clOptions.isConstructed() && clOptions->printCounterInfo)
    print(llvm::dbgs());
}

/// Add a counter for the given debug action tag. `countToSkip` is the number
/// of counter executions to skip before enabling execution of the action.
/// `countToStopAfter` is the number of executions of the counter to allow
/// before preventing the action from executing any more.
void DebugCounter::addCounter(StringRef actionTag, int64_t countToSkip,
                              int64_t countToStopAfter) {
  assert(!counters.count(actionTag) &&
         "a counter for the given action was already registered");
  counters.try_emplace(actionTag, countToSkip, countToStopAfter);
}

void DebugCounter::operator()(llvm::function_ref<void()> transform,
                              const Action &action) {
  if (shouldExecute(action.getTag()))
    transform();
}

bool DebugCounter::shouldExecute(StringRef tag) {
  auto counterIt = counters.find(tag);
  if (counterIt == counters.end())
    return true;

  ++counterIt->second.count;

  // We only execute while the `countToSkip` is not smaller than `count`, and
  // `countToStopAfter + countToSkip` is larger than `count`. Negative counters
  // always execute.
  if (counterIt->second.countToSkip < 0)
    return true;
  if (counterIt->second.countToSkip >= counterIt->second.count)
    return false;
  if (counterIt->second.countToStopAfter < 0)
    return true;
  return counterIt->second.countToStopAfter + counterIt->second.countToSkip >=
         counterIt->second.count;
}

void DebugCounter::print(raw_ostream &os) const {
  // Order the registered counters by name.
  SmallVector<const llvm::StringMapEntry<Counter> *, 16> sortedCounters(
      llvm::make_pointer_range(counters));
  llvm::array_pod_sort(sortedCounters.begin(), sortedCounters.end(),
                       [](const decltype(sortedCounters)::value_type *lhs,
                          const decltype(sortedCounters)::value_type *rhs) {
                         return (*lhs)->getKey().compare((*rhs)->getKey());
                       });

  os << "DebugCounter counters:\n";
  for (const llvm::StringMapEntry<Counter> *counter : sortedCounters) {
    os << llvm::left_justify(counter->getKey(), 32) << ": {"
       << counter->second.count << "," << counter->second.countToSkip << ","
       << counter->second.countToStopAfter << "}\n";
  }
}

/// Register a set of useful command-line options that can be used to configure
/// various flags within the DebugCounter. These flags are used when
/// constructing a DebugCounter for initialization.
void DebugCounter::registerCLOptions() {
  // Make sure that the options struct has been initialized.
  *clOptions;
}

bool DebugCounter::isActivated() {
  return clOptions->counters.getNumOccurrences() ||
         clOptions->printCounterInfo.getNumOccurrences();
}

// This is called by the command line parser when it sees a value for the
// debug-counter option defined above.
void DebugCounter::applyCLOptions() {
  if (!clOptions.isConstructed())
    return;

  for (StringRef arg : clOptions->counters) {
    if (arg.empty())
      continue;

    // Debug counter arguments are expected to be in the form: `counter=value`.
    auto [counterName, counterValueStr] = arg.split('=');
    if (counterValueStr.empty()) {
      llvm::errs() << "error: expected DebugCounter argument to have an `=` "
                      "separating the counter name and value, but the provided "
                      "argument was: `"
                   << arg << "`\n";
      llvm::report_fatal_error(
          "Invalid DebugCounter command-line configuration");
    }

    // Extract the counter value.
    int64_t counterValue;
    if (counterValueStr.getAsInteger(0, counterValue)) {
      llvm::errs() << "error: expected DebugCounter counter value to be "
                      "numeric, but got `"
                   << counterValueStr << "`\n";
      llvm::report_fatal_error(
          "Invalid DebugCounter command-line configuration");
    }

    // Now we need to see if this is the skip or the count, remove the suffix,
    // and add it to the counter values.
    if (counterName.consume_back("-skip")) {
      counters[counterName].countToSkip = counterValue;

    } else if (counterName.consume_back("-count")) {
      counters[counterName].countToStopAfter = counterValue;

    } else {
      llvm::errs() << "error: expected DebugCounter counter name to end with "
                      "either `-skip` or `-count`, but got`"
                   << counterName << "`\n";
      llvm::report_fatal_error(
          "Invalid DebugCounter command-line configuration");
    }
  }
}
