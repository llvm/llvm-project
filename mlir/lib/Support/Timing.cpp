//===- Timing.cpp - Execution time measurement facilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Facilities to measure and provide statistics on execution time.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/Timing.h"
#include "mlir/Support/ThreadLocalCache.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>

using namespace mlir;
using namespace detail;
using DisplayMode = DefaultTimingManager::DisplayMode;

constexpr llvm::StringLiteral kTimingDescription =
    "... Execution time report ...";

//===----------------------------------------------------------------------===//
// TimingManager
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {
/// Private implementation details of the `TimingManager`.
class TimingManagerImpl {
public:
  // Identifier allocator, map, and mutex for thread safety.
  llvm::BumpPtrAllocator identifierAllocator;
  llvm::StringSet<llvm::BumpPtrAllocator &> identifiers;
  llvm::sys::SmartRWMutex<true> identifierMutex;

  /// A thread local cache of identifiers to reduce lock contention.
  ThreadLocalCache<llvm::StringMap<llvm::StringMapEntry<llvm::NoneType> *>>
      localIdentifierCache;

  TimingManagerImpl() : identifiers(identifierAllocator) {}
};
} // namespace detail
} // namespace mlir

TimingManager::TimingManager() : impl(std::make_unique<TimingManagerImpl>()) {}

TimingManager::~TimingManager() = default;

/// Get the root timer of this timing manager.
Timer TimingManager::getRootTimer() {
  auto rt = rootTimer();
  return rt ? Timer(*this, *rt) : Timer();
}

/// Get the root timer of this timing manager wrapped in a `TimingScope`.
TimingScope TimingManager::getRootScope() {
  return TimingScope(getRootTimer());
}

//===----------------------------------------------------------------------===//
// Identifier uniquing
//===----------------------------------------------------------------------===//

/// Return an identifier for the specified string.
TimingIdentifier TimingIdentifier::get(StringRef str, TimingManager &tm) {
  // Check for an existing instance in the local cache.
  auto &impl = *tm.impl;
  auto *&localEntry = (*impl.localIdentifierCache)[str];
  if (localEntry)
    return TimingIdentifier(localEntry);

  // Check for an existing identifier in read-only mode.
  {
    llvm::sys::SmartScopedReader<true> contextLock(impl.identifierMutex);
    auto it = impl.identifiers.find(str);
    if (it != impl.identifiers.end()) {
      localEntry = &*it;
      return TimingIdentifier(localEntry);
    }
  }

  // Acquire a writer-lock so that we can safely create the new instance.
  llvm::sys::SmartScopedWriter<true> contextLock(impl.identifierMutex);
  auto it = impl.identifiers.insert(str).first;
  localEntry = &*it;
  return TimingIdentifier(localEntry);
}

//===----------------------------------------------------------------------===//
// Helpers for time record printing
//===----------------------------------------------------------------------===//

namespace {

/// Simple record class to record timing information.
struct TimeRecord {
  TimeRecord(double wall = 0.0, double user = 0.0) : wall(wall), user(user) {}

  TimeRecord &operator+=(const TimeRecord &other) {
    wall += other.wall;
    user += other.user;
    return *this;
  }

  TimeRecord &operator-=(const TimeRecord &other) {
    wall -= other.wall;
    user -= other.user;
    return *this;
  }

  /// Print the current time record to 'os', with a breakdown showing
  /// contributions to the give 'total' time record.
  void print(raw_ostream &os, const TimeRecord &total) {
    if (total.user != total.wall)
      os << llvm::format("  %8.4f (%5.1f%%)", user, 100.0 * user / total.user);
    os << llvm::format("  %8.4f (%5.1f%%)  ", wall, 100.0 * wall / total.wall);
  }

  double wall, user;
};

} // namespace

/// Utility to print a single line entry in the timer output.
static void printTimeEntry(raw_ostream &os, unsigned indent, StringRef name,
                           TimeRecord time, TimeRecord total) {
  time.print(os, total);
  os.indent(indent) << name << "\n";
}

/// Utility to print the timer heading information.
static void printTimeHeader(raw_ostream &os, TimeRecord total) {
  // Figure out how many spaces to description name.
  unsigned padding = (80 - kTimingDescription.size()) / 2;
  os << "===" << std::string(73, '-') << "===\n";
  os.indent(padding) << kTimingDescription << '\n';
  os << "===" << std::string(73, '-') << "===\n";

  // Print the total time followed by the section headers.
  os << llvm::format("  Total Execution Time: %.4f seconds\n\n", total.wall);
  if (total.user != total.wall)
    os << "  ----User Time----";
  os << "  ----Wall Time----  ----Name----\n";
}

//===----------------------------------------------------------------------===//
// Timer Implementation for DefaultTimingManager
//===----------------------------------------------------------------------===//

namespace {

/// A timer used to sample execution time.
///
/// Separately tracks wall time and user time to account for parallel threads of
/// execution. Timers are intended to be started and stopped multiple times.
/// Each start and stop will add to the timer's wall and user time.
class TimerImpl {
public:
  using ChildrenMap = llvm::MapVector<const void *, std::unique_ptr<TimerImpl>>;
  using AsyncChildrenMap = llvm::DenseMap<uint64_t, ChildrenMap>;

  TimerImpl(std::string &&name) : threadId(llvm::get_threadid()), name(name) {}

  /// Start the timer.
  void start() { startTime = std::chrono::steady_clock::now(); }

  /// Stop the timer.
  void stop() {
    auto newTime = std::chrono::steady_clock::now() - startTime;
    wallTime += newTime;
    userTime += newTime;
  }

  /// Create a child timer nested within this one. Multiple calls to this
  /// function with the same unique identifier `id` will return the same child
  /// timer.
  ///
  /// This function can be called from other threads, as long as this timer
  /// outlives any uses of the child timer on the other thread.
  TimerImpl *nest(const void *id, function_ref<std::string()> nameBuilder) {
    auto tid = llvm::get_threadid();
    if (tid == threadId)
      return nestTail(children[id], nameBuilder);
    std::unique_lock<std::mutex> lock(asyncMutex);
    return nestTail(asyncChildren[tid][id], nameBuilder);
  }

  /// Tail-called from `nest()`.
  TimerImpl *nestTail(std::unique_ptr<TimerImpl> &child,
                      function_ref<std::string()> nameBuilder) {
    if (!child)
      child = std::make_unique<TimerImpl>(nameBuilder());
    return child.get();
  }

  /// Finalize this timer and all its children.
  ///
  /// If this timer has async children, which happens if `nest()` was called
  /// from another thread, this function merges the async childr timers into the
  /// main list of child timers.
  ///
  /// Caution: Call this function only after all nested timers running on other
  /// threads no longer need their timers!
  void finalize() {
    addAsyncUserTime();
    mergeAsyncChildren();
  }

  /// Add the user time of all async children to this timer's user time. This is
  /// necessary since the user time already contains all regular child timers,
  /// but not the asynchronous ones (by the nesting nature of the timers).
  std::chrono::nanoseconds addAsyncUserTime() {
    auto added = std::chrono::nanoseconds(0);
    for (auto &child : children)
      added += child.second->addAsyncUserTime();
    for (auto &thread : asyncChildren) {
      for (auto &child : thread.second) {
        child.second->addAsyncUserTime();
        added += child.second->userTime;
      }
    }
    userTime += added;
    return added;
  }

  /// Ensure that this timer and recursively all its children have their async
  /// children folded into the main map of children.
  void mergeAsyncChildren() {
    for (auto &child : children)
      child.second->mergeAsyncChildren();
    mergeChildren(std::move(asyncChildren));
    assert(asyncChildren.empty());
  }

  /// Merge multiple child timers into this timer.
  ///
  /// Children in `other` are added as children to this timer, or, if this timer
  /// already contains a child with the corresponding unique identifier, are
  /// merged into the existing child.
  void mergeChildren(ChildrenMap &&other) {
    if (children.empty()) {
      children = std::move(other);
      for (auto &child : children)
        child.second->mergeAsyncChildren();
    } else {
      for (auto &child : other)
        mergeChild(child.first, std::move(child.second));
      other.clear();
    }
  }

  /// See above.
  void mergeChildren(AsyncChildrenMap &&other) {
    for (auto &thread : other) {
      mergeChildren(std::move(thread.second));
      assert(thread.second.empty());
    }
    other.clear();
  }

  /// Merge a child timer into this timer for a given unique identifier.
  ///
  /// Moves all child and async child timers of `other` into this timer's child
  /// for the given unique identifier.
  void mergeChild(const void *id, std::unique_ptr<TimerImpl> &&other) {
    auto &into = children[id];
    if (!into) {
      into = std::move(other);
      into->mergeAsyncChildren();
    } else {
      into->wallTime = std::max(into->wallTime, other->wallTime);
      into->userTime += other->userTime;
      into->mergeChildren(std::move(other->children));
      into->mergeChildren(std::move(other->asyncChildren));
      other.reset();
    }
  }

  /// Dump a human-readable tree representation of the timer and its children.
  /// This is useful for debugging the timing mechanisms and structure of the
  /// timers.
  void dump(raw_ostream &os, unsigned indent = 0, unsigned markThreadId = 0) {
    auto time = getTimeRecord();
    os << std::string(indent * 2, ' ') << name << " [" << threadId << "]"
       << llvm::format("  %7.4f / %7.4f", time.user, time.wall);
    if (threadId != markThreadId && markThreadId != 0)
      os << " (*)";
    os << "\n";
    for (auto &child : children)
      child.second->dump(os, indent + 1, threadId);
    for (auto &thread : asyncChildren)
      for (auto &child : thread.second)
        child.second->dump(os, indent + 1, threadId);
  }

  /// Returns the time for this timer in seconds.
  TimeRecord getTimeRecord() {
    return TimeRecord(
        std::chrono::duration_cast<std::chrono::duration<double>>(wallTime)
            .count(),
        std::chrono::duration_cast<std::chrono::duration<double>>(userTime)
            .count());
  }

  /// Print the timing result in list mode.
  void printAsList(raw_ostream &os, TimeRecord total) {
    // Flatten the leaf timers in the tree and merge them by name.
    llvm::StringMap<TimeRecord> mergedTimers;
    std::function<void(TimerImpl *)> addTimer = [&](TimerImpl *timer) {
      mergedTimers[timer->name] += timer->getTimeRecord();
      for (auto &children : timer->children)
        addTimer(children.second.get());
    };
    addTimer(this);

    // Sort the timing information by wall time.
    std::vector<std::pair<StringRef, TimeRecord>> timerNameAndTime;
    for (auto &it : mergedTimers)
      timerNameAndTime.emplace_back(it.first(), it.second);
    llvm::array_pod_sort(timerNameAndTime.begin(), timerNameAndTime.end(),
                         [](const std::pair<StringRef, TimeRecord> *lhs,
                            const std::pair<StringRef, TimeRecord> *rhs) {
                           return llvm::array_pod_sort_comparator<double>(
                               &rhs->second.wall, &lhs->second.wall);
                         });

    // Print the timing information sequentially.
    for (auto &timeData : timerNameAndTime)
      printTimeEntry(os, 0, timeData.first, timeData.second, total);
  }

  /// Print the timing result in tree mode.
  void printAsTree(raw_ostream &os, TimeRecord total, unsigned indent = 0) {
    unsigned childIndent = indent;
    if (!hidden) {
      printTimeEntry(os, indent, name, getTimeRecord(), total);
      childIndent += 2;
    }
    for (auto &child : children) {
      child.second->printAsTree(os, total, childIndent);
    }
  }

  /// Print the current timing information.
  void print(raw_ostream &os, DisplayMode displayMode) {
    // Print the banner.
    auto total = getTimeRecord();
    printTimeHeader(os, total);

    // Defer to a specialized printer for each display mode.
    switch (displayMode) {
    case DisplayMode::List:
      printAsList(os, total);
      break;
    case DisplayMode::Tree:
      printAsTree(os, total);
      break;
    }

    // Print the top-level time not accounted for by child timers, and the
    // total.
    auto rest = total;
    for (auto &child : children)
      rest -= child.second->getTimeRecord();
    printTimeEntry(os, 0, "Rest", rest, total);
    printTimeEntry(os, 0, "Total", total, total);
    os.flush();
  }

  /// The last time instant at which the timer was started.
  std::chrono::time_point<std::chrono::steady_clock> startTime;

  /// Accumulated wall time. If multiple threads of execution are merged into
  /// this timer, the wall time will hold the maximum wall time of each thread
  /// of execution.
  std::chrono::nanoseconds wallTime = std::chrono::nanoseconds(0);

  /// Accumulated user time. If multiple threads of execution are merged into
  /// this timer, each thread's user time is added here.
  std::chrono::nanoseconds userTime = std::chrono::nanoseconds(0);

  /// The thread on which this timer is running.
  uint64_t threadId;

  /// A descriptive name for this timer.
  std::string name;

  /// Whether to omit this timer from reports and directly show its children.
  bool hidden = false;

  /// Child timers on the same thread the timer itself. We keep at most one
  /// timer per unique identifier.
  ChildrenMap children;

  /// Child timers on other threads. We keep at most one timer per unique
  /// identifier.
  AsyncChildrenMap asyncChildren;

  /// Mutex for the async children.
  std::mutex asyncMutex;
};

} // namespace

//===----------------------------------------------------------------------===//
// DefaultTimingManager
//===----------------------------------------------------------------------===//

namespace mlir {
namespace detail {

/// Implementation details of the `DefaultTimingManager`.
class DefaultTimingManagerImpl {
public:
  /// Whether we should do our work or not.
  bool enabled = false;

  /// The configured display mode.
  DisplayMode displayMode = DisplayMode::Tree;

  /// The stream where we should print our output. This will always be non-null.
  raw_ostream *output = &llvm::errs();

  /// The root timer.
  std::unique_ptr<TimerImpl> rootTimer;
};

} // namespace detail
} // namespace mlir

DefaultTimingManager::DefaultTimingManager()
    : impl(std::make_unique<DefaultTimingManagerImpl>()) {
  clear(); // initializes the root timer
}

DefaultTimingManager::~DefaultTimingManager() { print(); }

/// Enable or disable execution time sampling.
void DefaultTimingManager::setEnabled(bool enabled) { impl->enabled = enabled; }

/// Return whether execution time sampling is enabled.
bool DefaultTimingManager::isEnabled() const { return impl->enabled; }

/// Change the display mode.
void DefaultTimingManager::setDisplayMode(DisplayMode displayMode) {
  impl->displayMode = displayMode;
}

/// Return the current display mode;
DefaultTimingManager::DisplayMode DefaultTimingManager::getDisplayMode() const {
  return impl->displayMode;
}

/// Change the stream where the output will be printed to.
void DefaultTimingManager::setOutput(raw_ostream &os) { impl->output = &os; }

/// Return the current output stream where the output will be printed to.
raw_ostream &DefaultTimingManager::getOutput() const {
  assert(impl->output);
  return *impl->output;
}

/// Print and clear the timing results.
void DefaultTimingManager::print() {
  if (impl->enabled) {
    impl->rootTimer->finalize();
    impl->rootTimer->print(*impl->output, impl->displayMode);
  }
  clear();
}

/// Clear the timing results.
void DefaultTimingManager::clear() {
  impl->rootTimer = std::make_unique<TimerImpl>("root");
  impl->rootTimer->hidden = true;
}

/// Debug print the timer data structures to an output stream.
void DefaultTimingManager::dumpTimers(raw_ostream &os) {
  impl->rootTimer->dump(os);
}

/// Debug print the timers as a list.
void DefaultTimingManager::dumpAsList(raw_ostream &os) {
  impl->rootTimer->finalize();
  impl->rootTimer->print(os, DisplayMode::List);
}

/// Debug print the timers as a tree.
void DefaultTimingManager::dumpAsTree(raw_ostream &os) {
  impl->rootTimer->finalize();
  impl->rootTimer->print(os, DisplayMode::Tree);
}

Optional<void *> DefaultTimingManager::rootTimer() {
  if (impl->enabled)
    return impl->rootTimer.get();
  return llvm::None;
}

void DefaultTimingManager::startTimer(void *handle) {
  static_cast<TimerImpl *>(handle)->start();
}

void DefaultTimingManager::stopTimer(void *handle) {
  static_cast<TimerImpl *>(handle)->stop();
}

void *DefaultTimingManager::nestTimer(void *handle, const void *id,
                                      function_ref<std::string()> nameBuilder) {
  return static_cast<TimerImpl *>(handle)->nest(id, nameBuilder);
}

void DefaultTimingManager::hideTimer(void *handle) {
  static_cast<TimerImpl *>(handle)->hidden = true;
}

//===----------------------------------------------------------------------===//
// DefaultTimingManager Command Line Options
//===----------------------------------------------------------------------===//

namespace {
struct DefaultTimingManagerOptions {
  llvm::cl::opt<bool> timing{"mlir-timing",
                             llvm::cl::desc("Display execution times"),
                             llvm::cl::init(false)};
  llvm::cl::opt<DisplayMode> displayMode{
      "mlir-timing-display", llvm::cl::desc("Display method for timing data"),
      llvm::cl::init(DisplayMode::Tree),
      llvm::cl::values(
          clEnumValN(DisplayMode::List, "list",
                     "display the results in a list sorted by total time"),
          clEnumValN(DisplayMode::Tree, "tree",
                     "display the results ina with a nested tree view"))};
};
} // namespace

static llvm::ManagedStatic<DefaultTimingManagerOptions> options;

void mlir::registerDefaultTimingManagerCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;
}

void mlir::applyDefaultTimingManagerCLOptions(DefaultTimingManager &tm) {
  if (!options.isConstructed())
    return;
  tm.setEnabled(options->timing);
  tm.setDisplayMode(options->displayMode);
}
