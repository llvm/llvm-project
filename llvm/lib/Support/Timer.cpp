//===-- Timer.cpp - Interval Timing Support -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Interval Timing implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Timer.h"

#include "DebugOptions.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Signposts.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_PROC_PID_RUSAGE
#include <libproc.h>
#endif

using namespace llvm;

//===----------------------------------------------------------------------===//
// Forward declarations for Managed Timer Globals (mtg) getters.
//
// Globals have been placed at the end of the file to restrict direct
// access. Use of getters also has the benefit of making it a bit more explicit
// that a global is being used.
//===----------------------------------------------------------------------===//
namespace {
class Name2PairMap;
}

namespace mtg {
static std::string &LibSupportInfoOutputFilename();
static const std::string &InfoOutputFilename();
static bool TrackSpace();
static bool SortTimers();
static SignpostEmitter &Signposts();
static sys::SmartMutex<true> &TimerLock();
static TimerGroup &DefaultTimerGroup();
static TimerGroup *claimDefaultTimerGroup();
static Name2PairMap &NamedGroupedTimers();
} // namespace mtg

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
void llvm::initTimerOptions() {
  mtg::TrackSpace();
  mtg::InfoOutputFilename();
  mtg::SortTimers();
}

std::unique_ptr<raw_ostream> llvm::CreateInfoOutputFile() {
  const std::string &OutputFilename = mtg::LibSupportInfoOutputFilename();
  if (OutputFilename.empty())
    return std::make_unique<raw_fd_ostream>(2, false); // stderr.
  if (OutputFilename == "-")
    return std::make_unique<raw_fd_ostream>(1, false); // stdout.

  // Append mode is used because the info output file is opened and closed
  // each time -stats or -time-passes wants to print output to it. To
  // compensate for this, the test-suite Makefiles have code to delete the
  // info output file before running commands which write to it.
  std::error_code EC;
  auto Result = std::make_unique<raw_fd_ostream>(
      OutputFilename, EC, sys::fs::OF_Append | sys::fs::OF_TextWithCRLF);
  if (!EC)
    return Result;

  errs() << "Error opening info-output-file '"
    << OutputFilename << " for appending!\n";
  return std::make_unique<raw_fd_ostream>(2, false); // stderr.
}

//===----------------------------------------------------------------------===//
// Timer Implementation
//===----------------------------------------------------------------------===//

void Timer::init(StringRef TimerName, StringRef TimerDescription) {
  init(TimerName, TimerDescription, mtg::DefaultTimerGroup());
}

void Timer::init(StringRef TimerName, StringRef TimerDescription,
                 TimerGroup &tg) {
  assert(!TG && "Timer already initialized");
  Name.assign(TimerName.begin(), TimerName.end());
  Description.assign(TimerDescription.begin(), TimerDescription.end());
  Running = Triggered = false;
  TG = &tg;
  TG->addTimer(*this);
}

Timer::~Timer() {
  if (!TG) return;  // Never initialized, or already cleared.
  TG->removeTimer(*this);
}

static inline size_t getMemUsage() {
  if (!mtg::TrackSpace())
    return 0;
  return sys::Process::GetMallocUsage();
}

static uint64_t getCurInstructionsExecuted() {
#if defined(HAVE_UNISTD_H) && defined(HAVE_PROC_PID_RUSAGE) &&                 \
    defined(RUSAGE_INFO_V4)
  struct rusage_info_v4 ru;
  if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ru) == 0) {
    return ru.ri_instructions;
  }
#endif
  return 0;
}

TimeRecord TimeRecord::getCurrentTime(bool Start) {
  using Seconds = std::chrono::duration<double, std::ratio<1>>;
  TimeRecord Result;
  sys::TimePoint<> now;
  std::chrono::nanoseconds user, sys;

  if (Start) {
    Result.MemUsed = getMemUsage();
    Result.InstructionsExecuted = getCurInstructionsExecuted();
    sys::Process::GetTimeUsage(now, user, sys);
  } else {
    sys::Process::GetTimeUsage(now, user, sys);
    Result.InstructionsExecuted = getCurInstructionsExecuted();
    Result.MemUsed = getMemUsage();
  }

  Result.WallTime = Seconds(now.time_since_epoch()).count();
  Result.UserTime = Seconds(user).count();
  Result.SystemTime = Seconds(sys).count();
  return Result;
}

void Timer::startTimer() {
  assert(!Running && "Cannot start a running timer");
  Running = Triggered = true;
  mtg::Signposts().startInterval(this, getName());
  StartTime = TimeRecord::getCurrentTime(true);
}

void Timer::stopTimer() {
  assert(Running && "Cannot stop a paused timer");
  Running = false;
  Time += TimeRecord::getCurrentTime(false);
  Time -= StartTime;
  mtg::Signposts().endInterval(this, getName());
}

void Timer::clear() {
  Running = Triggered = false;
  Time = StartTime = TimeRecord();
}

static void printVal(double Val, double Total, raw_ostream &OS) {
  if (Total < 1e-7)   // Avoid dividing by zero.
    OS << "        -----     ";
  else
    OS << format("  %7.4f (%5.1f%%)", Val, Val*100/Total);
}

void TimeRecord::print(const TimeRecord &Total, raw_ostream &OS) const {
  if (Total.getUserTime())
    printVal(getUserTime(), Total.getUserTime(), OS);
  if (Total.getSystemTime())
    printVal(getSystemTime(), Total.getSystemTime(), OS);
  if (Total.getProcessTime())
    printVal(getProcessTime(), Total.getProcessTime(), OS);
  printVal(getWallTime(), Total.getWallTime(), OS);

  OS << "  ";

  if (Total.getMemUsed())
    OS << format("%9" PRId64 "  ", (int64_t)getMemUsed());
  if (Total.getInstructionsExecuted())
    OS << format("%9" PRId64 "  ", (int64_t)getInstructionsExecuted());
}


//===----------------------------------------------------------------------===//
//   NamedRegionTimer Implementation
//===----------------------------------------------------------------------===//

namespace {

typedef StringMap<Timer> Name2TimerMap;

class Name2PairMap {
  StringMap<std::pair<TimerGroup*, Name2TimerMap> > Map;
public:
  ~Name2PairMap() {
    for (StringMap<std::pair<TimerGroup*, Name2TimerMap> >::iterator
         I = Map.begin(), E = Map.end(); I != E; ++I)
      delete I->second.first;
  }

  Timer &get(StringRef Name, StringRef Description, StringRef GroupName,
             StringRef GroupDescription) {
    sys::SmartScopedLock<true> L(mtg::TimerLock());

    std::pair<TimerGroup*, Name2TimerMap> &GroupEntry = Map[GroupName];

    if (!GroupEntry.first)
      GroupEntry.first = new TimerGroup(GroupName, GroupDescription);

    Timer &T = GroupEntry.second[Name];
    if (!T.isInitialized())
      T.init(Name, Description, *GroupEntry.first);
    return T;
  }
};

}

NamedRegionTimer::NamedRegionTimer(StringRef Name, StringRef Description,
                                   StringRef GroupName,
                                   StringRef GroupDescription, bool Enabled)
    : TimeRegion(!Enabled ? nullptr
                          : &mtg::NamedGroupedTimers().get(Name, Description,
                                                           GroupName,
                                                           GroupDescription)) {}

//===----------------------------------------------------------------------===//
//   TimerGroup Implementation
//===----------------------------------------------------------------------===//

/// This is the global list of TimerGroups, maintained by the TimerGroup
/// ctor/dtor and is protected by the TimerLock lock.
static TimerGroup *TimerGroupList = nullptr;

TimerGroup::TimerGroup(StringRef Name, StringRef Description,
                       sys::SmartMutex<true> &lock)
    : Name(Name.begin(), Name.end()),
      Description(Description.begin(), Description.end()) {
  // Add the group to TimerGroupList.
  sys::SmartScopedLock<true> L(lock);
  if (TimerGroupList)
    TimerGroupList->Prev = &Next;
  Next = TimerGroupList;
  Prev = &TimerGroupList;
  TimerGroupList = this;
}

TimerGroup::TimerGroup(StringRef Name, StringRef Description)
    : TimerGroup(Name, Description, mtg::TimerLock()) {}

TimerGroup::TimerGroup(StringRef Name, StringRef Description,
                       const StringMap<TimeRecord> &Records)
    : TimerGroup(Name, Description) {
  TimersToPrint.reserve(Records.size());
  for (const auto &P : Records)
    TimersToPrint.emplace_back(P.getValue(), std::string(P.getKey()),
                               std::string(P.getKey()));
  assert(TimersToPrint.size() == Records.size() && "Size mismatch");
}

TimerGroup::~TimerGroup() {
  // If the timer group is destroyed before the timers it owns, accumulate and
  // print the timing data.
  while (FirstTimer)
    removeTimer(*FirstTimer);

  // Remove the group from the TimerGroupList.
  sys::SmartScopedLock<true> L(mtg::TimerLock());
  *Prev = Next;
  if (Next)
    Next->Prev = Prev;
}


void TimerGroup::removeTimer(Timer &T) {
  sys::SmartScopedLock<true> L(mtg::TimerLock());

  // If the timer was started, move its data to TimersToPrint.
  if (T.hasTriggered())
    TimersToPrint.emplace_back(T.Time, T.Name, T.Description);

  T.TG = nullptr;

  // Unlink the timer from our list.
  *T.Prev = T.Next;
  if (T.Next)
    T.Next->Prev = T.Prev;

  // Print the report when all timers in this group are destroyed if some of
  // them were started.
  if (FirstTimer || TimersToPrint.empty())
    return;

  std::unique_ptr<raw_ostream> OutStream = CreateInfoOutputFile();
  PrintQueuedTimers(*OutStream);
}

void TimerGroup::addTimer(Timer &T) {
  sys::SmartScopedLock<true> L(mtg::TimerLock());

  // Add the timer to our list.
  if (FirstTimer)
    FirstTimer->Prev = &T.Next;
  T.Next = FirstTimer;
  T.Prev = &FirstTimer;
  FirstTimer = &T;
}

void TimerGroup::PrintQueuedTimers(raw_ostream &OS) {
  // Perhaps sort the timers in descending order by amount of time taken.
  if (mtg::SortTimers())
    llvm::sort(TimersToPrint);

  TimeRecord Total;
  for (const PrintRecord &Record : TimersToPrint)
    Total += Record.Time;

  // Print out timing header.
  OS << "===" << std::string(73, '-') << "===\n";
  // Figure out how many spaces to indent TimerGroup name.
  unsigned Padding = (80-Description.length())/2;
  if (Padding > 80) Padding = 0;         // Don't allow "negative" numbers
  OS.indent(Padding) << Description << '\n';
  OS << "===" << std::string(73, '-') << "===\n";

  // If this is not an collection of ungrouped times, print the total time.
  // Ungrouped timers don't really make sense to add up.  We still print the
  // TOTAL line to make the percentages make sense.
  if (this != &mtg::DefaultTimerGroup())
    OS << format("  Total Execution Time: %5.4f seconds (%5.4f wall clock)\n",
                 Total.getProcessTime(), Total.getWallTime());
  OS << '\n';

  if (Total.getUserTime())
    OS << "   ---User Time---";
  if (Total.getSystemTime())
    OS << "   --System Time--";
  if (Total.getProcessTime())
    OS << "   --User+System--";
  OS << "   ---Wall Time---";
  if (Total.getMemUsed())
    OS << "  ---Mem---";
  if (Total.getInstructionsExecuted())
    OS << "  ---Instr---";
  OS << "  --- Name ---\n";

  // Loop through all of the timing data, printing it out.
  for (const PrintRecord &Record : llvm::reverse(TimersToPrint)) {
    Record.Time.print(Total, OS);
    OS << Record.Description << '\n';
  }

  Total.print(Total, OS);
  OS << "Total\n\n";
  OS.flush();

  TimersToPrint.clear();
}

void TimerGroup::prepareToPrintList(bool ResetTime) {
  // See if any of our timers were started, if so add them to TimersToPrint.
  for (Timer *T = FirstTimer; T; T = T->Next) {
    if (!T->hasTriggered()) continue;
    bool WasRunning = T->isRunning();
    if (WasRunning)
      T->stopTimer();

    TimersToPrint.emplace_back(T->Time, T->Name, T->Description);

    if (ResetTime)
      T->clear();

    if (WasRunning)
      T->startTimer();
  }
}

void TimerGroup::print(raw_ostream &OS, bool ResetAfterPrint) {
  {
    // After preparing the timers we can free the lock
    sys::SmartScopedLock<true> L(mtg::TimerLock());
    prepareToPrintList(ResetAfterPrint);
  }

  // If any timers were started, print the group.
  if (!TimersToPrint.empty())
    PrintQueuedTimers(OS);
}

void TimerGroup::clear() {
  sys::SmartScopedLock<true> L(mtg::TimerLock());
  for (Timer *T = FirstTimer; T; T = T->Next)
    T->clear();
}

void TimerGroup::printAll(raw_ostream &OS) {
  sys::SmartScopedLock<true> L(mtg::TimerLock());

  for (TimerGroup *TG = TimerGroupList; TG; TG = TG->Next)
    TG->print(OS);
}

void TimerGroup::clearAll() {
  sys::SmartScopedLock<true> L(mtg::TimerLock());
  for (TimerGroup *TG = TimerGroupList; TG; TG = TG->Next)
    TG->clear();
}

void TimerGroup::printJSONValue(raw_ostream &OS, const PrintRecord &R,
                                const char *suffix, double Value) {
  assert(yaml::needsQuotes(Name) == yaml::QuotingType::None &&
         "TimerGroup name should not need quotes");
  assert(yaml::needsQuotes(R.Name) == yaml::QuotingType::None &&
         "Timer name should not need quotes");
  constexpr auto max_digits10 = std::numeric_limits<double>::max_digits10;
  OS << "\t\"time." << Name << '.' << R.Name << suffix
     << "\": " << format("%.*e", max_digits10 - 1, Value);
}

const char *TimerGroup::printJSONValues(raw_ostream &OS, const char *delim) {
  sys::SmartScopedLock<true> L(mtg::TimerLock());

  prepareToPrintList(false);
  for (const PrintRecord &R : TimersToPrint) {
    OS << delim;
    delim = ",\n";

    const TimeRecord &T = R.Time;
    printJSONValue(OS, R, ".wall", T.getWallTime());
    OS << delim;
    printJSONValue(OS, R, ".user", T.getUserTime());
    OS << delim;
    printJSONValue(OS, R, ".sys", T.getSystemTime());
    if (T.getMemUsed()) {
      OS << delim;
      printJSONValue(OS, R, ".mem", T.getMemUsed());
    }
    if (T.getInstructionsExecuted()) {
      OS << delim;
      printJSONValue(OS, R, ".instr", T.getInstructionsExecuted());
    }
  }
  TimersToPrint.clear();
  return delim;
}

const char *TimerGroup::printAllJSONValues(raw_ostream &OS, const char *delim) {
  sys::SmartScopedLock<true> L(mtg::TimerLock());
  for (TimerGroup *TG = TimerGroupList; TG; TG = TG->Next)
    delim = TG->printJSONValues(OS, delim);
  return delim;
}

void TimerGroup::constructForStatistics() {
  mtg::LibSupportInfoOutputFilename();
  mtg::NamedGroupedTimers();
}

std::unique_ptr<TimerGroup> TimerGroup::aquireDefaultGroup() {
  return std::unique_ptr<TimerGroup>(mtg::claimDefaultTimerGroup());
}

//===----------------------------------------------------------------------===//
// Timer Globals
//
// Previously, these were independent ManagedStatics. This led to bugs because
// there are dependencies between the globals, but no reliable mechanism to
// control relative lifetimes.
//
// Placing the globals within one class instance lets us control the lifetimes
// of the various data members and ensure that no global uses another that has
// been deleted.
//
// Globals fall into two categories. First are simple data types and
// command-line options. These are cheap to construct and/or required early
// during launch. They are created when the ManagedTimerGlobals singleton is
// constructed. Second are types that are more expensive to construct or not
// needed until later during compilation. These are lazily constructed in order
// to reduce launch time.
//===----------------------------------------------------------------------===//
class llvm::TimerGlobals {
public:
  std::string LibSupportInfoOutputFilename;
  cl::opt<std::string, true> InfoOutputFilename;
  cl::opt<bool> TrackSpace;
  cl::opt<bool> SortTimers;

private:
  // Order of these members and initialization below is important. For example
  // the DefaultTimerGroup uses the TimerLock. Most of these also depend on the
  // options above.
  std::once_flag InitDeferredFlag;
  std::unique_ptr<SignpostEmitter> SignpostsPtr;
  std::unique_ptr<sys::SmartMutex<true>> TimerLockPtr;
  std::unique_ptr<TimerGroup> DefaultTimerGroupPtr;
  std::unique_ptr<Name2PairMap> NamedGroupedTimersPtr;
  TimerGlobals &initDeferred() {
    std::call_once(InitDeferredFlag, [this]() {
      SignpostsPtr = std::make_unique<SignpostEmitter>();
      TimerLockPtr = std::make_unique<sys::SmartMutex<true>>();
      DefaultTimerGroupPtr.reset(new TimerGroup(
          "misc", "Miscellaneous Ungrouped Timers", *TimerLockPtr));
      NamedGroupedTimersPtr = std::make_unique<Name2PairMap>();
    });
    return *this;
  }

public:
  SignpostEmitter &Signposts() { return *initDeferred().SignpostsPtr; }
  sys::SmartMutex<true> &TimerLock() { return *initDeferred().TimerLockPtr; }
  TimerGroup &DefaultTimerGroup() {
    return *initDeferred().DefaultTimerGroupPtr;
  }
  TimerGroup *claimDefaultTimerGroup() {
    return initDeferred().DefaultTimerGroupPtr.release();
  }
  Name2PairMap &NamedGroupedTimers() {
    return *initDeferred().NamedGroupedTimersPtr;
  }

public:
  TimerGlobals()
      : InfoOutputFilename(
            "info-output-file", cl::value_desc("filename"),
            cl::desc("File to append -stats and -timer output to"), cl::Hidden,
            cl::location(LibSupportInfoOutputFilename)),
        TrackSpace(
            "track-memory",
            cl::desc("Enable -time-passes memory tracking (this may be slow)"),
            cl::Hidden),
        SortTimers(
            "sort-timers",
            cl::desc(
                "In the report, sort the timers in each group in wall clock"
                " time order"),
            cl::init(true), cl::Hidden) {}
};

static ManagedStatic<TimerGlobals> ManagedTimerGlobals;

static std::string &mtg::LibSupportInfoOutputFilename() {
  return ManagedTimerGlobals->LibSupportInfoOutputFilename;
}
static const std::string &mtg::InfoOutputFilename() {
  return ManagedTimerGlobals->InfoOutputFilename.getValue();
}
static bool mtg::TrackSpace() { return ManagedTimerGlobals->TrackSpace; }
static bool mtg::SortTimers() { return ManagedTimerGlobals->SortTimers; }
static SignpostEmitter &mtg::Signposts() {
  return ManagedTimerGlobals->Signposts();
}
static sys::SmartMutex<true> &mtg::TimerLock() {
  return ManagedTimerGlobals->TimerLock();
}
static TimerGroup &mtg::DefaultTimerGroup() {
  return ManagedTimerGlobals->DefaultTimerGroup();
}
static TimerGroup *mtg::claimDefaultTimerGroup() {
  return ManagedTimerGlobals->claimDefaultTimerGroup();
}
static Name2PairMap &mtg::NamedGroupedTimers() {
  return ManagedTimerGlobals->NamedGroupedTimers();
}
