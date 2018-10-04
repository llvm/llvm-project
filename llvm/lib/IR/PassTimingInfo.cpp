//===- PassTimingInfo.cpp - LLVM Pass Timing Implementation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass Timing infrastructure for both
// new and legacy pass managers.
//
// PassTimingInfo Class - This class is used to calculate information about the
// amount of time each pass takes to execute.  This only happens when
// -time-passes is enabled on the command line.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/PassTimingInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <string>

using namespace llvm;

namespace llvm {

bool TimePassesIsEnabled = false;

static cl::opt<bool, true> EnableTiming(
    "time-passes", cl::location(TimePassesIsEnabled), cl::Hidden,
    cl::desc("Time each pass, printing elapsed time for each on exit"));

namespace {
namespace legacy {

//===----------------------------------------------------------------------===//
// TimingInfo implementation

/// Provides an interface for collecting pass timing information.
///
/// It was intended to be generic but now we decided to split
/// interfaces completely. This is now exclusively for legacy-pass-manager use.
class PassTimingInfo {
public:
  using PassInstanceID = void *;

private:
  StringMap<unsigned> PassIDCountMap; ///< Map that counts instances of passes
  DenseMap<PassInstanceID, std::unique_ptr<Timer>> TimingData; ///< timers for pass instances
  TimerGroup TG;

public:
  /// Default constructor for yet-inactive timeinfo.
  /// Use \p init() to activate it.
  PassTimingInfo();

  /// Print out timing information and release timers.
  ~PassTimingInfo();

  /// Initializes the static \p TheTimeInfo member to a non-null value when
  /// -time-passes is enabled. Leaves it null otherwise.
  ///
  /// This method may be called multiple times.
  static void init();

  /// Prints out timing information and then resets the timers.
  void print();

  /// Returns the timer for the specified pass if it exists.
  Timer *getPassTimer(Pass *, PassInstanceID);

  static PassTimingInfo *TheTimeInfo;

private:
  Timer *newPassTimer(StringRef PassID, StringRef PassDesc);
};

static ManagedStatic<sys::SmartMutex<true>> TimingInfoMutex;

PassTimingInfo::PassTimingInfo()
    : TG("pass", "... Pass execution timing report ...") {}

PassTimingInfo::~PassTimingInfo() {
  // Deleting the timers accumulates their info into the TG member.
  // Then TG member is (implicitly) deleted, actually printing the report.
  TimingData.clear();
}

void PassTimingInfo::init() {
  if (!TimePassesIsEnabled || TheTimeInfo)
    return;

  // Constructed the first time this is called, iff -time-passes is enabled.
  // This guarantees that the object will be constructed after static globals,
  // thus it will be destroyed before them.
  static ManagedStatic<PassTimingInfo> TTI;
  TheTimeInfo = &*TTI;
}

/// Prints out timing information and then resets the timers.
void PassTimingInfo::print() { TG.print(*CreateInfoOutputFile()); }

Timer *PassTimingInfo::newPassTimer(StringRef PassID, StringRef PassDesc) {
  unsigned &num = PassIDCountMap[PassID];
  num++;
  // Appending description with a pass-instance number for all but the first one
  std::string PassDescNumbered =
      num <= 1 ? PassDesc.str() : formatv("{0} #{1}", PassDesc, num).str();
  return new Timer(PassID, PassDescNumbered, TG);
}

Timer *PassTimingInfo::getPassTimer(Pass *P, PassInstanceID Pass) {
  if (P->getAsPMDataManager())
    return nullptr;

  init();
  sys::SmartScopedLock<true> Lock(*TimingInfoMutex);
  std::unique_ptr<Timer> &T = TimingData[Pass];

  if (!T) {
    StringRef PassName = P->getPassName();
    StringRef PassArgument;
    if (const PassInfo *PI = Pass::lookupPassInfo(P->getPassID()))
      PassArgument = PI->getPassArgument();
    T.reset(newPassTimer(PassArgument.empty() ? PassName : PassArgument, PassName));
  }
  return T.get();
}

PassTimingInfo *PassTimingInfo::TheTimeInfo;
} // namespace legacy
} // namespace

Timer *getPassTimer(Pass *P) {
  legacy::PassTimingInfo::init();
  if (legacy::PassTimingInfo::TheTimeInfo)
    return legacy::PassTimingInfo::TheTimeInfo->getPassTimer(P, P);
  return nullptr;
}

/// If timing is enabled, report the times collected up to now and then reset
/// them.
void reportAndResetTimings() {
  if (legacy::PassTimingInfo::TheTimeInfo)
    legacy::PassTimingInfo::TheTimeInfo->print();
}

} // namespace llvm
