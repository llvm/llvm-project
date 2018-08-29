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
// TimingInfo Class - This class is used to calculate information about the
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
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace llvm;

namespace llvm {

//===----------------------------------------------------------------------===//
// TimingInfo implementation

bool TimePassesIsEnabled = false;
static cl::opt<bool, true> EnableTiming(
    "time-passes", cl::location(TimePassesIsEnabled), cl::Hidden,
    cl::desc("Time each pass, printing elapsed time for each on exit"));

namespace {
static ManagedStatic<sys::SmartMutex<true>> TimingInfoMutex;
}

template <typename PassT>
PassTimingInfo<PassT>::PassTimingInfo()
    : TG("pass", "... Pass execution timing report ...") {}

template <typename PassT> PassTimingInfo<PassT>::~PassTimingInfo() {
  // Deleting the timers accumulates their info into the TG member.
  // Then TG member is (implicitly) deleted, actually printing the report.
  for (auto &I : TimingData)
    delete I.getValue();
}

template <typename PassT> void PassTimingInfo<PassT>::init() {
  if (!TimePassesIsEnabled || TheTimeInfo)
    return;

  // Constructed the first time this is called, iff -time-passes is enabled.
  // This guarantees that the object will be constructed before static globals,
  // thus it will be destroyed before them.
  static ManagedStatic<PassTimingInfo> TTI;
  TheTimeInfo = &*TTI;
}

/// Prints out timing information and then resets the timers.
template <typename PassT> void PassTimingInfo<PassT>::print() {
  TG.print(*CreateInfoOutputFile());
}

/// Return the timer for the specified pass if it exists.
template <> Timer *PassTimingInfo<StringRef>::getPassTimer(StringRef PassName) {
  init();
  sys::SmartScopedLock<true> Lock(*TimingInfoMutex);
  Timer *&T = TimingData[PassName];
  if (!T)
    T = new Timer(PassName, PassName, TG);
  return T;
}

template <> Timer *PassTimingInfo<Pass *>::getPassTimer(Pass *P) {
  if (P->getAsPMDataManager())
    return nullptr;

  init();
  sys::SmartScopedLock<true> Lock(*TimingInfoMutex);
  StringRef PassName = P->getPassName();
  Timer *&T = TimingData[PassName];

  if (!T) {
    StringRef PassArgument;
    if (const PassInfo *PI = Pass::lookupPassInfo(P->getPassID()))
      PassArgument = PI->getPassArgument();
    T = new Timer(PassArgument.empty() ? PassName : PassArgument, PassName, TG);
  }
  return T;
}

template <typename PassInfoT>
PassTimingInfo<PassInfoT> *PassTimingInfo<PassInfoT>::TheTimeInfo;

template class PassTimingInfo<Pass *>;
template class PassTimingInfo<StringRef>;

Timer *getPassTimer(Pass *P) {
  PassTimingInfo<Pass *>::init();
  if (PassTimingInfo<Pass *>::TheTimeInfo)
    return PassTimingInfo<Pass *>::TheTimeInfo->getPassTimer(P);
  return nullptr;
}

Timer *getPassTimer(StringRef PassName) {
  PassTimingInfo<StringRef>::init();
  if (PassTimingInfo<StringRef>::TheTimeInfo)
    return PassTimingInfo<StringRef>::TheTimeInfo->getPassTimer(PassName);
  return nullptr;
}

/// If timing is enabled, report the times collected up to now and then reset
/// them.
void reportAndResetTimings() {
  if (PassTimingInfo<StringRef>::TheTimeInfo)
    PassTimingInfo<StringRef>::TheTimeInfo->print();
  if (PassTimingInfo<Pass *>::TheTimeInfo)
    PassTimingInfo<Pass *>::TheTimeInfo->print();
}

} // namespace llvm
