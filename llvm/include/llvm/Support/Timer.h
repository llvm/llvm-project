//===-- llvm/Support/Timer.h - Interval Timing Support ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_TIMER_H
#define LLVM_SUPPORT_TIMER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include <cassert>
#include <string>
#include <utility>
#include <vector>

namespace llvm {

class Timer;
class TimerGroup;
class raw_ostream;

class TimeRecord {
  double WallTime;       // Wall clock time elapsed in seconds
  double UserTime;       // User time elapsed
  double SystemTime;     // System time elapsed
  ssize_t MemUsed;       // Memory allocated (in bytes)
public:
  TimeRecord() : WallTime(0), UserTime(0), SystemTime(0), MemUsed(0) {}

  /// getCurrentTime - Get the current time and memory usage.  If Start is true
  /// we get the memory usage before the time, otherwise we get time before
  /// memory usage.  This matters if the time to get the memory usage is
  /// significant and shouldn't be counted as part of a duration.
  static TimeRecord getCurrentTime(bool Start = true);

  double getProcessTime() const { return UserTime + SystemTime; }
  double getUserTime() const { return UserTime; }
  double getSystemTime() const { return SystemTime; }
  double getWallTime() const { return WallTime; }
  ssize_t getMemUsed() const { return MemUsed; }

  // operator< - Allow sorting.
  bool operator<(const TimeRecord &T) const {
    // Sort by Wall Time elapsed, as it is the only thing really accurate
    return WallTime < T.WallTime;
  }

  void operator+=(const TimeRecord &RHS) {
    WallTime   += RHS.WallTime;
    UserTime   += RHS.UserTime;
    SystemTime += RHS.SystemTime;
    MemUsed    += RHS.MemUsed;
  }
  void operator-=(const TimeRecord &RHS) {
    WallTime   -= RHS.WallTime;
    UserTime   -= RHS.UserTime;
    SystemTime -= RHS.SystemTime;
    MemUsed    -= RHS.MemUsed;
  }

  /// Print the current time record to \p OS, with a breakdown showing
  /// contributions to the \p Total time record.
  void print(const TimeRecord &Total, raw_ostream &OS) const;
};

/// Timer - This class is used to track the amount of time spent between
/// invocations of its startTimer()/stopTimer() methods.  Given appropriate OS
/// support it can also keep track of the RSS of the program at various points.
/// By default, the Timer will print the amount of time it has captured to
/// standard error when the last timer is destroyed, otherwise it is printed
/// when its TimerGroup is destroyed.  Timers do not print their information
/// if they are never started.
///
class Timer {
  TimeRecord Time;       // The total time captured
  TimeRecord StartTime;  // The time startTimer() was last called
  std::string Name;      // The name of this time variable.
  bool Running;          // Is the timer currently running?
  bool Triggered;        // Has the timer ever been triggered?
  TimerGroup *TG;        // The TimerGroup this Timer is in.

  Timer **Prev, *Next;   // Doubly linked list of timers in the group.
public:
  explicit Timer(StringRef N) : TG(nullptr) { init(N); }
  Timer(StringRef N, TimerGroup &tg) : TG(nullptr) { init(N, tg); }
  Timer(const Timer &RHS) : TG(nullptr) {
    assert(!RHS.TG && "Can only copy uninitialized timers");
  }
  const Timer &operator=(const Timer &T) {
    assert(!TG && !T.TG && "Can only assign uninit timers");
    return *this;
  }
  ~Timer();

  // Create an uninitialized timer, client must use 'init'.
  explicit Timer() : TG(nullptr) {}
  void init(StringRef N);
  void init(StringRef N, TimerGroup &tg);

  const std::string &getName() const { return Name; }
  bool isInitialized() const { return TG != nullptr; }

  /// Check if the timer is currently running.
  bool isRunning() const { return Running; }

  /// Check if startTimer() has ever been called on this timer.
  bool hasTriggered() const { return Triggered; }

  /// Start the timer running.  Time between calls to startTimer/stopTimer is
  /// counted by the Timer class.  Note that these calls must be correctly
  /// paired.
  void startTimer();

  /// Stop the timer.
  void stopTimer();

  /// Clear the timer state.
  void clear();

  /// Return the duration for which this timer has been running.
  TimeRecord getTotalTime() const { return Time; }

private:
  friend class TimerGroup;
};

/// The TimeRegion class is used as a helper class to call the startTimer() and
/// stopTimer() methods of the Timer class.  When the object is constructed, it
/// starts the timer specified as its argument.  When it is destroyed, it stops
/// the relevant timer.  This makes it easy to time a region of code.
///
class TimeRegion {
  Timer *T;
  TimeRegion(const TimeRegion &) = delete;

public:
  explicit TimeRegion(Timer &t) : T(&t) {
    T->startTimer();
  }
  explicit TimeRegion(Timer *t) : T(t) {
    if (T) T->startTimer();
  }
  ~TimeRegion() {
    if (T) T->stopTimer();
  }
};

/// NamedRegionTimer - This class is basically a combination of TimeRegion and
/// Timer.  It allows you to declare a new timer, AND specify the region to
/// time, all in one statement.  All timers with the same name are merged.  This
/// is primarily used for debugging and for hunting performance problems.
///
struct NamedRegionTimer : public TimeRegion {
  explicit NamedRegionTimer(StringRef Name,
                            bool Enabled = true);
  explicit NamedRegionTimer(StringRef Name, StringRef GroupName,
                            bool Enabled = true);
};

/// The TimerGroup class is used to group together related timers into a single
/// report that is printed when the TimerGroup is destroyed.  It is illegal to
/// destroy a TimerGroup object before all of the Timers in it are gone.  A
/// TimerGroup can be specified for a newly created timer in its constructor.
///
class TimerGroup {
  std::string Name;
  Timer *FirstTimer;   // First timer in the group.
  std::vector<std::pair<TimeRecord, std::string>> TimersToPrint;

  TimerGroup **Prev, *Next; // Doubly linked list of TimerGroup's.
  TimerGroup(const TimerGroup &TG) = delete;
  void operator=(const TimerGroup &TG) = delete;

public:
  explicit TimerGroup(StringRef name);
  ~TimerGroup();

  void setName(StringRef name) { Name.assign(name.begin(), name.end()); }

  /// print - Print any started timers in this group and zero them.
  void print(raw_ostream &OS);

  /// printAll - This static method prints all timers and clears them all out.
  static void printAll(raw_ostream &OS);

private:
  friend class Timer;
  void addTimer(Timer &T);
  void removeTimer(Timer &T);
  void PrintQueuedTimers(raw_ostream &OS);
};

} // End llvm namespace

#endif
