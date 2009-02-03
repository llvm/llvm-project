//===-- Alarm.inc - Implement Unix Alarm Support --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the UNIX Alarm support.
//
//===----------------------------------------------------------------------===//

#include <signal.h>
#include <unistd.h>
#include <cassert>
using namespace llvm;

/// AlarmCancelled - This flag is set by the SIGINT signal handler if the
/// user presses CTRL-C.
static volatile bool AlarmCancelled = false;

/// AlarmTriggered - This flag is set by the SIGALRM signal handler if the 
/// alarm was triggered.
static volatile bool AlarmTriggered = false;

/// NestedSOI - Sanity check.  Alarms cannot be nested or run in parallel.  
/// This ensures that they never do.
static bool NestedSOI = false;

static RETSIGTYPE SigIntHandler(int Sig) {
  AlarmCancelled = true;
  signal(SIGINT, SigIntHandler);
}

static RETSIGTYPE SigAlarmHandler(int Sig) {
  AlarmTriggered = true;
}

static void (*OldSigIntHandler) (int);

void sys::SetupAlarm(unsigned seconds) {
  assert(!NestedSOI && "sys::SetupAlarm calls cannot be nested!");
  NestedSOI = true;
  AlarmCancelled = false;
  AlarmTriggered = false;
  ::signal(SIGALRM, SigAlarmHandler);
  OldSigIntHandler = ::signal(SIGINT, SigIntHandler);
  ::alarm(seconds);
}

void sys::TerminateAlarm() {
  assert(NestedSOI && "sys::TerminateAlarm called without sys::SetupAlarm!");
  ::alarm(0);
  ::signal(SIGALRM, SIG_DFL);
  ::signal(SIGINT, OldSigIntHandler);
  AlarmCancelled = false;
  AlarmTriggered = false;
  NestedSOI = false;
}

int sys::AlarmStatus() {
  if (AlarmCancelled)
    return -1;
  if (AlarmTriggered)
    return 1;
  return 0;
}
