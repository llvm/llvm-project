//===- llvm/System/Alarm.h - Alarm Generation support  ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides an operating system independent interface to alarm(2) 
// type functionality. The Alarm class allows a one-shot alarm to be set up 
// at some number of seconds in the future. When the alarm triggers, a method
// is called to process the event
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SYSTEM_ALARM_H
#define LLVM_SYSTEM_ALARM_H

namespace llvm {
namespace sys {

  /// This function registers an alarm to trigger some number of \p seconds in 
  /// the future. When that time arrives, the AlarmStatus function will begin
  /// to return 1 instead of 0. The user must poll the status of the alarm by
  /// making occasional calls to AlarmStatus. If the user sends an interrupt
  /// signal, AlarmStatus will begin returning -1, even if the alarm event
  /// occurred.
  /// @returns nothing
  void SetupAlarm(
    unsigned seconds ///< Number of seconds in future when alarm arrives
  );

  /// This function terminates the alarm previously set up 
  /// @returns nothing
  void TerminateAlarm();

  /// This function acquires the status of the alarm. 
  /// @returns -1=cancelled, 0=untriggered, 1=triggered
  int AlarmStatus();

} // End sys namespace
} // End llvm namespace

#endif
