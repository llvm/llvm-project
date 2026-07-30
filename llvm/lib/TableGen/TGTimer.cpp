//===- TGTimer.cpp - TableGen Timer implementation --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement the tablegen timer class.
//
//===----------------------------------------------------------------------===//

#include "llvm/TableGen/TGTimer.h"
using namespace llvm;

// These functions implement the phase timing facility. Starting a timer
// when one is already running stops the running one.
void TGTimer::startTimer(StringRef Name) {
  if (!TimingGroup)
    return;
  if (LastTimer && LastTimer->isRunning()) {
    LastTimer->stopTimer();
    if (BackendTimer) {
      LastTimer->clear();
      BackendTimer = false;
    }
  }

  LastTimer = std::make_unique<Timer>("", Name, *TimingGroup);
  LastTimer->startTimer();
}

void TGTimer::stopTimer() {
  if (!TimingGroup)
    return;

  assert(LastTimer && "No phase timer was started");
  LastTimer->stopTimer();
}

void TGTimer::startBackendTimer(StringRef Name) {
  if (!TimingGroup)
    return;

  startTimer(Name);
  BackendTimer = true;
}

void TGTimer::stopBackendTimer() {
  if (!TimingGroup || !BackendTimer)
    return;
  stopTimer();
  BackendTimer = false;
}
