//===- PPCMachineScheduler.cpp - MI Scheduler for PowerPC -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "PPCMachineScheduler.h"
using namespace llvm;

void PPCPostRASchedStrategy::enterMBB(MachineBasicBlock *MBB) {
  // Custom PPC PostRA specific behavior here.
  PostGenericScheduler::enterMBB(MBB);
}

void PPCPostRASchedStrategy::leaveMBB() {
  // Custom PPC PostRA specific behavior here.
  PostGenericScheduler::leaveMBB();
}

void PPCPostRASchedStrategy::initialize(ScheduleDAGMI *Dag) {
  // Custom PPC PostRA specific initialization here.
  PostGenericScheduler::initialize(Dag);
}

SUnit *PPCPostRASchedStrategy::pickNode(bool &IsTopNode) {
  // Custom PPC PostRA specific scheduling here.
  return PostGenericScheduler::pickNode(IsTopNode);
}

