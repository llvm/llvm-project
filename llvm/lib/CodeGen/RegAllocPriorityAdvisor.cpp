//===- RegAllocPriorityAdvisor.cpp - live ranges priority advisor ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the default priority advisor and of the Analysis pass.
//
//===----------------------------------------------------------------------===//

#include "RegAllocPriorityAdvisor.h"
#include "RegAllocGreedy.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

RegAllocPriorityAdvisor::RegAllocPriorityAdvisor(const MachineFunction &MF,
                                                 const RAGreedy &RA)
    : RA(RA), LIS(RA.getLiveIntervals()), VRM(RA.getVirtRegMap()),
      MRI(&VRM->getRegInfo()), TRI(MF.getSubtarget().getRegisterInfo()),
      RegClassInfo(RA.getRegClassInfo()), Indexes(RA.getIndexes()),
      RegClassPriorityTrumpsGlobalness(
          RA.getRegClassPriorityTrumpsGlobalness()),
      ReverseLocalAssignment(RA.getReverseLocalAssignment()) {}
