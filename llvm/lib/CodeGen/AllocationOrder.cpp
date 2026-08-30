//===-- llvm/CodeGen/AllocationOrder.cpp - Allocation Order ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements an allocation order for virtual registers.
//
// The preferred allocation order for a virtual register depends on allocation
// hints, anti-hints, and target hooks. The AllocationOrder class encapsulates
// all of that.
//
//===----------------------------------------------------------------------===//

#include "AllocationOrder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "regalloc"

// Compare VirtRegMap::getRegAllocPref().
AllocationOrder AllocationOrder::create(Register VirtReg, const VirtRegMap &VRM,
                                        const RegisterClassInfo &RegClassInfo,
                                        const LiveRegMatrix *Matrix) {
  const MachineFunction &MF = VRM.getMachineFunction();
  const TargetRegisterInfo *TRI = &VRM.getTargetRegInfo();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  auto Order = RegClassInfo.getOrder(MF.getRegInfo().getRegClass(VirtReg));

  // HintsAndCustomOrder holds Hints first followed by the shuffled order if the
  // anti-hints shuffle it.
  SmallVector<MCPhysReg, 16> HintsAndCustomOrder;

  // Get hints
  bool HardHints = TRI->getRegAllocationHints(
      VirtReg, Order, HintsAndCustomOrder, MF, &VRM, Matrix);
  const int NumHints = static_cast<int>(HintsAndCustomOrder.size());

  LLVM_DEBUG({
    if (NumHints) {
      dbgs() << "hints:";
      for (MCPhysReg Hint : HintsAndCustomOrder)
        dbgs() << ' ' << printReg(Hint, TRI);
      dbgs() << '\n';
    }
  });

  // Get anti-hints
  SmallVector<MCPhysReg, 16> AntiHintedPhysRegs;
  MRI.getPhysRegAntiHints(VirtReg, AntiHintedPhysRegs, VRM);

  LLVM_DEBUG({
    if (!AntiHintedPhysRegs.empty()) {
      dbgs() << "anti-hints:";
      for (MCPhysReg AntiHint : AntiHintedPhysRegs)
        dbgs() << ' ' << printReg(AntiHint, TRI);
      dbgs() << '\n';
    }
  });

  if (!AntiHintedPhysRegs.empty()) {
    HintsAndCustomOrder.reserve(NumHints + Order.size());
    TRI->applyRegAllocationAntiHints(VirtReg, Order, HintsAndCustomOrder,
                                     NumHints, AntiHintedPhysRegs, MF, &VRM,
                                     Matrix);
  }

  // Create allocation order object
  AllocationOrder AO(std::move(HintsAndCustomOrder), NumHints, Order,
                     HardHints);

  assert(all_of(AO.hints(),
                [&](MCPhysReg Hint) { return is_contained(AO.Order, Hint); }) &&
         "Target hint is outside allocation order.");
  return AO;
}
