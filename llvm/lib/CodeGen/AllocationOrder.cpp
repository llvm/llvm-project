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
#include "llvm/ADT/BitVector.h"
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

  // HintsAndCustomOrder holds Hints first followed by the custom order if the
  // anti-hints reorders it.
  SmallVector<MCPhysReg, 16> HintsAndCustomOrder;

  // Get Hints.
  bool HardHints = TRI->getRegAllocationHints(
      VirtReg, Order, HintsAndCustomOrder, MF, &VRM, Matrix);
  const int NumHints = static_cast<int>(HintsAndCustomOrder.size());

  // HintsAndCustomOrder only holds Hints (custom order is not added yet).
  LLVM_DEBUG({
    if (NumHints) {
      dbgs() << "hints:";
      for (MCPhysReg Hint : HintsAndCustomOrder)
        dbgs() << ' ' << printReg(Hint, TRI);
      dbgs() << '\n';
    }
  });

  // Get anti-hints.
  BitVector AntiHintedRegUnits;
  MRI.getBitVecRegAntiHints(VirtReg, AntiHintedRegUnits, VRM);

  LLVM_DEBUG({
    if (AntiHintedRegUnits.any()) {
      dbgs() << "anti-hints:";
      for (Register AntiHintVReg : MRI.getRegAllocationAntiHints(VirtReg)) {
        if (!VRM.hasPhys(AntiHintVReg))
          continue;
        dbgs() << ' ' << printReg(VRM.getPhys(AntiHintVReg), TRI);
      }
      dbgs() << '\n';
    }
  });

  if (AntiHintedRegUnits.any()) {
    HintsAndCustomOrder.reserve(NumHints + Order.size());
    TRI->applyRegAllocationAntiHints(VirtReg, Order, HintsAndCustomOrder,
                                     NumHints, AntiHintedRegUnits, MF, Matrix);
  }

  // Create allocation order object.
  AllocationOrder AO(std::move(HintsAndCustomOrder), NumHints, Order,
                     HardHints);

  assert(all_of(AO.hints(),
                [&](MCPhysReg Hint) { return is_contained(AO.Order, Hint); }) &&
         "Target hint is outside allocation order.");
  return AO;
}
