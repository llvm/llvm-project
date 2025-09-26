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
// hints and target hooks. The AllocationOrder class encapsulates all of that.
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
  SmallVector<MCPhysReg, 16> Hints;
  bool HardHints =
      TRI->getRegAllocationHints(VirtReg, Order, Hints, MF, &VRM, Matrix);

  LLVM_DEBUG({
    if (!Hints.empty()) {
      dbgs() << "hints:";
      for (MCPhysReg Hint : Hints)
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

  // Create allocation order object
  AllocationOrder AO(std::move(Hints), Order, HardHints);

  // Apply anti-hints filtering if needed
  if (!AntiHintedPhysRegs.empty()) {
    AO.applyAntiHints(AntiHintedPhysRegs, TRI);

    LLVM_DEBUG({
      if (!AO.Hints.empty()) {
        dbgs() << "filtered hints:";
        for (MCPhysReg Hint : AO.Hints)
          dbgs() << ' ' << printReg(Hint, TRI);
        dbgs() << '\n';
      }
    });
  }

  assert(all_of(AO.Hints,
                [&](MCPhysReg Hint) { return is_contained(AO.Order, Hint); }) &&
         "Target hint is outside allocation order.");
  return AO;
}

void AllocationOrder::applyAntiHints(ArrayRef<MCPhysReg> AntiHintedPhysRegs,
                                     const TargetRegisterInfo *TRI) {
  // Helper to check if a register overlaps with any anti-hint
  auto isAntiHinted = [&](MCPhysReg Reg) {
    return std::any_of(
        AntiHintedPhysRegs.begin(), AntiHintedPhysRegs.end(),
        [&](MCPhysReg AntiHint) { return TRI->regsOverlap(Reg, AntiHint); });
  };

  // Create filtered order
  FilteredOrderStorage.clear();
  FilteredOrderStorage.assign(Order.begin(), Order.end());

  // Partition: non-anti-hinted registers go first
  auto PartitionPoint = std::stable_partition(
      FilteredOrderStorage.begin(), FilteredOrderStorage.end(),
      [&](MCPhysReg Reg) { return !isAntiHinted(Reg); });

  // Update Order
  Order = FilteredOrderStorage;

  LLVM_DEBUG({
    size_t NonAntiHintedCount =
        std::distance(FilteredOrderStorage.begin(), PartitionPoint);
    size_t AntiHintedCount =
        std::distance(PartitionPoint, FilteredOrderStorage.end());
    dbgs() << "    Added " << NonAntiHintedCount
           << " non-anti-hinted registers first\n"
           << "    Added " << AntiHintedCount
           << " anti-hinted registers at the end\n"
           << "  Anti-hint filtering complete\n";
  });
}
