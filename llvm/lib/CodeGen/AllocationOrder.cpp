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
  MRI.getPhysRegAntiHints(VirtReg, AntiHintedPhysRegs, &VRM);
  
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
  
  // Apply anti-hint filtering if needed
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
  // Create filtered order
  FilteredOrderStorage.clear();
  FilteredOrderStorage.reserve(Order.size());
  
  // Add non-anti-hinted registers first
  for (MCPhysReg PhysReg : Order) {
    if (!is_contained(AntiHintedPhysRegs, PhysReg)) {
      FilteredOrderStorage.push_back(PhysReg);
    }
  }
  
  // Add anti-hinted registers at the end as last resort
  for (MCPhysReg PhysReg : Order) {
    if (is_contained(AntiHintedPhysRegs, PhysReg)) {
      FilteredOrderStorage.push_back(PhysReg);
    }
  }
  
  // Update Order to point to our filtered storage
  Order = FilteredOrderStorage;
  
  LLVM_DEBUG({
    dbgs() << "moved " << AntiHintedPhysRegs.size() 
           << " anti-hinted registers to end of allocation order\n";
  });
}
