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
  auto Order = RegClassInfo.getOrder(MF.getRegInfo().getRegClass(VirtReg));
  SmallVector<MCPhysReg, 16> Hints;
  bool HardHints =
      TRI->getRegAllocationHints(VirtReg, Order, Hints, MF, &VRM, Matrix);

  // TODO: Consider changing the interface of
  // TargetRegisterInfo::getRegAllocationHints to take a SetVector to enforce
  // uniqueness at the API boundary. Ensure hints are unique while preserving
  // priority order.
  SmallDenseSet<MCPhysReg, 16> Seen;
  llvm::erase_if(Hints,
                 [&](MCPhysReg Reg) { return !Seen.insert(Reg).second; });

  LLVM_DEBUG({
    if (!Hints.empty()) {
      dbgs() << "hints:";
      for (MCPhysReg Hint : Hints)
        dbgs() << ' ' << printReg(Hint, TRI);
      dbgs() << '\n';
    }
  });
  assert(all_of(Hints,
                [&](MCPhysReg Hint) { return is_contained(Order, Hint); }) &&
         "Target hint is outside allocation order.");
  assert(
      all_of(Hints, [&](MCPhysReg Hint) { return count(Hints, Hint) == 1; }) &&
      "Target hints contain duplicates.");
  return AllocationOrder(std::move(Hints), Order, HardHints);
}
