//===- RegAllocPriorityAdvisor.h - live ranges priority advisor -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCPRIORITYADVISOR_H
#define LLVM_CODEGEN_REGALLOCPRIORITYADVISOR_H

#include "RegAllocEvictionAdvisor.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Pass.h"

namespace llvm {

class MachineFunction;
class VirtRegMap;
class RAGreedy;

/// Interface to the priority advisor, which is responsible for prioritizing
/// live ranges.
class RegAllocPriorityAdvisor {
public:
  RegAllocPriorityAdvisor(const RegAllocPriorityAdvisor &) = delete;
  RegAllocPriorityAdvisor(RegAllocPriorityAdvisor &&) = delete;
  virtual ~RegAllocPriorityAdvisor() = default;

  /// Find the priority value for a live range. A float value is used since ML
  /// prefers it.
  virtual unsigned getPriority(const LiveInterval &LI) const = 0;

  RegAllocPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA);

protected:
  const RAGreedy &RA;
  LiveIntervals *const LIS;
  VirtRegMap *const VRM;
  MachineRegisterInfo *const MRI;
  const TargetRegisterInfo *const TRI;
  const RegisterClassInfo &RegClassInfo;
  SlotIndexes *const Indexes;
  const bool RegClassPriorityTrumpsGlobalness;
  const bool ReverseLocalAssignment;
};

class DefaultPriorityAdvisor : public RegAllocPriorityAdvisor {
public:
  DefaultPriorityAdvisor(const MachineFunction &MF, const RAGreedy &RA)
      : RegAllocPriorityAdvisor(MF, RA) {}

private:
  unsigned getPriority(const LiveInterval &LI) const override;
};
} // namespace llvm

#endif // LLVM_CODEGEN_REGALLOCPRIORITYADVISOR_H
