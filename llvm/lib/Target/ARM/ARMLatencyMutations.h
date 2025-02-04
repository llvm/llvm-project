//===- ARMLatencyMutations.h - ARM Latency Mutations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the ARM definition DAG scheduling mutations which
/// change inter-instruction latencies
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_LATENCYMUTATIONS_H
#define LLVM_LIB_TARGET_ARM_LATENCYMUTATIONS_H

#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/ScheduleDAGMutation.h"

namespace llvm {

class AAResults;
class ARMBaseInstrInfo;

/// Post-process the DAG to create cluster edges between instrs that may
/// be fused by the processor into a single operation.
class ARMOverrideBypasses : public ScheduleDAGMutation {
public:
  ARMOverrideBypasses(const ARMBaseInstrInfo *t, AAResults *a)
      : ScheduleDAGMutation(), TII(t), AA(a) {}

  void apply(ScheduleDAGInstrs *DAGInstrs) override;

private:
  virtual void modifyBypasses(SUnit &) = 0;

protected:
  const ARMBaseInstrInfo *TII;
  AAResults *AA;
  ScheduleDAGInstrs *DAG = nullptr;

  static void setBidirLatencies(SUnit &SrcSU, SDep &SrcDep, unsigned latency);
  static bool zeroOutputDependences(SUnit &ISU, SDep &Dep);
  unsigned makeBundleAssumptions(SUnit &ISU, SDep &Dep);
  bool memoryRAWHazard(SUnit &ISU, SDep &Dep, unsigned latency);
};

/// Note that you have to add:
///   DAG.addMutation(createARMLatencyMutation(ST, AA));
/// to ARMPassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation>
createARMLatencyMutations(const class ARMSubtarget &, AAResults *AA);

} // namespace llvm

#endif
