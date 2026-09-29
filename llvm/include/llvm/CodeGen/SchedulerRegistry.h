//===- llvm/CodeGen/SchedulerRegistry.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation for instruction scheduler function
// pass registry (RegisterScheduler).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULERREGISTRY_H
#define LLVM_CODEGEN_SCHEDULERREGISTRY_H

#include "llvm/CodeGen/MachinePassRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

//===----------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===----------------------------------------------------------------------===//

class ScheduleDAGSDNodes;
class SelectionDAGISel;

class RegisterScheduler
    : public MachinePassRegistryNode<ScheduleDAGSDNodes *(*)(SelectionDAGISel *,
                                                             CodeGenOptLevel)> {
public:
  using FunctionPassCtor = ScheduleDAGSDNodes *(*)(SelectionDAGISel *,
                                                   CodeGenOptLevel);

  LLVM_ABI static MachinePassRegistry<FunctionPassCtor> Registry;

  RegisterScheduler(const char *N, const char *D, FunctionPassCtor C)
      : MachinePassRegistryNode(N, D, C) {
    Registry.Add(this);
  }
  ~RegisterScheduler() { Registry.Remove(this); }


  // Accessors.
  RegisterScheduler *getNext() const {
    return (RegisterScheduler *)MachinePassRegistryNode::getNext();
  }

  static RegisterScheduler *getList() {
    return (RegisterScheduler *)Registry.getList();
  }

  static void setListener(MachinePassRegistryListener<FunctionPassCtor> *L) {
    Registry.setListener(L);
  }
};

/// createBURRListDAGScheduler - This creates a bottom up register usage
/// reduction list scheduler.
LLVM_ABI ScheduleDAGSDNodes *
createBURRListDAGScheduler(SelectionDAGISel *IS, CodeGenOptLevel OptLevel);

/// createSourceListDAGScheduler - This creates a bottom up list scheduler that
/// schedules nodes in source code order when possible.
LLVM_ABI ScheduleDAGSDNodes *
createSourceListDAGScheduler(SelectionDAGISel *IS, CodeGenOptLevel OptLevel);

/// createHybridListDAGScheduler - This creates a bottom up register pressure
/// aware list scheduler that make use of latency information to avoid stalls
/// for long latency instructions in low register pressure mode. In high
/// register pressure mode it schedules to reduce register pressure.
LLVM_ABI ScheduleDAGSDNodes *createHybridListDAGScheduler(SelectionDAGISel *IS,
                                                          CodeGenOptLevel);

/// createILPListDAGScheduler - This creates a bottom up register pressure
/// aware list scheduler that tries to increase instruction level parallelism
/// in low register pressure mode. In high register pressure mode it schedules
/// to reduce register pressure.
LLVM_ABI ScheduleDAGSDNodes *createILPListDAGScheduler(SelectionDAGISel *IS,
                                                       CodeGenOptLevel);

/// createFastDAGScheduler - This creates a "fast" scheduler.
///
LLVM_ABI ScheduleDAGSDNodes *createFastDAGScheduler(SelectionDAGISel *IS,
                                                    CodeGenOptLevel OptLevel);

/// createVLIWDAGScheduler - Scheduler for VLIW targets. This creates top down
/// DFA driven list scheduler with clustering heuristic to control
/// register pressure.
LLVM_ABI ScheduleDAGSDNodes *createVLIWDAGScheduler(SelectionDAGISel *IS,
                                                    CodeGenOptLevel OptLevel);
/// createDefaultScheduler - This creates an instruction scheduler appropriate
/// for the target.
LLVM_ABI ScheduleDAGSDNodes *createDefaultScheduler(SelectionDAGISel *IS,
                                                    CodeGenOptLevel OptLevel);

/// createDAGLinearizer - This creates a "no-scheduling" scheduler which
/// linearize the DAG using topological order.
LLVM_ABI ScheduleDAGSDNodes *createDAGLinearizer(SelectionDAGISel *IS,
                                                 CodeGenOptLevel OptLevel);

} // end namespace llvm

#endif // LLVM_CODEGEN_SCHEDULERREGISTRY_H
