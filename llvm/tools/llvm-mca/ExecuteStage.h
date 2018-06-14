//===---------------------- ExecuteStage.h ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the execution stage of an instruction pipeline.
///
/// The ExecuteStage is responsible for managing the hardware scheduler
/// and issuing notifications that an instruction has been executed.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_EXECUTE_STAGE_H
#define LLVM_TOOLS_LLVM_MCA_EXECUTE_STAGE_H

#include "Instruction.h"
#include "RetireControlUnit.h"
#include "Scheduler.h"
#include "Stage.h"
#include "llvm/ADT/ArrayRef.h"

namespace mca {

class Backend;

class ExecuteStage : public Stage {
  // Owner will go away when we move listeners/eventing to the stages.
  Backend *Owner;
  RetireControlUnit &RCU;
  Scheduler &HWS;

  // The following routines are used to maintain the HWS.
  void reclaimSchedulerResources();
  void updateSchedulerQueues();
  void issueReadyInstructions();

public:
  ExecuteStage(Backend *B, RetireControlUnit &R, Scheduler &S)
      : Stage(), Owner(B), RCU(R), HWS(S) {}
  ExecuteStage(const ExecuteStage &Other) = delete;
  ExecuteStage &operator=(const ExecuteStage &Other) = delete;

  virtual void preExecute(const InstRef &IR) override final;
  virtual bool execute(InstRef &IR) override final;

  void
  notifyInstructionIssued(const InstRef &IR,
                          llvm::ArrayRef<std::pair<ResourceRef, double>> Used);
  void notifyInstructionExecuted(const InstRef &IR);
  void notifyInstructionReady(const InstRef &IR);
  void notifyResourceAvailable(const ResourceRef &RR);

  // Notify listeners that buffered resources were consumed.
  void notifyReservedBuffers(llvm::ArrayRef<uint64_t> Buffers);

  // Notify listeners that buffered resources were freed.
  void notifyReleasedBuffers(llvm::ArrayRef<uint64_t> Buffers);
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_EXECUTE_STAGE_H
