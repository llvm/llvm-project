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
/// This file defines the execution stage of a default instruction pipeline.
///
/// The ExecuteStage is responsible for managing the hardware scheduler
/// and issuing notifications that an instruction has been executed.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_EXECUTE_STAGE_H
#define LLVM_TOOLS_LLVM_MCA_EXECUTE_STAGE_H

#include "HardwareUnits/Scheduler.h"
#include "Instruction.h"
#include "Stages/Stage.h"
#include "llvm/ADT/ArrayRef.h"

namespace mca {

class ExecuteStage final : public Stage {
  Scheduler &HWS;

  llvm::Error issueInstruction(InstRef &IR);

  // Called at the beginning of each cycle to issue already dispatched
  // instructions to the underlying pipelines.
  llvm::Error issueReadyInstructions();

  ExecuteStage(const ExecuteStage &Other) = delete;
  ExecuteStage &operator=(const ExecuteStage &Other) = delete;

public:
  ExecuteStage(Scheduler &S) : Stage(), HWS(S) {}

  // This stage works under the assumption that the Pipeline will eventually
  // execute a retire stage. We don't need to check if pipelines and/or
  // schedulers have instructions to process, because those instructions are
  // also tracked by the retire control unit. That means,
  // RetireControlUnit::hasWorkToComplete() is responsible for checking if there
  // are still instructions in-flight in the out-of-order backend.
  bool hasWorkToComplete() const override { return false; }
  bool isAvailable(const InstRef &IR) const override;

  // Notifies the scheduler that a new cycle just started.
  //
  // This method notifies the scheduler that a new cycle started.
  // This method is also responsible for notifying listeners about instructions
  // state changes, and processor resources freed by the scheduler.
  // Instructions that transitioned to the 'Executed' state are automatically
  // moved to the next stage (i.e. RetireStage).
  llvm::Error cycleStart() override;
  llvm::Error execute(InstRef &IR) override;

  void
  notifyInstructionIssued(const InstRef &IR,
                          llvm::ArrayRef<std::pair<ResourceRef, double>> Used);
  void notifyInstructionExecuted(const InstRef &IR);
  void notifyInstructionReady(const InstRef &IR);
  void notifyResourceAvailable(const ResourceRef &RR);

  // Notify listeners that buffered resources have been consumed or freed.
  void notifyReservedOrReleasedBuffers(const InstRef &IR, bool Reserved);
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_EXECUTE_STAGE_H
