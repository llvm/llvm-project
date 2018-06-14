//===---------------------- ExecuteStage.cpp --------------------*- C++ -*-===//
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

#include "ExecuteStage.h"
#include "Backend.h"
#include "Scheduler.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm-mca"

namespace mca {

using namespace llvm;

// Reclaim the simulated resources used by the scheduler.
void ExecuteStage::reclaimSchedulerResources() {
  SmallVector<ResourceRef, 8> ResourcesFreed;
  HWS.reclaimSimulatedResources(ResourcesFreed);
  for (const ResourceRef &RR : ResourcesFreed)
    notifyResourceAvailable(RR);
}

// Update the scheduler's instruction queues.
void ExecuteStage::updateSchedulerQueues() {
  SmallVector<InstRef, 4> InstructionIDs;
  HWS.updateIssuedQueue(InstructionIDs);
  for (const InstRef &IR : InstructionIDs)
    notifyInstructionExecuted(IR);
  InstructionIDs.clear();

  HWS.updatePendingQueue(InstructionIDs);
  for (const InstRef &IR : InstructionIDs)
    notifyInstructionReady(IR);
}

// Issue instructions that are waiting in the scheduler's ready queue.
void ExecuteStage::issueReadyInstructions() {
  SmallVector<InstRef, 4> InstructionIDs;
  InstRef IR = HWS.select();
  while (IR.isValid()) {
    SmallVector<std::pair<ResourceRef, double>, 4> Used;
    HWS.issueInstruction(IR, Used);

    // Reclaim instruction resources and perform notifications.
    const InstrDesc &Desc = IR.getInstruction()->getDesc();
    notifyReleasedBuffers(Desc.Buffers);
    notifyInstructionIssued(IR, Used);
    if (IR.getInstruction()->isExecuted())
      notifyInstructionExecuted(IR);

    // Instructions that have been issued during this cycle might have unblocked
    // other dependent instructions. Dependent instructions may be issued during
    // this same cycle if operands have ReadAdvance entries.  Promote those
    // instructions to the ReadyQueue and tell to the caller that we need
    // another round of 'issue()'.
    HWS.promoteToReadyQueue(InstructionIDs);
    for (const InstRef &I : InstructionIDs)
      notifyInstructionReady(I);
    InstructionIDs.clear();

    // Select the next instruction to issue.
    IR = HWS.select();
  }
}

// The following routine is the maintenance routine of the ExecuteStage.
// It is responsible for updating the hardware scheduler (HWS), including
// reclaiming the HWS's simulated hardware resources, as well as updating the
// HWS's queues.
//
// This routine also processes the instructions that are ready for issuance.
// These instructions are managed by the HWS's ready queue and can be accessed
// via the Scheduler::select() routine.
//
// Notifications are issued to this stage's listeners when instructions are
// moved between the HWS's queues.  In particular, when an instruction becomes
// ready or executed.
void ExecuteStage::preExecute(const InstRef &Unused) {
  reclaimSchedulerResources();
  updateSchedulerQueues();
  issueReadyInstructions();
}

// Schedule the instruction for execution on the hardware.
bool ExecuteStage::execute(InstRef &IR) {
#ifndef NDEBUG
  // Ensure that the HWS has not stored this instruction in its queues.
  HWS.sanityCheck(IR);
#endif
  // Reserve a slot in each buffered resource. Also, mark units with
  // BufferSize=0 as reserved. Resources with a buffer size of zero will only
  // be released after MCIS is issued, and all the ResourceCycles for those
  // units have been consumed.
  const InstrDesc &Desc = IR.getInstruction()->getDesc();
  HWS.reserveBuffers(Desc.Buffers);
  notifyReservedBuffers(Desc.Buffers);

  // Obtain a slot in the LSU.
  if (!HWS.reserveResources(IR))
    return false;

  // If we did not return early, then the scheduler is ready for execution.
  notifyInstructionReady(IR);

  // Don't add a zero-latency instruction to the Wait or Ready queue.
  // A zero-latency instruction doesn't consume any scheduler resources. That is
  // because it doesn't need to be executed, and it is often removed at register
  // renaming stage. For example, register-register moves are often optimized at
  // register renaming stage by simply updating register aliases. On some
  // targets, zero-idiom instructions (for example: a xor that clears the value
  // of a register) are treated specially, and are often eliminated at register
  // renaming stage.
  //
  // Instructions that use an in-order dispatch/issue processor resource must be
  // issued immediately to the pipeline(s). Any other in-order buffered
  // resources (i.e. BufferSize=1) is consumed.
  //
  // If we cannot issue immediately, the HWS will add IR to its ready queue for
  // execution later, so we must return early here.
  if (!HWS.issueImmediately(IR))
    return true;

  LLVM_DEBUG(dbgs() << "[SCHEDULER] Instruction " << IR
                    << " issued immediately\n");

  // Issue IR.  The resources for this issuance will be placed in 'Used.'
  SmallVector<std::pair<ResourceRef, double>, 4> Used;
  HWS.issueInstruction(IR, Used);

  // Perform notifications.
  notifyReleasedBuffers(Desc.Buffers);
  notifyInstructionIssued(IR, Used);
  if (IR.getInstruction()->isExecuted())
    notifyInstructionExecuted(IR);

  return true;
}

void ExecuteStage::notifyInstructionExecuted(const InstRef &IR) {
  HWS.onInstructionExecuted(IR);
  LLVM_DEBUG(dbgs() << "[E] Instruction Executed: " << IR << '\n');
  Owner->notifyInstructionEvent(
      HWInstructionEvent(HWInstructionEvent::Executed, IR));
  RCU.onInstructionExecuted(IR.getInstruction()->getRCUTokenID());
}

void ExecuteStage::notifyInstructionReady(const InstRef &IR) {
  LLVM_DEBUG(dbgs() << "[E] Instruction Ready: " << IR << '\n');
  Owner->notifyInstructionEvent(
      HWInstructionEvent(HWInstructionEvent::Ready, IR));
}

void ExecuteStage::notifyResourceAvailable(const ResourceRef &RR) {
  Owner->notifyResourceAvailable(RR);
}

void ExecuteStage::notifyInstructionIssued(
    const InstRef &IR, ArrayRef<std::pair<ResourceRef, double>> Used) {
  LLVM_DEBUG({
    dbgs() << "[E] Instruction Issued: " << IR << '\n';
    for (const std::pair<ResourceRef, unsigned> &Resource : Used) {
      dbgs() << "[E] Resource Used: [" << Resource.first.first << '.'
             << Resource.first.second << "]\n";
      dbgs() << "           cycles: " << Resource.second << '\n';
    }
  });
  Owner->notifyInstructionEvent(HWInstructionIssuedEvent(IR, Used));
}

void ExecuteStage::notifyReservedBuffers(ArrayRef<uint64_t> Buffers) {
  if (Buffers.empty())
    return;

  SmallVector<unsigned, 4> BufferIDs(Buffers.begin(), Buffers.end());
  std::transform(Buffers.begin(), Buffers.end(), BufferIDs.begin(),
                 [&](uint64_t Op) { return HWS.getResourceID(Op); });
  Owner->notifyReservedBuffers(BufferIDs);
}

void ExecuteStage::notifyReleasedBuffers(ArrayRef<uint64_t> Buffers) {
  if (Buffers.empty())
    return;

  SmallVector<unsigned, 4> BufferIDs(Buffers.begin(), Buffers.end());
  std::transform(Buffers.begin(), Buffers.end(), BufferIDs.begin(),
                 [&](uint64_t Op) { return HWS.getResourceID(Op); });
  Owner->notifyReleasedBuffers(BufferIDs);
}

} // namespace mca
