//===--------------------- Backend.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements an OoO backend for the llvm-mca tool.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_BACKEND_H
#define LLVM_TOOLS_LLVM_MCA_BACKEND_H

#include "Dispatch.h"
#include "FetchStage.h"
#include "InstrBuilder.h"
#include "Scheduler.h"

namespace mca {

class HWEventListener;
class HWInstructionEvent;
class HWStallEvent;

/// An out of order backend for a specific subtarget.
///
/// It emulates an out-of-order execution of instructions. Instructions are
/// fetched from a MCInst sequence managed by an initial 'Fetch' stage.
/// Instructions are firstly fetched, then dispatched to the schedulers, and
/// then executed.
///
/// This class tracks the lifetime of an instruction from the moment where
/// it gets dispatched to the schedulers, to the moment where it finishes
/// executing and register writes are architecturally committed.
/// In particular, it monitors changes in the state of every instruction
/// in flight.
///
/// Instructions are executed in a loop of iterations. The number of iterations
/// is defined by the SourceMgr object, which is managed by the initial stage
/// of the instruction pipeline.
///
/// The Backend entry point is method 'run()' which executes cycles in a loop
/// until there are new instructions to dispatch, and not every instruction
/// has been retired.
///
/// Internally, the Backend collects statistical information in the form of
/// histograms. For example, it tracks how the dispatch group size changes
/// over time.
class Backend {
  const llvm::MCSubtargetInfo &STI;

  /// This is the initial stage of the pipeline.
  /// TODO: Eventually this will become a list of unique Stage* that this
  /// backend pipeline executes.
  std::unique_ptr<FetchStage> Fetch;

  std::unique_ptr<Scheduler> HWS;
  std::unique_ptr<DispatchUnit> DU;
  std::set<HWEventListener *> Listeners;
  unsigned Cycles;

  void runCycle(unsigned Cycle);

public:
  Backend(const llvm::MCSubtargetInfo &Subtarget,
          const llvm::MCRegisterInfo &MRI,
          std::unique_ptr<FetchStage> InitialStage, unsigned DispatchWidth = 0,
          unsigned RegisterFileSize = 0, unsigned LoadQueueSize = 0,
          unsigned StoreQueueSize = 0, bool AssumeNoAlias = false)
      : STI(Subtarget), Fetch(std::move(InitialStage)),
        HWS(llvm::make_unique<Scheduler>(this, Subtarget.getSchedModel(),
                                         LoadQueueSize, StoreQueueSize,
                                         AssumeNoAlias)),
        DU(llvm::make_unique<DispatchUnit>(this, Subtarget.getSchedModel(), MRI,
                                           RegisterFileSize, DispatchWidth,
                                           HWS.get())),
        Cycles(0) {
    HWS->setDispatchUnit(DU.get());
  }

  void run();
  void addEventListener(HWEventListener *Listener);
  void notifyCycleBegin(unsigned Cycle);
  void notifyInstructionEvent(const HWInstructionEvent &Event);
  void notifyStallEvent(const HWStallEvent &Event);
  void notifyResourceAvailable(const ResourceRef &RR);
  void notifyReservedBuffers(llvm::ArrayRef<unsigned> Buffers);
  void notifyReleasedBuffers(llvm::ArrayRef<unsigned> Buffers);
  void notifyCycleEnd(unsigned Cycle);
};
} // namespace mca

#endif
