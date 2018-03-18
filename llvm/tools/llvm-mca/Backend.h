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
#include "InstrBuilder.h"
#include "Scheduler.h"
#include "SourceMgr.h"

namespace mca {

class HWEventListener;
class HWInstructionEvent;

/// \brief An out of order backend for a specific subtarget.
///
/// It emulates an out-of-order execution of instructions. Instructions are
/// fetched from a MCInst sequence managed by an object of class SourceMgr.
/// Instructions are firstly dispatched to the schedulers and then executed.
/// This class tracks the lifetime of an instruction from the moment where
/// it gets dispatched to the schedulers, to the moment where it finishes
/// executing and register writes are architecturally committed.
/// In particular, it monitors changes in the state of every instruction
/// in flight.
/// Instructions are executed in a loop of iterations. The number of iterations
/// is defined by the SourceMgr object.
/// The Backend entrypoint is method 'Run()' which execute cycles in a loop
/// until there are new instructions to dispatch, and not every instruction
/// has been retired.
/// Internally, the Backend collects statistical information in the form of
/// histograms. For example, it tracks how the dispatch group size changes
/// over time.
class Backend {
  const llvm::MCSubtargetInfo &STI;

  std::unique_ptr<InstrBuilder> IB;
  std::unique_ptr<Scheduler> HWS;
  std::unique_ptr<DispatchUnit> DU;
  SourceMgr &SM;
  unsigned Cycles;

  llvm::DenseMap<unsigned, std::unique_ptr<Instruction>> Instructions;
  std::set<HWEventListener *> Listeners;

  void runCycle(unsigned Cycle);

public:
  Backend(const llvm::MCSubtargetInfo &Subtarget, const llvm::MCInstrInfo &MCII,
          const llvm::MCRegisterInfo &MRI, SourceMgr &Source,
          unsigned DispatchWidth = 0, unsigned RegisterFileSize = 0,
          unsigned MaxRetirePerCycle = 0, unsigned LoadQueueSize = 0,
          unsigned StoreQueueSize = 0, bool AssumeNoAlias = false)
      : STI(Subtarget),
        HWS(llvm::make_unique<Scheduler>(this, Subtarget.getSchedModel(),
                                         LoadQueueSize, StoreQueueSize,
                                         AssumeNoAlias)),
        DU(llvm::make_unique<DispatchUnit>(
            this, MRI, Subtarget.getSchedModel().MicroOpBufferSize,
            RegisterFileSize, MaxRetirePerCycle, DispatchWidth, HWS.get())),
        SM(Source), Cycles(0) {
    IB = llvm::make_unique<InstrBuilder>(MCII, HWS->getProcResourceMasks());
    HWS->setDispatchUnit(DU.get());
  }

  void run() {
    while (SM.hasNext() || !DU->isRCUEmpty())
      runCycle(Cycles++);
  }

  const Instruction &getInstruction(unsigned Index) const {
    const auto It = Instructions.find(Index);
    assert(It != Instructions.end() && "no running instructions with index");
    assert(It->second);
    return *It->second;
  }
  void eraseInstruction(unsigned Index) { Instructions.erase(Index); }
  unsigned getTotalRegisterMappingsCreated() const {
    return DU->getTotalRegisterMappingsCreated();
  }
  unsigned getMaxUsedRegisterMappings() const {
    return DU->getMaxUsedRegisterMappings();
  }
  void getBuffersUsage(std::vector<BufferUsageEntry> &Usage) const {
    return HWS->getBuffersUsage(Usage);
  }

  unsigned getNumRATStalls() const { return DU->getNumRATStalls(); }
  unsigned getNumRCUStalls() const { return DU->getNumRCUStalls(); }
  unsigned getNumSQStalls() const { return DU->getNumSQStalls(); }
  unsigned getNumLDQStalls() const { return DU->getNumLDQStalls(); }
  unsigned getNumSTQStalls() const { return DU->getNumSTQStalls(); }
  unsigned getNumDispatchGroupStalls() const {
    return DU->getNumDispatchGroupStalls();
  }

  void addEventListener(HWEventListener *Listener);
  void notifyCycleBegin(unsigned Cycle);
  void notifyInstructionEvent(const HWInstructionEvent &Event);
  void notifyResourceAvailable(const ResourceRef &RR);
  void notifyCycleEnd(unsigned Cycle);
};

} // namespace mca

#endif
