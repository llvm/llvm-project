//==- llvm/CodeGen/ScheduleDAGInstrs.h - MachineInstr Scheduling -*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ScheduleDAGInstrs class, which implements
// scheduling for a MachineInstr-based dependency graph.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDAGINSTRS_H
#define LLVM_CODEGEN_SCHEDULEDAGINSTRS_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  class MachineLoopInfo;
  class MachineDominatorTree;

  class ScheduleDAGInstrs : public ScheduleDAG {
    const MachineLoopInfo &MLI;
    const MachineDominatorTree &MDT;

    /// Defs, Uses - Remember where defs and uses of each physical register
    /// are as we iterate upward through the instructions. This is allocated
    /// here instead of inside BuildSchedGraph to avoid the need for it to be
    /// initialized and destructed for each block.
    std::vector<SUnit *> Defs[TargetRegisterInfo::FirstVirtualRegister];
    std::vector<SUnit *> Uses[TargetRegisterInfo::FirstVirtualRegister];

    /// PendingLoads - Remember where unknown loads are after the most recent
    /// unknown store, as we iterate. As with Defs and Uses, this is here
    /// to minimize construction/destruction.
    std::vector<SUnit *> PendingLoads;

  public:
    explicit ScheduleDAGInstrs(MachineFunction &mf,
                               const MachineLoopInfo &mli,
                               const MachineDominatorTree &mdt);

    virtual ~ScheduleDAGInstrs() {}

    /// NewSUnit - Creates a new SUnit and return a ptr to it.
    ///
    SUnit *NewSUnit(MachineInstr *MI) {
#ifndef NDEBUG
      const SUnit *Addr = SUnits.empty() ? 0 : &SUnits[0];
#endif
      SUnits.push_back(SUnit(MI, (unsigned)SUnits.size()));
      assert((Addr == 0 || Addr == &SUnits[0]) &&
             "SUnits std::vector reallocated on the fly!");
      SUnits.back().OrigNode = &SUnits.back();
      return &SUnits.back();
    }

    /// BuildSchedGraph - Build SUnits from the MachineBasicBlock that we are
    /// input.
    virtual void BuildSchedGraph();

    /// ComputeLatency - Compute node latency.
    ///
    virtual void ComputeLatency(SUnit *SU);

    virtual MachineBasicBlock *EmitSchedule();

    /// Schedule - Order nodes according to selected style, filling
    /// in the Sequence member.
    ///
    virtual void Schedule() = 0;

    virtual void dumpNode(const SUnit *SU) const;

    virtual std::string getGraphNodeLabel(const SUnit *SU) const;
  };
}

#endif
