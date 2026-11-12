//===- PPCMachineScheduler.h - Custom PowerPC MI scheduler --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Custom PowerPC MI scheduler.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_POWERPC_POWERPCMACHINESCHEDULER_H
#define LLVM_LIB_TARGET_POWERPC_POWERPCMACHINESCHEDULER_H

#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

struct PPCOptions;

/// A MachineSchedStrategy implementation for PowerPC pre RA scheduling.
class PPCPreRASchedStrategy : public GenericScheduler {
  const PPCOptions &CLOpts;

public:
  PPCPreRASchedStrategy(const MachineSchedContext *C);

protected:
  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) const override;

private:
  bool biasAddiLoadCandidate(SchedCandidate &Cand,
                             SchedCandidate &TryCand,
                             SchedBoundary &Zone) const;
};

/// A MachineSchedStrategy implementation for PowerPC post RA scheduling.
class PPCPostRASchedStrategy : public PostGenericScheduler {
  const PPCOptions &CLOpts;

public:
  PPCPostRASchedStrategy(const MachineSchedContext *C);

protected:
  void initialize(ScheduleDAGMI *Dag) override;
  SUnit *pickNode(bool &IsTopNode) override;
  void enterMBB(MachineBasicBlock *MBB) override;
  void leaveMBB() override;

  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand) override;
  bool biasAddiCandidate(SchedCandidate &Cand, SchedCandidate &TryCand) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_POWERPC_POWERPCMACHINESCHEDULER_H
