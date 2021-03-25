//===---------------------- InOrderIssueStage.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// InOrderIssueStage implements an in-order execution pipeline.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCA_IN_ORDER_ISSUE_STAGE_H
#define LLVM_MCA_IN_ORDER_ISSUE_STAGE_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MCA/SourceMgr.h"
#include "llvm/MCA/Stages/Stage.h"

#include <queue>

namespace llvm {
struct MCSchedModel;
class MCSubtargetInfo;

namespace mca {
class RegisterFile;
class ResourceManager;

class InOrderIssueStage final : public Stage {
  const MCSchedModel &SM;
  const MCSubtargetInfo &STI;
  RegisterFile &PRF;
  std::unique_ptr<ResourceManager> RM;

  /// Instructions that were issued, but not executed yet.
  SmallVector<InstRef, 4> IssuedInst;

  /// Number of instructions issued in the current cycle.
  unsigned NumIssued;

  /// If an instruction cannot execute due to an unmet register or resource
  /// dependency, the it is stalled for StallCyclesLeft.
  InstRef StalledInst;
  unsigned StallCyclesLeft;

  /// Number of instructions that can be issued in the current cycle.
  unsigned Bandwidth;

  /// Number of cycles (counted from the current cycle) until the last write is
  /// committed. This is taken into account to ensure that writes commit in the
  /// program order.
  unsigned LastWriteBackCycle;

  InOrderIssueStage(const InOrderIssueStage &Other) = delete;
  InOrderIssueStage &operator=(const InOrderIssueStage &Other) = delete;

  /// If IR has an unmet register or resource dependency, canExecute returns
  /// false. StallCycles is set to the number of cycles left before the
  /// instruction can be issued.
  bool canExecute(const InstRef &IR, unsigned *StallCycles) const;

  /// Issue the instruction, or update StallCycles if IR is stalled.
  Error tryIssue(InstRef &IR, unsigned *StallCycles);

  /// Update status of instructions from IssuedInst.
  void updateIssuedInst();

  /// Retire instruction once it is executed.
  void retireInstruction(InstRef &IR);

public:
  InOrderIssueStage(RegisterFile &PRF, const MCSchedModel &SM,
                    const MCSubtargetInfo &STI)
      : SM(SM), STI(STI), PRF(PRF), RM(std::make_unique<ResourceManager>(SM)),
        NumIssued(0), StallCyclesLeft(0), Bandwidth(0), LastWriteBackCycle(0) {}

  bool isAvailable(const InstRef &) const override;
  bool hasWorkToComplete() const override;
  Error execute(InstRef &IR) override;
  Error cycleStart() override;
  Error cycleEnd() override;
};

} // namespace mca
} // namespace llvm

#endif // LLVM_MCA_IN_ORDER_ISSUE_STAGE_H
