//===---------------------- RetireStage.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file defines the retire stage of a default instruction pipeline.
/// The RetireStage represents the process logic that interacts with the
/// simulated RetireControlUnit hardware.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_RETIRE_STAGE_H
#define LLVM_TOOLS_LLVM_MCA_RETIRE_STAGE_H

#include "RegisterFile.h"
#include "RetireControlUnit.h"
#include "Stage.h"

namespace mca {

class RetireStage final : public Stage {
  // Owner will go away when we move listeners/eventing to the stages.
  RetireControlUnit &RCU;
  RegisterFile &PRF;

  RetireStage(const RetireStage &Other) = delete;
  RetireStage &operator=(const RetireStage &Other) = delete;

public:
  RetireStage(RetireControlUnit &R, RegisterFile &F)
      : Stage(), RCU(R), PRF(F) {}

  bool hasWorkToComplete() const override { return !RCU.isEmpty(); }
  void cycleStart() override;
  Status execute(InstRef &IR) override { return Stage::Continue; }
  void notifyInstructionRetired(const InstRef &IR);
  void onInstructionExecuted(unsigned TokenID);
};

} // namespace mca

#endif // LLVM_TOOLS_LLVM_MCA_RETIRE_STAGE_H
