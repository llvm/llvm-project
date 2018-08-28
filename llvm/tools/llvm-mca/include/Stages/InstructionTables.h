//===--------------------- InstructionTables.h ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements a custom stage to generate instruction tables.
/// See the description of command-line flag -instruction-tables in
/// docs/CommandGuide/lvm-mca.rst
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_MCA_INSTRUCTIONTABLES_H
#define LLVM_TOOLS_LLVM_MCA_INSTRUCTIONTABLES_H

#include "HardwareUnits/Scheduler.h"
#include "InstrBuilder.h"
#include "Stages/Stage.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSchedule.h"

namespace mca {

class InstructionTables final : public Stage {
  const llvm::MCSchedModel &SM;
  InstrBuilder &IB;
  llvm::SmallVector<std::pair<ResourceRef, double>, 4> UsedResources;

public:
  InstructionTables(const llvm::MCSchedModel &Model, InstrBuilder &Builder)
      : Stage(), SM(Model), IB(Builder) {}

  bool hasWorkToComplete() const override { return false; }
  llvm::Error execute(InstRef &IR) override;
};
} // namespace mca

#endif
