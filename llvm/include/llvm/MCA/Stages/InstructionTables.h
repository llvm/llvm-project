//===--------------------- InstructionTables.h ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file implements a custom stage to generate instruction tables.
/// See the description of command-line flag -instruction-tables in
/// docs/CommandGuide/lvm-mca.rst
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCA_STAGES_INSTRUCTIONTABLES_H
#define LLVM_MCA_STAGES_INSTRUCTIONTABLES_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCSchedule.h"
#include "llvm/MCA/HardwareUnits/Scheduler.h"
#include "llvm/MCA/Stages/Stage.h"
#include "llvm/MCA/Support.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "llvm-mca"

namespace llvm {
namespace mca {

class LLVM_ABI InstructionTables final : public Stage {
  const MCSchedModel &SM;
  SmallVector<ResourceUse, 4> UsedResources;
  SmallVector<uint64_t, 8> Masks;

public:
  InstructionTables(const MCSchedModel &Model)
      : SM(Model), Masks(Model.getNumProcResourceKinds()) {
    computeProcResourceMasks(Model, Masks);
    LLVM_DEBUG({
      dbgs() << "\nProcessor resource masks:\n";
      for (unsigned I = 0, E = Model.getNumProcResourceKinds(); I < E; ++I) {
        const MCProcResourceDesc &Desc = *Model.getProcResource(I);
        dbgs() << '[' << format_decimal(I, 2) << "] " << " - "
               << format_hex(Masks[I], 16) << " - " << Desc.Name << '\n';
      }
    });
  }

  bool hasWorkToComplete() const override { return false; }
  Error execute(InstRef &IR) override;
};
} // namespace mca
} // namespace llvm

#endif // LLVM_MCA_STAGES_INSTRUCTIONTABLES_H
