//===- ARMMacroFusion.h - ARM Macro Fusion ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the ARM definition of the DAG scheduling mutation
/// to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

/// Note that you have to add:
///   DAG.addMutation(createARMMacroFusionDAGMutation());
/// to ARMPassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createARMMacroFusionDAGMutation();

} // llvm
