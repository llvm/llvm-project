//===- AArch64MacroFusion.h - AArch64 Macro Fusion ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// \fileThis file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the AArch64 definition of the DAG scheduling mutation
// to pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"

//===----------------------------------------------------------------------===//
// AArch64MacroFusion - DAG post-processing to encourage fusion of macro ops.
//===----------------------------------------------------------------------===//

namespace llvm {

/// Note that you have to add:
///   DAG.addMutation(createAArch64MacroFusionDAGMutation());
/// to AArch64PassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createAArch64MacroFusionDAGMutation();

} // llvm
