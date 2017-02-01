//===- X86MacroFusion.h - X86 Macro Fusion --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// \file This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 definition of the DAG scheduling mutation to pair
// instructions back to back.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"

//===----------------------------------------------------------------------===//
// X86MacroFusion - DAG post-processing to encourage fusion of macro ops.
//===----------------------------------------------------------------------===//

namespace llvm {

/// Note that you have to add:
///   DAG.addMutation(createX86MacroFusionDAGMutation());
/// to X86PassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation>
createX86MacroFusionDAGMutation();

} // end namespace llvm
