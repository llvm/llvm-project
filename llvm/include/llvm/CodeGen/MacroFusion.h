//===- MacroFusion.h - Macro Fusion ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file This file contains the definition of the DAG scheduling mutation to
/// pair instructions back to back.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

/// \brief Check if the instr pair, FirstMI and SecondMI, should be fused
/// together. Given SecondMI, when FirstMI is unspecified, then check if
/// SecondMI may be part of a fused pair at all.
typedef std::function<bool(const TargetInstrInfo &TII,
                           const TargetSubtargetInfo &TSI,
                           const MachineInstr *FirstMI,
                           const MachineInstr &SecondMI)>  ShouldSchedulePredTy;

/// \brief Create a DAG scheduling mutation to pair instructions back to back
/// for instructions that benefit according to the target-specific
/// shouldScheduleAdjacent predicate function.
std::unique_ptr<ScheduleDAGMutation>
createMacroFusionDAGMutation(ShouldSchedulePredTy shouldScheduleAdjacent);

/// \brief Create a DAG scheduling mutation to pair branch instructions with one
/// of their predecessors back to back for instructions that benefit according
/// to the target-specific shouldScheduleAdjacent predicate function.
std::unique_ptr<ScheduleDAGMutation>
createBranchMacroFusionDAGMutation(ShouldSchedulePredTy shouldScheduleAdjacent);

} // end namespace llvm
