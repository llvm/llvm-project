//===- DPUMacroFusion.h - DPU Macro Fusion --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUMACROFUSION_H
#define LLVM_LIB_TARGET_DPU_DPUMACROFUSION_H

#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/RegisterBankInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"

namespace llvm {
/// Note that you have to add:
///   DAG.addMutation(createDPUMacroFusionDAGMutation());
/// to DPUPassConfig::createMachineScheduler() to have an effect.
std::unique_ptr<ScheduleDAGMutation> createDPUMacroFusionDAGMutation();
} // end namespace llvm

#endif // LLVM_LIB_TARGET_DPU_DPUMACROFUSION_H
