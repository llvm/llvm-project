//===-- CodeGen.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the common initialization routines for the
// CodeGen library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Initialization.h"
#include "llvm/InitializePasses.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void llvm::initializeCodeGen(PassRegistry &Registry) {
  initializeAtomicExpandPass(Registry);
  initializeBBSectionsPreparePass(Registry);
  initializeBranchFolderPassPass(Registry);
  initializeBranchRelaxationPass(Registry);
  initializeCFGuardLongjmpPass(Registry);
  initializeCFIInstrInserterPass(Registry);
  initializeCodeGenPreparePass(Registry);
  initializeDeadMachineInstructionElimPass(Registry);
  initializeDetectDeadLanesPass(Registry);
  initializeDwarfEHPreparePass(Registry);
  initializeEarlyIfConverterPass(Registry);
  initializeEarlyIfPredicatorPass(Registry);
  initializeEarlyMachineLICMPass(Registry);
  initializeEarlyTailDuplicatePass(Registry);
  initializeExpandMemCmpPassPass(Registry);
  initializeExpandPostRAPass(Registry);
  initializeFEntryInserterPass(Registry);
  initializeFinalizeISelPass(Registry);
  initializeFinalizeMachineBundlesPass(Registry);
  initializeFuncletLayoutPass(Registry);
  initializeGCMachineCodeAnalysisPass(Registry);
  initializeGCModuleInfoPass(Registry);
  initializeHardwareLoopsPass(Registry);
  initializeIfConverterPass(Registry);
  initializeImplicitNullChecksPass(Registry);
  initializeIndirectBrExpandPassPass(Registry);
  initializeInterleavedLoadCombinePass(Registry);
  initializeInterleavedAccessPass(Registry);
  initializeLiveDebugValuesPass(Registry);
  initializeLiveDebugVariablesPass(Registry);
  initializeLiveIntervalsPass(Registry);
  initializeLiveRangeShrinkPass(Registry);
  initializeLiveStacksPass(Registry);
  initializeLiveVariablesPass(Registry);
  initializeLocalStackSlotPassPass(Registry);
  initializeLowerIntrinsicsPass(Registry);
  initializeMIRCanonicalizerPass(Registry);
  initializeMIRNamerPass(Registry);
  initializeMachineBlockFrequencyInfoPass(Registry);
  initializeMachineBlockPlacementPass(Registry);
  initializeMachineBlockPlacementStatsPass(Registry);
  initializeMachineCSEPass(Registry);
  initializeMachineCombinerPass(Registry);
  initializeMachineCopyPropagationPass(Registry);
  initializeMachineDominatorTreePass(Registry);
  initializeMachineFunctionPrinterPassPass(Registry);
  initializeMachineLICMPass(Registry);
  initializeMachineLoopInfoPass(Registry);
  initializeMachineModuleInfoWrapperPassPass(Registry);
  initializeMachineOptimizationRemarkEmitterPassPass(Registry);
  initializeMachineOutlinerPass(Registry);
  initializeMachinePipelinerPass(Registry);
  initializeModuloScheduleTestPass(Registry);
  initializeMachinePostDominatorTreePass(Registry);
  initializeMachineRegionInfoPassPass(Registry);
  initializeMachineSchedulerPass(Registry);
  initializeMachineSinkingPass(Registry);
  initializeMachineVerifierPassPass(Registry);
  initializeOptimizePHIsPass(Registry);
  initializePEIPass(Registry);
  initializePHIEliminationPass(Registry);
  initializePatchableFunctionPass(Registry);
  initializePeepholeOptimizerPass(Registry);
  initializePostMachineSchedulerPass(Registry);
  initializePostRAHazardRecognizerPass(Registry);
  initializePostRAMachineSinkingPass(Registry);
  initializePostRASchedulerPass(Registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  initializeProcessImplicitDefsPass(Registry);
  initializeRABasicPass(Registry);
  initializeRAGreedyPass(Registry);
  initializeRegAllocFastPass(Registry);
  initializeRegUsageInfoCollectorPass(Registry);
  initializeRegUsageInfoPropagationPass(Registry);
  initializeRegisterCoalescerPass(Registry);
  initializeRenameIndependentSubregsPass(Registry);
  initializeSafeStackLegacyPassPass(Registry);
  initializeScalarizeMaskedMemIntrinPass(Registry);
  initializeShrinkWrapPass(Registry);
  initializeSjLjEHPreparePass(Registry);
  initializeSlotIndexesPass(Registry);
  initializeStackColoringPass(Registry);
  initializeStackMapLivenessPass(Registry);
  initializeStackProtectorPass(Registry);
  initializeStackSlotColoringPass(Registry);
  initializeTailDuplicatePass(Registry);
  initializeTargetPassConfigPass(Registry);
  initializeTwoAddressInstructionPassPass(Registry);
  initializeTypePromotionPass(Registry);
  initializeUnpackMachineBundlesPass(Registry);
  initializeUnreachableBlockElimLegacyPassPass(Registry);
  initializeUnreachableMachineBlockElimPass(Registry);
  initializeVirtRegMapPass(Registry);
  initializeVirtRegRewriterPass(Registry);
  initializeWasmEHPreparePass(Registry);
  initializeWinEHPreparePass(Registry);
  initializeXRayInstrumentationPass(Registry);
}

void LLVMInitializeCodeGen(LLVMPassRegistryRef R) {
  initializeCodeGen(*unwrap(R));
}
