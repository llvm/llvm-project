//===-- ARM.h - Top-level interface for ARM representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// ARM back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARM_H
#define LLVM_LIB_TARGET_ARM_ARM_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CodeGen.h"
#include <functional>
#include <vector>

namespace llvm {

class ARMAsmPrinter;
class ARMBaseTargetMachine;
class ARMRegisterBankInfo;
class ARMSubtarget;
struct BasicBlockInfo;
class Function;
class FunctionPass;
class InstructionSelector;
class MachineBasicBlock;
class MachineFunction;
class MachineInstr;
class MCInst;
class PassRegistry;

Pass *createMVETailPredicationPass();
FunctionPass *createARMLowOverheadLoopsPass();
FunctionPass *createARMBlockPlacementPass();
Pass *createARMParallelDSPPass();
FunctionPass *createARMISelDag(ARMBaseTargetMachine &TM,
                               CodeGenOpt::Level OptLevel);
FunctionPass *createA15SDOptimizerPass();
FunctionPass *createARMLoadStoreOptimizationPass(bool PreAlloc = false);
FunctionPass *createARMExpandPseudoPass();
FunctionPass *createARMConstantIslandPass();
FunctionPass *createMLxExpansionPass();
FunctionPass *createThumb2ITBlockPass();
FunctionPass *createMVEVPTBlockPass();
FunctionPass *createMVETPAndVPTOptimisationsPass();
FunctionPass *createARMOptimizeBarriersPass();
FunctionPass *createThumb2SizeReductionPass(
    std::function<bool(const Function &)> Ftor = nullptr);
InstructionSelector *
createARMInstructionSelector(const ARMBaseTargetMachine &TM, const ARMSubtarget &STI,
                             const ARMRegisterBankInfo &RBI);
Pass *createMVEGatherScatterLoweringPass();
FunctionPass *createARMSLSHardeningPass();
FunctionPass *createARMIndirectThunks();
Pass *createMVELaneInterleavingPass();

void LowerARMMachineInstrToMCInst(const MachineInstr *MI, MCInst &OutMI,
                                  ARMAsmPrinter &AP);

void initializeARMParallelDSPPass(PassRegistry &);
void initializeARMLoadStoreOptPass(PassRegistry &);
void initializeARMPreAllocLoadStoreOptPass(PassRegistry &);
void initializeARMConstantIslandsPass(PassRegistry &);
void initializeARMExpandPseudoPass(PassRegistry &);
void initializeThumb2SizeReducePass(PassRegistry &);
void initializeThumb2ITBlockPass(PassRegistry &);
void initializeMVEVPTBlockPass(PassRegistry &);
void initializeMVETPAndVPTOptimisationsPass(PassRegistry &);
void initializeARMLowOverheadLoopsPass(PassRegistry &);
void initializeARMBlockPlacementPass(PassRegistry &);
void initializeMVETailPredicationPass(PassRegistry &);
void initializeMVEGatherScatterLoweringPass(PassRegistry &);
void initializeARMSLSHardeningPass(PassRegistry &);
void initializeMVELaneInterleavingPass(PassRegistry &);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_ARM_ARM_H
