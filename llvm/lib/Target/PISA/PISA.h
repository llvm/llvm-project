//===-- PISA.h - Top-level interface for PISA representation --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISA_H
#define LLVM_LIB_TARGET_PISA_PISA_H

#include "MCTargetDesc/PISAMCTargetDesc.h"
#include "PISADefines.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class ImmutablePass;
class InstructionSelector;
class RegisterBankInfo;
class PISASubtarget;
class PISATargetMachine;

ModulePass *createPISALegalizeCallsPass();
ModulePass *createPISAVerifierPass();
ModulePass *createPISAKernelByValArgsLoweringLegacyPass();
ModulePass *createPISAPropagateNullPointersPass();

FunctionPass *createPISAExpandIntrinsicsPass();
FunctionPass *createPISAEmitIntrinsicsPass();
FunctionPass *createPISALegalizeSubregAccess();
FunctionPass *createPISAOptimizeSubregAccess();
FunctionPass *createPISAOptimizeRedundantCopies();
FunctionPass *createPISAInsertLifetimeStart();
FunctionPass *createPISAMarkConvergentNoMerge();
FunctionPass *createPISAPreLegalizerCombiner();
FunctionPass *createPISAPostLegalizerCombiner();
FunctionPass *createPISAReplaceIntrinsicsPass();
MachineFunctionPass *createPISALegalizePredicatesPass();
MachineFunctionPass *createCacheHintSelectorPass();
MachineFunctionPass *createPISAScopeSelectorPass();

InstructionSelector *
createPISAInstructionSelector(const PISATargetMachine &TM,
                              const PISASubtarget &Subtarget,
                              const RegisterBankInfo &RBI);

MachineFunctionPass *
createPISAMachineFunctionPrinterPass(const std::string &Banner,
                                     unsigned Counter);
FunctionPass *createPISALayoutPass();
MachineFunctionPass *createPISAVerifyTypesPass();

void initializeCacheHintSelectorPass(PassRegistry &);
void initializePISAEmitIntrinsicsPass(PassRegistry &);
void initializePISAExpandIntrinsicsPass(PassRegistry &);
void initializePISAInsertLifetimeStartPass(PassRegistry &);
void initializePISAKernelByValArgsLoweringLegacyPass(PassRegistry &);
void initializePISALegalizeCallsPass(PassRegistry &);
void initializePISALegalizeSubregAccessPass(PassRegistry &);
void initializePISAMachineFunctionPrinterPass(PassRegistry &);
void initializePISAMarkConvergentNoMergePass(PassRegistry &);
void initializePISAOptimizeRedundantCopiesPass(PassRegistry &);
void initializePISAOptimizeSubregAccessPass(PassRegistry &);
void initializePISALegalizePredicatesPass(PassRegistry &);
void initializePISAPostLegalizerCombinerPass(PassRegistry &);
void initializePISAPreLegalizerCombinerPass(PassRegistry &);
void initializePISAPropagateNullPointersPass(PassRegistry &);
void initializePISAReplaceIntrinsicsPass(PassRegistry &);
void initializePISAScopeSelectorPass(PassRegistry &);
void initializePISAVerifierPass(PassRegistry &);
void initializePISALayoutPass(PassRegistry &);
void initializePISAVerifyTypesPass(PassRegistry &);

namespace PISA {
LLVM_READONLY int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx);
} // namespace PISA
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISA_H
