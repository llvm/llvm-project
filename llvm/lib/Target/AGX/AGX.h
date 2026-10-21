//===-- AGX.h - Top-level interface for AGX ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Entry points of the AGX code generator passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AGX_AGX_H
#define LLVM_LIB_TARGET_AGX_AGX_H

#include "llvm/Pass.h"

namespace llvm {

class AGXTargetMachine;
class FunctionPass;
class ModulePass;
class PassRegistry;

// IR passes.
ModulePass *createAGXAlwaysInlinePass();
ModulePass *createAGXPromoteKernelArgsPass();
FunctionPass *createAGXCodeGenPreparePass();
FunctionPass *createAGXStrictFPPass();
FunctionPass *createAGXLowerFPEnvPass();
FunctionPass *createAGXSoftF64Pass();
FunctionPass *createAGXExpandWideAtomicsPass(const AGXTargetMachine *TM = nullptr);
FunctionPass *createAGXAnnotateControlFlowPass();
FunctionPass *createAGXLowerGenericAddressesPass();

// Instruction selection.
FunctionPass *createAGXISelDag(AGXTargetMachine &TM);

// Machine passes.
FunctionPass *createAGXFoldOperandsPass();
FunctionPass *createAGXHoistFromElseFlowPass();
FunctionPass *createAGXLowerControlFlowPass();
FunctionPass *createAGXInsertWaitsPass();

void initializeAGXAlwaysInlinePass(PassRegistry &);
void initializeAGXPromoteKernelArgsPass(PassRegistry &);
void initializeAGXCodeGenPreparePass(PassRegistry &);
void initializeAGXStrictFPPass(PassRegistry &);
void initializeAGXLowerFPEnvPass(PassRegistry &);
void initializeAGXSoftF64Pass(PassRegistry &);
void initializeAGXExpandWideAtomicsPass(PassRegistry &);
void initializeAGXAnnotateControlFlowPass(PassRegistry &);
void initializeAGXLowerGenericAddressesPass(PassRegistry &);
void initializeAGXDAGToDAGISelLegacyPass(PassRegistry &);
void initializeAGXFoldOperandsPass(PassRegistry &);
void initializeAGXHoistFromElseFlowPass(PassRegistry &);
void initializeAGXLowerControlFlowPass(PassRegistry &);
void initializeAGXInsertWaitsPass(PassRegistry &);
void initializeAGXAsmPrinterPass(PassRegistry &);

class Instruction;
class MachineInstr;

namespace AGX {
/// Whether a device memory access must reach memory shared with the host while
/// the dispatch runs: volatile and system-scope atomic loads and stores
/// (PTX ld/st.volatile, .sys). They are encoded coherent, loads after an f59d
/// and stores before an f533 (ptx-re-blockers B7).
bool isHostCoherentAccess(const Instruction &I);
bool isHostCoherentAccess(const MachineInstr &MI);
} // namespace AGX

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGX_H
