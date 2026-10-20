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
FunctionPass *createAGXCodeGenPreparePass();
FunctionPass *createAGXStrictFPPass();
FunctionPass *createAGXLowerFPEnvPass();
FunctionPass *createAGXSoftF64Pass();
FunctionPass *createAGXExpandWideAtomicsPass(const AGXTargetMachine *TM = nullptr);
FunctionPass *createAGXAnnotateControlFlowPass();

// Instruction selection.
FunctionPass *createAGXISelDag(AGXTargetMachine &TM);

// Machine passes.
FunctionPass *createAGXFoldOperandsPass();
FunctionPass *createAGXLowerControlFlowPass();
FunctionPass *createAGXInsertWaitsPass();

void initializeAGXAlwaysInlinePass(PassRegistry &);
void initializeAGXCodeGenPreparePass(PassRegistry &);
void initializeAGXStrictFPPass(PassRegistry &);
void initializeAGXLowerFPEnvPass(PassRegistry &);
void initializeAGXSoftF64Pass(PassRegistry &);
void initializeAGXExpandWideAtomicsPass(PassRegistry &);
void initializeAGXAnnotateControlFlowPass(PassRegistry &);
void initializeAGXDAGToDAGISelLegacyPass(PassRegistry &);
void initializeAGXFoldOperandsPass(PassRegistry &);
void initializeAGXLowerControlFlowPass(PassRegistry &);
void initializeAGXInsertWaitsPass(PassRegistry &);
void initializeAGXAsmPrinterPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_AGX_AGX_H
