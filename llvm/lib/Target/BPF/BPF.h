//===-- BPF.h - Top-level interface for BPF representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPF_H
#define LLVM_LIB_TARGET_BPF_BPF_H

#include "MCTargetDesc/BPFMCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class BPFTargetMachine;

ModulePass *createBPFAbstractMemberAccess(BPFTargetMachine *TM);

FunctionPass *createBPFISelDag(BPFTargetMachine &TM);
FunctionPass *createBPFMISimplifyPatchablePass();
FunctionPass *createBPFMIPeepholePass();
FunctionPass *createBPFMIPeepholeTruncElimPass();
FunctionPass *createBPFMIPreEmitPeepholePass();
FunctionPass *createBPFMIPreEmitCheckingPass();

void initializeBPFAbstractMemberAccessPass(PassRegistry&);
void initializeBPFMISimplifyPatchablePass(PassRegistry&);
void initializeBPFMIPeepholePass(PassRegistry&);
void initializeBPFMIPeepholeTruncElimPass(PassRegistry&);
void initializeBPFMIPreEmitPeepholePass(PassRegistry&);
void initializeBPFMIPreEmitCheckingPass(PassRegistry&);
}

#endif
