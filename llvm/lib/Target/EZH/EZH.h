//===-- EZH.h - Top-level interface for EZH representation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// EZH back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_EZH_EZH_H
#define LLVM_LIB_TARGET_EZH_EZH_H

#include "llvm/Pass.h"

namespace llvm {
class FunctionPass;
class EZHTargetMachine;
class PassRegistry;

// createEZHISelDag - This pass converts a legalized DAG into a
// EZH-specific DAG, ready for instruction scheduling.
FunctionPass *createEZHISelDag(EZHTargetMachine &TM);

FunctionPass *createEZHBranchFixupPass();
FunctionPass *createEZHConstantIslandPass();
FunctionPass *createEZHBitSliceInjectionPass();

void initializeEZHAsmPrinterPass(PassRegistry &);
void initializeEZHDAGToDAGISelLegacyPass(PassRegistry &);

} // namespace llvm

#endif // LLVM_LIB_TARGET_EZH_EZH_H
