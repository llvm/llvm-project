//===-- SuperH.h - Top-level interface for SuperH representation *- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// SuperH back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SUPERH_SUPERH_H
#define LLVM_LIB_TARGET_SUPERH_SUPERH_H

#include "MCTargetDesc/SuperHMCTargetDesc.h"

using namespace llvm;
namespace llvm {
class AsmPrinter;
class FunctionPass;
class MCInst;
class MachineInstr;
class PassRegistry;
class SuperHTargetMachine;

FunctionPass *createSuperHISelDag(SuperHTargetMachine &TM, CodeGenOptLevel OptLevel);
FunctionPass *createSuperHFillDelaySlotsPass();
FunctionPass *createSuperHConstantIslandPass();

void initializeSuperHDAGToDAGISelLegacyPass(PassRegistry &);
void initializeSuperHAsmPrinterPass(PassRegistry &);
void initializeSuperHAsmPrinterPass(PassRegistry &);
void initializeSuperHFillDelaySlotsPass(PassRegistry &);
void initializeSuperHConstantIslandsPass(PassRegistry &);
} // namespace llvm


#endif