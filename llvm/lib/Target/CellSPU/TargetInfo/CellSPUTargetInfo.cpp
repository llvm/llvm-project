//===-- CellSPUTargetInfo.cpp - CellSPU Target Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SPU.h"
#include "llvm/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheCellSPUTarget;

extern "C" void LLVMInitializeCellSPUTargetInfo() { 
  RegisterTarget<Triple::cellspu> 
    X(TheCellSPUTarget, "cellspu", "STI CBEA Cell SPU [experimental]");
}
