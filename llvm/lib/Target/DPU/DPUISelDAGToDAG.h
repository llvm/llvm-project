//===-- DPUISelDAGToDAG.h - A dag to dag inst selector for DPU ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the DPU target.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_DPU_DPUISELDAGTODAG_H
#define LLVM_LIB_TARGET_DPU_DPUISELDAGTODAG_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include <DPUTargetMachine.h>

namespace llvm {
FunctionPass *createDPUISelDag(DPUTargetMachine &TM,
                               CodeGenOpt::Level OptLevel);
}

#endif
