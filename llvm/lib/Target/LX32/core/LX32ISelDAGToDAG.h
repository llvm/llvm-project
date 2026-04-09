//===-- LX32ISelDAGToDAG.h - LX32 DAG->DAG Instruction Selector ----------===//
//
// Part of the LX32 Project
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_LX32_CORE_LX32ISELDAGTODAG_H
#define LLVM_LIB_TARGET_LX32_CORE_LX32ISELDAGTODAG_H

#include "llvm/Support/CodeGen.h"

namespace llvm {

class FunctionPass;
class LX32TargetMachine;

FunctionPass *createLX32ISelDag(LX32TargetMachine &TM, CodeGenOptLevel OptLevel);

} // namespace llvm

#endif // LLVM_LIB_TARGET_LX32_CORE_LX32ISELDAGTODAG_H


