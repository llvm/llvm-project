//===-- Next32.h - Top-level interface for Next32 representation ------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_Next32_Next32_H
#define LLVM_LIB_TARGET_Next32_Next32_H

#include "MCTargetDesc/Next32MCTargetDesc.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class Next32TargetMachine;
class FunctionPass;

FunctionPass *createNext32ISelDag(Next32TargetMachine &TM,
                                  CodeGenOptLevel OptLevel);
FunctionPass *createNext32PromotePass();
FunctionPass *createNext32CondBranchFixup();
FunctionPass *createNext32CallSplits();
FunctionPass *createNext32AddRetFid();
FunctionPass *createNext32CallTerminators();
FunctionPass *createNext32EliminateCallTerminators();
FunctionPass *createNext32CalculateFeeders();
FunctionPass *createNext32OrderCallChain();
FunctionPass *createNext32WriterChains();
} // namespace llvm

#endif
