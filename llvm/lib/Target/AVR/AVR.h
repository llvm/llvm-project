//===-- AVR.h - Top-level interface for AVR representation ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in the LLVM
// AVR back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AVR_H
#define LLVM_AVR_H

#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class AVRTargetMachine;
class FunctionPass;

Pass *createAVRShiftExpandPass();
FunctionPass *createAVRISelDag(AVRTargetMachine &TM,
                               CodeGenOpt::Level OptLevel);
FunctionPass *createAVRExpandPseudoPass();
FunctionPass *createAVRFrameAnalyzerPass();
FunctionPass *createAVRRelaxMemPass();
FunctionPass *createAVRDynAllocaSRPass();
FunctionPass *createAVRBranchSelectionPass();

void initializeAVRShiftExpandPass(PassRegistry &);
void initializeAVRExpandPseudoPass(PassRegistry&);
void initializeAVRRelaxMemPass(PassRegistry&);

/// Contains the AVR backend.
namespace AVR {

/// An integer that identifies all of the supported AVR address spaces.
enum AddressSpace { DataMemory, ProgramMemory };

/// Checks if a given type is a pointer to program memory.
template <typename T> bool isProgramMemoryAddress(T *V) {
  return cast<PointerType>(V->getType())->getAddressSpace() == ProgramMemory;
}

inline bool isProgramMemoryAccess(MemSDNode const *N) {
  auto V = N->getMemOperand()->getValue();

  return (V != nullptr) ? isProgramMemoryAddress(V) : false;
}

} // end of namespace AVR

} // end namespace llvm

#endif // LLVM_AVR_H
