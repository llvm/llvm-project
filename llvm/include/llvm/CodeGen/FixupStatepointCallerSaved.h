//===- llvm/CodeGen/FixupStatepointCallerSaved.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_FIXUPSTATEPOINTCALLERSAVED_H
#define LLVM_CODEGEN_FIXUPSTATEPOINTCALLERSAVED_H

#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

class FixupStatepointCallerSavedPass
    : public PassInfoMixin<FixupStatepointCallerSavedPass> {
public:
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_FIXUPSTATEPOINTCALLERSAVED_H
