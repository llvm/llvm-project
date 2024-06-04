//===-- MachineFunctionHashBuilder.h ----------------------------------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of pass about calculating machine
/// function hash.
//===----------------------------------------------------------------------===//
#ifndef LLVM_MACHINE_FUNCTION_HASH_BUILDER_H
#define LLVM_MACHINE_FUNCTION_HASH_BUILDER_H

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"

namespace llvm {

class MachineFunctionHashBuilder : public MachineFunctionPass {
public:
  static char ID;

  MachineFunctionHashBuilder() : MachineFunctionPass(ID) {
    initializeMachineFunctionHashBuilderPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Calculate machine function hash.";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  // Return machine function's cfg hash calculated by its machine basic block's
  // BaseID.
  uint64_t getCFGHash(StringRef Name) { return CFGHash.lookup(Name); }

private:
  void setCFGHash(StringRef Name, uint64_t Hash) { CFGHash[Name] = Hash; }
  StringMap<uint64_t> CFGHash;
};

} // namespace llvm
#endif