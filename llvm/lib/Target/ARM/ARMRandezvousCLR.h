//===- ARMRandezvousCLR.h - ARM Randezvous Code Layout Randomization ------===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces of a pass that randomizes the code layout
// of ARM machine code.
//
//===----------------------------------------------------------------------===//

#ifndef ARM_RANDEZVOUS_CLR
#define ARM_RANDEZVOUS_CLR

#include "ARMRandezvousInstrumentor.h"
#include "llvm/Pass.h"
#include "llvm/Support/RandomNumberGenerator.h"

namespace llvm {
  struct ARMRandezvousCLR : public ModulePass, ARMRandezvousInstrumentor {
    // Pass Identifier
    static char ID;

    ARMRandezvousCLR(bool LateStage);
    virtual StringRef getPassName() const override;
    void getAnalysisUsage(AnalysisUsage & AU) const override;
    virtual bool runOnModule(Module & M) override;

  private:
    // Which stage we are at:
    //
    // * Early stage: insert most of trap instructions for trap block consumers
    //                between early and late stages
    //
    // * Late stage: shuffle code layout and insert the rest of trap
    //               instructions
    bool LateStage = false;

    std::unique_ptr<RandomNumberGenerator> RNG;

    void shuffleMachineBasicBlocks(MachineFunction & MF);
    void shuffleMachineBasicBlockClusters(MachineFunction & MF);
    void insertTrapBlocks(Function & F, MachineFunction & MF,
                          uint64_t NumTrapInsts);
  };

  ModulePass * createARMRandezvousCLR(bool LateStage);
}

#endif