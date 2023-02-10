//===- ARMRandezvousCLR.cpp - ARM Randezvous Code Layout Randomization ----===//
//
// Copyright (c) 2021-2022, University of Rochester
//
// Part of the Randezvous Project, under the Apache License v2.0 with
// LLVM Exceptions.  See LICENSE.txt in the llvm directory for license
// information.
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of a pass that randomizes the code
// layout of ARM machine code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm-randezvous-clr"

#include "ARMRandezvousCLR.h"
#include "ARMRandezvousOptions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

char ARMRandezvousCLR::ID = 0;

STATISTIC(NumTraps, "Number of trap instructions inserted");
STATISTIC(NumFuncsBBLR, "Number of functions with basic blocks reordered");
STATISTIC(NumJumps4BBLR, "Number of jump instructions inserted due to BBLR");
STATISTIC(NumFuncsBBCLR, "Number of functions with basic block clusters reordered");

ARMRandezvousCLR::ARMRandezvousCLR(bool LateStage)
    : ModulePass(ID), LateStage(LateStage) {
}

StringRef
ARMRandezvousCLR::getPassName() const {
  return "ARM Randezvous Code Layout Randomization Pass";
}

void
ARMRandezvousCLR::getAnalysisUsage(AnalysisUsage & AU) const {
  // We need this to access MachineFunctions
  AU.addRequired<MachineModuleInfoWrapperPass>();

  AU.setPreservesCFG();
  ModulePass::getAnalysisUsage(AU);
}

//
// Method: shuffleMachineBasicBlocks()
//
// Description:
//   This method shuffles the order of MachineBasicBlocks in a MachineFunction.
//   It shuffles all the basic blocks except the entry block, so fall-through
//   blocks will be taken apart and branch instructions will be inserted
//   appropriately to preserve the CFG.
//
// Input:
//   MF - A reference to the MachineFunction.
//
// Output:
//   MF - The transformed MachineFunction.
//
void
ARMRandezvousCLR::shuffleMachineBasicBlocks(MachineFunction & MF) {
  // Shuffling has no effect on functions with fewer than 3 MachineBasicBlocks
  // (because we are not reordering the entry block)
  if (MF.size() < 3) {
    return;
  }

  // Add an unconditional branch to all MachineBasicBlocks that fall through so
  // that we can safely take them apart from their fall-through blocks
  std::vector<MachineBasicBlock *> MBBs;
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();
  for (MachineBasicBlock & MBB : MF) {
    MachineBasicBlock * FallThruMBB = MBB.getFallThrough();
    if (FallThruMBB != nullptr) {
      BuildMI(MBB, MBB.end(), DebugLoc(), TII->get(ARM::t2B))
      .addMBB(FallThruMBB)
      .add(predOps(ARMCC::AL));
      ++NumJumps4BBLR;
    }
    MBBs.push_back(&MBB);
  }

  // Now do shuffling; ilist (iplist_impl) does not support iterator
  // increment/decrement so we have to first do out-of-place shuffling and then
  // do in-place removal and insertion
  auto & MBBList = (&MF)->*(MachineFunction::getSublistAccess)(nullptr);
  llvm::shuffle(MBBs.begin() + 1, MBBs.end(), *RNG);
  for (MachineBasicBlock * MBB : MBBs) {
    MBBList.remove(MBB);
  }
  for (MachineBasicBlock * MBB : MBBs) {
    MBBList.push_back(MBB);
  }
  ++NumFuncsBBLR;
}

//
// Method: shuffleMachineBasicBlockClusters()
//
// Description:
//   This method shuffles the order of clusters of MachineBasicBlocks that fall
//   through in the order as they appear.  It shuffles all the basic block
//   clusters except the entry cluster.
//
// Input:
//   MF - A reference to the MachineFunction.
//
// Output:
//   MF - The transformed MachineFunction.
//
void
ARMRandezvousCLR::shuffleMachineBasicBlockClusters(MachineFunction & MF) {
  auto & MBBList = (&MF)->*(MachineFunction::getSublistAccess)(nullptr);

  // Construct a list of clusters
  std::vector<std::vector<MachineBasicBlock *> *> Clusters;
  std::vector<MachineBasicBlock *> * CurrentCluster = nullptr;
  for (MachineBasicBlock & MBB : MF) {
    if (CurrentCluster == nullptr) {
      CurrentCluster = new std::vector<MachineBasicBlock *>();
    }
    CurrentCluster->push_back(&MBB);
    if (!MBB.canFallThrough()) {
      Clusters.push_back(CurrentCluster);
      CurrentCluster = nullptr;
    }
  }

  do {
    // Shuffling has no effect on functions with fewer than 3 clusters (because
    // we are not reordering the entry cluster)
    if (Clusters.size() < 3) {
      break;
    }

    // Now do shuffling; ilist (iplist_impl) does not support iterator
    // increment/decrement so we have to first do out-of-place shuffling and
    // then do in-place removal and insertion
    llvm::shuffle(Clusters.begin() + 1, Clusters.end(), *RNG);
    for (auto * Cluster : Clusters) {
      for (MachineBasicBlock * MBB : *Cluster) {
        MBBList.remove(MBB);
      }
    }
    for (auto * Cluster : Clusters) {
      for (MachineBasicBlock * MBB : *Cluster) {
        MBBList.push_back(MBB);
      }
    }
    ++NumFuncsBBCLR;
  } while (false);

  // Garbage collection
  for (auto * Cluster : Clusters) {
    delete Cluster;
  }
}

//
// Method: insertTrapBlocks()
//
// Description:
//   This method inserts a given number of trap instructions into a Function
//   and keeps track of each inserted trap instruction as a single basic block.
//
// Inputs:
//   F            - A reference to the Function.
//   MF           - A reference to the MachineFunction to which F corresponds.
//   NumTrapInsts - Total number of trap instructions to insert.
//
// Output:
//   MF - The transformed MachineFunction.
//
void
ARMRandezvousCLR::insertTrapBlocks(Function & F, MachineFunction & MF,
                                   uint64_t NumTrapInsts) {
  LLVMContext & Ctx = F.getContext();
  const TargetInstrInfo * TII = MF.getSubtarget().getInstrInfo();

  //
  // In machine IR, disperse trap blocks throughout the MachineFunction, where
  // the insertion points are the MachineBasicBlocks that do not fall through;
  // this allows us to preserve the CFG while adding randomness to the inside
  // of the MachineFunction.
  //
  // In LLVM IR, simply place trap blocks at the end of the Function.
  //

  // Determine where to insert trap instructions
  std::vector<MachineBasicBlock *> InsertionPts;
  for (MachineBasicBlock & MBB : MF) {
    if (!MBB.canFallThrough() && !MBB.isRandezvousTrapBlock()) {
      InsertionPts.push_back(&MBB);
    }
  }

  // Determine the numbers of trap instructions to insert at each point
  uint64_t SumShares = 0;
  std::vector<uint64_t> Shares(InsertionPts.size());
  for (uint64_t i = 0; i < InsertionPts.size(); ++i) {
    Shares[i] = (*RNG)() & 0xffffffff; // Prevent overflow
    SumShares += Shares[i];
  }
  for (uint64_t i = 0; i < InsertionPts.size(); ++i) {
    Shares[i] = Shares[i] * NumTrapInsts / SumShares;
  }

  // Do insertion
  for (uint64_t i = 0; i < InsertionPts.size(); ++i) {
    for (uint64_t j = 0; j < Shares[i]; ++j) {
      // Build an IR basic block
      BasicBlock * BB = BasicBlock::Create(Ctx, "", &F);
      IRBuilder<> IRB(BB);
      IRB.CreateUnreachable();

      // Build a machine IR basic block
      MachineBasicBlock * MBB = MF.CreateMachineBasicBlock(BB);
      BuildMI(MBB, DebugLoc(), TII->get(ARM::t2UDF_ga)).addImm(0);
      MF.push_back(MBB);
      MBB->moveAfter(InsertionPts[i]);
      MBB->setMachineBlockAddressTaken();
      MBB->setIsRandezvousTrapBlock();

      ++NumTraps;
    }
  }
}

//
// Method: runOnModule()
//
// Description:
//   This method is called when the PassManager wants this pass to transform
//   the specified Module.  This method shuffles the order of functions within
//   the module and/or the order of basic blocks within each function, and
//   inserts trap instructions to fill the text section.
//
// Input:
//   M - A reference to the Module to transform.
//
// Output:
//   M - The transformed Module.
//
// Return value:
//   true  - The Module was transformed.
//   false - The Module was not transformed.
//
bool
ARMRandezvousCLR::runOnModule(Module & M) {
  if (!EnableRandezvousCLR) {
    return false;
  }

  MachineModuleInfo & MMI = getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
  Twine RNGName = getPassName() + "-" + Twine(RandezvousCLRSeed);
  RNG = M.createRNG(RNGName.str());

  // First, shuffle the order of basic blocks in each function (if requested
  // and at the late stage) and calculate how much space existing functions
  // have taken up
  uint64_t TotalTextSize = 0;
  std::vector<std::pair<Function *, MachineFunction *> > Functions;
  for (Function & F : M) {
    MachineFunction * MF = MMI.getMachineFunction(F);
    if (MF == nullptr) {
      continue;
    }

    if (LateStage) {
      if (EnableRandezvousBBLR) {
        shuffleMachineBasicBlocks(*MF);
      } else if (EnableRandezvousBBCLR) {
        shuffleMachineBasicBlockClusters(*MF);
      }
    }

    uint64_t TextSize = getFunctionCodeSize(*MF);
    if (TextSize != 0) {
      Functions.push_back(std::make_pair(&F, MF));
      TotalTextSize += TextSize;
    }
  }
  assert(TotalTextSize <= RandezvousMaxTextSize && "Text size exceeds the limit");

  if (LateStage) {
    // Second, shuffle the order of functions; SymbolTableList (iplist_impl)
    // does not support iterator increment/decrement so we have to first do
    // out-of-place shuffling and then do in-place removal and insertion
    SymbolTableList<Function> & FunctionList = M.getFunctionList();
    llvm::shuffle(Functions.begin(), Functions.end(), *RNG);
    for (auto & FMF : Functions) {
      FunctionList.remove(FMF.first);
    }
    for (auto & FMF : Functions) {
      FunctionList.push_back(FMF.first);
    }
  }

  // Third, determine the numbers of trap instructions to insert
  uint64_t NumTrapInsts = (RandezvousMaxTextSize - TotalTextSize) / 4;
  uint64_t SumShares = 0;
  std::vector<uint64_t> Shares(Functions.size());
  if (!LateStage) {
    // Insert 80% of trap instructions during the early stage; this allows most
    // of trap blocks to be consumed by later passes while still keeping a
    // considerable code size budget for later passes and the late-stage CLR
    // pass
    NumTrapInsts = NumTrapInsts * 80 / 100;
  }
  for (uint64_t i = 0; i < Functions.size(); ++i) {
    Shares[i] = (*RNG)() & 0xffffffff; // Prevent overflow
    SumShares += Shares[i];
  }
  for (uint64_t i = 0; i < Functions.size(); ++i) {
    Shares[i] = Shares[i] * NumTrapInsts / SumShares;
  }

  // Lastly, insert trap instructions into each function
  for (uint64_t i = 0; i < Functions.size(); ++i) {
    insertTrapBlocks(*Functions[i].first, *Functions[i].second, Shares[i]);
  }

  return true;
}

ModulePass *
llvm::createARMRandezvousCLR(bool EarlyTrapInsertion) {
  return new ARMRandezvousCLR(EarlyTrapInsertion);
}