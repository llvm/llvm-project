//===-- UnisonMIRPrepare.cpp - Unison-style MIR printing preparation --=======//
//
//  Main authors:
//    Roberto Castaneda Lozano <roberto.castaneda@ri.se>
//
//  This file is part of Unison, see http://unison-code.github.io
//
//  Copyright (c) 2017, RISE SICS AB
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//  3. Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived from this
//     software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Implementation of the UnisonMIRPrepare pass.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/UnisonMIRPrepare.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/MDBuilder.h"

using namespace llvm;

#define DEBUG_TYPE "unison-mir-prepare"

INITIALIZE_PASS_BEGIN(UnisonMIRPrepare, DEBUG_TYPE,
                      "Unison-style MIR printing preparation", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_END(UnisonMIRPrepare, DEBUG_TYPE,
                    "Unison-style MIR printing preparation", true, true)

char UnisonMIRPrepare::ID = 0;

MDNode *createMDTaggedTuple(MachineFunction &MF, std::string Tag,
                            uint64_t Val) {
  LLVMContext &Context = MF.getFunction().getContext();
  MDBuilder Builder(Context);
  return MDNode::get(Context,
                     {Builder.createString(Tag),
                      Builder.createConstant(
                          ConstantInt::get(Type::getInt64Ty(Context), Val))});
}

UnisonMIRPrepare::UnisonMIRPrepare() : MachineFunctionPass(ID) {
  initializeUnisonMIRPreparePass(*PassRegistry::getPassRegistry());
}

void UnisonMIRPrepare::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<MachineBlockFrequencyInfoWrapperPass>();
  AU.addRequired<AAResultsWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool UnisonMIRPrepare::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  MBFI = &getAnalysis<MachineBlockFrequencyInfoWrapperPass>();
  AA = &getAnalysis<AAResultsWrapperPass>().getAAResults();
  for (auto &MBB : MF) {
    annotateFrequency(MBB);
    annotateMemoryPartitions(MBB);
  }
  return !MF.empty();
}

void UnisonMIRPrepare::annotateFrequency(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  uint64_t Freq = MBFI->getMBFI().getBlockFreq(&MBB).getFrequency();
  MDNode *MD = createMDTaggedTuple(MF, "unison-block-frequency", Freq);
  auto MI = MBB.instr_begin();
  DebugLoc DL;
  BuildMI(MBB, MI, DL, TII->get(TargetOpcode::ANNOTATION_LABEL))
      .addMetadata(MD);
}

void UnisonMIRPrepare::annotateMemoryPartitions(MachineBasicBlock &MBB) {
  MachineFunction &MF = *MBB.getParent();
  // Create initial partitions with all the memory references in the block.
  MemAccessPartition MAP;
  for (auto &MI : MBB)
    if (!MI.isBundle() && (MI.mayStore() || MI.mayLoad()))
      MAP.insert(&MI);
  // Pairwise compare all memory references and merge those which may alias.
  for (auto &MI1 : MAP)
    for (auto &MI2 : MAP)
      // If MI1 and MI2 may alias. We use the same interface to 'AliasAnalysis'
      // as 'ScheduleDAGInstrs::addChainDependency' (that is, invoking
      // 'MachineInstr::mayAlias'). Therefore we share the same assumptions, see
      // the comments for 'MachineInstr::mayAlias'.
      if ((MI1->getData()->mayStore() || MI2->getData()->mayStore()) &&
          MI1->getData()->mayAlias(AA, *(MI2->getData()), true))
        MAP.unionSets(MI1->getData(), MI2->getData());
  // Populate the memory partition map.
  unsigned int P = 0;
  for (MemAccessPartition::iterator MA = MAP.begin(); MA != MAP.end(); ++MA) {
    if (!(*MA)->isLeader())
      continue;
    for (MemAccessPartition::member_iterator MI = MAP.member_begin(**MA);
         MI != MAP.member_end(); ++MI)
      MP[*MI] = P;
    ++P;
  }
  // Add a debug operand to each unbundled memory access instruction with the
  // partition of its memory reference.
  for (auto &MI : MBB)
    if (!MI.isBundle() && (MI.mayStore() || MI.mayLoad())) {
      MDNode *MD =
          createMDTaggedTuple(MF, "unison-memory-partition", MP.at(&MI));
      MI.addOperand(MF, MachineOperand::CreateMetadata(MD));
    }
}
