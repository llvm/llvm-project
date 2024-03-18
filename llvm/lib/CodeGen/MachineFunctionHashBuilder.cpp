//===-- MachineFunctionHashBuilder.cpp ----------------------------------*-
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
#include "llvm/CodeGen/MachineFunctionHashBuilder.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>

using namespace llvm;

// Calculate machine function hash in level order traversal.
// The control flow graph is uniquely represented by its level-order traversal.
static uint64_t calculateMBBCFGHash(const MachineFunction &MF) {
  if (MF.empty())
    return 0;
  std::unordered_set<const MachineBasicBlock *> Visited;
  MD5 Hash;
  std::queue<const MachineBasicBlock *> WorkList;
  WorkList.push(&*MF.begin());
  while (!WorkList.empty()) {
    const MachineBasicBlock *CurrentBB = WorkList.front();
    WorkList.pop();
    uint32_t Value = support::endian::byte_swap<uint32_t, endianness::little>(
        CurrentBB->getBBID()->BaseID);
    uint32_t Size = support::endian::byte_swap<uint32_t, endianness::little>(
        CurrentBB->succ_size());
    Hash.update(ArrayRef((uint8_t *)&Value, sizeof(Value)));
    Hash.update(ArrayRef((uint8_t *)&Size, sizeof(Size)));
    if (Visited.count(CurrentBB))
      continue;
    Visited.insert(CurrentBB);
    std::vector<MachineBasicBlock *> Successors(CurrentBB->succ_begin(),
                                                CurrentBB->succ_end());
    std::sort(Successors.begin(), Successors.end(),
              [](const MachineBasicBlock *MBB1, const MachineBasicBlock *MBB2) {
                return MBB1->getBBID()->BaseID < MBB2->getBBID()->BaseID;
              });
    for (MachineBasicBlock *Succ : Successors) {
      WorkList.push(Succ);
    }
  }
  MD5::MD5Result Result;
  Hash.final(Result);
  return Result.low();
}

bool MachineFunctionHashBuilder::runOnMachineFunction(MachineFunction &MF) {
  setCFGHash(MF.getName(), calculateMBBCFGHash(MF));
  return true;
}

char MachineFunctionHashBuilder::ID = 0;
INITIALIZE_PASS(MachineFunctionHashBuilder, "machine-function-hash",
                "Calculate machine function hash", false, false)
char &llvm::MachineFunctionHashBuilderID = MachineFunctionHashBuilder::ID;
MachineFunctionPass *llvm::createMachineFunctionHashBuilderPass() {
  return new MachineFunctionHashBuilder();
}
