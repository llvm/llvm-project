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
static cl::opt<bool>
    MFCFGHashDump("mf-cfg-hash-dump",
                  cl::desc("Dump machine function's control flow grpah hash"),
                  cl::init(false), cl::Hidden);

// Calculate machine function hash in level order traversal.
// For each machine basic block, using its mbb's BaseID,
// size of successors and  successors' mbb's BaseID to update hash.
// These informations can make graph unique.
static uint64_t calculateMBBCFGHash(MachineFunction &MF) {
  std::unordered_set<MachineBasicBlock *> Visited;
  MD5 Hash;
  std::queue<MachineBasicBlock *> Q;
  if (!MF.empty()) {
    Q.push(&*MF.begin());
  }
  while (!Q.empty()) {
    MachineBasicBlock *Now = Q.front();
    Q.pop();
    using namespace llvm::support;
    uint32_t Value = endian::byte_swap<uint32_t, llvm::endianness::little>(
        Now->getBBID()->BaseID);
    uint32_t Size =
        endian::byte_swap<uint32_t, llvm::endianness::little>(Now->succ_size());
    Hash.update(llvm::ArrayRef((uint8_t *)&Value, sizeof(Value)));
    Hash.update(llvm::ArrayRef((uint8_t *)&Size, sizeof(Size)));
    if (Visited.count(Now)) {
      continue;
    }
    Visited.insert(Now);
    for (MachineBasicBlock *Succ : Now->successors()) {
      Q.push(Succ);
    }
  }
  llvm::MD5::MD5Result Result;
  Hash.final(Result);
  return Result.low();
}

bool MachineFunctionHashBuilder::runOnMachineFunction(MachineFunction &MF) {
  setCFGHash(MF.getName(), calculateMBBCFGHash(MF));
  if (MFCFGHashDump) {
    llvm::outs() << "Function name: " << MF.getName().str()
                 << " Hash: " << getCFGHash(MF.getName()) << "\n";
  }
  return true;
}

char MachineFunctionHashBuilder::ID = 0;
INITIALIZE_PASS(MachineFunctionHashBuilder, "machine-function-hash",
                "Calculate machine function hash", false, false)
char &llvm::MachineFunctionHashBuilderID = MachineFunctionHashBuilder::ID;
MachineFunctionPass *llvm::createMachineFunctionHashBuilderPass() {
  return new MachineFunctionHashBuilder();
}
