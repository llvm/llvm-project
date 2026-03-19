//===- llvm/CodeGen/MachineBlockHashInfo.cpp---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Compute the hashes of basic blocks.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBlockHashInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

uint64_t hashBlock(const MachineBasicBlock &MBB, bool HashOperands) {
  uint64_t Hash = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction() || MI.isTerminator())
      continue;
    Hash = hashing::detail::hash_16_bytes(Hash, MI.getOpcode());
    if (HashOperands) {
      for (unsigned i = 0; i < MI.getNumOperands(); i++) {
        Hash =
            hashing::detail::hash_16_bytes(Hash, hash_value(MI.getOperand(i)));
      }
    }
  }
  return Hash;
}

/// Fold a 64-bit integer to a 16-bit one.
uint16_t fold_64_to_16(const uint64_t Value) {
  uint16_t Res = static_cast<uint16_t>(Value);
  Res ^= static_cast<uint16_t>(Value >> 16);
  Res ^= static_cast<uint16_t>(Value >> 32);
  Res ^= static_cast<uint16_t>(Value >> 48);
  return Res;
}

INITIALIZE_PASS(MachineBlockHashInfo, "machine-block-hash",
                "Machine Block Hash Analysis", true, true)

char MachineBlockHashInfo::ID = 0;

MachineBlockHashInfo::MachineBlockHashInfo() : MachineFunctionPass(ID) {}

void MachineBlockHashInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

struct CollectHashInfo {
  uint64_t Offset;
  uint64_t OpcodeHash;
  uint64_t InstrHash;
  uint64_t NeighborHash;
};

bool MachineBlockHashInfo::runOnMachineFunction(MachineFunction &F) {
  DenseMap<const MachineBasicBlock *, CollectHashInfo> HashInfos;
  uint16_t Offset = 0;
  // Initialize hash components
  for (const MachineBasicBlock &MBB : F) {
    // offset of the machine basic block
    HashInfos[&MBB].Offset = Offset;
    Offset += MBB.size();
    // Hashing opcodes
    HashInfos[&MBB].OpcodeHash = hashBlock(MBB, /*HashOperands=*/false);
    // Hash complete instructions
    HashInfos[&MBB].InstrHash = hashBlock(MBB, /*HashOperands=*/true);
  }

  // Initialize neighbor hash
  for (const MachineBasicBlock &MBB : F) {
    uint64_t Hash = HashInfos[&MBB].OpcodeHash;
    // Append hashes of successors
    for (const MachineBasicBlock *SuccMBB : MBB.successors()) {
      uint64_t SuccHash = HashInfos[SuccMBB].OpcodeHash;
      Hash = hashing::detail::hash_16_bytes(Hash, SuccHash);
    }
    // Append hashes of predecessors
    for (const MachineBasicBlock *PredMBB : MBB.predecessors()) {
      uint64_t PredHash = HashInfos[PredMBB].OpcodeHash;
      Hash = hashing::detail::hash_16_bytes(Hash, PredHash);
    }
    HashInfos[&MBB].NeighborHash = Hash;
  }

  // Assign hashes
  for (const MachineBasicBlock &MBB : F) {
    const auto &HashInfo = HashInfos[&MBB];
    BlendedBlockHash BlendedHash(fold_64_to_16(HashInfo.Offset),
                                 fold_64_to_16(HashInfo.OpcodeHash),
                                 fold_64_to_16(HashInfo.InstrHash),
                                 fold_64_to_16(HashInfo.NeighborHash));
    MBBHashInfo[&MBB] = BlendedHash.combine();
  }

  return false;
}

uint64_t MachineBlockHashInfo::getMBBHash(const MachineBasicBlock &MBB) {
  return MBBHashInfo[&MBB];
}

MachineFunctionPass *llvm::createMachineBlockHashInfoPass() {
  return new MachineBlockHashInfo();
}
