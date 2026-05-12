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
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

// Frozen mixer; the block hashes computed below are serialized into BB
// section profile data, so this function's exact output is part of the
// on-disk format. Do not change without versioning that format.
static constexpr uint64_t hash_16_bytes(uint64_t low, uint64_t high) {
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (low ^ high) * kMul;
  a ^= (a >> 47);
  uint64_t b = (high ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}

static uint64_t hashBlock(const MachineBasicBlock &MBB, bool HashOperands) {
  uint64_t Hash = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction() || MI.isTerminator())
      continue;
    Hash = hash_16_bytes(Hash, MI.getOpcode());
    if (HashOperands) {
      for (unsigned i = 0; i < MI.getNumOperands(); i++) {
        Hash = hash_16_bytes(Hash, stableHashValue(MI.getOperand(i)));
      }
    }
  }
  return Hash;
}

/// Fold a 64-bit integer to a 16-bit one.
static constexpr uint16_t fold_64_to_16(const uint64_t Value) {
  uint16_t Res = static_cast<uint16_t>(Value);
  Res ^= static_cast<uint16_t>(Value >> 16);
  Res ^= static_cast<uint16_t>(Value >> 32);
  Res ^= static_cast<uint16_t>(Value >> 48);
  return Res;
}

static_assert(hash_16_bytes(1, 2) == 9684580150926652833ull,
              "Hash function must be stable");
static_assert(hash_16_bytes(-1, -2) == 7819786907124864172ull,
              "Hash function must be stable");
static_assert(fold_64_to_16(1) == 1, "Fold function must be stable");
static_assert(fold_64_to_16(12345678) == 25074, "Fold function must be stable");

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

MachineBlockHashInfoResult::MachineBlockHashInfoResult() = default;

MachineBlockHashInfoResult::MachineBlockHashInfoResult(
    const MachineFunction &F) {
  DenseMap<const MachineBasicBlock *, CollectHashInfo> HashInfos;
  uint16_t Offset = 0;
  // Initialize hash components
  for (const MachineBasicBlock &MBB : F) {
    auto &HashInfo = HashInfos[&MBB];
    // offset of the machine basic block
    HashInfo.Offset = Offset + MBB.size();
    // Hashing opcodes
    HashInfo.OpcodeHash = hashBlock(MBB, /*HashOperands=*/false);
    // Hash complete instructions
    HashInfo.InstrHash = hashBlock(MBB, /*HashOperands=*/true);
  }

  // Initialize neighbor hash
  for (const MachineBasicBlock &MBB : F) {
    auto &HashInfo = HashInfos[&MBB];
    uint64_t Hash = HashInfo.OpcodeHash;
    // Append hashes of successors
    for (const MachineBasicBlock *SuccMBB : MBB.successors()) {
      uint64_t SuccHash = HashInfos[SuccMBB].OpcodeHash;
      Hash = hash_16_bytes(Hash, SuccHash);
    }
    // Append hashes of predecessors
    for (const MachineBasicBlock *PredMBB : MBB.predecessors()) {
      uint64_t PredHash = HashInfos[PredMBB].OpcodeHash;
      Hash = hash_16_bytes(Hash, PredHash);
    }
    HashInfo.NeighborHash = Hash;
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
}

uint64_t
MachineBlockHashInfoResult::getMBBHash(const MachineBasicBlock &MBB) const {
  auto it = MBBHashInfo.find(&MBB);
  return it->second;
}

bool MachineBlockHashInfo::runOnMachineFunction(MachineFunction &F) {
  Result = MachineBlockHashInfoResult{F};
  return false;
}

uint64_t MachineBlockHashInfo::getMBBHash(const MachineBasicBlock &MBB) const {
  return Result.getMBBHash(MBB);
}

MachineFunctionPass *llvm::createMachineBlockHashInfoPass() {
  return new MachineBlockHashInfo();
}

AnalysisKey MachineBlockHashInfoAnalysis::Key;

MachineBlockHashInfoResult
MachineBlockHashInfoAnalysis::run(MachineFunction &MF,
                                  MachineFunctionAnalysisManager &MFAM) {
  return MachineBlockHashInfoResult{MF};
}

PreservedAnalyses
MachineBlockHashInfoPrinterPass::run(MachineFunction &MF,
                                     MachineFunctionAnalysisManager &MFAM) {
  auto &MBHI = MFAM.getResult<MachineBlockHashInfoAnalysis>(MF);
  OS << "Machine Block Hash Info for function: " << MF.getName() << "\n";
  for (const auto &MBB : MF) {
    OS << "  BB#" << MBB.getNumber() << ": "
       << format_hex(MBHI.getMBBHash(MBB), 16) << "\n";
  }
  return PreservedAnalyses::all();
}
