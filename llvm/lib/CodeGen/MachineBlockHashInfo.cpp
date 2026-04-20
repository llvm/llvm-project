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
#include "llvm/ADT/Hashing.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

static uint64_t hashBlock(const MachineBasicBlock &MBB, bool HashOperands) {
  uint64_t Hash = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isMetaInstruction() || MI.isTerminator())
      continue;
    Hash = hashing::detail::hash_16_bytes(Hash, MI.getOpcode());
    if (HashOperands) {
      for (unsigned i = 0; i < MI.getNumOperands(); i++) {
        Hash = hashing::detail::hash_16_bytes(
            Hash, stableHashValue(MI.getOperand(i)));
      }
    }
  }
  return Hash;
}

/// Fold a 64-bit integer to a 16-bit one.
static uint16_t fold_64_to_16(const uint64_t Value) {
  uint16_t Res = static_cast<uint16_t>(Value);
  Res ^= static_cast<uint16_t>(Value >> 16);
  Res ^= static_cast<uint16_t>(Value >> 32);
  Res ^= static_cast<uint16_t>(Value >> 48);
  return Res;
}

namespace {
class MachineBlockHashInfoPrinter : public MachineFunctionPass {
  raw_ostream &OS;

public:
  static char ID;
  MachineBlockHashInfoPrinter() : MachineFunctionPass(ID), OS(errs()) {}
  MachineBlockHashInfoPrinter(raw_ostream &OS)
      : MachineFunctionPass(ID), OS(OS) {}

  StringRef getPassName() const override {
    return "Machine Block Hash Info Printer";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<MachineBlockHashInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    MachineBlockHashInfo &MBHI = getAnalysis<MachineBlockHashInfo>();
    OS << "Machine Block Hash Info for function: " << MF.getName() << "\n";
    for (const auto &MBB : MF) {
      OS << "  BB#" << MBB.getNumber() << ": "
         << format_hex(MBHI.getMBBHash(MBB), 16) << "\n";
    }
    return false;
  }
};
char MachineBlockHashInfoPrinter::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(MachineBlockHashInfo, "machine-block-hash",
                "Machine Block Hash Analysis", true, true)

INITIALIZE_PASS_BEGIN(MachineBlockHashInfoPrinter, "print-machine-block-hash",
                      "Machine Block Hash Info Printer", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBlockHashInfo)
INITIALIZE_PASS_END(MachineBlockHashInfoPrinter, "print-machine-block-hash",
                    "Machine Block Hash Info Printer", true, true)

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
      Hash = hashing::detail::hash_16_bytes(Hash, SuccHash);
    }
    // Append hashes of predecessors
    for (const MachineBasicBlock *PredMBB : MBB.predecessors()) {
      uint64_t PredHash = HashInfos[PredMBB].OpcodeHash;
      Hash = hashing::detail::hash_16_bytes(Hash, PredHash);
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

  return false;
}

uint64_t MachineBlockHashInfo::getMBBHash(const MachineBasicBlock &MBB) {
  return MBBHashInfo[&MBB];
}

MachineFunctionPass *llvm::createMachineBlockHashInfoPass() {
  return new MachineBlockHashInfo();
}

MachineFunctionPass *
llvm::createMachineBlockHashInfoPrinterPass(raw_ostream &OS) {
  return new MachineBlockHashInfoPrinter(OS);
}