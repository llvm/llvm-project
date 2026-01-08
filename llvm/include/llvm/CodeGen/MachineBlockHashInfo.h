//===- llvm/CodeGen/MachineBlockHashInfo.h ----------------------*- C++ -*-===//
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

#ifndef LLVM_CODEGEN_MACHINEBLOCKHASHINFO_H
#define LLVM_CODEGEN_MACHINEBLOCKHASHINFO_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

/// An object wrapping several components of a basic block hash. The combined
/// (blended) hash is represented and stored as one uint64_t, while individual
/// components are of smaller size (e.g., uint16_t or uint8_t).
struct BlendedBlockHash {
public:
  explicit BlendedBlockHash(uint16_t Offset, uint16_t OpcodeHash,
                            uint16_t InstrHash, uint16_t NeighborHash)
      : Offset(Offset), OpcodeHash(OpcodeHash), InstrHash(InstrHash),
        NeighborHash(NeighborHash) {}

  explicit BlendedBlockHash(uint64_t CombinedHash) {
    Offset = CombinedHash & 0xffff;
    CombinedHash >>= 16;
    OpcodeHash = CombinedHash & 0xffff;
    CombinedHash >>= 16;
    InstrHash = CombinedHash & 0xffff;
    CombinedHash >>= 16;
    NeighborHash = CombinedHash & 0xffff;
  }

  /// Combine the blended hash into uint64_t.
  uint64_t combine() const {
    uint64_t Hash = 0;
    Hash |= uint64_t(NeighborHash);
    Hash <<= 16;
    Hash |= uint64_t(InstrHash);
    Hash <<= 16;
    Hash |= uint64_t(OpcodeHash);
    Hash <<= 16;
    Hash |= uint64_t(Offset);
    return Hash;
  }

  /// Compute a distance between two given blended hashes. The smaller the
  /// distance, the more similar two blocks are. For identical basic blocks,
  /// the distance is zero.
  /// Since OpcodeHash is highly stable, we consider a match good only if
  /// the OpcodeHashes are identical. Mismatched OpcodeHashes lead to low
  /// matching accuracy, and poor matches undermine the quality of final
  /// inference. Notably, during inference, we also consider the matching
  /// ratio of basic blocks. For MachineFunctions with a low matching
  /// ratio, we directly skip optimization to reduce the impact of
  /// mismatches. This ensures even very poor profiles won’t cause negative
  /// optimization.
  /// In the context of matching, we consider NeighborHash to be more
  /// important. This is especially true when accounting for inlining
  /// scenarios, where the position of a basic block in the control
  /// flow graph is more critical.
  uint64_t distance(const BlendedBlockHash &BBH) const {
    assert(OpcodeHash == BBH.OpcodeHash &&
           "incorrect blended hash distance computation");
    uint64_t Dist = 0;
    // Account for NeighborHash
    Dist += NeighborHash == BBH.NeighborHash ? 0 : 1;
    Dist <<= 16;
    // Account for InstrHash
    Dist += InstrHash == BBH.InstrHash ? 0 : 1;
    Dist <<= 16;
    // Account for Offset
    Dist += (Offset >= BBH.Offset ? Offset - BBH.Offset : BBH.Offset - Offset);
    return Dist;
  }

private:
  /// The offset of the basic block from the function start.
  uint16_t Offset{0};
  /// Hash of the basic block instructions, excluding operands.
  uint16_t OpcodeHash{0};
  /// Hash of the basic block instructions, including opcodes and
  /// operands.
  uint16_t InstrHash{0};
  /// OpcodeHash of the basic block together with OpcodeHashes of its
  /// successors and predecessors.
  uint16_t NeighborHash{0};
};

class MachineBlockHashInfo : public MachineFunctionPass {
  DenseMap<const MachineBasicBlock *, uint64_t> MBBHashInfo;

public:
  static char ID;
  MachineBlockHashInfo();

  StringRef getPassName() const override { return "Basic Block Hash Compute"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &F) override;

  uint64_t getMBBHash(const MachineBasicBlock &MBB);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEBLOCKHASHINFO_H
