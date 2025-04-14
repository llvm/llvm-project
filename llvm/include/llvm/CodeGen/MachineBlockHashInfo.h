#ifndef LLVM_CODEGEN_MACHINEBLOCKHASHINFO_H
#define LLVM_CODEGEN_MACHINEBLOCKHASHINFO_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

/// An object wrapping several components of a basic block hash. The combined
/// (blended) hash is represented and stored as one uint64_t, while individual
/// components are of smaller size (e.g., uint16_t or uint8_t).
struct BlendedBlockHash {
private:
  static uint64_t combineHashes(uint16_t Hash1, uint16_t Hash2, uint16_t Hash3,
                                uint16_t Hash4) {
    uint64_t Hash = 0;

    Hash |= uint64_t(Hash4);
    Hash <<= 16;

    Hash |= uint64_t(Hash3);
    Hash <<= 16;

    Hash |= uint64_t(Hash2);
    Hash <<= 16;

    Hash |= uint64_t(Hash1);

    return Hash;
  }

  static void parseHashes(uint64_t Hash, uint16_t &Hash1, uint16_t &Hash2,
                          uint16_t &Hash3, uint16_t &Hash4) {
    Hash1 = Hash & 0xffff;
    Hash >>= 16;

    Hash2 = Hash & 0xffff;
    Hash >>= 16;

    Hash3 = Hash & 0xffff;
    Hash >>= 16;

    Hash4 = Hash & 0xffff;
    Hash >>= 16;
  }

public:
  explicit BlendedBlockHash() {}

  explicit BlendedBlockHash(uint64_t CombinedHash) {
    parseHashes(CombinedHash, Offset, OpcodeHash, InstrHash, NeighborHash);
  }

  /// Combine the blended hash into uint64_t.
  uint64_t combine() const {
    return combineHashes(Offset, OpcodeHash, InstrHash, NeighborHash);
  }

  /// Compute a distance between two given blended hashes. The smaller the
  /// distance, the more similar two blocks are. For identical basic blocks,
  /// the distance is zero.
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

  /// The offset of the basic block from the function start.
  uint16_t Offset{0};
  /// (Loose) Hash of the basic block instructions, excluding operands.
  uint16_t OpcodeHash{0};
  /// (Strong) Hash of the basic block instructions, including opcodes and
  /// operands.
  uint16_t InstrHash{0};
  /// Hash of the (loose) basic block together with (loose) hashes of its
  /// successors and predecessors.
  uint16_t NeighborHash{0};
};

class MachineBlockHashInfo : public MachineFunctionPass {
  DenseMap<unsigned, uint64_t> MBBHashInfo;

public:
  static char ID;
  MachineBlockHashInfo();

  StringRef getPassName() const override {
    return "Basic Block Hash Compute";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &F) override;

  uint64_t getMBBHash(const MachineBasicBlock &MBB);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEBLOCKHASHINFO_H