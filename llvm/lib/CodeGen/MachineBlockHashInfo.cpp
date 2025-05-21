#include "llvm/CodeGen/MachineBlockHashInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

using OperandHashFuncTy = function_ref<uint64_t(uint64_t &, const MachineOperand &)>;

uint64_t hashBlock(const MachineBasicBlock &MBB, OperandHashFuncTy OperandHashFunc) {
  uint64_t Hash = 0;
  for (const MachineInstr &MI : MBB) {
    if (MI.isPseudo())
      continue;
    // Ignore unconditional jumps
    if (MI.isUnconditionalBranch())
      continue;
    Hash = hashing::detail::hash_16_bytes(Hash, MI.getOpcode());
    for (unsigned i = 0; i < MI.getNumOperands(); i++) {
      Hash = OperandHashFunc(Hash, MI.getOperand(i));
    }
  }
  return Hash;
}

/// Hashing a 64-bit integer to a 16-bit one.
uint16_t hash_64_to_16(const uint64_t Hash) {
  uint16_t Res = (uint16_t)(Hash & 0xFFFF);
  Res ^= (uint16_t)((Hash >> 16) & 0xFFFF);
  Res ^= (uint16_t)((Hash >> 32) & 0xFFFF);
  Res ^= (uint16_t)((Hash >> 48) & 0xFFFF);
  return Res;
}

uint64_t hashInstOperand(uint64_t &Hash, const MachineOperand &Operand) {
  return hashing::detail::hash_16_bytes(Hash, hash_value(Operand));
}

INITIALIZE_PASS(MachineBlockHashInfo, "machine-block-hash",
                      "Machine Block Hash Analysis", true, true)

char MachineBlockHashInfo::ID = 0;

MachineBlockHashInfo::MachineBlockHashInfo() : MachineFunctionPass(ID) {
  initializeMachineBlockHashInfoPass(*PassRegistry::getPassRegistry());
}

void MachineBlockHashInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineBlockHashInfo::runOnMachineFunction(MachineFunction &F) {
  DenseMap<MachineBasicBlock *, BlendedBlockHash> BlendedHashes;
  DenseMap<MachineBasicBlock *, uint64_t> OpcodeHashes;
  uint16_t Offset = 0;
  // Initialize hash components
  for (MachineBasicBlock &MBB : F) {
    BlendedBlockHash BlendedHash;
    // offset of the machine basic block
    BlendedHash.Offset = Offset;
    Offset += MBB.size();
    // Hashing opcodes
    uint64_t OpcodeHash = hashBlock(MBB, [](uint64_t &Hash, const MachineOperand &Op) { return Hash; });
    OpcodeHashes[&MBB] = OpcodeHash;
    BlendedHash.OpcodeHash = hash_64_to_16(OpcodeHash);
    // Hash complete instructions
    uint64_t InstrHash = hashBlock(MBB, hashInstOperand);
    BlendedHash.InstrHash = hash_64_to_16(InstrHash);
    BlendedHashes[&MBB] = BlendedHash;
  }

  // Initialize neighbor hash
  for (MachineBasicBlock &MBB : F) {
    uint64_t Hash = OpcodeHashes[&MBB];
    // Append hashes of successors
    for (MachineBasicBlock *SuccMBB : MBB.successors()) {
      uint64_t SuccHash = OpcodeHashes[SuccMBB];
      Hash = hashing::detail::hash_16_bytes(Hash, SuccHash);
    }
    // Append hashes of predecessors
    for (MachineBasicBlock *PredMBB : MBB.predecessors()) {
      uint64_t PredHash = OpcodeHashes[PredMBB];
      Hash = hashing::detail::hash_16_bytes(Hash, PredHash);
    }
    BlendedHashes[&MBB].NeighborHash = hash_64_to_16(Hash);
  }

  // Assign hashes
  for (MachineBasicBlock &MBB : F) {
    if (MBB.getBBID()) {
      MBBHashInfo[MBB.getBBID()->BaseID] = BlendedHashes[&MBB].combine();
    }
  }

  return false;
}

uint64_t MachineBlockHashInfo::getMBBHash(const MachineBasicBlock &MBB) {
  if (MBB.getBBID()) {
    return MBBHashInfo[MBB.getBBID()->BaseID];
  }
  return 0;
}