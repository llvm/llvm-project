//===- MIR2Vec.h - Implementation of MIR2Vec ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the MIR2Vec framework for generating Machine IR
/// embeddings.
///
/// Design Overview:
/// ----------------------
/// 1. MIR2VecVocabProvider - Core vocabulary loading logic (no PM dependency)
///    - Can be used standalone or wrapped by the pass manager
///    - Requires MachineModuleInfo with parsed machine functions
///
/// 2. MIR2VecVocabLegacyAnalysis - Pass manager wrapper (ImmutablePass)
///    - Integrated and used by llc -print-mir2vec
///
/// 3. MIREmbedder - Generates embeddings from vocabulary
///    - SymbolicMIREmbedder: MIR2Vec embedding implementation
///
/// MIR2Vec extends IR2Vec to support Machine IR embeddings. It represents the
/// LLVM Machine IR as embeddings which can be used as input to machine learning
/// algorithms.
///
/// The original idea of MIR2Vec is described in the following paper:
///
/// RL4ReAl: Reinforcement Learning for Register Allocation. S. VenkataKeerthy,
/// Siddharth Jain, Anilava Kundu, Rohit Aggarwal, Albert Cohen, and Ramakrishna
/// Upadrasta. 2023. RL4ReAl: Reinforcement Learning for Register Allocation.
/// Proceedings of the 32nd ACM SIGPLAN International Conference on Compiler
/// Construction (CC 2023). https://doi.org/10.1145/3578360.3580273.
/// https://arxiv.org/abs/2204.02013
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MIR2VEC_H
#define LLVM_CODEGEN_MIR2VEC_H

#include "llvm/Analysis/IR2Vec.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace llvm {

class Module;
class raw_ostream;
class MIR2VecVocabLegacyAnalysis;
class TargetInstrInfo;

enum class MIR2VecKind { Symbolic };

namespace mir2vec {

// Forward declarations
class MIREmbedder;
class SymbolicMIREmbedder;

LLVM_ABI extern llvm::cl::OptionCategory MIR2VecCategory;

using Embedding = ir2vec::Embedding;
using MachineInstEmbeddingsMap = DenseMap<const MachineInstr *, Embedding>;
using MachineBlockEmbeddingsMap =
    DenseMap<const MachineBasicBlock *, Embedding>;

/// Class for storing and accessing the MIR2Vec vocabulary.
/// The MIRVocabulary class manages seed embeddings for LLVM Machine IR
class MIRVocabulary {
  friend class llvm::MIR2VecVocabLegacyAnalysis;
  using VocabMap = std::map<std::string, ir2vec::Embedding>;

  // MIRVocabulary Layout:
  // +-------------------+-----------------------------------------------------+
  // | Entity Type       | Description                                         |
  // +-------------------+-----------------------------------------------------+
  // | 1. Opcodes        | Target specific opcodes derived from TII, grouped   |
  // |                   | by instruction semantics.                           |
  // | 2. Common Operands| All common operand types, except register operands, |
  // |                   | defined by MachineOperand::MachineOperandType enum. |
  // | 3. Physical       | Register classes defined by the target, specialized |
  // |    Reg classes    | by physical registers.                              |
  // | 4. Virtual        | Register classes defined by the target, specialized |
  // |    Reg classes    | by virtual and physical registers.                  |
  // +-------------------+-----------------------------------------------------+
  // Both register sections end with a NoRegClass entry for classless registers.

  /// Layout information for the MIR vocabulary. Defines the starting index
  /// and size of each section in the vocabulary.
  struct {
    size_t OpcodeBase = 0;
    size_t CommonOperandBase = 0;
    size_t PhyRegBase = 0;
    size_t VirtRegBase = 0;
    size_t TotalEntries = 0;
  } Layout;

  // TODO: See if we can have only one reg classes section instead of physical
  // and virtual separate sections in the vocabulary. This would reduce the
  // number of vocabulary entities significantly.
  // We can potentially distinguish physical and virtual registers by
  // considering them as a separate feature.
  enum class Section : unsigned {
    Opcodes = 0,
    CommonOperands = 1,
    PhyRegisters = 2,
    VirtRegisters = 3,
    MaxSections
  };

  ir2vec::VocabStorage Storage;
  /// Sorted unique base opcode names. The position is the canonical index.
  SmallVector<std::string, 0> UniqueBaseOpcodeNames;
  SmallVector<unsigned, 0> OpcodeToCanonicalIndex;
  SmallVector<std::string, 24> RegisterOperandNames;
  static constexpr StringLiteral NoRegClassName = "NoRegClass";

  // Some instructions have optional register operands that may be NoRegister.
  // We return a zero vector in such cases.
  Embedding ZeroEmbedding;

  // MO_Register is handled by the register sections. Operand types without an
  // entry, such as MO_DbgInstrRef, get a zero embedding.
  struct CommonOperandEntry {
    MachineOperand::MachineOperandType Type;
    StringLiteral Name;
  };
  static constexpr CommonOperandEntry CommonOperands[] = {
      {MachineOperand::MO_Immediate, "Immediate"},
      {MachineOperand::MO_CImmediate, "CImmediate"},
      {MachineOperand::MO_FPImmediate, "FPImmediate"},
      {MachineOperand::MO_MachineBasicBlock, "MBB"},
      {MachineOperand::MO_FrameIndex, "FrameIndex"},
      {MachineOperand::MO_ConstantPoolIndex, "ConstantPoolIndex"},
      {MachineOperand::MO_TargetIndex, "TargetIndex"},
      {MachineOperand::MO_JumpTableIndex, "JumpTableIndex"},
      {MachineOperand::MO_ExternalSymbol, "ExternalSymbol"},
      {MachineOperand::MO_GlobalAddress, "GlobalAddress"},
      {MachineOperand::MO_BlockAddress, "BlockAddress"},
      {MachineOperand::MO_RegisterMask, "RegisterMask"},
      {MachineOperand::MO_RegisterLiveOut, "RegisterLiveOut"},
      {MachineOperand::MO_Metadata, "Metadata"},
      {MachineOperand::MO_MCSymbol, "MCSymbol"},
      {MachineOperand::MO_CFIIndex, "CFIIndex"},
      {MachineOperand::MO_IntrinsicID, "IntrinsicID"},
      {MachineOperand::MO_Predicate, "Predicate"},
      {MachineOperand::MO_ShuffleMask, "ShuffleMask"},
      {MachineOperand::MO_LaneMask, "LaneMask"}};

  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  void generateStorage(const VocabMap &OpcodeMap,
                       const VocabMap &CommonOperandMap,
                       const VocabMap &PhyRegMap, const VocabMap &VirtRegMap);
  void buildCanonicalOpcodeMapping();
  void buildRegisterOperandMapping();

  /// Get canonical index for a machine opcode
  LLVM_ABI unsigned getCanonicalOpcodeIndex(unsigned Opcode) const;

  /// Get index for a common (non-register) machine operand. Returns
  /// std::nullopt if OperandType has no vocabulary entry.
  LLVM_ABI std::optional<unsigned>
  getCommonOperandIndex(MachineOperand::MachineOperandType OperandType) const;

  /// Get index for a register machine operand. Registers without a register
  /// class map to the NoRegClass entry.
  LLVM_ABI unsigned
  getRegisterOperandIndex(Register Reg, const MachineRegisterInfo &MRI) const;

  // Accessors for operand types
  const Embedding &
  operator[](MachineOperand::MachineOperandType OperandType) const {
    std::optional<unsigned> LocalIndex = getCommonOperandIndex(OperandType);
    if (!LocalIndex)
      return ZeroEmbedding;
    return Storage[static_cast<unsigned>(Section::CommonOperands)][*LocalIndex];
  }

  const Embedding &getRegisterEmbedding(Register Reg,
                                        const MachineRegisterInfo &MRI) const {
    // Reg is sometimes NoRegister (0) for optional operands. We return a zero
    // vector in this case.
    if (!Reg.isValid())
      return ZeroEmbedding;
    // TODO: Implement proper stack slot handling for MIR2Vec embeddings.
    // Stack slots represent frame indices and should have their own
    // embedding strategy rather than defaulting to register class 0.
    // Consider: 1) Separate vocabulary section for stack slots
    //          2) Stack slot size/alignment based embeddings
    //          3) Frame index based categorization
    if (Reg.isStack())
      return ZeroEmbedding;

    auto SectionID =
        Reg.isPhysical() ? Section::PhyRegisters : Section::VirtRegisters;
    return Storage[static_cast<unsigned>(SectionID)]
                  [getRegisterOperandIndex(Reg, MRI)];
  }

  /// Get entity ID (flat index) for a common operand type
  /// This is used for triplet generation
  unsigned getEntityIDForCommonOperand(
      MachineOperand::MachineOperandType OperandType) const {
    std::optional<unsigned> LocalIndex = getCommonOperandIndex(OperandType);
    assert(LocalIndex && "Operand type has no vocabulary entry");
    return Layout.CommonOperandBase + *LocalIndex;
  }

  /// Get entity ID (flat index) for a register
  /// This is used for triplet generation. Reg must be a physical or virtual
  /// register.
  unsigned getEntityIDForRegister(Register Reg,
                                  const MachineRegisterInfo &MRI) const {
    size_t BaseOffset =
        Reg.isPhysical() ? Layout.PhyRegBase : Layout.VirtRegBase;
    return BaseOffset + getRegisterOperandIndex(Reg, MRI);
  }

public:
  /// Static method for extracting base opcode names (public for testing)
  LLVM_ABI static std::string extractBaseOpcodeName(StringRef InstrName);

  /// Get indices from opcode or operand names. These are public for testing.
  /// String based lookups are inefficient and should be avoided in general.
  LLVM_ABI unsigned getCanonicalIndexForBaseName(StringRef BaseName) const;
  LLVM_ABI unsigned
  getCanonicalIndexForOperandName(StringRef OperandName) const;
  LLVM_ABI unsigned
  getCanonicalIndexForRegisterClass(StringRef RegName,
                                    bool IsPhysical = true) const;

  /// Get the string key for a vocabulary entry at the given position
  LLVM_ABI std::string getStringKey(unsigned Pos) const;

  unsigned getDimension() const { return Storage.getDimension(); }

  /// Get entity ID (flat index) for an opcode
  /// This is used for triplet generation
  unsigned getEntityIDForOpcode(unsigned Opcode) const {
    return Layout.OpcodeBase + getCanonicalOpcodeIndex(Opcode);
  }

  /// Get entity ID (flat index) for a machine operand
  /// This is used for triplet generation
  unsigned getEntityIDForMachineOperand(const MachineOperand &MO,
                                        const MachineRegisterInfo &MRI) const {
    if (MO.getType() == MachineOperand::MO_Register)
      return getEntityIDForRegister(MO.getReg(), MRI);
    return getEntityIDForCommonOperand(MO.getType());
  }

  // Accessor methods
  const Embedding &operator[](unsigned Opcode) const {
    unsigned LocalIndex = getCanonicalOpcodeIndex(Opcode);
    return Storage[static_cast<unsigned>(Section::Opcodes)][LocalIndex];
  }

  const Embedding &getOperandEmbedding(const MachineOperand &Operand,
                                       const MachineRegisterInfo &MRI) const {
    auto OperandType = Operand.getType();
    if (OperandType == MachineOperand::MO_Register)
      return getRegisterEmbedding(Operand.getReg(), MRI);
    else
      return operator[](OperandType);
  }

  // Iterator access
  using const_iterator = ir2vec::VocabStorage::const_iterator;
  const_iterator begin() const { return Storage.begin(); }

  const_iterator end() const { return Storage.end(); }

  MIRVocabulary() = delete;

  /// Factory method to create MIRVocabulary from vocabulary map
  LLVM_ABI static Expected<MIRVocabulary>
  create(VocabMap &&OpcMap, VocabMap &&CommonOperandsMap, VocabMap &&PhyRegMap,
         VocabMap &&VirtRegMap, const TargetInstrInfo &TII,
         const TargetRegisterInfo &TRI);

  /// Create a dummy vocabulary for testing purposes.
  LLVM_ABI static Expected<MIRVocabulary>
  createDummyVocabForTest(const TargetInstrInfo &TII,
                          const TargetRegisterInfo &TRI, unsigned Dim = 1);

  /// Total number of entries in the vocabulary
  size_t getCanonicalSize() const { return Storage.size(); }

private:
  MIRVocabulary(VocabMap &&OpcMap, VocabMap &&CommonOperandsMap,
                VocabMap &&PhyRegMap, VocabMap &&VirtRegMap,
                const TargetInstrInfo &TII, const TargetRegisterInfo &TRI);
};

/// Base class for MIR embedders
class MIREmbedder {
protected:
  const MachineFunction &MF;
  const MIRVocabulary &Vocab;

  /// Dimension of the embeddings; Captured from the vocabulary
  const unsigned Dimension;

  /// Weight for opcode embeddings
  const float OpcWeight, CommonOperandWeight, RegOperandWeight;

  LLVM_ABI MIREmbedder(const MachineFunction &MF, const MIRVocabulary &Vocab);

  /// Function to compute embeddings.
  LLVM_ABI Embedding computeEmbeddings() const;

  /// Function to compute the embedding for a given machine basic block.
  LLVM_ABI Embedding computeEmbeddings(const MachineBasicBlock &MBB) const;

  /// Function to compute the embedding for a given machine instruction.
  /// Specific to the kind of embeddings being computed.
  virtual Embedding computeEmbeddings(const MachineInstr &MI) const = 0;

public:
  virtual ~MIREmbedder() = default;

  /// Factory method to create an Embedder object of the specified kind
  /// Returns nullptr if the requested kind is not supported.
  LLVM_ABI static std::unique_ptr<MIREmbedder>
  create(MIR2VecKind Mode, const MachineFunction &MF,
         const MIRVocabulary &Vocab);

  /// Computes and returns the embedding for a given machine instruction MI in
  /// the machine function MF.
  Embedding getMInstVector(const MachineInstr &MI) const {
    return computeEmbeddings(MI);
  }

  /// Computes and returns the embedding for a given machine basic block in the
  /// machine function MF.
  Embedding getMBBVector(const MachineBasicBlock &MBB) const {
    return computeEmbeddings(MBB);
  }

  /// Computes and returns the embedding for the current machine function.
  Embedding getMFunctionVector() const {
    // Currently, we always (re)compute the embeddings for the function. This is
    // cheaper than caching the vector.
    return computeEmbeddings();
  }
};

/// Class for computing Symbolic embeddings
/// Symbolic embeddings are constructed based on the entity-level
/// representations obtained from the MIR Vocabulary.
class LLVM_ABI SymbolicMIREmbedder : public MIREmbedder {
private:
  Embedding computeEmbeddings(const MachineInstr &MI) const override;

public:
  SymbolicMIREmbedder(const MachineFunction &F, const MIRVocabulary &Vocab);
  static std::unique_ptr<SymbolicMIREmbedder>
  create(const MachineFunction &MF, const MIRVocabulary &Vocab);
};

} // namespace mir2vec

/// MIR2Vec vocabulary provider used by pass managers and standalone tools.
/// This class encapsulates the core vocabulary loading logic and can be used
/// independently of the pass manager infrastructure. For pass-based usage,
/// see MIR2VecVocabLegacyAnalysis and MIR2VecVocabAnalysis.
class MIR2VecVocabProvider {
public:
  MIR2VecVocabProvider(const MachineModuleInfo &MMI) : MMI(MMI) {}

  LLVM_ABI Expected<mir2vec::MIRVocabulary> getVocabulary(const Module &M);

private:
  const MachineModuleInfo &MMI;
};

/// Pass to analyze and populate MIR2Vec vocabulary from a module
class LLVM_ABI MIR2VecVocabLegacyAnalysis : public ImmutablePass {
  using VocabMap = std::map<std::string, mir2vec::Embedding>;

  StringRef getPassName() const override;

protected:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.setPreservesAll();
  }
  std::optional<mir2vec::MIRVocabulary> Vocab;

public:
  static char ID;
  MIR2VecVocabLegacyAnalysis() : ImmutablePass(ID) {}

  /// Build the vocabulary on the first call and return the cached one after.
  Expected<const mir2vec::MIRVocabulary &>
  getMIR2VecVocabulary(const Module &M) {
    if (!Vocab) {
      MIR2VecVocabProvider Provider(
          getAnalysis<MachineModuleInfoWrapperPass>().getMMI());
      auto VocabOrErr = Provider.getVocabulary(M);
      if (!VocabOrErr)
        return VocabOrErr.takeError();
      Vocab.emplace(std::move(*VocabOrErr));
    }
    return *Vocab;
  }
};

/// This pass prints the embeddings in the MIR2Vec vocabulary
class LLVM_ABI MIR2VecVocabPrinterLegacyPass : public MachineFunctionPass {
  raw_ostream &OS;

public:
  static char ID;
  explicit MIR2VecVocabPrinterLegacyPass(raw_ostream &OS)
      : MachineFunctionPass(ID), OS(OS) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  bool doFinalization(Module &M) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MIR2VecVocabLegacyAnalysis>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "MIR2Vec Vocabulary Printer Pass";
  }
};

/// This pass prints the MIR2Vec embeddings for machine functions, basic blocks,
/// and instructions
class LLVM_ABI MIR2VecPrinterLegacyPass : public MachineFunctionPass {
  raw_ostream &OS;

public:
  static char ID;
  explicit MIR2VecPrinterLegacyPass(raw_ostream &OS)
      : MachineFunctionPass(ID), OS(OS) {}

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MIR2VecVocabLegacyAnalysis>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "MIR2Vec Embedder Printer Pass";
  }
};

/// Create a machine pass that prints MIR2Vec embeddings
LLVM_ABI MachineFunctionPass *createMIR2VecPrinterLegacyPass(raw_ostream &OS);

/// New pass manager analysis that provides the MIR2Vec vocabulary. The
/// vocabulary is built on the first request and shared by all machine
/// functions.
class MIR2VecVocabAnalysis : public AnalysisInfoMixin<MIR2VecVocabAnalysis> {
  friend AnalysisInfoMixin<MIR2VecVocabAnalysis>;
  LLVM_ABI static AnalysisKey Key;
  std::shared_ptr<const mir2vec::MIRVocabulary> Vocab;

public:
  using Result = std::shared_ptr<const mir2vec::MIRVocabulary>;
  LLVM_ABI Result run(MachineFunction &MF,
                      MachineFunctionAnalysisManager &MFAM);
};

/// New pass manager pass that prints the MIR2Vec embeddings for machine
/// functions, basic blocks, and instructions.
class MIR2VecPrinterPass : public RequiredPassInfoMixin<MIR2VecPrinterPass> {
  raw_ostream &OS;

public:
  explicit MIR2VecPrinterPass(raw_ostream &OS) : OS(OS) {}
  LLVM_ABI PreservedAnalyses run(MachineFunction &MF,
                                 MachineFunctionAnalysisManager &MFAM);
};

} // namespace llvm

#endif // LLVM_CODEGEN_MIR2VEC_H
