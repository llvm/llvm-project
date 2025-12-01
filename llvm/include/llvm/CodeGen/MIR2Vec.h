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
#include <set>
#include <string>

namespace llvm {

class Module;
class raw_ostream;
class LLVMContext;
class MIR2VecVocabLegacyAnalysis;
class TargetInstrInfo;

enum class MIR2VecKind { Symbolic };

namespace mir2vec {

// Forward declarations
class MIREmbedder;
class SymbolicMIREmbedder;

LLVM_ABI extern llvm::cl::OptionCategory MIR2VecCategory;
LLVM_ABI extern cl::opt<float> OpcWeight, CommonOperandWeight, RegOperandWeight;

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
  std::set<std::string> UniqueBaseOpcodeNames;
  SmallVector<std::string, 24> RegisterOperandNames;

  // Some instructions have optional register operands that may be NoRegister.
  // We return a zero vector in such cases.
  Embedding ZeroEmbedding;

  // We have specialized MO_Register handling in the Register operand section,
  // so we don't include it here. Also, no MO_DbgInstrRef for now.
  static constexpr StringLiteral CommonOperandNames[] = {
      "Immediate",       "CImmediate",        "FPImmediate",  "MBB",
      "FrameIndex",      "ConstantPoolIndex", "TargetIndex",  "JumpTableIndex",
      "ExternalSymbol",  "GlobalAddress",     "BlockAddress", "RegisterMask",
      "RegisterLiveOut", "Metadata",          "MCSymbol",     "CFIIndex",
      "IntrinsicID",     "Predicate",         "ShuffleMask"};
  static_assert(std::size(CommonOperandNames) == MachineOperand::MO_Last - 1 &&
                "Common operand names size changed, update accordingly");

  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  const MachineRegisterInfo &MRI;

  void generateStorage(const VocabMap &OpcodeMap,
                       const VocabMap &CommonOperandMap,
                       const VocabMap &PhyRegMap, const VocabMap &VirtRegMap);
  void buildCanonicalOpcodeMapping();
  void buildRegisterOperandMapping();

  /// Get canonical index for a machine opcode
  LLVM_ABI unsigned getCanonicalOpcodeIndex(unsigned Opcode) const;

  /// Get index for a common (non-register) machine operand
  unsigned
  getCommonOperandIndex(MachineOperand::MachineOperandType OperandType) const;

  /// Get index for a register machine operand
  LLVM_ABI unsigned getRegisterOperandIndex(Register Reg) const;

  // Accessors for operand types
  const Embedding &
  operator[](MachineOperand::MachineOperandType OperandType) const {
    unsigned LocalIndex = getCommonOperandIndex(OperandType);
    return Storage[static_cast<unsigned>(Section::CommonOperands)][LocalIndex];
  }

  const Embedding &operator[](Register Reg) const {
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

    unsigned LocalIndex = getRegisterOperandIndex(Reg);
    auto SectionID =
        Reg.isPhysical() ? Section::PhyRegisters : Section::VirtRegisters;
    return Storage[static_cast<unsigned>(SectionID)][LocalIndex];
  }

  /// Get entity ID (flat index) for a common operand type
  /// This is used for triplet generation
  LLVM_ABI unsigned getEntityIDForCommonOperand(
      MachineOperand::MachineOperandType OperandType) const {
    return Layout.CommonOperandBase + getCommonOperandIndex(OperandType);
  }

  /// Get entity ID (flat index) for a register
  /// This is used for triplet generation
  unsigned getEntityIDForRegister(Register Reg) const {
    if (!Reg.isValid() || Reg.isStack())
      return Layout
          .VirtRegBase; // Return VirtRegBase for invalid/stack registers
    unsigned LocalIndex = getRegisterOperandIndex(Reg);
    size_t BaseOffset =
        Reg.isPhysical() ? Layout.PhyRegBase : Layout.VirtRegBase;
    return BaseOffset + LocalIndex;
  }

public:
  /// Static method for extracting base opcode names (public for testing)
  LLVM_ABI_FOR_TEST static std::string
  extractBaseOpcodeName(StringRef InstrName);

  /// Get indices from opcode or operand names. These are public for testing.
  /// String based lookups are inefficient and should be avoided in general.
  LLVM_ABI_FOR_TEST unsigned
  getCanonicalIndexForBaseName(StringRef BaseName) const;
  LLVM_ABI_FOR_TEST unsigned
  getCanonicalIndexForOperandName(StringRef OperandName) const;
  LLVM_ABI_FOR_TEST unsigned
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
  unsigned getEntityIDForMachineOperand(const MachineOperand &MO) const {
    if (MO.getType() == MachineOperand::MO_Register)
      return getEntityIDForRegister(MO.getReg());
    return getEntityIDForCommonOperand(MO.getType());
  }

  // Accessor methods
  const Embedding &operator[](unsigned Opcode) const {
    unsigned LocalIndex = getCanonicalOpcodeIndex(Opcode);
    return Storage[static_cast<unsigned>(Section::Opcodes)][LocalIndex];
  }

  const Embedding &operator[](MachineOperand Operand) const {
    auto OperandType = Operand.getType();
    if (OperandType == MachineOperand::MO_Register)
      return operator[](Operand.getReg());
    else
      return operator[](OperandType);
  }

  // Iterator access
  using const_iterator = ir2vec::VocabStorage::const_iterator;
  const_iterator begin() const { return Storage.begin(); }

  const_iterator end() const { return Storage.end(); }

  MIRVocabulary() = delete;

  /// Factory method to create MIRVocabulary from vocabulary map
  LLVM_ABI_FOR_TEST static Expected<MIRVocabulary>
  create(VocabMap &&OpcMap, VocabMap &&CommonOperandsMap, VocabMap &&PhyRegMap,
         VocabMap &&VirtRegMap, const TargetInstrInfo &TII,
         const TargetRegisterInfo &TRI, const MachineRegisterInfo &MRI);

  /// Create a dummy vocabulary for testing purposes.
  LLVM_ABI static Expected<MIRVocabulary>
  createDummyVocabForTest(const TargetInstrInfo &TII,
                          const TargetRegisterInfo &TRI,
                          const MachineRegisterInfo &MRI, unsigned Dim = 1);

  /// Total number of entries in the vocabulary
  size_t getCanonicalSize() const { return Storage.size(); }

private:
  MIRVocabulary(VocabMap &&OpcMap, VocabMap &&CommonOperandsMap,
                VocabMap &&PhyRegMap, VocabMap &&VirtRegMap,
                const TargetInstrInfo &TII, const TargetRegisterInfo &TRI,
                const MachineRegisterInfo &MRI);
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

  MIREmbedder(const MachineFunction &MF, const MIRVocabulary &Vocab)
      : MF(MF), Vocab(Vocab), Dimension(Vocab.getDimension()),
        OpcWeight(mir2vec::OpcWeight),
        CommonOperandWeight(mir2vec::CommonOperandWeight),
        RegOperandWeight(mir2vec::RegOperandWeight) {}

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
class SymbolicMIREmbedder : public MIREmbedder {
private:
  Embedding computeEmbeddings(const MachineInstr &MI) const override;

public:
  SymbolicMIREmbedder(const MachineFunction &F, const MIRVocabulary &Vocab);
  LLVM_ABI_FOR_TEST static std::unique_ptr<SymbolicMIREmbedder>
  create(const MachineFunction &MF, const MIRVocabulary &Vocab);
};

} // namespace mir2vec

/// MIR2Vec vocabulary provider used by pass managers and standalone tools.
/// This class encapsulates the core vocabulary loading logic and can be used
/// independently of the pass manager infrastructure. For pass-based usage,
/// see MIR2VecVocabLegacyAnalysis.
///
/// Note: This provider pattern makes new PM migration straightforward when
/// needed. A new PM analysis wrapper can be added that delegates to this
/// provider, similar to how MIR2VecVocabLegacyAnalysis currently wraps it.
class MIR2VecVocabProvider {
  using VocabMap = std::map<std::string, mir2vec::Embedding>;

public:
  MIR2VecVocabProvider(const MachineModuleInfo &MMI) : MMI(MMI) {}

  LLVM_ABI Expected<mir2vec::MIRVocabulary> getVocabulary(const Module &M);

private:
  Error readVocabulary(VocabMap &OpcVocab, VocabMap &CommonOperandVocab,
                       VocabMap &PhyRegVocabMap, VocabMap &VirtRegVocabMap);
  const MachineModuleInfo &MMI;
};

/// Pass to analyze and populate MIR2Vec vocabulary from a module
class MIR2VecVocabLegacyAnalysis : public ImmutablePass {
  using VocabVector = std::vector<mir2vec::Embedding>;
  using VocabMap = std::map<std::string, mir2vec::Embedding>;

  StringRef getPassName() const override;

protected:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.setPreservesAll();
  }
  std::unique_ptr<MIR2VecVocabProvider> Provider;

public:
  static char ID;
  MIR2VecVocabLegacyAnalysis() : ImmutablePass(ID) {}

  Expected<mir2vec::MIRVocabulary> getMIR2VecVocabulary(const Module &M) {
    MachineModuleInfo &MMI =
        getAnalysis<MachineModuleInfoWrapperPass>().getMMI();
    if (!Provider)
      Provider = std::make_unique<MIR2VecVocabProvider>(MMI);
    return Provider->getVocabulary(M);
  }

  MIR2VecVocabProvider &getProvider() {
    assert(Provider && "Provider not initialized");
    return *Provider;
  }
};

/// This pass prints the embeddings in the MIR2Vec vocabulary
class MIR2VecVocabPrinterLegacyPass : public MachineFunctionPass {
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
class MIR2VecPrinterLegacyPass : public MachineFunctionPass {
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

} // namespace llvm

#endif // LLVM_CODEGEN_MIR2VEC_H
