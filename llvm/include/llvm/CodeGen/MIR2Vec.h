//===- MIR2Vec.h - Implementation of MIR2Vec ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the MIR2Vec vocabulary
/// analysis(MIR2VecVocabLegacyAnalysis), the core mir2vec::MIREmbedder
/// interface for generating Machine IR embeddings, and related utilities.
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

namespace mir2vec {
extern llvm::cl::OptionCategory MIR2VecCategory;
extern cl::opt<float> OpcWeight;

using Embedding = ir2vec::Embedding;

/// Class for storing and accessing the MIR2Vec vocabulary.
/// The MIRVocabulary class manages seed embeddings for LLVM Machine IR
class MIRVocabulary {
  friend class llvm::MIR2VecVocabLegacyAnalysis;
  using VocabMap = std::map<std::string, ir2vec::Embedding>;

private:
  // Define vocabulary layout - adapted for MIR
  struct {
    size_t OpcodeBase = 0;
    size_t OperandBase = 0;
    size_t TotalEntries = 0;
  } Layout;

  enum class Section : unsigned { Opcodes = 0, MaxSections };

  ir2vec::VocabStorage Storage;
  mutable std::set<std::string> UniqueBaseOpcodeNames;
  const TargetInstrInfo &TII;
  void generateStorage(const VocabMap &OpcodeMap);
  void buildCanonicalOpcodeMapping();

  /// Get canonical index for a machine opcode
  unsigned getCanonicalOpcodeIndex(unsigned Opcode) const;

public:
  /// Static method for extracting base opcode names (public for testing)
  static std::string extractBaseOpcodeName(StringRef InstrName);

  /// Get canonical index for base name (public for testing)
  unsigned getCanonicalIndexForBaseName(StringRef BaseName) const;

  /// Get the string key for a vocabulary entry at the given position
  std::string getStringKey(unsigned Pos) const;

  unsigned getDimension() const { return Storage.getDimension(); }

  // Accessor methods
  const Embedding &operator[](unsigned Opcode) const {
    unsigned LocalIndex = getCanonicalOpcodeIndex(Opcode);
    return Storage[static_cast<unsigned>(Section::Opcodes)][LocalIndex];
  }

  // Iterator access
  using const_iterator = ir2vec::VocabStorage::const_iterator;
  const_iterator begin() const { return Storage.begin(); }

  const_iterator end() const { return Storage.end(); }

  /// Total number of entries in the vocabulary
  size_t getCanonicalSize() const { return Storage.size(); }

  MIRVocabulary() = delete;

  /// Factory method to create MIRVocabulary from vocabulary map
  static Expected<MIRVocabulary> create(VocabMap &&Entries,
                                        const TargetInstrInfo &TII);

private:
  MIRVocabulary(VocabMap &&Entries, const TargetInstrInfo &TII);
};

} // namespace mir2vec

/// Pass to analyze and populate MIR2Vec vocabulary from a module
class MIR2VecVocabLegacyAnalysis : public ImmutablePass {
  using VocabVector = std::vector<mir2vec::Embedding>;
  using VocabMap = std::map<std::string, mir2vec::Embedding>;
  VocabMap StrVocabMap;
  VocabVector Vocab;

  StringRef getPassName() const override;
  Error readVocabulary();

protected:
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineModuleInfoWrapperPass>();
    AU.setPreservesAll();
  }

public:
  static char ID;
  MIR2VecVocabLegacyAnalysis() : ImmutablePass(ID) {}
  Expected<mir2vec::MIRVocabulary> getMIR2VecVocabulary(const Module &M);
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

} // namespace llvm

#endif // LLVM_CODEGEN_MIR2VEC_H