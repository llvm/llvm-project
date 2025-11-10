//===- MIR2VecTool.h - MIR2Vec Tool Interface -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the MIR2Vec tool interface for Machine IR embedding
/// generation, triplet generation, and entity mapping operations.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_MIR2VECTOOL_H
#define LLVM_TOOLS_LLVM_IR2VEC_MIR2VECTOOL_H

#include "EmbeddingCommon.h"
#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Errc.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
namespace mir2vec {

/// Relation types for MIR2Vec triplet generation
enum MIRRelationType {
  MIRNextRelation = 0, ///< Sequential instruction relationship
  MIRArgRelation = 1   ///< Instruction to operand relationship (ArgRelation + N)
};

/// Context for MIR parsing and processing
/// Manages lifetime of MIR parser, target machine, and module info
struct MIRContext {
  LLVMContext Context;
  std::unique_ptr<MIRParser> Parser;
  std::unique_ptr<Module> M;
  std::unique_ptr<TargetMachine> TM;
  std::unique_ptr<MachineModuleInfo> MMI;

  MIRContext() = default;

  MIRContext(const MIRContext &) = delete;
  MIRContext &operator=(const MIRContext &) = delete;

  MIRContext(MIRContext &&) = default;
  MIRContext &operator=(MIRContext &&) = default;
};

/// Core MIR2Vec tool for embedding generation
class MIR2VecTool {
private:
  MachineModuleInfo &MMI;
  std::unique_ptr<MIRVocabulary> Vocab;

  /// Generate triplets for a single machine function (internal helper)
  /// Returns the maximum relation ID used in this function
  unsigned generateTripletsForMF(const MachineFunction &MF,
                                 std::vector<Triplet> &Triplets) const;

public:
  explicit MIR2VecTool(MachineModuleInfo &MMI) : MMI(MMI) {}

  /// Initialize MIR2Vec vocabulary from file (for embeddings generation)
  /// This loads a fully trained vocabulary with embeddings.
  bool initializeVocabulary(const Module &M);

  /// Initialize vocabulary with layout information only.
  /// This creates a minimal vocabulary with correct layout but no actual
  /// embeddings. Sufficient for generating training data and entity mappings.
  ///
  /// Note: Requires target-specific information from the first machine function
  /// to determine the vocabulary layout (number of opcodes, register classes).
  bool initializeVocabularyForLayout(const Module &M);

  // ========================================================================
  // Data structure methods (for Python bindings)
  // ========================================================================

  /// Get triplets for the entire module
  TripletResult getTriplets(const Module &M) const;

  /// Get triplets for a single machine function
  TripletResult getTriplets(const MachineFunction &MF) const;

  /// Get entity mappings
  EntityMap getEntityMappings() const;

  /// Get function-level embeddings for all functions in the module
  FuncVecMap getFunctionEmbeddings(const Module &M) const;

  /// Get function-level embedding for a single machine function
  std::pair<std::string, Embedding>
  getFunctionEmbedding(MachineFunction &MF) const;

  /// Get basic block-level embeddings for a machine function
  BBVecList getMBBEmbeddings(MachineFunction &MF) const;

  /// Get instruction-level embeddings for a machine function
  InstVecList getMInstEmbeddings(MachineFunction &MF) const;

  // ========================================================================
  // Stream output methods (for CLI tool)
  // ========================================================================

  /// Generate triplets for the module and write to stream
  /// Output format: MAX_RELATION=N header followed by relationships
  void generateTriplets(const Module &M, raw_ostream &OS) const;

  /// Generate triplets for a single machine function and write to stream
  void generateTriplets(const MachineFunction &MF, raw_ostream &OS) const;

  /// Generate entity mappings and write to stream
  void generateEntityMappings(raw_ostream &OS) const;

  /// Generate embeddings for all machine functions in the module
  void generateEmbeddings(const Module &M, raw_ostream &OS) const;

  /// Generate embeddings for a specific machine function
  void generateEmbeddings(MachineFunction &MF, raw_ostream &OS) const;

  /// Get the vocabulary (for testing/debugging)
  const MIRVocabulary *getVocabulary() const { return Vocab.get(); }

  /// Get the MachineModuleInfo
  MachineModuleInfo &getMachineModuleInfo() { return MMI; }
};

// ========================================================================
// MIR Context Setup Functions
// ========================================================================

/// Initialize target backends (call once at startup)
void initializeTargets();

/// Setup MIR context from input file
/// This parses the MIR file, creates target machine, and parses machine functions
Error setupMIRContext(const std::string &InputFile, MIRContext &Ctx);

} // namespace mir2vec
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_IR2VEC_MIR2VECTOOL_H