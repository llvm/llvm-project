//===- Utils.h - IR2Vec/MIR2Vec Tool Classes ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the IR2VecTool and MIR2VecTool class definitions for
/// the llvm-ir2vec embedding generation tool.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_UTILS_UTILS_H
#define LLVM_TOOLS_LLVM_IR2VEC_UTILS_UTILS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>
#include <vector>

#define DEBUG_TYPE "ir2vec"

namespace llvm {

/// Tool name for error reporting
static const char *ToolName = "llvm-ir2vec";

/// Specifies the granularity at which embeddings are generated.
enum EmbeddingLevel {
  InstructionLevel, // Generate instruction-level embeddings
  BasicBlockLevel,  // Generate basic block-level embeddings
  FunctionLevel     // Generate function-level embeddings
};

/// Represents a single knowledge graph triplet (Head, Relation, Tail)
/// where indices reference entities in an EntityList
struct Triplet {
  unsigned Head = 0;     ///< Index of the head entity in the entity list
  unsigned Tail = 0;     ///< Index of the tail entity in the entity list
  unsigned Relation = 0; ///< Relation type (see RelationType enum)
};

/// Result structure containing all generated triplets and metadata
struct TripletResult {
  unsigned MaxRelation =
      0; ///< Highest relation index used (for ArgRelation + N)
  std::vector<Triplet> Triplets; ///< Collection of all generated triplets
};

/// Entity mappings: [entity_name]
using EntityList = std::vector<std::string>;

namespace ir2vec {

/// Relation types for triplet generation
enum RelationType {
  TypeRelation = 0, ///< Instruction to type relationship
  NextRelation = 1, ///< Sequential instruction relationship
  ArgRelation = 2   ///< Instruction to operand relationship (ArgRelation + N)
};

/// Helper class for collecting IR triplets and generating embeddings
class IR2VecTool {
private:
  Module &M;
  ModuleAnalysisManager MAM;
  const Vocabulary *Vocab = nullptr;

public:
  explicit IR2VecTool(Module &M) : M(M) {}

  /// Initialize the IR2Vec vocabulary analysis
  bool initializeVocabulary();

  /// Generate triplets for a single function
  /// Returns a TripletResult with:
  ///   - Triplets: vector of all (subject, object, relation) tuples
  ///   - MaxRelation: highest Arg relation ID used, or NextRelation if none
  TripletResult generateTriplets(const Function &F) const;

  /// Get triplets for the entire module
  TripletResult generateTriplets() const;

  /// Collect triplets for the module and dump output to stream
  /// Output format: MAX_RELATION=N header followed by relationships
  void writeTripletsToStream(raw_ostream &OS) const;

  /// Generate entity mappings for the entire vocabulary
  /// Returns EntityList containing all entity strings
  static EntityList collectEntityMappings();

  /// Dump entity ID to string mappings
  static void writeEntitiesToStream(raw_ostream &OS);

  /// Generate embeddings for the entire module
  void writeEmbeddingsToStream(raw_ostream &OS, EmbeddingLevel Level) const;

  /// Generate embeddings for a single function
  void writeEmbeddingsToStream(const Function &F, raw_ostream &OS,
                               EmbeddingLevel Level) const;
};

} // namespace ir2vec

namespace mir2vec {

/// Relation types for MIR2Vec triplet generation
enum MIRRelationType {
  MIRNextRelation = 0, ///< Sequential instruction relationship
  MIRArgRelation = 1 ///< Instruction to operand relationship (ArgRelation + N)
};

/// Helper class for MIR2Vec embedding generation
class MIR2VecTool {
private:
  MachineModuleInfo &MMI;
  std::unique_ptr<MIRVocabulary> Vocab;

public:
  explicit MIR2VecTool(MachineModuleInfo &MMI) : MMI(MMI) {}

  /// Initialize MIR2Vec vocabulary from file (for embeddings generation)
  bool initializeVocabulary(const Module &M);

  /// Initialize vocabulary with layout information only.
  /// This creates a minimal vocabulary with correct layout but no actual
  /// embeddings. Sufficient for generating training data and entity mappings.
  ///
  /// Note: Requires target-specific information from the first machine function
  /// to determine the vocabulary layout (number of opcodes, register classes).
  ///
  /// FIXME: Use --target option to get target info directly, avoiding the need
  /// to parse machine functions for pre-training operations.
  bool initializeVocabularyForLayout(const Module &M);

  /// Get triplets for a single machine function
  /// Returns TripletResult containing MaxRelation and vector of Triplets
  TripletResult generateTriplets(const MachineFunction &MF) const;

  /// Get triplets for the entire module
  /// Returns TripletResult containing aggregated MaxRelation and all Triplets
  TripletResult generateTriplets(const Module &M) const;

  /// Collect triplets for the module and write to output stream
  /// Output format: MAX_RELATION=N header followed by relationships
  void writeTripletsToStream(const Module &M, raw_ostream &OS) const;

  /// Generate entity mappings for the entire vocabulary
  EntityList collectEntityMappings() const;

  /// Generate entity mappings and write to output stream
  void writeEntitiesToStream(raw_ostream &OS) const;

  /// Generate embeddings for all machine functions in the module
  void writeEmbeddingsToStream(const Module &M, raw_ostream &OS,
                               EmbeddingLevel Level) const;

  /// Generate embeddings for a specific machine function
  void writeEmbeddingsToStream(MachineFunction &MF, raw_ostream &OS,
                               EmbeddingLevel Level) const;

  /// Get the MIR vocabulary instance
  const MIRVocabulary *getVocabulary() const { return Vocab.get(); }
};

/// Helper structure to hold MIR context
struct MIRContext {
  LLVMContext Context; // CRITICAL: Must be first for proper destruction order
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<TargetMachine> TM;
};

} // namespace mir2vec

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_IR2VEC_UTILS_UTILS_H
