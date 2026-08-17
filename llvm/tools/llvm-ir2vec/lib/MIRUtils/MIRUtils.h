//===- MIRUtils.h - MIR2Vec Tool Class ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the MIR2VecTool class definition for generating
/// embeddings and triplets from LLVM Machine IR. It has no dependency on
/// the LLVM IR embedding API (IR2VecTool).
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_UTILS_MIRUTILS_H
#define LLVM_TOOLS_LLVM_IR2VEC_UTILS_MIRUTILS_H

#include "Common.h"
#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>

namespace llvm {

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

/// Helper structure to hold MIR context.
/// CRITICAL: Member declaration order matters for correct destruction.
struct MIRContext {
  LLVMContext Context; // Must be first: other members hold references into it
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<TargetMachine> TM;
};

} // namespace mir2vec
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_IR2VEC_UTILS_MIRUTILS_H
