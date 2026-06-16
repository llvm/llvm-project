//===- IRUtils.h - IR2Vec Tool Class ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the IR2VecTool class definition for generating
/// embeddings and triplets from LLVM IR. It has no dependency on Machine IR.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_UTILS_IRUTILS_H
#define LLVM_TOOLS_LLVM_IR2VEC_UTILS_IRUTILS_H

#include "Common.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#define DEBUG_TYPE "ir2vec"

namespace llvm {

/// Per-function embedding map: Function* -> Embedding
using FuncEmbMap = DenseMap<const Function *, ir2vec::Embedding>;

namespace ir2vec {

/// Relation types for IR triplet generation
enum RelationType {
  TypeRelation = 0, ///< Instruction to type relationship
  NextRelation = 1, ///< Sequential instruction relationship
  ArgRelation = 2   ///< Instruction to operand relationship (ArgRelation + N)
};

/// Load an IR2Vec vocabulary from a JSON file on disk.
Expected<std::shared_ptr<Vocabulary>> loadVocabulary(StringRef VocabPath);

/// Helper class for collecting IR triplets and generating embeddings
class IR2VecTool {
private:
  Module &M;
  ModuleAnalysisManager MAM;

  /// \note The API around vocab object is not thread-safe.
  /// Specifically, calling setVocabulary() on an instance while
  /// another thread reading the Vocab object with the same instance
  /// can cause a data race on this internal shared_ptr<Vocabulary> member.
  std::shared_ptr<Vocabulary> Vocab;

public:
  explicit IR2VecTool(Module &M) : M(M) {}

  /// Creates the embedding object for downstream embedding streaming
  Expected<std::unique_ptr<Embedder>>
  createIR2VecEmbedder(const Function &F, IR2VecKind Kind) const;

  /// Sets the vocabulary for this tool instance.
  /// This allows sharing the same vocabulary instance across multiple
  /// IR2VecTool instances, which is useful for generating embeddings for
  /// multiple functions without needing to reload the vocabulary each time.
  Error setVocabulary(std::shared_ptr<Vocabulary> V);

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

  /// Get embedding for a single function
  Expected<Embedding> getFunctionEmbedding(const Function &F,
                                           IR2VecKind Kind) const;

  /// Get embeddings for all functions in the module
  Expected<FuncEmbMap> getFunctionEmbeddingsMap(IR2VecKind Kind) const;

  /// Get embeddings for all basic blocks in a function
  Expected<BBEmbeddingsMap> getBBEmbeddingsMap(const Function &F,
                                               IR2VecKind Kind) const;

  /// Get embeddings for all instructions in a function
  Expected<InstEmbeddingsMap> getInstEmbeddingsMap(const Function &F,
                                                   IR2VecKind Kind) const;

  /// Generate embeddings for the entire module
  void writeEmbeddingsToStream(raw_ostream &OS, EmbeddingLevel Level) const;

  /// Generate embeddings for a single function
  void writeEmbeddingsToStream(const Function &F, raw_ostream &OS,
                               EmbeddingLevel Level) const;
};

} // namespace ir2vec
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_IR2VEC_UTILS_IRUTILS_H
