//===- IR2VecTool.h - IR2Vec Tool Interface ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_IR2VECTOOL_H
#define LLVM_TOOLS_LLVM_IR2VEC_IR2VECTOOL_H

#include "EmbeddingCommon.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include <string>
#include <vector>

namespace llvm {
namespace ir2vec {
/// Relation types for triplet generation
enum RelationType {
  TypeRelation = 0,
  NextRelation = 1,
  ArgRelation = 2
};

/// Core IR2Vec tool
class IR2VecTool {
private:
  Module &M;
  ModuleAnalysisManager MAM;
  const Vocabulary *Vocab = nullptr;

public:
  explicit IR2VecTool(Module &M) : M(M), Vocab(nullptr) {}

  /// Initialize vocabulary
  bool initializeVocabulary();

  /// Generate triplets for module and write to stream
  void generateTriplets(raw_ostream &OS) const;

  /// Generate embeddings for module and write to stream
  void generateEmbeddings(raw_ostream &OS) const;

  /// Generate embeddings for single function and write to stream
  void generateEmbeddings(const Function &F, raw_ostream &OS) const;

  /// Get entity mappings
  static EntityMap getEntityMappings();

  /// Generate entity mappings (static - no module needed)
  static void generateEntityMappings(raw_ostream &OS);

  // Data structure methods for Python bindings

  /// Get triplets
  TripletResult getTriplets() const;

  /// Get triplets for function
  TripletResult getTriplets(const Function &F) const;

  /// Get single function embedding
  std::pair<std::string, std::pair<std::string, Embedding>>
  getFunctionEmbedding(const Function &F) const;

  /// Get function embeddings
  FuncVecMap getFunctionEmbeddings() const;

  /// Get BB embeddings for a specific function
  BBVecList getBBEmbeddings(const Function &F) const;

  /// Get BB embeddings
  BBVecList getBBEmbeddings() const;

  /// Get instruction embeddings for a specific function
  InstVecList getInstEmbeddings(const Function &F) const;

  /// Get instruction embeddings
  InstVecList getInstEmbeddings() const;

  /// Check if vocabulary is valid
  bool isVocabularyValid() const { return Vocab && Vocab->isValid(); }

  Module &getModule() { return M; }
};

} // namespace ir2vec
} // namespace llvm

#endif