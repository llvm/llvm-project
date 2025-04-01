//===- IR2VecAnalysis.h - IR2Vec Analysis Implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of IR2VecAnalysis that computes
/// IR2Vec Embeddings of the program.
///
/// Program Embeddings are typically or derived-from a learned
/// representation of the program. Such embeddings are used to represent the
/// programs as input to machine learning algorithms. IR2Vec represents the
/// LLVM IR as embeddings.
///
/// The IR2Vec algorithm is described in the following paper:
///
///   IR2Vec: LLVM IR Based Scalable Program Embeddings, S. VenkataKeerthy,
///   Rohit Aggarwal, Shalini Jain, Maunendra Sankar Desarkar, Ramakrishna
///   Upadrasta, and Y. N. Srikant, ACM Transactions on Architecture and
///   Code Optimization (TACO), 2020. https://doi.org/10.1145/3418463.
///   https://arxiv.org/abs/1909.06228
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IR2VECANALYSIS_H
#define LLVM_ANALYSIS_IR2VECANALYSIS_H

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/PassManager.h"
#include <map>

namespace llvm {

class Module;
class BasicBlock;
class Instruction;
class Function;

namespace ir2vec {
using Embedding = std::vector<double>;
// ToDo: Current the keys are strings. This can be changed to
// use integers for cheaper lookups.
using Vocab = std::map<std::string, Embedding>;
} // namespace ir2vec

class VocabResult;
class IR2VecResult;

/// This analysis provides the vocabulary for IR2Vec. The vocabulary provides a
/// mapping between an entity of the IR (like opcode, type, argument, etc.) and
/// its corresponding embedding.
class VocabAnalysis : public AnalysisInfoMixin<VocabAnalysis> {
  unsigned DIM = 0;
  ir2vec::Vocab Vocabulary;
  Error readVocabulary();

public:
  static AnalysisKey Key;
  VocabAnalysis() = default;
  using Result = VocabResult;
  Result run(Module &M, ModuleAnalysisManager &MAM);
};

class VocabResult {
  ir2vec::Vocab Vocabulary;
  bool Valid = false;
  unsigned DIM = 0;

public:
  VocabResult() = default;
  VocabResult(const ir2vec::Vocab &Vocabulary, unsigned Dim);

  // Helper functions
  bool isValid() const { return Valid; }
  const ir2vec::Vocab &getVocabulary() const;
  unsigned getDimension() const { return DIM; }
  bool invalidate(Module &M, const PreservedAnalyses &PA,
                  ModuleAnalysisManager::Invalidator &Inv);
};

class IR2VecResult {
  SmallMapVector<const Instruction *, ir2vec::Embedding, 128> InstVecMap;
  SmallMapVector<const BasicBlock *, ir2vec::Embedding, 16> BBVecMap;
  ir2vec::Embedding FuncVector;
  unsigned DIM = 0;
  bool Valid = false;

public:
  IR2VecResult() = default;
  IR2VecResult(
      SmallMapVector<const Instruction *, ir2vec::Embedding, 128> InstMap,
      SmallMapVector<const BasicBlock *, ir2vec::Embedding, 16> BBMap,
      const ir2vec::Embedding &FuncVector, unsigned Dim);
  bool isValid() const { return Valid; }

  const SmallMapVector<const Instruction *, ir2vec::Embedding, 128> &
  getInstVecMap() const;
  const SmallMapVector<const BasicBlock *, ir2vec::Embedding, 16> &
  getBBVecMap() const;
  const ir2vec::Embedding &getFunctionVector() const;
  unsigned getDimension() const;
};

/// This analysis provides the IR2Vec embeddings for instructions, basic blocks,
/// and functions.
class IR2VecAnalysis : public AnalysisInfoMixin<IR2VecAnalysis> {
  bool Avg;
  float WO = 1, WT = 0.5, WA = 0.2;

public:
  IR2VecAnalysis() = default;
  static AnalysisKey Key;
  using Result = IR2VecResult;
  Result run(Function &F, FunctionAnalysisManager &FAM);
};

/// This pass prints the IR2Vec embeddings for instructions, basic blocks, and
/// functions.
class IR2VecPrinterPass : public PassInfoMixin<IR2VecPrinterPass> {
  raw_ostream &OS;
  void printVector(const ir2vec::Embedding &Vec) const;

public:
  explicit IR2VecPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_ANALYSIS_IR2VECANALYSIS_H
