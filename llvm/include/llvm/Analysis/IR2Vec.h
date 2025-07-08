//===- IR2Vec.h - Implementation of IR2Vec ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the IR2Vec vocabulary analysis(IR2VecVocabAnalysis),
/// the core ir2vec::Embedder interface for generating IR embeddings,
/// and related utilities like the IR2VecPrinterPass.
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

#ifndef LLVM_ANALYSIS_IR2VEC_H
#define LLVM_ANALYSIS_IR2VEC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include <map>

namespace llvm {

class Module;
class BasicBlock;
class Instruction;
class Function;
class Type;
class Value;
class raw_ostream;
class LLVMContext;

/// IR2Vec computes two kinds of embeddings: Symbolic and Flow-aware.
/// Symbolic embeddings capture the "syntactic" and "statistical correlation"
/// of the IR entities. Flow-aware embeddings build on top of symbolic
/// embeddings and additionally capture the flow information in the IR.
/// IR2VecKind is used to specify the type of embeddings to generate.
/// Currently, only Symbolic embeddings are supported.
enum class IR2VecKind { Symbolic };

namespace ir2vec {

LLVM_ABI extern cl::opt<float> OpcWeight;
LLVM_ABI extern cl::opt<float> TypeWeight;
LLVM_ABI extern cl::opt<float> ArgWeight;

/// Embedding is a datatype that wraps std::vector<double>. It provides
/// additional functionality for arithmetic and comparison operations.
/// It is meant to be used *like* std::vector<double> but is more restrictive
/// in the sense that it does not allow the user to change the size of the
/// embedding vector. The dimension of the embedding is fixed at the time of
/// construction of Embedding object. But the elements can be modified in-place.
struct Embedding {
private:
  std::vector<double> Data;

public:
  Embedding() = default;
  Embedding(const std::vector<double> &V) : Data(V) {}
  Embedding(std::vector<double> &&V) : Data(std::move(V)) {}
  Embedding(std::initializer_list<double> IL) : Data(IL) {}

  explicit Embedding(size_t Size) : Data(Size) {}
  Embedding(size_t Size, double InitialValue) : Data(Size, InitialValue) {}

  size_t size() const { return Data.size(); }
  bool empty() const { return Data.empty(); }

  double &operator[](size_t Itr) {
    assert(Itr < Data.size() && "Index out of bounds");
    return Data[Itr];
  }

  const double &operator[](size_t Itr) const {
    assert(Itr < Data.size() && "Index out of bounds");
    return Data[Itr];
  }

  using iterator = typename std::vector<double>::iterator;
  using const_iterator = typename std::vector<double>::const_iterator;

  iterator begin() { return Data.begin(); }
  iterator end() { return Data.end(); }
  const_iterator begin() const { return Data.begin(); }
  const_iterator end() const { return Data.end(); }
  const_iterator cbegin() const { return Data.cbegin(); }
  const_iterator cend() const { return Data.cend(); }

  const std::vector<double> &getData() const { return Data; }

  /// Arithmetic operators
  LLVM_ABI Embedding &operator+=(const Embedding &RHS);
  LLVM_ABI Embedding operator+(const Embedding &RHS) const;
  LLVM_ABI Embedding &operator-=(const Embedding &RHS);
  LLVM_ABI Embedding operator-(const Embedding &RHS) const;
  LLVM_ABI Embedding &operator*=(double Factor);
  LLVM_ABI Embedding operator*(double Factor) const;

  /// Adds Src Embedding scaled by Factor with the called Embedding.
  /// Called_Embedding += Src * Factor
  LLVM_ABI Embedding &scaleAndAdd(const Embedding &Src, float Factor);

  /// Returns true if the embedding is approximately equal to the RHS embedding
  /// within the specified tolerance.
  LLVM_ABI bool approximatelyEquals(const Embedding &RHS,
                                    double Tolerance = 1e-4) const;

  void print(raw_ostream &OS) const;
};

using InstEmbeddingsMap = DenseMap<const Instruction *, Embedding>;
using BBEmbeddingsMap = DenseMap<const BasicBlock *, Embedding>;
// FIXME: Current the keys are strings. This can be changed to
// use integers for cheaper lookups.
using Vocab = std::map<std::string, Embedding>;

/// Embedder provides the interface to generate embeddings (vector
/// representations) for instructions, basic blocks, and functions. The
/// vector representations are generated using IR2Vec algorithms.
///
/// The Embedder class is an abstract class and it is intended to be
/// subclassed for different IR2Vec algorithms like Symbolic and Flow-aware.
class Embedder {
protected:
  const Function &F;
  const Vocab &Vocabulary;

  /// Dimension of the vector representation; captured from the input vocabulary
  const unsigned Dimension;

  /// Weights for different entities (like opcode, arguments, types)
  /// in the IR instructions to generate the vector representation.
  const float OpcWeight, TypeWeight, ArgWeight;

  // Utility maps - these are used to store the vector representations of
  // instructions, basic blocks and functions.
  mutable Embedding FuncVector;
  mutable BBEmbeddingsMap BBVecMap;
  mutable InstEmbeddingsMap InstVecMap;

  LLVM_ABI Embedder(const Function &F, const Vocab &Vocabulary);

  /// Helper function to compute embeddings. It generates embeddings for all
  /// the instructions and basic blocks in the function F. Logic of computing
  /// the embeddings is specific to the kind of embeddings being computed.
  virtual void computeEmbeddings() const = 0;

  /// Helper function to compute the embedding for a given basic block.
  /// Specific to the kind of embeddings being computed.
  virtual void computeEmbeddings(const BasicBlock &BB) const = 0;

  /// Lookup vocabulary for a given Key. If the key is not found, it returns a
  /// zero vector.
  LLVM_ABI Embedding lookupVocab(const std::string &Key) const;

public:
  virtual ~Embedder() = default;

  /// Factory method to create an Embedder object.
  LLVM_ABI static std::unique_ptr<Embedder>
  create(IR2VecKind Mode, const Function &F, const Vocab &Vocabulary);

  /// Returns a map containing instructions and the corresponding embeddings for
  /// the function F if it has been computed. If not, it computes the embeddings
  /// for the function and returns the map.
  LLVM_ABI const InstEmbeddingsMap &getInstVecMap() const;

  /// Returns a map containing basic block and the corresponding embeddings for
  /// the function F if it has been computed. If not, it computes the embeddings
  /// for the function and returns the map.
  LLVM_ABI const BBEmbeddingsMap &getBBVecMap() const;

  /// Returns the embedding for a given basic block in the function F if it has
  /// been computed. If not, it computes the embedding for the basic block and
  /// returns it.
  LLVM_ABI const Embedding &getBBVector(const BasicBlock &BB) const;

  /// Computes and returns the embedding for the current function.
  LLVM_ABI const Embedding &getFunctionVector() const;
};

/// Class for computing the Symbolic embeddings of IR2Vec.
/// Symbolic embeddings are constructed based on the entity-level
/// representations obtained from the Vocabulary.
class LLVM_ABI SymbolicEmbedder : public Embedder {
private:
  /// Utility function to compute the embedding for a given type.
  Embedding getTypeEmbedding(const Type *Ty) const;

  /// Utility function to compute the embedding for a given operand.
  Embedding getOperandEmbedding(const Value *Op) const;

  void computeEmbeddings() const override;
  void computeEmbeddings(const BasicBlock &BB) const override;

public:
  SymbolicEmbedder(const Function &F, const Vocab &Vocabulary)
      : Embedder(F, Vocabulary) {
    FuncVector = Embedding(Dimension, 0);
  }
};

} // namespace ir2vec

/// Class for storing the result of the IR2VecVocabAnalysis.
class IR2VecVocabResult {
  ir2vec::Vocab Vocabulary;
  bool Valid = false;

public:
  IR2VecVocabResult() = default;
  LLVM_ABI IR2VecVocabResult(ir2vec::Vocab &&Vocabulary);

  bool isValid() const { return Valid; }
  LLVM_ABI const ir2vec::Vocab &getVocabulary() const;
  LLVM_ABI unsigned getDimension() const;
  LLVM_ABI bool invalidate(Module &M, const PreservedAnalyses &PA,
                           ModuleAnalysisManager::Invalidator &Inv) const;
};

/// This analysis provides the vocabulary for IR2Vec. The vocabulary provides a
/// mapping between an entity of the IR (like opcode, type, argument, etc.) and
/// its corresponding embedding.
class IR2VecVocabAnalysis : public AnalysisInfoMixin<IR2VecVocabAnalysis> {
  ir2vec::Vocab Vocabulary;
  Error readVocabulary();
  Error parseVocabSection(StringRef Key, const json::Value &ParsedVocabValue,
                          ir2vec::Vocab &TargetVocab, unsigned &Dim);
  void emitError(Error Err, LLVMContext &Ctx);

public:
  LLVM_ABI static AnalysisKey Key;
  IR2VecVocabAnalysis() = default;
  LLVM_ABI explicit IR2VecVocabAnalysis(const ir2vec::Vocab &Vocab);
  LLVM_ABI explicit IR2VecVocabAnalysis(ir2vec::Vocab &&Vocab);
  using Result = IR2VecVocabResult;
  LLVM_ABI Result run(Module &M, ModuleAnalysisManager &MAM);
};

/// This pass prints the IR2Vec embeddings for instructions, basic blocks, and
/// functions.
class IR2VecPrinterPass : public PassInfoMixin<IR2VecPrinterPass> {
  raw_ostream &OS;

public:
  explicit IR2VecPrinterPass(raw_ostream &OS) : OS(OS) {}
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};

/// This pass prints the embeddings in the vocabulary
class IR2VecVocabPrinterPass : public PassInfoMixin<IR2VecVocabPrinterPass> {
  raw_ostream &OS;

public:
  explicit IR2VecVocabPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_ANALYSIS_IR2VEC_H
