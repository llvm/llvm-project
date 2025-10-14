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
/// To obtain embeddings:
/// First run IR2VecVocabAnalysis to populate the vocabulary.
/// Then, use the Embedder interface to generate embeddings for the desired IR
/// entities. See the documentation for more details -
/// https://llvm.org/docs/MLGO.html#ir2vec-embeddings
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IR2VEC_H
#define LLVM_ANALYSIS_IR2VEC_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"
#include <array>
#include <map>
#include <optional>

namespace llvm {

class Module;
class BasicBlock;
class Instruction;
class Function;
class Value;
class raw_ostream;
class LLVMContext;
class IR2VecVocabAnalysis;

/// IR2Vec computes two kinds of embeddings: Symbolic and Flow-aware.
/// Symbolic embeddings capture the "syntactic" and "statistical correlation"
/// of the IR entities. Flow-aware embeddings build on top of symbolic
/// embeddings and additionally capture the flow information in the IR.
/// IR2VecKind is used to specify the type of embeddings to generate.
/// Note: Implementation of FlowAware embeddings is not same as the one
/// described in the paper. The current implementation is a simplified version
/// that captures the flow information (SSA-based use-defs) without tracing
/// through memory level use-defs in the embedding computation described in the
/// paper.
enum class IR2VecKind { Symbolic, FlowAware };

namespace ir2vec {

extern llvm::cl::OptionCategory IR2VecCategory;
LLVM_ABI extern cl::opt<float> OpcWeight;
LLVM_ABI extern cl::opt<float> TypeWeight;
LLVM_ABI extern cl::opt<float> ArgWeight;
LLVM_ABI extern cl::opt<IR2VecKind> IR2VecEmbeddingKind;

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

  explicit Embedding(size_t Size) : Data(Size, 0.0) {}
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

  LLVM_ABI void print(raw_ostream &OS) const;
};

using InstEmbeddingsMap = DenseMap<const Instruction *, Embedding>;
using BBEmbeddingsMap = DenseMap<const BasicBlock *, Embedding>;

/// Generic storage class for section-based vocabularies.
/// VocabStorage provides a generic foundation for storing and accessing
/// embeddings organized into sections.
class VocabStorage {
private:
  /// Section-based storage
  std::vector<std::vector<Embedding>> Sections;

  // Fixme: Check if these members can be made const (and delete move
  // assignment) after changing Vocabulary creation by using static factory
  // methods.
  size_t TotalSize = 0;
  unsigned Dimension = 0;

public:
  /// Default constructor creates empty storage (invalid state)
  VocabStorage() : Sections(), TotalSize(0), Dimension(0) {}

  /// Create a VocabStorage with pre-organized section data
  VocabStorage(std::vector<std::vector<Embedding>> &&SectionData);

  VocabStorage(VocabStorage &&) = default;
  VocabStorage &operator=(VocabStorage &&) = default;

  VocabStorage(const VocabStorage &) = delete;
  VocabStorage &operator=(const VocabStorage &) = delete;

  /// Get total number of entries across all sections
  size_t size() const { return TotalSize; }

  /// Get number of sections
  unsigned getNumSections() const {
    return static_cast<unsigned>(Sections.size());
  }

  /// Section-based access: Storage[sectionId][localIndex]
  const std::vector<Embedding> &operator[](unsigned SectionId) const {
    assert(SectionId < Sections.size() && "Invalid section ID");
    return Sections[SectionId];
  }

  /// Get vocabulary dimension
  unsigned getDimension() const { return Dimension; }

  /// Check if vocabulary is valid (has data)
  bool isValid() const { return TotalSize > 0; }

  /// Iterator support for section-based access
  class const_iterator {
    const VocabStorage *Storage;
    unsigned SectionId = 0;
    size_t LocalIndex = 0;

  public:
    const_iterator(const VocabStorage *Storage, unsigned SectionId,
                   size_t LocalIndex)
        : Storage(Storage), SectionId(SectionId), LocalIndex(LocalIndex) {}

    LLVM_ABI const Embedding &operator*() const;
    LLVM_ABI const_iterator &operator++();
    LLVM_ABI bool operator==(const const_iterator &Other) const;
    LLVM_ABI bool operator!=(const const_iterator &Other) const;
  };

  const_iterator begin() const { return const_iterator(this, 0, 0); }
  const_iterator end() const {
    return const_iterator(this, getNumSections(), 0);
  }

  using VocabMap = std::map<std::string, Embedding>;
  /// Parse a vocabulary section from JSON and populate the target vocabulary
  /// map.
  static Error parseVocabSection(StringRef Key,
                                 const json::Value &ParsedVocabValue,
                                 VocabMap &TargetVocab, unsigned &Dim);
};

/// Class for storing and accessing the IR2Vec vocabulary.
/// The Vocabulary class manages seed embeddings for LLVM IR entities. The
/// seed embeddings are the initial learned representations of the entities
/// of LLVM IR. The IR2Vec representation for a given IR is derived from these
/// seed embeddings.
///
/// The vocabulary contains the seed embeddings for three types of entities:
/// instruction opcodes, types, and operands. Types are grouped/canonicalized
/// for better learning (e.g., all float variants map to FloatTy). The
/// vocabulary abstracts away the canonicalization effectively, the exposed APIs
/// handle all the known LLVM IR opcodes, types and operands.
///
/// This class helps populate the seed embeddings in an internal vector-based
/// ADT. It provides logic to map every IR entity to a specific slot index or
/// position in this vector, enabling O(1) embedding lookup while avoiding
/// unnecessary computations involving string based lookups while generating the
/// embeddings.
class Vocabulary {
  friend class llvm::IR2VecVocabAnalysis;

  // Vocabulary Layout:
  // +----------------+------------------------------------------------------+
  // | Entity Type    | Index Range                                          |
  // +----------------+------------------------------------------------------+
  // | Opcodes        | [0 .. (MaxOpcodes-1)]                                |
  // | Canonical Types| [MaxOpcodes .. (MaxOpcodes+MaxCanonicalTypeIDs-1)]   |
  // | Operands       | [(MaxOpcodes+MaxCanonicalTypeIDs) .. NumCanEntries]  |
  // +----------------+------------------------------------------------------+
  // Note: MaxOpcodes is the number of unique opcodes supported by LLVM IR.
  //       MaxCanonicalTypeIDs is the number of canonicalized type IDs.
  //       "Similar" LLVM Types are grouped/canonicalized together. E.g., all
  //       float variants (FloatTy, DoubleTy, HalfTy, etc.) map to
  //       CanonicalTypeID::FloatTy. This helps reduce the vocabulary size
  //       and improves learning. Operands include Comparison predicates
  //       (ICmp/FCmp) along with other operand types. This can be extended to
  //       include other specializations in future.
  enum class Section : unsigned {
    Opcodes = 0,
    CanonicalTypes = 1,
    Operands = 2,
    Predicates = 3,
    MaxSections
  };

  // Use section-based storage for better organization and efficiency
  VocabStorage Storage;

  static constexpr unsigned NumICmpPredicates =
      static_cast<unsigned>(CmpInst::LAST_ICMP_PREDICATE) -
      static_cast<unsigned>(CmpInst::FIRST_ICMP_PREDICATE) + 1;
  static constexpr unsigned NumFCmpPredicates =
      static_cast<unsigned>(CmpInst::LAST_FCMP_PREDICATE) -
      static_cast<unsigned>(CmpInst::FIRST_FCMP_PREDICATE) + 1;

public:
  /// Canonical type IDs supported by IR2Vec Vocabulary
  enum class CanonicalTypeID : unsigned {
    FloatTy,
    VoidTy,
    LabelTy,
    MetadataTy,
    VectorTy,
    TokenTy,
    IntegerTy,
    FunctionTy,
    PointerTy,
    StructTy,
    ArrayTy,
    UnknownTy,
    MaxCanonicalType
  };

  /// Operand kinds supported by IR2Vec Vocabulary
  enum class OperandKind : unsigned {
    FunctionID,
    PointerID,
    ConstantID,
    VariableID,
    MaxOperandKind
  };

  /// Vocabulary layout constants
#define LAST_OTHER_INST(NUM) static constexpr unsigned MaxOpcodes = NUM;
#include "llvm/IR/Instruction.def"
#undef LAST_OTHER_INST

  static constexpr unsigned MaxTypeIDs = Type::TypeID::TargetExtTyID + 1;
  static constexpr unsigned MaxCanonicalTypeIDs =
      static_cast<unsigned>(CanonicalTypeID::MaxCanonicalType);
  static constexpr unsigned MaxOperandKinds =
      static_cast<unsigned>(OperandKind::MaxOperandKind);
  // CmpInst::Predicate has gaps. We want the vocabulary to be dense without
  // empty slots.
  static constexpr unsigned MaxPredicateKinds =
      NumICmpPredicates + NumFCmpPredicates;

  Vocabulary() = default;
  LLVM_ABI Vocabulary(VocabStorage &&Storage) : Storage(std::move(Storage)) {}

  Vocabulary(const Vocabulary &) = delete;
  Vocabulary &operator=(const Vocabulary &) = delete;

  Vocabulary(Vocabulary &&) = default;
  Vocabulary &operator=(Vocabulary &&Other) = delete;

  LLVM_ABI bool isValid() const {
    return Storage.size() == NumCanonicalEntries;
  }

  LLVM_ABI unsigned getDimension() const {
    assert(isValid() && "IR2Vec Vocabulary is invalid");
    return Storage.getDimension();
  }

  /// Total number of entries (opcodes + canonicalized types + operand kinds +
  /// predicates)
  static constexpr size_t getCanonicalSize() { return NumCanonicalEntries; }

  /// Function to get vocabulary key for a given Opcode
  LLVM_ABI static StringRef getVocabKeyForOpcode(unsigned Opcode);

  /// Function to get vocabulary key for a given TypeID
  LLVM_ABI static StringRef getVocabKeyForTypeID(Type::TypeID TypeID) {
    return getVocabKeyForCanonicalTypeID(getCanonicalTypeID(TypeID));
  }

  /// Function to get vocabulary key for a given OperandKind
  LLVM_ABI static StringRef getVocabKeyForOperandKind(OperandKind Kind) {
    unsigned Index = static_cast<unsigned>(Kind);
    assert(Index < MaxOperandKinds && "Invalid OperandKind");
    return OperandKindNames[Index];
  }

  /// Function to classify an operand into OperandKind
  LLVM_ABI static OperandKind getOperandKind(const Value *Op);

  /// Function to get vocabulary key for a given predicate
  LLVM_ABI static StringRef getVocabKeyForPredicate(CmpInst::Predicate P);

  /// Functions to return flat index
  LLVM_ABI static unsigned getIndex(unsigned Opcode) {
    assert(Opcode >= 1 && Opcode <= MaxOpcodes && "Invalid opcode");
    return Opcode - 1; // Convert to zero-based index
  }

  LLVM_ABI static unsigned getIndex(Type::TypeID TypeID) {
    assert(static_cast<unsigned>(TypeID) < MaxTypeIDs && "Invalid type ID");
    return MaxOpcodes + static_cast<unsigned>(getCanonicalTypeID(TypeID));
  }

  LLVM_ABI static unsigned getIndex(const Value &Op) {
    unsigned Index = static_cast<unsigned>(getOperandKind(&Op));
    assert(Index < MaxOperandKinds && "Invalid OperandKind");
    return OperandBaseOffset + Index;
  }

  LLVM_ABI static unsigned getIndex(CmpInst::Predicate P) {
    return PredicateBaseOffset + getPredicateLocalIndex(P);
  }

  /// Accessors to get the embedding for a given entity.
  LLVM_ABI const ir2vec::Embedding &operator[](unsigned Opcode) const {
    assert(Opcode >= 1 && Opcode <= MaxOpcodes && "Invalid opcode");
    return Storage[static_cast<unsigned>(Section::Opcodes)][Opcode - 1];
  }

  LLVM_ABI const ir2vec::Embedding &operator[](Type::TypeID TypeID) const {
    assert(static_cast<unsigned>(TypeID) < MaxTypeIDs && "Invalid type ID");
    unsigned LocalIndex = static_cast<unsigned>(getCanonicalTypeID(TypeID));
    return Storage[static_cast<unsigned>(Section::CanonicalTypes)][LocalIndex];
  }

  LLVM_ABI const ir2vec::Embedding &operator[](const Value &Arg) const {
    unsigned LocalIndex = static_cast<unsigned>(getOperandKind(&Arg));
    assert(LocalIndex < MaxOperandKinds && "Invalid OperandKind");
    return Storage[static_cast<unsigned>(Section::Operands)][LocalIndex];
  }

  LLVM_ABI const ir2vec::Embedding &operator[](CmpInst::Predicate P) const {
    unsigned LocalIndex = getPredicateLocalIndex(P);
    return Storage[static_cast<unsigned>(Section::Predicates)][LocalIndex];
  }

  /// Const Iterator type aliases
  using const_iterator = VocabStorage::const_iterator;

  const_iterator begin() const {
    assert(isValid() && "IR2Vec Vocabulary is invalid");
    return Storage.begin();
  }

  const_iterator cbegin() const { return begin(); }

  const_iterator end() const {
    assert(isValid() && "IR2Vec Vocabulary is invalid");
    return Storage.end();
  }

  const_iterator cend() const { return end(); }

  /// Returns the string key for a given index position in the vocabulary.
  /// This is useful for debugging or printing the vocabulary. Do not use this
  /// for embedding generation as string based lookups are inefficient.
  LLVM_ABI static StringRef getStringKey(unsigned Pos);

  /// Create a dummy vocabulary for testing purposes.
  LLVM_ABI static VocabStorage createDummyVocabForTest(unsigned Dim = 1);

  LLVM_ABI bool invalidate(Module &M, const PreservedAnalyses &PA,
                           ModuleAnalysisManager::Invalidator &Inv) const;

private:
  constexpr static unsigned NumCanonicalEntries =
      MaxOpcodes + MaxCanonicalTypeIDs + MaxOperandKinds + MaxPredicateKinds;

  // Base offsets for flat index computation
  constexpr static unsigned OperandBaseOffset =
      MaxOpcodes + MaxCanonicalTypeIDs;
  constexpr static unsigned PredicateBaseOffset =
      OperandBaseOffset + MaxOperandKinds;

  /// Functions for predicate index calculations
  static unsigned getPredicateLocalIndex(CmpInst::Predicate P);
  static CmpInst::Predicate getPredicateFromLocalIndex(unsigned LocalIndex);

  /// String mappings for CanonicalTypeID values
  static constexpr StringLiteral CanonicalTypeNames[] = {
      "FloatTy",   "VoidTy",   "LabelTy",   "MetadataTy",
      "VectorTy",  "TokenTy",  "IntegerTy", "FunctionTy",
      "PointerTy", "StructTy", "ArrayTy",   "UnknownTy"};
  static_assert(std::size(CanonicalTypeNames) ==
                    static_cast<unsigned>(CanonicalTypeID::MaxCanonicalType),
                "CanonicalTypeNames array size must match MaxCanonicalType");

  /// String mappings for OperandKind values
  static constexpr StringLiteral OperandKindNames[] = {"Function", "Pointer",
                                                       "Constant", "Variable"};
  static_assert(std::size(OperandKindNames) ==
                    static_cast<unsigned>(OperandKind::MaxOperandKind),
                "OperandKindNames array size must match MaxOperandKind");

  /// Every known TypeID defined in llvm/IR/Type.h is expected to have a
  /// corresponding mapping here in the same order as enum Type::TypeID.
  static constexpr std::array<CanonicalTypeID, MaxTypeIDs> TypeIDMapping = {{
      CanonicalTypeID::FloatTy,    // HalfTyID = 0
      CanonicalTypeID::FloatTy,    // BFloatTyID
      CanonicalTypeID::FloatTy,    // FloatTyID
      CanonicalTypeID::FloatTy,    // DoubleTyID
      CanonicalTypeID::FloatTy,    // X86_FP80TyID
      CanonicalTypeID::FloatTy,    // FP128TyID
      CanonicalTypeID::FloatTy,    // PPC_FP128TyID
      CanonicalTypeID::VoidTy,     // VoidTyID
      CanonicalTypeID::LabelTy,    // LabelTyID
      CanonicalTypeID::MetadataTy, // MetadataTyID
      CanonicalTypeID::VectorTy,   // X86_AMXTyID
      CanonicalTypeID::TokenTy,    // TokenTyID
      CanonicalTypeID::IntegerTy,  // IntegerTyID
      CanonicalTypeID::FunctionTy, // FunctionTyID
      CanonicalTypeID::PointerTy,  // PointerTyID
      CanonicalTypeID::StructTy,   // StructTyID
      CanonicalTypeID::ArrayTy,    // ArrayTyID
      CanonicalTypeID::VectorTy,   // FixedVectorTyID
      CanonicalTypeID::VectorTy,   // ScalableVectorTyID
      CanonicalTypeID::PointerTy,  // TypedPointerTyID
      CanonicalTypeID::UnknownTy   // TargetExtTyID
  }};
  static_assert(TypeIDMapping.size() == MaxTypeIDs,
                "TypeIDMapping must cover all Type::TypeID values");

  /// Function to get vocabulary key for canonical type by enum
  LLVM_ABI static StringRef
  getVocabKeyForCanonicalTypeID(CanonicalTypeID CType) {
    unsigned Index = static_cast<unsigned>(CType);
    assert(Index < MaxCanonicalTypeIDs && "Invalid CanonicalTypeID");
    return CanonicalTypeNames[Index];
  }

  /// Function to convert TypeID to CanonicalTypeID
  LLVM_ABI static CanonicalTypeID getCanonicalTypeID(Type::TypeID TypeID) {
    unsigned Index = static_cast<unsigned>(TypeID);
    assert(Index < MaxTypeIDs && "Invalid TypeID");
    return TypeIDMapping[Index];
  }

  /// Function to get the predicate enum value for a given index. Index is
  /// relative to the predicates section of the vocabulary. E.g., Index 0
  /// corresponds to the first predicate.
  LLVM_ABI static CmpInst::Predicate getPredicate(unsigned Index) {
    assert(Index < MaxPredicateKinds && "Invalid predicate index");
    return getPredicateFromLocalIndex(Index);
  }
};

/// Embedder provides the interface to generate embeddings (vector
/// representations) for instructions, basic blocks, and functions. The
/// vector representations are generated using IR2Vec algorithms.
///
/// The Embedder class is an abstract class and it is intended to be
/// subclassed for different IR2Vec algorithms like Symbolic and Flow-aware.
class Embedder {
protected:
  const Function &F;
  const Vocabulary &Vocab;

  /// Dimension of the vector representation; captured from the input vocabulary
  const unsigned Dimension;

  /// Weights for different entities (like opcode, arguments, types)
  /// in the IR instructions to generate the vector representation.
  const float OpcWeight, TypeWeight, ArgWeight;

  LLVM_ABI Embedder(const Function &F, const Vocabulary &Vocab)
      : F(F), Vocab(Vocab), Dimension(Vocab.getDimension()),
        OpcWeight(ir2vec::OpcWeight), TypeWeight(ir2vec::TypeWeight),
        ArgWeight(ir2vec::ArgWeight) {}

  /// Function to compute embeddings.
  Embedding computeEmbeddings() const;

  /// Function to compute the embedding for a given basic block.
  Embedding computeEmbeddings(const BasicBlock &BB) const;

  /// Function to compute the embedding for a given instruction.
  /// Specific to the kind of embeddings being computed.
  virtual Embedding computeEmbeddings(const Instruction &I) const = 0;

public:
  virtual ~Embedder() = default;

  /// Factory method to create an Embedder object.
  LLVM_ABI static std::unique_ptr<Embedder>
  create(IR2VecKind Mode, const Function &F, const Vocabulary &Vocab);

  /// Computes and returns the embedding for a given instruction in the function
  /// F
  LLVM_ABI Embedding getInstVector(const Instruction &I) const {
    return computeEmbeddings(I);
  }

  /// Computes and returns the embedding for a given basic block in the function
  /// F
  LLVM_ABI Embedding getBBVector(const BasicBlock &BB) const {
    return computeEmbeddings(BB);
  }

  /// Computes and returns the embedding for the current function.
  LLVM_ABI Embedding getFunctionVector() const { return computeEmbeddings(); }

  /// Invalidate embeddings if cached. The embeddings may not be relevant
  /// anymore when the IR changes due to transformations. In such cases, the
  /// cached embeddings should be invalidated to ensure
  /// correctness/recomputation. This is a no-op for SymbolicEmbedder but
  /// removes all the cached entries in FlowAwareEmbedder.
  virtual void invalidateEmbeddings() { return; }
};

/// Class for computing the Symbolic embeddings of IR2Vec.
/// Symbolic embeddings are constructed based on the entity-level
/// representations obtained from the Vocabulary.
class LLVM_ABI SymbolicEmbedder : public Embedder {
private:
  Embedding computeEmbeddings(const Instruction &I) const override;

public:
  SymbolicEmbedder(const Function &F, const Vocabulary &Vocab)
      : Embedder(F, Vocab) {}
};

/// Class for computing the Flow-aware embeddings of IR2Vec.
/// Flow-aware embeddings build on the vocabulary, just like Symbolic
/// embeddings, and additionally capture the flow information in the IR.
class LLVM_ABI FlowAwareEmbedder : public Embedder {
private:
  // FlowAware embeddings would benefit from caching instruction embeddings as
  // they are reused while computing the embeddings of other instructions.
  mutable InstEmbeddingsMap InstVecMap;
  Embedding computeEmbeddings(const Instruction &I) const override;

public:
  FlowAwareEmbedder(const Function &F, const Vocabulary &Vocab)
      : Embedder(F, Vocab) {}
  void invalidateEmbeddings() override { InstVecMap.clear(); }
};

} // namespace ir2vec

/// This analysis provides the vocabulary for IR2Vec. The vocabulary provides a
/// mapping between an entity of the IR (like opcode, type, argument, etc.) and
/// its corresponding embedding.
class IR2VecVocabAnalysis : public AnalysisInfoMixin<IR2VecVocabAnalysis> {
  using VocabMap = std::map<std::string, ir2vec::Embedding>;
  std::optional<ir2vec::VocabStorage> Vocab;

  Error readVocabulary(VocabMap &OpcVocab, VocabMap &TypeVocab,
                       VocabMap &ArgVocab);
  void generateVocabStorage(VocabMap &OpcVocab, VocabMap &TypeVocab,
                            VocabMap &ArgVocab);
  void emitError(Error Err, LLVMContext &Ctx);

public:
  LLVM_ABI static AnalysisKey Key;
  IR2VecVocabAnalysis() = default;
  LLVM_ABI explicit IR2VecVocabAnalysis(ir2vec::VocabStorage &&Vocab)
      : Vocab(std::move(Vocab)) {}
  using Result = ir2vec::Vocabulary;
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
  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_ANALYSIS_IR2VEC_H
