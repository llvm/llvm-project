//===- IR2Vec.cpp - Implementation of IR2Vec -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the IR2Vec algorithm.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IR2Vec.h"

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace ir2vec;

#define DEBUG_TYPE "ir2vec"

STATISTIC(VocabMissCounter,
          "Number of lookups to entities not present in the vocabulary");

namespace llvm {
namespace ir2vec {
cl::OptionCategory IR2VecCategory("IR2Vec Options");

// FIXME: Use a default vocab when not specified
static cl::opt<std::string>
    VocabFile("ir2vec-vocab-path", cl::Optional,
              cl::desc("Path to the vocabulary file for IR2Vec"), cl::init(""),
              cl::cat(IR2VecCategory));
cl::opt<float> OpcWeight("ir2vec-opc-weight", cl::Optional, cl::init(1.0),
                         cl::desc("Weight for opcode embeddings"),
                         cl::cat(IR2VecCategory));
cl::opt<float> TypeWeight("ir2vec-type-weight", cl::Optional, cl::init(0.5),
                          cl::desc("Weight for type embeddings"),
                          cl::cat(IR2VecCategory));
cl::opt<float> ArgWeight("ir2vec-arg-weight", cl::Optional, cl::init(0.2),
                         cl::desc("Weight for argument embeddings"),
                         cl::cat(IR2VecCategory));
cl::opt<IR2VecKind> IR2VecEmbeddingKind(
    "ir2vec-kind", cl::Optional,
    cl::values(clEnumValN(IR2VecKind::Symbolic, "symbolic",
                          "Generate symbolic embeddings"),
               clEnumValN(IR2VecKind::FlowAware, "flow-aware",
                          "Generate flow-aware embeddings")),
    cl::init(IR2VecKind::Symbolic), cl::desc("IR2Vec embedding kind"),
    cl::cat(IR2VecCategory));

} // namespace ir2vec
} // namespace llvm

AnalysisKey IR2VecVocabAnalysis::Key;

// ==----------------------------------------------------------------------===//
// Local helper functions
//===----------------------------------------------------------------------===//
namespace llvm::json {
inline bool fromJSON(const llvm::json::Value &E, Embedding &Out,
                     llvm::json::Path P) {
  std::vector<double> TempOut;
  if (!llvm::json::fromJSON(E, TempOut, P))
    return false;
  Out = Embedding(std::move(TempOut));
  return true;
}
} // namespace llvm::json

// ==----------------------------------------------------------------------===//
// Embedding
//===----------------------------------------------------------------------===//
Embedding &Embedding::operator+=(const Embedding &RHS) {
  assert(this->size() == RHS.size() && "Vectors must have the same dimension");
  std::transform(this->begin(), this->end(), RHS.begin(), this->begin(),
                 std::plus<double>());
  return *this;
}

Embedding Embedding::operator+(const Embedding &RHS) const {
  Embedding Result(*this);
  Result += RHS;
  return Result;
}

Embedding &Embedding::operator-=(const Embedding &RHS) {
  assert(this->size() == RHS.size() && "Vectors must have the same dimension");
  std::transform(this->begin(), this->end(), RHS.begin(), this->begin(),
                 std::minus<double>());
  return *this;
}

Embedding Embedding::operator-(const Embedding &RHS) const {
  Embedding Result(*this);
  Result -= RHS;
  return Result;
}

Embedding &Embedding::operator*=(double Factor) {
  std::transform(this->begin(), this->end(), this->begin(),
                 [Factor](double Elem) { return Elem * Factor; });
  return *this;
}

Embedding Embedding::operator*(double Factor) const {
  Embedding Result(*this);
  Result *= Factor;
  return Result;
}

Embedding &Embedding::scaleAndAdd(const Embedding &Src, float Factor) {
  assert(this->size() == Src.size() && "Vectors must have the same dimension");
  for (size_t Itr = 0; Itr < this->size(); ++Itr)
    (*this)[Itr] += Src[Itr] * Factor;
  return *this;
}

bool Embedding::approximatelyEquals(const Embedding &RHS,
                                    double Tolerance) const {
  assert(this->size() == RHS.size() && "Vectors must have the same dimension");
  for (size_t Itr = 0; Itr < this->size(); ++Itr)
    if (std::abs((*this)[Itr] - RHS[Itr]) > Tolerance) {
      LLVM_DEBUG(errs() << "Embedding mismatch at index " << Itr << ": "
                        << (*this)[Itr] << " vs " << RHS[Itr]
                        << "; Tolerance: " << Tolerance << "\n");
      return false;
    }
  return true;
}

void Embedding::print(raw_ostream &OS) const {
  OS << " [";
  for (const auto &Elem : Data)
    OS << " " << format("%.2f", Elem) << " ";
  OS << "]\n";
}

// ==----------------------------------------------------------------------===//
// Embedder and its subclasses
//===----------------------------------------------------------------------===//

Embedder::Embedder(const Function &F, const Vocabulary &Vocab)
    : F(F), Vocab(Vocab), Dimension(Vocab.getDimension()),
      OpcWeight(::OpcWeight), TypeWeight(::TypeWeight), ArgWeight(::ArgWeight),
      FuncVector(Embedding(Dimension, 0)) {}

std::unique_ptr<Embedder> Embedder::create(IR2VecKind Mode, const Function &F,
                                           const Vocabulary &Vocab) {
  switch (Mode) {
  case IR2VecKind::Symbolic:
    return std::make_unique<SymbolicEmbedder>(F, Vocab);
  case IR2VecKind::FlowAware:
    return std::make_unique<FlowAwareEmbedder>(F, Vocab);
  }
  return nullptr;
}

const InstEmbeddingsMap &Embedder::getInstVecMap() const {
  if (InstVecMap.empty())
    computeEmbeddings();
  return InstVecMap;
}

const BBEmbeddingsMap &Embedder::getBBVecMap() const {
  if (BBVecMap.empty())
    computeEmbeddings();
  return BBVecMap;
}

const Embedding &Embedder::getBBVector(const BasicBlock &BB) const {
  auto It = BBVecMap.find(&BB);
  if (It != BBVecMap.end())
    return It->second;
  computeEmbeddings(BB);
  return BBVecMap[&BB];
}

const Embedding &Embedder::getFunctionVector() const {
  // Currently, we always (re)compute the embeddings for the function.
  // This is cheaper than caching the vector.
  computeEmbeddings();
  return FuncVector;
}

void Embedder::computeEmbeddings() const {
  if (F.isDeclaration())
    return;

  // Consider only the basic blocks that are reachable from entry
  for (const BasicBlock *BB : depth_first(&F)) {
    computeEmbeddings(*BB);
    FuncVector += BBVecMap[BB];
  }
}

void SymbolicEmbedder::computeEmbeddings(const BasicBlock &BB) const {
  Embedding BBVector(Dimension, 0);

  // We consider only the non-debug and non-pseudo instructions
  for (const auto &I : BB.instructionsWithoutDebug()) {
    Embedding ArgEmb(Dimension, 0);
    for (const auto &Op : I.operands())
      ArgEmb += Vocab[*Op];
    auto InstVector =
        Vocab[I.getOpcode()] + Vocab[I.getType()->getTypeID()] + ArgEmb;
    InstVecMap[&I] = InstVector;
    BBVector += InstVector;
  }
  BBVecMap[&BB] = BBVector;
}

void FlowAwareEmbedder::computeEmbeddings(const BasicBlock &BB) const {
  Embedding BBVector(Dimension, 0);

  // We consider only the non-debug and non-pseudo instructions
  for (const auto &I : BB.instructionsWithoutDebug()) {
    // TODO: Handle call instructions differently.
    // For now, we treat them like other instructions
    Embedding ArgEmb(Dimension, 0);
    for (const auto &Op : I.operands()) {
      // If the operand is defined elsewhere, we use its embedding
      if (const auto *DefInst = dyn_cast<Instruction>(Op)) {
        auto DefIt = InstVecMap.find(DefInst);
        assert(DefIt != InstVecMap.end() &&
               "Instruction should have been processed before its operands");
        ArgEmb += DefIt->second;
        continue;
      }
      // If the operand is not defined by an instruction, we use the vocabulary
      else {
        LLVM_DEBUG(errs() << "Using embedding from vocabulary for operand: "
                          << *Op << "=" << Vocab[*Op][0] << "\n");
        ArgEmb += Vocab[*Op];
      }
    }
    // Create the instruction vector by combining opcode, type, and arguments
    // embeddings
    auto InstVector =
        Vocab[I.getOpcode()] + Vocab[I.getType()->getTypeID()] + ArgEmb;
    InstVecMap[&I] = InstVector;
    BBVector += InstVector;
  }
  BBVecMap[&BB] = BBVector;
}

// ==----------------------------------------------------------------------===//
// Vocabulary
//===----------------------------------------------------------------------===//

Vocabulary::Vocabulary(VocabVector &&Vocab)
    : Vocab(std::move(Vocab)), Valid(true) {}

bool Vocabulary::isValid() const {
  return Vocab.size() == NumCanonicalEntries && Valid;
}

unsigned Vocabulary::getDimension() const {
  assert(Valid && "IR2Vec Vocabulary is invalid");
  return Vocab[0].size();
}

unsigned Vocabulary::getSlotIndex(unsigned Opcode) {
  assert(Opcode >= 1 && Opcode <= MaxOpcodes && "Invalid opcode");
  return Opcode - 1; // Convert to zero-based index
}

unsigned Vocabulary::getSlotIndex(Type::TypeID TypeID) {
  assert(static_cast<unsigned>(TypeID) < MaxTypeIDs && "Invalid type ID");
  return MaxOpcodes + static_cast<unsigned>(getCanonicalTypeID(TypeID));
}

unsigned Vocabulary::getSlotIndex(const Value &Op) {
  unsigned Index = static_cast<unsigned>(getOperandKind(&Op));
  assert(Index < MaxOperandKinds && "Invalid OperandKind");
  return MaxOpcodes + MaxCanonicalTypeIDs + Index;
}

const Embedding &Vocabulary::operator[](unsigned Opcode) const {
  return Vocab[getSlotIndex(Opcode)];
}

const Embedding &Vocabulary::operator[](Type::TypeID TypeID) const {
  return Vocab[getSlotIndex(TypeID)];
}

const ir2vec::Embedding &Vocabulary::operator[](const Value &Arg) const {
  return Vocab[getSlotIndex(Arg)];
}

StringRef Vocabulary::getVocabKeyForOpcode(unsigned Opcode) {
  assert(Opcode >= 1 && Opcode <= MaxOpcodes && "Invalid opcode");
#define HANDLE_INST(NUM, OPCODE, CLASS)                                        \
  if (Opcode == NUM) {                                                         \
    return #OPCODE;                                                            \
  }
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
  return "UnknownOpcode";
}

StringRef Vocabulary::getVocabKeyForCanonicalTypeID(CanonicalTypeID CType) {
  unsigned Index = static_cast<unsigned>(CType);
  assert(Index < MaxCanonicalTypeIDs && "Invalid CanonicalTypeID");
  return CanonicalTypeNames[Index];
}

Vocabulary::CanonicalTypeID
Vocabulary::getCanonicalTypeID(Type::TypeID TypeID) {
  unsigned Index = static_cast<unsigned>(TypeID);
  assert(Index < MaxTypeIDs && "Invalid TypeID");
  return TypeIDMapping[Index];
}

StringRef Vocabulary::getVocabKeyForTypeID(Type::TypeID TypeID) {
  return getVocabKeyForCanonicalTypeID(getCanonicalTypeID(TypeID));
}

StringRef Vocabulary::getVocabKeyForOperandKind(Vocabulary::OperandKind Kind) {
  unsigned Index = static_cast<unsigned>(Kind);
  assert(Index < MaxOperandKinds && "Invalid OperandKind");
  return OperandKindNames[Index];
}

// Helper function to classify an operand into OperandKind
Vocabulary::OperandKind Vocabulary::getOperandKind(const Value *Op) {
  if (isa<Function>(Op))
    return OperandKind::FunctionID;
  if (isa<PointerType>(Op->getType()))
    return OperandKind::PointerID;
  if (isa<Constant>(Op))
    return OperandKind::ConstantID;
  return OperandKind::VariableID;
}

StringRef Vocabulary::getStringKey(unsigned Pos) {
  assert(Pos < NumCanonicalEntries && "Position out of bounds in vocabulary");
  // Opcode
  if (Pos < MaxOpcodes)
    return getVocabKeyForOpcode(Pos + 1);
  // Type
  if (Pos < MaxOpcodes + MaxCanonicalTypeIDs)
    return getVocabKeyForCanonicalTypeID(
        static_cast<CanonicalTypeID>(Pos - MaxOpcodes));
  // Operand
  return getVocabKeyForOperandKind(
      static_cast<OperandKind>(Pos - MaxOpcodes - MaxCanonicalTypeIDs));
}

// For now, assume vocabulary is stable unless explicitly invalidated.
bool Vocabulary::invalidate(Module &M, const PreservedAnalyses &PA,
                            ModuleAnalysisManager::Invalidator &Inv) const {
  auto PAC = PA.getChecker<IR2VecVocabAnalysis>();
  return !(PAC.preservedWhenStateless());
}

Vocabulary::VocabVector Vocabulary::createDummyVocabForTest(unsigned Dim) {
  VocabVector DummyVocab;
  DummyVocab.reserve(NumCanonicalEntries);
  float DummyVal = 0.1f;
  // Create a dummy vocabulary with entries for all opcodes, types, and
  // operands
  for ([[maybe_unused]] unsigned _ :
       seq(0u, Vocabulary::MaxOpcodes + Vocabulary::MaxCanonicalTypeIDs +
                   Vocabulary::MaxOperandKinds)) {
    DummyVocab.push_back(Embedding(Dim, DummyVal));
    DummyVal += 0.1f;
  }
  return DummyVocab;
}

// ==----------------------------------------------------------------------===//
// IR2VecVocabAnalysis
//===----------------------------------------------------------------------===//

Error IR2VecVocabAnalysis::parseVocabSection(
    StringRef Key, const json::Value &ParsedVocabValue, VocabMap &TargetVocab,
    unsigned &Dim) {
  json::Path::Root Path("");
  const json::Object *RootObj = ParsedVocabValue.getAsObject();
  if (!RootObj)
    return createStringError(errc::invalid_argument,
                             "JSON root is not an object");

  const json::Value *SectionValue = RootObj->get(Key);
  if (!SectionValue)
    return createStringError(errc::invalid_argument,
                             "Missing '" + std::string(Key) +
                                 "' section in vocabulary file");
  if (!json::fromJSON(*SectionValue, TargetVocab, Path))
    return createStringError(errc::illegal_byte_sequence,
                             "Unable to parse '" + std::string(Key) +
                                 "' section from vocabulary");

  Dim = TargetVocab.begin()->second.size();
  if (Dim == 0)
    return createStringError(errc::illegal_byte_sequence,
                             "Dimension of '" + std::string(Key) +
                                 "' section of the vocabulary is zero");

  if (!std::all_of(TargetVocab.begin(), TargetVocab.end(),
                   [Dim](const std::pair<StringRef, Embedding> &Entry) {
                     return Entry.second.size() == Dim;
                   }))
    return createStringError(
        errc::illegal_byte_sequence,
        "All vectors in the '" + std::string(Key) +
            "' section of the vocabulary are not of the same dimension");

  return Error::success();
}

// FIXME: Make this optional. We can avoid file reads
// by auto-generating a default vocabulary during the build time.
Error IR2VecVocabAnalysis::readVocabulary() {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(VocabFile, /*IsText=*/true);
  if (!BufOrError)
    return createFileError(VocabFile, BufOrError.getError());

  auto Content = BufOrError.get()->getBuffer();

  Expected<json::Value> ParsedVocabValue = json::parse(Content);
  if (!ParsedVocabValue)
    return ParsedVocabValue.takeError();

  unsigned OpcodeDim = 0, TypeDim = 0, ArgDim = 0;
  if (auto Err =
          parseVocabSection("Opcodes", *ParsedVocabValue, OpcVocab, OpcodeDim))
    return Err;

  if (auto Err =
          parseVocabSection("Types", *ParsedVocabValue, TypeVocab, TypeDim))
    return Err;

  if (auto Err =
          parseVocabSection("Arguments", *ParsedVocabValue, ArgVocab, ArgDim))
    return Err;

  if (!(OpcodeDim == TypeDim && TypeDim == ArgDim))
    return createStringError(errc::illegal_byte_sequence,
                             "Vocabulary sections have different dimensions");

  return Error::success();
}

void IR2VecVocabAnalysis::generateNumMappedVocab() {

  // Helper for handling missing entities in the vocabulary.
  // Currently, we use a zero vector. In the future, we will throw an error to
  // ensure that *all* known entities are present in the vocabulary.
  auto handleMissingEntity = [](const std::string &Val) {
    LLVM_DEBUG(errs() << Val
                      << " is not in vocabulary, using zero vector; This "
                         "would result in an error in future.\n");
    ++VocabMissCounter;
  };

  unsigned Dim = OpcVocab.begin()->second.size();
  assert(Dim > 0 && "Vocabulary dimension must be greater than zero");

  // Handle Opcodes
  std::vector<Embedding> NumericOpcodeEmbeddings(Vocabulary::MaxOpcodes,
                                                 Embedding(Dim, 0));
  NumericOpcodeEmbeddings.reserve(Vocabulary::MaxOpcodes);
  for (unsigned Opcode : seq(0u, Vocabulary::MaxOpcodes)) {
    StringRef VocabKey = Vocabulary::getVocabKeyForOpcode(Opcode + 1);
    auto It = OpcVocab.find(VocabKey.str());
    if (It != OpcVocab.end())
      NumericOpcodeEmbeddings[Opcode] = It->second;
    else
      handleMissingEntity(VocabKey.str());
  }
  Vocab.insert(Vocab.end(), NumericOpcodeEmbeddings.begin(),
               NumericOpcodeEmbeddings.end());

  // Handle Types - only canonical types are present in vocabulary
  std::vector<Embedding> NumericTypeEmbeddings(Vocabulary::MaxCanonicalTypeIDs,
                                               Embedding(Dim, 0));
  NumericTypeEmbeddings.reserve(Vocabulary::MaxCanonicalTypeIDs);
  for (unsigned CTypeID : seq(0u, Vocabulary::MaxCanonicalTypeIDs)) {
    StringRef VocabKey = Vocabulary::getVocabKeyForCanonicalTypeID(
        static_cast<Vocabulary::CanonicalTypeID>(CTypeID));
    if (auto It = TypeVocab.find(VocabKey.str()); It != TypeVocab.end()) {
      NumericTypeEmbeddings[CTypeID] = It->second;
      continue;
    }
    handleMissingEntity(VocabKey.str());
  }
  Vocab.insert(Vocab.end(), NumericTypeEmbeddings.begin(),
               NumericTypeEmbeddings.end());

  // Handle Arguments/Operands
  std::vector<Embedding> NumericArgEmbeddings(Vocabulary::MaxOperandKinds,
                                              Embedding(Dim, 0));
  NumericArgEmbeddings.reserve(Vocabulary::MaxOperandKinds);
  for (unsigned OpKind : seq(0u, Vocabulary::MaxOperandKinds)) {
    Vocabulary::OperandKind Kind = static_cast<Vocabulary::OperandKind>(OpKind);
    StringRef VocabKey = Vocabulary::getVocabKeyForOperandKind(Kind);
    auto It = ArgVocab.find(VocabKey.str());
    if (It != ArgVocab.end()) {
      NumericArgEmbeddings[OpKind] = It->second;
      continue;
    }
    handleMissingEntity(VocabKey.str());
  }
  Vocab.insert(Vocab.end(), NumericArgEmbeddings.begin(),
               NumericArgEmbeddings.end());
}

IR2VecVocabAnalysis::IR2VecVocabAnalysis(const VocabVector &Vocab)
    : Vocab(Vocab) {}

IR2VecVocabAnalysis::IR2VecVocabAnalysis(VocabVector &&Vocab)
    : Vocab(std::move(Vocab)) {}

void IR2VecVocabAnalysis::emitError(Error Err, LLVMContext &Ctx) {
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
    Ctx.emitError("Error reading vocabulary: " + EI.message());
  });
}

IR2VecVocabAnalysis::Result
IR2VecVocabAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  auto Ctx = &M.getContext();
  // If vocabulary is already populated by the constructor, use it.
  if (!Vocab.empty())
    return Vocabulary(std::move(Vocab));

  // Otherwise, try to read from the vocabulary file.
  if (VocabFile.empty()) {
    // FIXME: Use default vocabulary
    Ctx->emitError("IR2Vec vocabulary file path not specified; You may need to "
                   "set it using --ir2vec-vocab-path");
    return Vocabulary(); // Return invalid result
  }
  if (auto Err = readVocabulary()) {
    emitError(std::move(Err), *Ctx);
    return Vocabulary();
  }

  // Scale the vocabulary sections based on the provided weights
  auto scaleVocabSection = [](VocabMap &Vocab, double Weight) {
    for (auto &Entry : Vocab)
      Entry.second *= Weight;
  };
  scaleVocabSection(OpcVocab, OpcWeight);
  scaleVocabSection(TypeVocab, TypeWeight);
  scaleVocabSection(ArgVocab, ArgWeight);

  // Generate the numeric lookup vocabulary
  generateNumMappedVocab();

  return Vocabulary(std::move(Vocab));
}

// ==----------------------------------------------------------------------===//
// Printer Passes
//===----------------------------------------------------------------------===//

PreservedAnalyses IR2VecPrinterPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  auto Vocabulary = MAM.getResult<IR2VecVocabAnalysis>(M);
  assert(Vocabulary.isValid() && "IR2Vec Vocabulary is invalid");

  for (Function &F : M) {
    auto Emb = Embedder::create(IR2VecEmbeddingKind, F, Vocabulary);
    if (!Emb) {
      OS << "Error creating IR2Vec embeddings \n";
      continue;
    }

    OS << "IR2Vec embeddings for function " << F.getName() << ":\n";
    OS << "Function vector: ";
    Emb->getFunctionVector().print(OS);

    OS << "Basic block vectors:\n";
    const auto &BBMap = Emb->getBBVecMap();
    for (const BasicBlock &BB : F) {
      auto It = BBMap.find(&BB);
      if (It != BBMap.end()) {
        OS << "Basic block: " << BB.getName() << ":\n";
        It->second.print(OS);
      }
    }

    OS << "Instruction vectors:\n";
    const auto &InstMap = Emb->getInstVecMap();
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        auto It = InstMap.find(&I);
        if (It != InstMap.end()) {
          OS << "Instruction: ";
          I.print(OS);
          It->second.print(OS);
        }
      }
    }
  }
  return PreservedAnalyses::all();
}

PreservedAnalyses IR2VecVocabPrinterPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  auto IR2VecVocabulary = MAM.getResult<IR2VecVocabAnalysis>(M);
  assert(IR2VecVocabulary.isValid() && "IR2Vec Vocabulary is invalid");

  // Print each entry
  unsigned Pos = 0;
  for (const auto &Entry : IR2VecVocabulary) {
    OS << "Key: " << IR2VecVocabulary.getStringKey(Pos++) << ": ";
    Entry.print(OS);
  }
  return PreservedAnalyses::all();
}
