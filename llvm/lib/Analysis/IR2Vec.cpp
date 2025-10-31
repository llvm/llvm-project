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
#include "llvm/ADT/SmallVector.h"
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

Embedding Embedder::computeEmbeddings() const {
  Embedding FuncVector(Dimension, 0.0);

  if (F.isDeclaration())
    return FuncVector;

  // Consider only the basic blocks that are reachable from entry
  for (const BasicBlock *BB : depth_first(&F))
    FuncVector += computeEmbeddings(*BB);
  return FuncVector;
}

Embedding Embedder::computeEmbeddings(const BasicBlock &BB) const {
  Embedding BBVector(Dimension, 0);

  // We consider only the non-debug and non-pseudo instructions
  for (const auto &I : BB.instructionsWithoutDebug())
    BBVector += computeEmbeddings(I);
  return BBVector;
}

Embedding SymbolicEmbedder::computeEmbeddings(const Instruction &I) const {
  // Currently, we always (re)compute the embeddings for symbolic embedder.
  // This is cheaper than caching the vectors.
  Embedding ArgEmb(Dimension, 0);
  for (const auto &Op : I.operands())
    ArgEmb += Vocab[*Op];
  auto InstVector =
      Vocab[I.getOpcode()] + Vocab[I.getType()->getTypeID()] + ArgEmb;
  if (const auto *IC = dyn_cast<CmpInst>(&I))
    InstVector += Vocab[IC->getPredicate()];
  return InstVector;
}

Embedding FlowAwareEmbedder::computeEmbeddings(const Instruction &I) const {
  // If we have already computed the embedding for this instruction, return it
  auto It = InstVecMap.find(&I);
  if (It != InstVecMap.end())
    return It->second;

  // TODO: Handle call instructions differently.
  // For now, we treat them like other instructions
  Embedding ArgEmb(Dimension, 0);
  for (const auto &Op : I.operands()) {
    // If the operand is defined elsewhere, we use its embedding
    if (const auto *DefInst = dyn_cast<Instruction>(Op)) {
      auto DefIt = InstVecMap.find(DefInst);
      // Fixme (#159171): Ideally we should never miss an instruction
      // embedding here.
      // But when we have cyclic dependencies (e.g., phi
      // nodes), we might miss the embedding. In such cases, we fall back to
      // using the vocabulary embedding. This can be fixed by iterating to a
      // fixed-point, or by using a simple solver for the set of simultaneous
      // equations.
      // Another case when we might miss an instruction embedding is when
      // the operand instruction is in a different basic block that has not
      // been processed yet. This can be fixed by processing the basic blocks
      // in a topological order.
      if (DefIt != InstVecMap.end())
        ArgEmb += DefIt->second;
      else
        ArgEmb += Vocab[*Op];
    }
    // If the operand is not defined by an instruction, we use the
    // vocabulary
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
  if (const auto *IC = dyn_cast<CmpInst>(&I))
    InstVector += Vocab[IC->getPredicate()];
  InstVecMap[&I] = InstVector;
  return InstVector;
}

// ==----------------------------------------------------------------------===//
// VocabStorage
//===----------------------------------------------------------------------===//

VocabStorage::VocabStorage(std::vector<std::vector<Embedding>> &&SectionData)
    : Sections(std::move(SectionData)), TotalSize([&] {
        assert(!Sections.empty() && "Vocabulary has no sections");
        // Compute total size across all sections
        size_t Size = 0;
        for (const auto &Section : Sections) {
          assert(!Section.empty() && "Vocabulary section is empty");
          Size += Section.size();
        }
        return Size;
      }()),
      Dimension([&] {
        // Get dimension from the first embedding in the first section - all
        // embeddings must have the same dimension
        assert(!Sections.empty() && "Vocabulary has no sections");
        assert(!Sections[0].empty() && "First section of vocabulary is empty");
        unsigned ExpectedDim = static_cast<unsigned>(Sections[0][0].size());

        // Verify that all embeddings across all sections have the same
        // dimension
        [[maybe_unused]] auto allSameDim =
            [ExpectedDim](const std::vector<Embedding> &Section) {
              return std::all_of(Section.begin(), Section.end(),
                                 [ExpectedDim](const Embedding &Emb) {
                                   return Emb.size() == ExpectedDim;
                                 });
            };
        assert(std::all_of(Sections.begin(), Sections.end(), allSameDim) &&
               "All embeddings must have the same dimension");

        return ExpectedDim;
      }()) {}

const Embedding &VocabStorage::const_iterator::operator*() const {
  assert(SectionId < Storage->Sections.size() && "Invalid section ID");
  assert(LocalIndex < Storage->Sections[SectionId].size() &&
         "Local index out of range");
  return Storage->Sections[SectionId][LocalIndex];
}

VocabStorage::const_iterator &VocabStorage::const_iterator::operator++() {
  ++LocalIndex;
  // Check if we need to move to the next section
  if (SectionId < Storage->getNumSections() &&
      LocalIndex >= Storage->Sections[SectionId].size()) {
    assert(LocalIndex == Storage->Sections[SectionId].size() &&
           "Local index should be at the end of the current section");
    LocalIndex = 0;
    ++SectionId;
  }
  return *this;
}

bool VocabStorage::const_iterator::operator==(
    const const_iterator &Other) const {
  return Storage == Other.Storage && SectionId == Other.SectionId &&
         LocalIndex == Other.LocalIndex;
}

bool VocabStorage::const_iterator::operator!=(
    const const_iterator &Other) const {
  return !(*this == Other);
}

Error VocabStorage::parseVocabSection(StringRef Key,
                                      const json::Value &ParsedVocabValue,
                                      VocabMap &TargetVocab, unsigned &Dim) {
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

// ==----------------------------------------------------------------------===//
// Vocabulary
//===----------------------------------------------------------------------===//

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

unsigned Vocabulary::getPredicateLocalIndex(CmpInst::Predicate P) {
  if (P >= CmpInst::FIRST_FCMP_PREDICATE && P <= CmpInst::LAST_FCMP_PREDICATE)
    return P - CmpInst::FIRST_FCMP_PREDICATE;
  else
    return P - CmpInst::FIRST_ICMP_PREDICATE +
           (CmpInst::LAST_FCMP_PREDICATE - CmpInst::FIRST_FCMP_PREDICATE + 1);
}

CmpInst::Predicate Vocabulary::getPredicateFromLocalIndex(unsigned LocalIndex) {
  unsigned fcmpRange =
      CmpInst::LAST_FCMP_PREDICATE - CmpInst::FIRST_FCMP_PREDICATE + 1;
  if (LocalIndex < fcmpRange)
    return static_cast<CmpInst::Predicate>(CmpInst::FIRST_FCMP_PREDICATE +
                                           LocalIndex);
  else
    return static_cast<CmpInst::Predicate>(CmpInst::FIRST_ICMP_PREDICATE +
                                           LocalIndex - fcmpRange);
}

StringRef Vocabulary::getVocabKeyForPredicate(CmpInst::Predicate Pred) {
  static SmallString<16> PredNameBuffer;
  if (Pred < CmpInst::FIRST_ICMP_PREDICATE)
    PredNameBuffer = "FCMP_";
  else
    PredNameBuffer = "ICMP_";
  PredNameBuffer += CmpInst::getPredicateName(Pred);
  return PredNameBuffer;
}

StringRef Vocabulary::getStringKey(unsigned Pos) {
  assert(Pos < NumCanonicalEntries && "Position out of bounds in vocabulary");
  // Opcode
  if (Pos < MaxOpcodes)
    return getVocabKeyForOpcode(Pos + 1);
  // Type
  if (Pos < OperandBaseOffset)
    return getVocabKeyForCanonicalTypeID(
        static_cast<CanonicalTypeID>(Pos - MaxOpcodes));
  // Operand
  if (Pos < PredicateBaseOffset)
    return getVocabKeyForOperandKind(
        static_cast<OperandKind>(Pos - OperandBaseOffset));
  // Predicates
  return getVocabKeyForPredicate(getPredicate(Pos - PredicateBaseOffset));
}

// For now, assume vocabulary is stable unless explicitly invalidated.
bool Vocabulary::invalidate(Module &M, const PreservedAnalyses &PA,
                            ModuleAnalysisManager::Invalidator &Inv) const {
  auto PAC = PA.getChecker<IR2VecVocabAnalysis>();
  return !(PAC.preservedWhenStateless());
}

VocabStorage Vocabulary::createDummyVocabForTest(unsigned Dim) {
  float DummyVal = 0.1f;

  // Create sections for opcodes, types, operands, and predicates
  // Order must match Vocabulary::Section enum
  std::vector<std::vector<Embedding>> Sections;
  Sections.reserve(4);

  // Opcodes section
  std::vector<Embedding> OpcodeSec;
  OpcodeSec.reserve(MaxOpcodes);
  for (unsigned I = 0; I < MaxOpcodes; ++I) {
    OpcodeSec.emplace_back(Dim, DummyVal);
    DummyVal += 0.1f;
  }
  Sections.push_back(std::move(OpcodeSec));

  // Types section
  std::vector<Embedding> TypeSec;
  TypeSec.reserve(MaxCanonicalTypeIDs);
  for (unsigned I = 0; I < MaxCanonicalTypeIDs; ++I) {
    TypeSec.emplace_back(Dim, DummyVal);
    DummyVal += 0.1f;
  }
  Sections.push_back(std::move(TypeSec));

  // Operands section
  std::vector<Embedding> OperandSec;
  OperandSec.reserve(MaxOperandKinds);
  for (unsigned I = 0; I < MaxOperandKinds; ++I) {
    OperandSec.emplace_back(Dim, DummyVal);
    DummyVal += 0.1f;
  }
  Sections.push_back(std::move(OperandSec));

  // Predicates section
  std::vector<Embedding> PredicateSec;
  PredicateSec.reserve(MaxPredicateKinds);
  for (unsigned I = 0; I < MaxPredicateKinds; ++I) {
    PredicateSec.emplace_back(Dim, DummyVal);
    DummyVal += 0.1f;
  }
  Sections.push_back(std::move(PredicateSec));

  return VocabStorage(std::move(Sections));
}

// ==----------------------------------------------------------------------===//
// IR2VecVocabAnalysis
//===----------------------------------------------------------------------===//

// FIXME: Make this optional. We can avoid file reads
// by auto-generating a default vocabulary during the build time.
Error IR2VecVocabAnalysis::readVocabulary(VocabMap &OpcVocab,
                                          VocabMap &TypeVocab,
                                          VocabMap &ArgVocab) {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(VocabFile, /*IsText=*/true);
  if (!BufOrError)
    return createFileError(VocabFile, BufOrError.getError());

  auto Content = BufOrError.get()->getBuffer();

  Expected<json::Value> ParsedVocabValue = json::parse(Content);
  if (!ParsedVocabValue)
    return ParsedVocabValue.takeError();

  unsigned OpcodeDim = 0, TypeDim = 0, ArgDim = 0;
  if (auto Err = VocabStorage::parseVocabSection("Opcodes", *ParsedVocabValue,
                                                 OpcVocab, OpcodeDim))
    return Err;

  if (auto Err = VocabStorage::parseVocabSection("Types", *ParsedVocabValue,
                                                 TypeVocab, TypeDim))
    return Err;

  if (auto Err = VocabStorage::parseVocabSection("Arguments", *ParsedVocabValue,
                                                 ArgVocab, ArgDim))
    return Err;

  if (!(OpcodeDim == TypeDim && TypeDim == ArgDim))
    return createStringError(errc::illegal_byte_sequence,
                             "Vocabulary sections have different dimensions");

  return Error::success();
}

void IR2VecVocabAnalysis::generateVocabStorage(VocabMap &OpcVocab,
                                               VocabMap &TypeVocab,
                                               VocabMap &ArgVocab) {

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
                                                 Embedding(Dim));
  for (unsigned Opcode : seq(0u, Vocabulary::MaxOpcodes)) {
    StringRef VocabKey = Vocabulary::getVocabKeyForOpcode(Opcode + 1);
    auto It = OpcVocab.find(VocabKey.str());
    if (It != OpcVocab.end())
      NumericOpcodeEmbeddings[Opcode] = It->second;
    else
      handleMissingEntity(VocabKey.str());
  }

  // Handle Types - only canonical types are present in vocabulary
  std::vector<Embedding> NumericTypeEmbeddings(Vocabulary::MaxCanonicalTypeIDs,
                                               Embedding(Dim));
  for (unsigned CTypeID : seq(0u, Vocabulary::MaxCanonicalTypeIDs)) {
    StringRef VocabKey = Vocabulary::getVocabKeyForCanonicalTypeID(
        static_cast<Vocabulary::CanonicalTypeID>(CTypeID));
    if (auto It = TypeVocab.find(VocabKey.str()); It != TypeVocab.end()) {
      NumericTypeEmbeddings[CTypeID] = It->second;
      continue;
    }
    handleMissingEntity(VocabKey.str());
  }

  // Handle Arguments/Operands
  std::vector<Embedding> NumericArgEmbeddings(Vocabulary::MaxOperandKinds,
                                              Embedding(Dim));
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

  // Handle Predicates: part of Operands section. We look up predicate keys
  // in ArgVocab.
  std::vector<Embedding> NumericPredEmbeddings(Vocabulary::MaxPredicateKinds,
                                               Embedding(Dim, 0));
  for (unsigned PK : seq(0u, Vocabulary::MaxPredicateKinds)) {
    StringRef VocabKey =
        Vocabulary::getVocabKeyForPredicate(Vocabulary::getPredicate(PK));
    auto It = ArgVocab.find(VocabKey.str());
    if (It != ArgVocab.end()) {
      NumericPredEmbeddings[PK] = It->second;
      continue;
    }
    handleMissingEntity(VocabKey.str());
  }

  // Create section-based storage instead of flat vocabulary
  // Order must match Vocabulary::Section enum
  std::vector<std::vector<Embedding>> Sections(4);
  Sections[static_cast<unsigned>(Vocabulary::Section::Opcodes)] =
      std::move(NumericOpcodeEmbeddings); // Section::Opcodes
  Sections[static_cast<unsigned>(Vocabulary::Section::CanonicalTypes)] =
      std::move(NumericTypeEmbeddings); // Section::CanonicalTypes
  Sections[static_cast<unsigned>(Vocabulary::Section::Operands)] =
      std::move(NumericArgEmbeddings); // Section::Operands
  Sections[static_cast<unsigned>(Vocabulary::Section::Predicates)] =
      std::move(NumericPredEmbeddings); // Section::Predicates

  // Create VocabStorage from organized sections
  Vocab.emplace(std::move(Sections));
}

void IR2VecVocabAnalysis::emitError(Error Err, LLVMContext &Ctx) {
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
    Ctx.emitError("Error reading vocabulary: " + EI.message());
  });
}

IR2VecVocabAnalysis::Result
IR2VecVocabAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  auto Ctx = &M.getContext();
  // If vocabulary is already populated by the constructor, use it.
  if (Vocab.has_value())
    return Vocabulary(std::move(Vocab.value()));

  // Otherwise, try to read from the vocabulary file.
  if (VocabFile.empty()) {
    // FIXME: Use default vocabulary
    Ctx->emitError("IR2Vec vocabulary file path not specified; You may need to "
                   "set it using --ir2vec-vocab-path");
    return Vocabulary(); // Return invalid result
  }

  VocabMap OpcVocab, TypeVocab, ArgVocab;
  if (auto Err = readVocabulary(OpcVocab, TypeVocab, ArgVocab)) {
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
  generateVocabStorage(OpcVocab, TypeVocab, ArgVocab);

  return Vocabulary(std::move(Vocab.value()));
}

// ==----------------------------------------------------------------------===//
// Printer Passes
//===----------------------------------------------------------------------===//

PreservedAnalyses IR2VecPrinterPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  auto &Vocabulary = MAM.getResult<IR2VecVocabAnalysis>(M);
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
    for (const BasicBlock &BB : F) {
      OS << "Basic block: " << BB.getName() << ":\n";
      Emb->getBBVector(BB).print(OS);
    }

    OS << "Instruction vectors:\n";
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        OS << "Instruction: ";
        I.print(OS);
        Emb->getInstVector(I).print(OS);
      }
    }
  }
  return PreservedAnalyses::all();
}

PreservedAnalyses IR2VecVocabPrinterPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  auto &IR2VecVocabulary = MAM.getResult<IR2VecVocabAnalysis>(M);
  assert(IR2VecVocabulary.isValid() && "IR2Vec Vocabulary is invalid");

  // Print each entry
  unsigned Pos = 0;
  for (const auto &Entry : IR2VecVocabulary) {
    OS << "Key: " << IR2VecVocabulary.getStringKey(Pos++) << ": ";
    Entry.print(OS);
  }
  return PreservedAnalyses::all();
}
