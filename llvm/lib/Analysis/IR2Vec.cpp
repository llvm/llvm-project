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

#include "llvm/ADT/Statistic.h"
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
          "Number of lookups to entites not present in the vocabulary");

namespace llvm {
namespace ir2vec {
static cl::OptionCategory IR2VecCategory("IR2Vec Options");

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
} // namespace ir2vec
} // namespace llvm

AnalysisKey IR2VecVocabAnalysis::Key;

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

Embedding &Embedding::operator-=(const Embedding &RHS) {
  assert(this->size() == RHS.size() && "Vectors must have the same dimension");
  std::transform(this->begin(), this->end(), RHS.begin(), this->begin(),
                 std::minus<double>());
  return *this;
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
    if (std::abs((*this)[Itr] - RHS[Itr]) > Tolerance)
      return false;
  return true;
}

// ==----------------------------------------------------------------------===//
// Embedder and its subclasses
//===----------------------------------------------------------------------===//

Embedder::Embedder(const Function &F, const Vocab &Vocabulary)
    : F(F), Vocabulary(Vocabulary),
      Dimension(Vocabulary.begin()->second.size()), OpcWeight(::OpcWeight),
      TypeWeight(::TypeWeight), ArgWeight(::ArgWeight) {}

Expected<std::unique_ptr<Embedder>>
Embedder::create(IR2VecKind Mode, const Function &F, const Vocab &Vocabulary) {
  switch (Mode) {
  case IR2VecKind::Symbolic:
    return std::make_unique<SymbolicEmbedder>(F, Vocabulary);
  }
  return make_error<StringError>("Unknown IR2VecKind", errc::invalid_argument);
}

// FIXME: Currently lookups are string based. Use numeric Keys
// for efficiency
Embedding Embedder::lookupVocab(const std::string &Key) const {
  Embedding Vec(Dimension, 0);
  // FIXME: Use zero vectors in vocab and assert failure for
  // unknown entities rather than silently returning zeroes here.
  auto It = Vocabulary.find(Key);
  if (It != Vocabulary.end())
    return It->second;
  LLVM_DEBUG(errs() << "cannot find key in map : " << Key << "\n");
  ++VocabMissCounter;
  return Vec;
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

#define RETURN_LOOKUP_IF(CONDITION, KEY_STR)                                   \
  if (CONDITION)                                                               \
    return lookupVocab(KEY_STR);

Embedding SymbolicEmbedder::getTypeEmbedding(const Type *Ty) const {
  RETURN_LOOKUP_IF(Ty->isVoidTy(), "voidTy");
  RETURN_LOOKUP_IF(Ty->isFloatingPointTy(), "floatTy");
  RETURN_LOOKUP_IF(Ty->isIntegerTy(), "integerTy");
  RETURN_LOOKUP_IF(Ty->isFunctionTy(), "functionTy");
  RETURN_LOOKUP_IF(Ty->isStructTy(), "structTy");
  RETURN_LOOKUP_IF(Ty->isArrayTy(), "arrayTy");
  RETURN_LOOKUP_IF(Ty->isPointerTy(), "pointerTy");
  RETURN_LOOKUP_IF(Ty->isVectorTy(), "vectorTy");
  RETURN_LOOKUP_IF(Ty->isEmptyTy(), "emptyTy");
  RETURN_LOOKUP_IF(Ty->isLabelTy(), "labelTy");
  RETURN_LOOKUP_IF(Ty->isTokenTy(), "tokenTy");
  RETURN_LOOKUP_IF(Ty->isMetadataTy(), "metadataTy");
  return lookupVocab("unknownTy");
}

Embedding SymbolicEmbedder::getOperandEmbedding(const Value *Op) const {
  RETURN_LOOKUP_IF(isa<Function>(Op), "function");
  RETURN_LOOKUP_IF(isa<PointerType>(Op->getType()), "pointer");
  RETURN_LOOKUP_IF(isa<Constant>(Op), "constant");
  return lookupVocab("variable");
}

#undef RETURN_LOOKUP_IF

void SymbolicEmbedder::computeEmbeddings(const BasicBlock &BB) const {
  Embedding BBVector(Dimension, 0);

  for (const auto &I : BB) {
    Embedding InstVector(Dimension, 0);

    const auto OpcVec = lookupVocab(I.getOpcodeName());
    InstVector.scaleAndAdd(OpcVec, OpcWeight);

    // FIXME: Currently lookups are string based. Use numeric Keys
    // for efficiency.
    const auto Type = I.getType();
    const auto TypeVec = getTypeEmbedding(Type);
    InstVector.scaleAndAdd(TypeVec, TypeWeight);

    for (const auto &Op : I.operands()) {
      const auto OperandVec = getOperandEmbedding(Op.get());
      InstVector.scaleAndAdd(OperandVec, ArgWeight);
    }
    InstVecMap[&I] = InstVector;
    BBVector += InstVector;
  }
  BBVecMap[&BB] = BBVector;
}

void SymbolicEmbedder::computeEmbeddings() const {
  if (F.isDeclaration())
    return;
  for (const auto &BB : F) {
    computeEmbeddings(BB);
    FuncVector += BBVecMap[&BB];
  }
}

// ==----------------------------------------------------------------------===//
// IR2VecVocabResult and IR2VecVocabAnalysis
//===----------------------------------------------------------------------===//

IR2VecVocabResult::IR2VecVocabResult(ir2vec::Vocab &&Vocabulary)
    : Vocabulary(std::move(Vocabulary)), Valid(true) {}

const ir2vec::Vocab &IR2VecVocabResult::getVocabulary() const {
  assert(Valid && "IR2Vec Vocabulary is invalid");
  return Vocabulary;
}

unsigned IR2VecVocabResult::getDimension() const {
  assert(Valid && "IR2Vec Vocabulary is invalid");
  return Vocabulary.begin()->second.size();
}

// For now, assume vocabulary is stable unless explicitly invalidated.
bool IR2VecVocabResult::invalidate(
    Module &M, const PreservedAnalyses &PA,
    ModuleAnalysisManager::Invalidator &Inv) const {
  auto PAC = PA.getChecker<IR2VecVocabAnalysis>();
  return !(PAC.preservedWhenStateless());
}

// FIXME: Make this optional. We can avoid file reads
// by auto-generating a default vocabulary during the build time.
Error IR2VecVocabAnalysis::readVocabulary() {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(VocabFile, /*IsText=*/true);
  if (!BufOrError)
    return createFileError(VocabFile, BufOrError.getError());

  auto Content = BufOrError.get()->getBuffer();
  json::Path::Root Path("");
  Expected<json::Value> ParsedVocabValue = json::parse(Content);
  if (!ParsedVocabValue)
    return ParsedVocabValue.takeError();

  bool Res = json::fromJSON(*ParsedVocabValue, Vocabulary, Path);
  if (!Res)
    return createStringError(errc::illegal_byte_sequence,
                             "Unable to parse the vocabulary");

  if (Vocabulary.empty())
    return createStringError(errc::illegal_byte_sequence,
                             "Vocabulary is empty");

  unsigned Dim = Vocabulary.begin()->second.size();
  if (Dim == 0)
    return createStringError(errc::illegal_byte_sequence,
                             "Dimension of vocabulary is zero");

  if (!std::all_of(Vocabulary.begin(), Vocabulary.end(),
                   [Dim](const std::pair<StringRef, Embedding> &Entry) {
                     return Entry.second.size() == Dim;
                   }))
    return createStringError(
        errc::illegal_byte_sequence,
        "All vectors in the vocabulary are not of the same dimension");

  return Error::success();
}

IR2VecVocabAnalysis::IR2VecVocabAnalysis(const Vocab &Vocabulary)
    : Vocabulary(Vocabulary) {}

IR2VecVocabAnalysis::IR2VecVocabAnalysis(Vocab &&Vocabulary)
    : Vocabulary(std::move(Vocabulary)) {}

void IR2VecVocabAnalysis::emitError(Error Err, LLVMContext &Ctx) {
  handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
    Ctx.emitError("Error reading vocabulary: " + EI.message());
  });
}

IR2VecVocabAnalysis::Result
IR2VecVocabAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  auto Ctx = &M.getContext();
  // FIXME: Scale the vocabulary once. This would avoid scaling per use later.
  // If vocabulary is already populated by the constructor, use it.
  if (!Vocabulary.empty())
    return IR2VecVocabResult(std::move(Vocabulary));

  // Otherwise, try to read from the vocabulary file.
  if (VocabFile.empty()) {
    // FIXME: Use default vocabulary
    Ctx->emitError("IR2Vec vocabulary file path not specified");
    return IR2VecVocabResult(); // Return invalid result
  }
  if (auto Err = readVocabulary()) {
    emitError(std::move(Err), *Ctx);
    return IR2VecVocabResult();
  }
  return IR2VecVocabResult(std::move(Vocabulary));
}

// ==----------------------------------------------------------------------===//
// IR2VecPrinterPass
//===----------------------------------------------------------------------===//

void IR2VecPrinterPass::printVector(const Embedding &Vec) const {
  OS << " [";
  for (const auto &Elem : Vec)
    OS << " " << format("%.2f", Elem) << " ";
  OS << "]\n";
}

PreservedAnalyses IR2VecPrinterPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  auto IR2VecVocabResult = MAM.getResult<IR2VecVocabAnalysis>(M);
  assert(IR2VecVocabResult.isValid() && "IR2Vec Vocabulary is invalid");

  auto Vocab = IR2VecVocabResult.getVocabulary();
  for (Function &F : M) {
    Expected<std::unique_ptr<Embedder>> EmbOrErr =
        Embedder::create(IR2VecKind::Symbolic, F, Vocab);
    if (auto Err = EmbOrErr.takeError()) {
      handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
        OS << "Error creating IR2Vec embeddings: " << EI.message() << "\n";
      });
      continue;
    }

    std::unique_ptr<Embedder> Emb = std::move(*EmbOrErr);

    OS << "IR2Vec embeddings for function " << F.getName() << ":\n";
    OS << "Function vector: ";
    printVector(Emb->getFunctionVector());

    OS << "Basic block vectors:\n";
    const auto &BBMap = Emb->getBBVecMap();
    for (const BasicBlock &BB : F) {
      auto It = BBMap.find(&BB);
      if (It != BBMap.end()) {
        OS << "Basic block: " << BB.getName() << ":\n";
        printVector(It->second);
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
          printVector(It->second);
        }
      }
    }
  }
  return PreservedAnalyses::all();
}
