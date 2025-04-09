//===- IR2VecAnalysis.cpp - IR2Vec Analysis Implementation ----------------===//
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

#include "llvm/Analysis/IR2VecAnalysis.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace ir2vec;

#define DEBUG_TYPE "ir2vec"

STATISTIC(DataMissCounter, "Number of data misses in the vocabulary");

/// IR2Vec computes two kinds of embeddings: Symbolic and Flow-aware.
/// Symbolic embeddings capture the "syntactic" and "statistical correlation"
/// of the IR entities. Flow-aware embeddings build on top of symbolic
/// embeddings and additionally capture the flow information in the IR.
/// IR2VecKind is used to specify the type of embeddings to generate.
// FIXME: Currently we support only Symbolic.  Add support for
// Flow-aware in upcoming patches.
enum class IR2VecKind { Symbolic, FlowAware };

static cl::OptionCategory IR2VecAnalysisCategory("IR2Vec Analysis Options");

cl::opt<IR2VecKind>
    IR2VecMode("ir2vec-mode",
               cl::desc("Choose type of embeddings to generate:"),
               cl::values(clEnumValN(IR2VecKind::Symbolic, "symbolic",
                                     "Generates symbolic embeddings"),
                          clEnumValN(IR2VecKind::FlowAware, "flow-aware",
                                     "Generates flow-aware embeddings")),
               cl::init(IR2VecKind::Symbolic), cl::cat(IR2VecAnalysisCategory));

// FIXME: Use a default vocab when not specified
static cl::opt<std::string>
    VocabFile("ir2vec-vocab-path", cl::Optional,
              cl::desc("Path to the vocabulary file for IR2Vec"), cl::init(""),
              cl::cat(IR2VecAnalysisCategory));

AnalysisKey IR2VecVocabAnalysis::Key;
AnalysisKey IR2VecAnalysis::Key;

// ==----------------------------------------------------------------------===//
// Embeddings and its subclasses
//===----------------------------------------------------------------------===//

namespace {
/// Embeddings provides the interface to generate vector representations for
/// instructions, basic blocks, and functions. The vector
/// representations are generated using IR2Vec algorithms.
///
/// The Embeddings class is an abstract class and it is intended to be
/// subclassed for different IR2Vec algorithms like Symbolic and Flow-aware.
class Embeddings {
protected:
  const Function &F;
  Vocab Vocabulary;

  /// Weights for different entities (like opcode, arguments, types)
  /// in the IR instructions to generate the vector representation.
  // FIXME: Defaults to the values used in the original algorithm. Can be
  // parameterized later.
  const float OpcWeight = 1.0, TypeWeight = 0.5, ArgWeight = 0.2;

  /// Dimension of the vector representation; captured from the input vocabulary
  const unsigned Dimension = 300;

  // Utility maps - these are used to store the vector representations of
  // instructions, basic blocks and functions.
  Embedding FuncVector;
  SmallMapVector<const BasicBlock *, Embedding, 16> BBVecMap;
  SmallMapVector<const Instruction *, Embedding, 128> InstVecMap;

  Embeddings(const Function &F, const Vocab &Vocabulary, unsigned Dimension)
      : F(F), Vocabulary(Vocabulary), Dimension(Dimension) {}

  /// Lookup vocabulary for a given Key. If the key is not found, it returns a
  /// zero vector.
  Embedding lookupVocab(const std::string &Key);

public:
  virtual ~Embeddings() = default;

  /// Top level function to compute embeddings. Given a function, it
  /// generates embeddings for all the instructions and basic blocks in that
  /// function. Logic of computing the embeddings is specific to the kind of
  /// embeddings being computed.
  virtual void computeEmbeddings() = 0;

  /// Returns a map containing instructions and the corresponding vector
  /// representations for a given module corresponding to the IR2Vec
  /// algorithm.
  const SmallMapVector<const Instruction *, Embedding, 128> &
  getInstVecMap() const {
    return InstVecMap;
  }

  /// Returns a map containing basic block and the corresponding vector
  /// representations for a given module corresponding to the IR2Vec
  /// algorithm.
  const SmallMapVector<const BasicBlock *, Embedding, 16> &getBBVecMap() const {
    return BBVecMap;
  }

  /// Returns the vector representation for a given function corresponding to
  /// the IR2Vec algorithm.
  const Embedding &getFunctionVector() const { return FuncVector; }
};

/// Class for computing the Symbolic embeddings of IR2Vec
class Symbolic : public Embeddings {
private:
  /// Utility function to compute the vector representation for a given basic
  /// block.
  Embedding computeBB2Vec(const BasicBlock &BB);

  /// Utility function to compute the vector representation for a given
  /// function.
  Embedding computeFunc2Vec();

public:
  Symbolic(const Function &F, const Vocab &Vocabulary, unsigned Dimension)
      : Embeddings(F, Vocabulary, Dimension) {
    FuncVector = Embedding(Dimension, 0);
  }
  void computeEmbeddings() override;
};

/// Scales the vector Vec by Factor
void scaleVector(Embedding &Vec, const float Factor) {
  std::transform(Vec.begin(), Vec.end(), Vec.begin(),
                 [Factor](double X) { return X * Factor; });
}

/// Adds two vectors: Vec += Vec2
void addVectors(Embedding &Vec, const Embedding &Vec2) {
  std::transform(Vec.begin(), Vec.end(), Vec2.begin(), Vec.begin(),
                 std::plus<double>());
}

// FIXME: Currently lookups are string based. Use numeric Keys
// for efficiency.
Embedding Embeddings::lookupVocab(const std::string &Key) {
  Embedding Vec(Dimension, 0);
  // FIXME: Use zero vectors in vocab and assert failure for
  // unknown entities rather than silently returning zeroes here.
  auto It = Vocabulary.find(Key);
  if (It == Vocabulary.end()) {
    LLVM_DEBUG(errs() << "cannot find key in map : " << Key << "\n");
    ++DataMissCounter;
  } else {
    Vec = It->second;
  }
  return Vec;
}

void Symbolic::computeEmbeddings() {
  if (F.isDeclaration())
    return;
  for (auto &BB : F) {
    auto Result = BBVecMap.try_emplace(&BB);
    if (!Result.second)
      continue;
    auto It = Result.first;
    It->second = std::move(computeBB2Vec(BB));
    addVectors(FuncVector, It->second);
  }
}

Embedding Symbolic::computeBB2Vec(const BasicBlock &BB) {
  Embedding BBVector(Dimension, 0);

  for (auto &I : BB) {
    Embedding InstVector(Dimension, 0);

    auto Vec = lookupVocab(I.getOpcodeName());
    scaleVector(Vec, OpcWeight);
    addVectors(InstVector, Vec);

    // FIXME: Currently lookups are string based. Use numeric Keys
    // for efficiency.
    auto Type = I.getType();
    if (Type->isVoidTy()) {
      Vec = lookupVocab("voidTy");
    } else if (Type->isFloatingPointTy()) {
      Vec = lookupVocab("floatTy");
    } else if (Type->isIntegerTy()) {
      Vec = lookupVocab("integerTy");
    } else if (Type->isFunctionTy()) {
      Vec = lookupVocab("functionTy");
    } else if (Type->isStructTy()) {
      Vec = lookupVocab("structTy");
    } else if (Type->isArrayTy()) {
      Vec = lookupVocab("arrayTy");
    } else if (Type->isPointerTy()) {
      Vec = lookupVocab("pointerTy");
    } else if (Type->isVectorTy()) {
      Vec = lookupVocab("vectorTy");
    } else if (Type->isEmptyTy()) {
      Vec = lookupVocab("emptyTy");
    } else if (Type->isLabelTy()) {
      Vec = lookupVocab("labelTy");
    } else if (Type->isTokenTy()) {
      Vec = lookupVocab("tokenTy");
    } else if (Type->isMetadataTy()) {
      Vec = lookupVocab("metadataTy");
    } else {
      Vec = lookupVocab("unknownTy");
    }
    scaleVector(Vec, TypeWeight);
    addVectors(InstVector, Vec);

    for (const auto &Op : I.operands()) {
      Embedding Vec;
      if (isa<Function>(Op)) {
        Vec = lookupVocab("function");
      } else if (isa<PointerType>(Op->getType())) {
        Vec = lookupVocab("pointer");
      } else if (isa<Constant>(Op)) {
        Vec = lookupVocab("constant");
      } else {
        Vec = lookupVocab("variable");
      }
      scaleVector(Vec, ArgWeight);
      addVectors(InstVector, Vec);
    }
    InstVecMap[&I] = InstVector;
    addVectors(BBVector, InstVector);
  }
  return BBVector;
}
} // namespace

// ==----------------------------------------------------------------------===//
// IR2VecVocabResult and IR2VecVocabAnalysis
//===----------------------------------------------------------------------===//

IR2VecVocabResult::IR2VecVocabResult(ir2vec::Vocab &&Vocabulary)
    : Vocabulary(std::move(Vocabulary)), Valid(true) {}

const ir2vec::Vocab &IR2VecVocabResult::getVocabulary() const {
  assert(Valid);
  return Vocabulary;
}

unsigned IR2VecVocabResult::getDimension() const {
  assert(Valid);
  return Vocabulary.begin()->second.size();
}

// For now, assume vocabulary is stable unless explicitly invalidated.
bool IR2VecVocabResult::invalidate(Module &M, const PreservedAnalyses &PA,
                                   ModuleAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<IR2VecVocabAnalysis>();
  return !(PAC.preservedWhenStateless());
}

// FIXME: Make this optional. We can avoid file reads
// by auto-generating a default vocabulary during the build time.
Error IR2VecVocabAnalysis::readVocabulary() {
  auto BufOrError = MemoryBuffer::getFileOrSTDIN(VocabFile, /*IsText=*/true);
  if (!BufOrError) {
    return createFileError(VocabFile, BufOrError.getError());
  }
  auto Content = BufOrError.get()->getBuffer();
  json::Path::Root Path("");
  Expected<json::Value> ParsedVocabValue = json::parse(Content);
  if (!ParsedVocabValue)
    return ParsedVocabValue.takeError();

  bool Res = json::fromJSON(*ParsedVocabValue, Vocabulary, Path);
  if (!Res) {
    return createStringError(errc::illegal_byte_sequence,
                             "Unable to parse the vocabulary");
  }
  assert(Vocabulary.size() > 0 && "Vocabulary is empty");

  unsigned Dim = Vocabulary.begin()->second.size();
  assert(Dim > 0 && "Dimension of vocabulary is zero");
  assert(std::all_of(Vocabulary.begin(), Vocabulary.end(),
                     [Dim](const std::pair<StringRef, Embedding> &Entry) {
                       return Entry.second.size() == Dim;
                     }) &&
         "All vectors in the vocabulary are not of the same dimension");
  return Error::success();
}

IR2VecVocabAnalysis::Result
IR2VecVocabAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  auto Ctx = &M.getContext();
  if (VocabFile.empty()) {
    // FIXME: Use default vocabulary
    Ctx->emitError("IR2Vec vocabulary file path not specified");
    return IR2VecVocabResult(); // Return invalid result
  }
  if (auto Err = readVocabulary()) {
    handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EI) {
      Ctx->emitError("Error reading vocabulary: " + EI.message());
    });
    return IR2VecVocabResult();
  }
  return IR2VecVocabResult(std::move(Vocabulary));
}

// ==----------------------------------------------------------------------===//
// IR2VecResult and IR2VecAnalysis
//===----------------------------------------------------------------------===//

IR2VecResult::IR2VecResult(
    const SmallMapVector<const Instruction *, Embedding, 128> &&InstMap,
    const SmallMapVector<const BasicBlock *, Embedding, 16> &&BBMap,
    const Embedding &&FuncVector)
    : InstVecMap(std::move(InstMap)), BBVecMap(std::move(BBMap)),
      FuncVector(std::move(FuncVector)), Valid(true) {}

const SmallMapVector<const Instruction *, Embedding, 128> &
IR2VecResult::getInstVecMap() const {
  assert(Valid);
  return InstVecMap;
}
const SmallMapVector<const BasicBlock *, Embedding, 16> &
IR2VecResult::getBBVecMap() const {
  assert(Valid);
  return BBVecMap;
}
const Embedding &IR2VecResult::getFunctionVector() const {
  assert(Valid);
  return FuncVector;
}

IR2VecAnalysis::Result IR2VecAnalysis::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  auto *VocabRes = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F)
                       .getCachedResult<IR2VecVocabAnalysis>(*F.getParent());
  auto Ctx = &F.getContext();
  if (!VocabRes->isValid()) {
    Ctx->emitError("IR2Vec vocabulary is invalid");
    return IR2VecResult();
  }

  auto Dim = VocabRes->getDimension();
  if (Dim <= 0) {
    Ctx->emitError("IR2Vec vocabulary dimension is zero");
    return IR2VecResult();
  }

  auto Vocabulary = VocabRes->getVocabulary();
  std::unique_ptr<Embeddings> Emb;
  switch (IR2VecMode) {
  case IR2VecKind::Symbolic:
    Emb = std::make_unique<Symbolic>(F, Vocabulary, Dim);
    break;
  case IR2VecKind::FlowAware:
    // FIXME: Add support for flow-aware embeddings
  default:
    Ctx->emitError("Invalid IR2Vec mode");
    return IR2VecResult();
  }

  Emb->computeEmbeddings();
  auto &InstMap = Emb->getInstVecMap();
  auto &BBMap = Emb->getBBVecMap();
  auto &FuncVec = Emb->getFunctionVector();
  return IR2VecResult(std::move(InstMap), std::move(BBMap), std::move(FuncVec));
}

// ==----------------------------------------------------------------------===//
// IR2VecPrinterPass
//===----------------------------------------------------------------------===//

void IR2VecPrinterPass::printVector(const Embedding &Vec) const {
  OS << " [";
  for (auto &Elem : Vec)
    OS << " " << format("%.2f", Elem) << " ";
  OS << "]\n";
}

PreservedAnalyses IR2VecPrinterPass::run(Module &M,
                                         ModuleAnalysisManager &MAM) {
  auto IR2VecVocabResult = MAM.getResult<IR2VecVocabAnalysis>(M);
  assert(IR2VecVocabResult.isValid() && "Vocab is invalid");

  for (Function &F : M) {
    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    auto IR2VecRes = FAM.getResult<IR2VecAnalysis>(F);

    if (!IR2VecRes.isValid()) {
      auto Ctx = &F.getContext();
      Ctx->emitError("IR2Vec embeddings are invalid");
      return PreservedAnalyses::all();
    }

    OS << "IR2Vec embeddings for function " << F.getName() << ":\n";
    OS << "Function vector: ";
    printVector(IR2VecRes.getFunctionVector());

    OS << "Basic block vectors:\n";
    for (const auto &BBVector : IR2VecRes.getBBVecMap()) {
      OS << "Basic block: " << BBVector.first->getName() << ":\n";
      printVector(BBVector.second);
    }

    OS << "Instruction vectors:\n";
    for (const auto &InstVector : IR2VecRes.getInstVecMap()) {
      OS << "Instruction: ";
      InstVector.first->print(OS);
      printVector(InstVector.second);
    }
  }
  return PreservedAnalyses::all();
}
