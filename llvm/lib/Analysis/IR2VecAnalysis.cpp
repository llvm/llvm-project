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

STATISTIC(dataMissCounter, "Number of data misses in the vocabulary");

/// IR2Vec computes two kinds of embeddings: Symbolic and Flow-aware.
/// Symbolic embeddings capture the "syntactic" and "statistical correlation"
/// of the IR entities. Flow-aware embeddings build on top of symbolic
/// embeddings and additionally capture the flow information in the IR.
/// IR2VecKind is used to specify the type of embeddings to generate.
// ToDo: Currently we support only Symbolic.
// We shall add support for Flow-aware in upcoming patches.
enum IR2VecKind { symbolic, flowaware };

static cl::OptionCategory IR2VecAnalysisCategory("IR2Vec Analysis Options");

cl::opt<IR2VecKind>
    IR2VecMode("ir2vec-mode",
               cl::desc("Choose type of embeddings to generate:"),
               cl::values(clEnumValN(symbolic, "symbolic",
                                     "Generates symbolic embeddings"),
                          clEnumValN(flowaware, "flowaware",
                                     "Generates flow-aware embeddings")),
               cl::init(symbolic), cl::cat(IR2VecAnalysisCategory));

// ToDo: Use a default vocab when not specified
static cl::opt<std::string>
    VocabFile("ir2vec-vocab-path", cl::Optional,
              cl::desc("Path to the vocabulary file for IR2Vec"), cl::init(""),
              cl::cat(IR2VecAnalysisCategory));

AnalysisKey VocabAnalysis::Key;
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
  // ToDo: Defaults to the values used in the original algorithm. Can be
  // parameterized later.
  float WO = 1.0, WT = 0.5, WA = 0.2;

  /// Dimension of the vector representation; captured from the input vocabulary
  unsigned DIM = 300;

  // Utility maps - these are used to store the vector representations of
  // instructions, basic blocks and functions.
  Embedding FuncVector;
  SmallMapVector<const BasicBlock *, Embedding, 16> BBVecMap;
  SmallMapVector<const Instruction *, Embedding, 128> InstVecMap;

  Embeddings(const Function &F, const Vocab &Vocabulary, unsigned DIM)
      : F(F), Vocabulary(Vocabulary), DIM(DIM) {}

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

  /// Returns the dimension of the embedding vector.
  unsigned getDimension() const { return DIM; }

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
  Symbolic(const Function &F, const Vocab &Vocabulary, unsigned DIM)
      : Embeddings(F, Vocabulary, DIM) {
    FuncVector = Embedding(DIM, 0);
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

// ToDo: Currently lookups are string based. Use numeric Keys
// for efficiency.
Embedding Embeddings::lookupVocab(const std::string &Key) {
  Embedding Vec(DIM, 0);
  // ToDo: Use zero vectors in vocab and assert failure for
  // unknown entities rather than silently returning zeroes here.
  if (Vocabulary.find(Key) == Vocabulary.end()) {
    LLVM_DEBUG(errs() << "cannot find key in map : " << Key << "\n");
    dataMissCounter++;
  } else {
    Vec = Vocabulary[Key];
  }
  return Vec;
}

void Symbolic::computeEmbeddings() {
  if (F.isDeclaration())
    return;
  for (auto &BB : F) {
    BBVecMap[&BB] = computeBB2Vec(BB);
    addVectors(FuncVector, BBVecMap[&BB]);
  }
}

Embedding Symbolic::computeBB2Vec(const BasicBlock &BB) {
  auto It = BBVecMap.find(&BB);
  if (It != BBVecMap.end()) {
    return It->second;
  }
  Embedding BBVector(DIM, 0);

  for (auto &I : BB) {
    Embedding InstVector(DIM, 0);

    auto Vec = lookupVocab(I.getOpcodeName());
    scaleVector(Vec, WO);
    addVectors(InstVector, Vec);

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
    scaleVector(Vec, WT);
    addVectors(InstVector, Vec);

    for (auto &Op : I.operands()) {
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
      scaleVector(Vec, WA);
      addVectors(InstVector, Vec);
      InstVecMap[&I] = InstVector;
    }
    addVectors(BBVector, InstVector);
  }
  return BBVector;
}
} // namespace

// ==----------------------------------------------------------------------===//
// VocabResult and VocabAnalysis
//===----------------------------------------------------------------------===//

VocabResult::VocabResult(const ir2vec::Vocab &Vocabulary, unsigned Dim)
    : Vocabulary(std::move(Vocabulary)), Valid(true), DIM(Dim) {}

const ir2vec::Vocab &VocabResult::getVocabulary() const {
  assert(Valid);
  return Vocabulary;
}

// For now, assume vocabulary is stable unless explicitly invalidated.
bool VocabResult::invalidate(Module &M, const PreservedAnalyses &PA,
                             ModuleAnalysisManager::Invalidator &Inv) {
  auto PAC = PA.getChecker<VocabAnalysis>();
  return !(PAC.preservedWhenStateless());
}

// ToDo: Make this optional. We can avoid file reads
// by auto-generating the vocabulary during the build time.
Error VocabAnalysis::readVocabulary() {
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
  this->DIM = Dim;
  return Error::success();
}

VocabAnalysis::Result VocabAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  if (VocabFile.empty()) {
    // ToDo: Use default vocabulary
    errs() << "Error: IR2Vec vocabulary file path not specified.\n";
    return VocabResult(); // Return invalid result
  }

  if (auto Err = readVocabulary())
    return VocabResult();

  return VocabResult(std::move(Vocabulary), DIM);
}

// ==----------------------------------------------------------------------===//
// IR2VecResult and IR2VecAnalysis
//===----------------------------------------------------------------------===//

IR2VecResult::IR2VecResult(
    const SmallMapVector<const Instruction *, Embedding, 128> InstMap,
    const SmallMapVector<const BasicBlock *, Embedding, 16> BBMap,
    const Embedding &FuncVector, unsigned Dim)
    : InstVecMap(std::move(InstMap)), BBVecMap(std::move(BBMap)),
      FuncVector(std::move(FuncVector)), DIM(Dim), Valid(true) {}

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
unsigned IR2VecResult::getDimension() const { return DIM; }
IR2VecAnalysis::Result IR2VecAnalysis::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  auto *VocabRes = FAM.getResult<ModuleAnalysisManagerFunctionProxy>(F)
                       .getCachedResult<VocabAnalysis>(*F.getParent());
  if (!VocabRes->isValid()) {
    errs() << "Error: IR2Vec vocabulary is invalid.\n";
    return IR2VecResult();
  }

  auto Dim = VocabRes->getDimension();
  if (Dim <= 0) {
    errs() << "Error: IR2Vec vocabulary dimension is zero.\n";
    return IR2VecResult();
  }

  auto Vocabulary = VocabRes->getVocabulary();
  std::unique_ptr<Embeddings> Emb;
  switch (IR2VecMode) {
  case IR2VecKind::symbolic:
    Emb = std::make_unique<Symbolic>(F, Vocabulary, Dim);
    break;
  case flowaware:
    // ToDo: Add support for flow-aware embeddings
    llvm_unreachable("Flow-aware embeddings are not supported yet");
    break;
  default:
    llvm_unreachable("Invalid IR2Vec mode");
  }
  Emb->computeEmbeddings();
  auto InstMap = Emb->getInstVecMap();
  auto BBMap = Emb->getBBVecMap();
  auto FuncVec = Emb->getFunctionVector();
  return IR2VecResult(std::move(InstMap), std::move(BBMap), std::move(FuncVec),
                      Dim);
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
  auto VocabResult = MAM.getResult<VocabAnalysis>(M);
  assert(VocabResult.isValid() && "Vocab is invalid");

  for (Function &F : M) {
    auto &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    auto IR2VecRes = FAM.getResult<IR2VecAnalysis>(F);
    if (!IR2VecRes.isValid()) {
      errs() << "Error: IR2Vec embeddings are invalid.\n";
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
