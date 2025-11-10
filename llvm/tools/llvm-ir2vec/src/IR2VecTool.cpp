//===- IR2VecTool.cpp - IR2Vec Tool Implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IR2VecTool.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <cxxabi.h>

#define DEBUG_TYPE "ir2vec"

using namespace llvm;
using namespace llvm::ir2vec;

namespace llvm {
extern cl::opt<EmbeddingLevel> Level;
} // namespace llvm

namespace llvm {
namespace ir2vec {

bool IR2VecTool::initializeVocabulary() {
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MAM.registerPass([&] { return IR2VecVocabAnalysis(); });
  Vocab = &MAM.getResult<IR2VecVocabAnalysis>(M);
  return Vocab && Vocab->isValid();
}

TripletResult IR2VecTool::getTriplets(const Function &F) const {
  TripletResult Result;
  Result.MaxRelation = 0;

  if (F.isDeclaration())
    return Result;

  unsigned MaxRelation = NextRelation;
  unsigned PrevOpcode = 0;
  bool HasPrevOpcode = false;

  for (const BasicBlock &BB : F) {
    for (const auto &I : BB.instructionsWithoutDebug()) {
      unsigned Opcode = Vocabulary::getIndex(I.getOpcode());
      unsigned TypeID = Vocabulary::getIndex(I.getType()->getTypeID());

      if (HasPrevOpcode) {
        Result.Triplets.push_back({PrevOpcode, Opcode, NextRelation});
        LLVM_DEBUG(dbgs()
                   << Vocabulary::getVocabKeyForOpcode(PrevOpcode + 1) << '\t'
                   << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                   << "Next\n");
      }

      Result.Triplets.push_back({Opcode, TypeID, TypeRelation});
      LLVM_DEBUG(
          dbgs() << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                 << Vocabulary::getVocabKeyForTypeID(I.getType()->getTypeID())
                 << '\t' << "Type\n");

      unsigned ArgIndex = 0;
      for (const Use &U : I.operands()) {
        unsigned OperandID = Vocabulary::getIndex(*U.get());
        unsigned RelationID = ArgRelation + ArgIndex;
        Result.Triplets.push_back({Opcode, OperandID, RelationID});

        LLVM_DEBUG({
          StringRef OperandStr = Vocabulary::getVocabKeyForOperandKind(
              Vocabulary::getOperandKind(U.get()));
          dbgs() << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                 << OperandStr << '\t' << "Arg" << ArgIndex << '\n';
        });

        ++ArgIndex;
      }

      if (ArgIndex > 0) {
        MaxRelation = std::max(MaxRelation, ArgRelation + ArgIndex - 1);
      }

      PrevOpcode = Opcode;
      HasPrevOpcode = true;
    }
  }

  Result.MaxRelation = MaxRelation;
  return Result;
}

TripletResult IR2VecTool::getTriplets() const {
  TripletResult Result;
  Result.MaxRelation = NextRelation;

  for (const Function &F : M) {
    TripletResult FuncResult = getTriplets(F);
    Result.MaxRelation = std::max(Result.MaxRelation, FuncResult.MaxRelation);
    Result.Triplets.insert(Result.Triplets.end(),
                           FuncResult.Triplets.begin(),
                           FuncResult.Triplets.end());
  }

  return Result;
}

void IR2VecTool::generateTriplets(raw_ostream &OS) const {
  auto Result = getTriplets();
  OS << "MAX_RELATION=" << Result.MaxRelation << '\n';
  for (const auto &T : Result.Triplets) {
    OS << T.Head << '\t' << T.Tail << '\t' << T.Relation << '\n';
  }
}

EntityMap IR2VecTool::getEntityMappings() {
  auto EntityLen = Vocabulary::getCanonicalSize();
  EntityMap Result;
  Result.reserve(EntityLen);

  for (unsigned EntityID = 0; EntityID < EntityLen; ++EntityID)
    Result.push_back(Vocabulary::getStringKey(EntityID).str());

  return Result;
}

void IR2VecTool::generateEntityMappings(raw_ostream &OS) {
  auto Entities = getEntityMappings();
  OS << Entities.size() << "\n";
  for (unsigned EntityID = 0; EntityID < Entities.size(); ++EntityID)
    OS << Entities[EntityID] << '\t' << EntityID << '\n';
}

std::pair<std::string, std::pair<std::string, Embedding>>
IR2VecTool::getFunctionEmbedding(const Function &F) const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  if (F.isDeclaration())
    return {};

  auto Emb = Embedder::create(IR2VecEmbeddingKind, F, *Vocab);
  if (!Emb) {
    return {};
  }

  auto FuncVec = Emb->getFunctionVector();
  auto DemangledName = getDemagledName(&F);
  auto ActualName = getActualName(&F);

  return {std::move(DemangledName), {std::move(ActualName), std::move(FuncVec)}};
}

FuncVecMap IR2VecTool::getFunctionEmbeddings() const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  FuncVecMap Result;

  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;

    auto Emb = getFunctionEmbedding(F);
    if (!Emb.first.empty()) {
      Result.try_emplace(
        std::move(Emb.first),
        std::move(Emb.second.first),
        std::move(Emb.second.second)
      );
    }
  }

  return Result;
}

void IR2VecTool::generateEmbeddings(const Function &F, raw_ostream &OS) const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  if (F.isDeclaration()) {
    OS << "Function " << F.getName() << " is a declaration, skipping.\n";
    return;
  }

  OS << "Function: " << F.getName() << "\n";

  auto printError = [&]() {
    OS << "Error: Failed to create embedder for function " << F.getName() << '\n';
  };

  auto printListLevel = [&](const auto& list) {
    if (list.empty()) return printError();
    for (const auto& [name, embedding] : list) {
      OS << name;
      embedding.print(OS);
      OS << '\n';
    }
  };

  switch (Level) {
    case EmbeddingLevel::FunctionLevel:
      if (auto FuncEmb = getFunctionEmbedding(F); !FuncEmb.first.empty())
        FuncEmb.second.second.print(OS);
      else printError();
      break;
    case EmbeddingLevel::BasicBlockLevel:
      printListLevel(getBBEmbeddings(F));
      break;
    case EmbeddingLevel::InstructionLevel:
      printListLevel(getInstEmbeddings(F));
      break;
  }
}

void IR2VecTool::generateEmbeddings(raw_ostream &OS) const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  for (const Function &F : M)
    generateEmbeddings(F, OS);
}

BBVecList IR2VecTool::getBBEmbeddings(const Function &F) const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  BBVecList Result;

  if (F.isDeclaration())
    return Result;

  auto Emb = Embedder::create(IR2VecEmbeddingKind, F, *Vocab);
  if (!Emb)
    return Result;

  for (const BasicBlock &BB : F)
    Result.push_back({BB.getName().str(), Emb->getBBVector(BB)});

  return Result;
}

BBVecList IR2VecTool::getBBEmbeddings() const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  BBVecList Result;

  for (const Function &F : M) {
    if (F.isDeclaration()) continue;

    BBVecList FuncBBVecs = getBBEmbeddings(F);
    Result.insert(Result.end(), FuncBBVecs.begin(), FuncBBVecs.end());
  }

  return Result;
}

InstVecList IR2VecTool::getInstEmbeddings(const Function &F) const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  InstVecList Result;

  if (F.isDeclaration())
    return Result;

  auto Emb = Embedder::create(IR2VecEmbeddingKind, F, *Vocab);
  if (!Emb)
    return Result;

  for (const Instruction &I : instructions(F)) {
    std::string InstStr = [&]() {
      std::string str;
      raw_string_ostream RSO(str);
      I.print(RSO);
      RSO.flush();
      return str;
    }();

    Result.push_back({InstStr, Emb->getInstVector(I)});
  }

  return Result;
}

InstVecList IR2VecTool::getInstEmbeddings() const {
  assert(Vocab && Vocab->isValid() && "Vocabulary not initialized");

  InstVecList Result;

  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;

    InstVecList FuncInstVecs = getInstEmbeddings(F);
    Result.insert(Result.end(), FuncInstVecs.begin(), FuncInstVecs.end());
  }

  return Result;
}

} // namespace ir2vec
} // namespace llvm