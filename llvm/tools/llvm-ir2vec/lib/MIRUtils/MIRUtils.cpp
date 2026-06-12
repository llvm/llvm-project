//===- MIRUtils.cpp - MIR2Vec Embedding Generation --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the MIR2VecTool class for MIR2Vec embedding generation
/// from LLVM Machine IR. It has no dependency on the IR2Vec embedding API.
///
//===----------------------------------------------------------------------===//

#include "MIRUtils.h"
#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "ir2vec"

namespace llvm {
namespace mir2vec {

bool MIR2VecTool::initializeVocabulary(const Module &M) {
  MIR2VecVocabProvider Provider(MMI);
  auto VocabOrErr = Provider.getVocabulary(M);
  if (!VocabOrErr) {
    WithColor::error(errs(), ToolName)
        << "Failed to load MIR2Vec vocabulary - "
        << toString(VocabOrErr.takeError()) << "\n";
    return false;
  }
  Vocab = std::make_unique<MIRVocabulary>(std::move(*VocabOrErr));
  return true;
}

bool MIR2VecTool::initializeVocabularyForLayout(const Module &M) {
  for (const Function &F : M.getFunctionDefs()) {
    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF)
      continue;

    const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
    const TargetRegisterInfo &TRI = *MF->getSubtarget().getRegisterInfo();
    const MachineRegisterInfo &MRI = MF->getRegInfo();

    auto VocabOrErr = MIRVocabulary::createDummyVocabForTest(TII, TRI, MRI, 1);
    if (!VocabOrErr) {
      WithColor::error(errs(), ToolName)
          << "Failed to create dummy vocabulary - "
          << toString(VocabOrErr.takeError()) << "\n";
      return false;
    }
    Vocab = std::make_unique<MIRVocabulary>(std::move(*VocabOrErr));
    return true;
  }

  WithColor::error(errs(), ToolName)
      << "No machine functions found to initialize vocabulary\n";
  return false;
}

TripletResult MIR2VecTool::generateTriplets(const MachineFunction &MF) const {
  TripletResult Result;
  Result.MaxRelation = MIRNextRelation;

  if (!Vocab) {
    WithColor::error(errs(), ToolName)
        << "MIR Vocabulary must be initialized for triplet generation.\n";
    return Result;
  }

  unsigned PrevOpcode = 0;
  bool HasPrevOpcode = false;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;

      unsigned OpcodeID = Vocab->getEntityIDForOpcode(MI.getOpcode());

      if (HasPrevOpcode) {
        Result.Triplets.push_back({PrevOpcode, OpcodeID, MIRNextRelation});
        LLVM_DEBUG(dbgs() << Vocab->getStringKey(PrevOpcode) << '\t'
                          << Vocab->getStringKey(OpcodeID) << '\t' << "Next\n");
      }

      unsigned ArgIndex = 0;
      for (const MachineOperand &MO : MI.operands()) {
        auto OperandID = Vocab->getEntityIDForMachineOperand(MO);
        unsigned RelationID = MIRArgRelation + ArgIndex;
        Result.Triplets.push_back({OpcodeID, OperandID, RelationID});
        LLVM_DEBUG({
          std::string OperandStr = Vocab->getStringKey(OperandID);
          dbgs() << Vocab->getStringKey(OpcodeID) << '\t' << OperandStr << '\t'
                 << "Arg" << ArgIndex << '\n';
        });

        ++ArgIndex;
      }

      if (ArgIndex > 0)
        Result.MaxRelation =
            std::max(Result.MaxRelation, MIRArgRelation + ArgIndex - 1);

      PrevOpcode = OpcodeID;
      HasPrevOpcode = true;
    }
  }

  return Result;
}

TripletResult MIR2VecTool::generateTriplets(const Module &M) const {
  TripletResult Result;
  Result.MaxRelation = MIRNextRelation;

  for (const Function &F : M.getFunctionDefs()) {
    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF) {
      WithColor::warning(errs(), ToolName)
          << "No MachineFunction for " << F.getName() << "\n";
      continue;
    }

    TripletResult FuncResult = generateTriplets(*MF);
    Result.MaxRelation = std::max(Result.MaxRelation, FuncResult.MaxRelation);
    Result.Triplets.insert(Result.Triplets.end(), FuncResult.Triplets.begin(),
                           FuncResult.Triplets.end());
  }

  return Result;
}

void MIR2VecTool::writeTripletsToStream(const Module &M,
                                        raw_ostream &OS) const {
  auto Result = generateTriplets(M);
  OS << "MAX_RELATION=" << Result.MaxRelation << '\n';
  for (const auto &T : Result.Triplets)
    OS << T.Head << '\t' << T.Tail << '\t' << T.Relation << '\n';
}

EntityList MIR2VecTool::collectEntityMappings() const {
  if (!Vocab) {
    WithColor::error(errs(), ToolName)
        << "Vocabulary must be initialized for entity mappings.\n";
    return {};
  }

  const unsigned EntityCount = Vocab->getCanonicalSize();
  EntityList Result;
  for (unsigned EntityID = 0; EntityID < EntityCount; ++EntityID)
    Result.push_back(Vocab->getStringKey(EntityID));

  return Result;
}

void MIR2VecTool::writeEntitiesToStream(raw_ostream &OS) const {
  auto Entities = collectEntityMappings();
  if (Entities.empty())
    return;

  OS << Entities.size() << "\n";
  for (unsigned EntityID = 0; EntityID < Entities.size(); ++EntityID)
    OS << Entities[EntityID] << '\t' << EntityID << '\n';
}

void MIR2VecTool::writeEmbeddingsToStream(const Module &M, raw_ostream &OS,
                                          EmbeddingLevel Level) const {
  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return;
  }

  for (const Function &F : M.getFunctionDefs()) {
    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF) {
      WithColor::warning(errs(), ToolName)
          << "No MachineFunction for " << F.getName() << "\n";
      continue;
    }

    writeEmbeddingsToStream(*MF, OS, Level);
  }
}

void MIR2VecTool::writeEmbeddingsToStream(MachineFunction &MF, raw_ostream &OS,
                                          EmbeddingLevel Level) const {
  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return;
  }

  auto Emb = MIREmbedder::create(MIR2VecKind::Symbolic, MF, *Vocab);
  if (!Emb) {
    WithColor::error(errs(), ToolName)
        << "Failed to create embedder for " << MF.getName() << "\n";
    return;
  }

  OS << "MIR2Vec embeddings for machine function " << MF.getName() << ":\n";

  switch (Level) {
  case FunctionLevel:
    OS << "Function vector: ";
    Emb->getMFunctionVector().print(OS);
    break;
  case BasicBlockLevel:
    OS << "Basic block vectors:\n";
    for (const MachineBasicBlock &MBB : MF) {
      OS << "MBB " << MBB.getName() << ": ";
      Emb->getMBBVector(MBB).print(OS);
    }
    break;
  case InstructionLevel:
    OS << "Instruction vectors:\n";
    for (const MachineBasicBlock &MBB : MF) {
      for (const MachineInstr &MI : MBB) {
        OS << MI << " -> ";
        Emb->getMInstVector(MI).print(OS);
      }
    }
    break;
  }
}

} // namespace mir2vec
} // namespace llvm
