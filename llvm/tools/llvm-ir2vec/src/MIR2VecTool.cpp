//===- MIR2VecTool.cpp - MIR2Vec Tool Implementation ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the MIR2Vec tool for Machine IR embedding generation.
///
//===----------------------------------------------------------------------===//

#include "MIR2VecTool.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"

#define DEBUG_TYPE "mir2vec"

namespace llvm {
// Generate embeddings based on the specified level
// Note: Level is expected to be a global variable accessible here
// (This matches the original design where Level is in CommonCategory)
extern cl::opt<llvm::EmbeddingLevel> Level;

namespace mir2vec {

static const char *ToolName = "llvm-ir2vec";

// ========================================================================
// MIR Context Setup Functions
// ========================================================================

void initializeTargets() {
  static bool Initialized = false;
  if (!Initialized) {
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
    Initialized = true;
  }
}

static Error parseMIRFile(const std::string &Filename, MIRContext &Ctx) {
  SMDiagnostic Err;

  Ctx.Parser = createMIRParserFromFile(Filename, Err, Ctx.Context);
  if (!Ctx.Parser) {
    std::string ErrMsg;
    raw_string_ostream OS(ErrMsg);
    Err.print(ToolName, OS);
    return createStringError(errc::invalid_argument, OS.str());
  }

  auto SetDataLayout = [&Ctx](StringRef DataLayoutTargetTriple,
                              StringRef OldDLStr) -> std::optional<std::string> {
    std::string IRTargetTriple = DataLayoutTargetTriple.str();
    Triple TheTriple = Triple(IRTargetTriple);
    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    auto TMOrErr = codegen::createTargetMachineForTriple(TheTriple.str());
    if (!TMOrErr) {
      WithColor::error(errs(), ToolName)
          << "Failed to create target machine: "
          << toString(TMOrErr.takeError()) << "\n";
      exit(1);
    }

    Ctx.TM = std::move(*TMOrErr);
    return Ctx.TM->createDataLayout().getStringRepresentation();
  };

  Ctx.M = Ctx.Parser->parseIRModule(SetDataLayout);
  if (!Ctx.M) {
    return createStringError(errc::invalid_argument,
                            "Failed to parse IR module from MIR file");
  }

  return Error::success();
}

static Error parseMachineFunctions(MIRContext &Ctx) {
  SMDiagnostic Err;
  Ctx.MMI = std::make_unique<MachineModuleInfo>(Ctx.TM.get());
  if (!Ctx.MMI) {
    return createStringError(errc::not_enough_memory,
                            "Failed to create MachineModuleInfo");
  }

  if (Ctx.Parser->parseMachineFunctions(*Ctx.M, *Ctx.MMI)) {
    return createStringError(errc::invalid_argument,
                            "Failed to parse machine functions");
  }

  return Error::success();
}

Error setupMIRContext(const std::string &InputFile, MIRContext &Ctx) {
  initializeTargets();

  if (auto Err = parseMIRFile(InputFile, Ctx))
    return Err;

  if (auto Err = parseMachineFunctions(Ctx))
    return Err;

  return Error::success();
}

// ========================================================================
// MIR2VecTool Implementation
// ========================================================================

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

/// Initialize vocabulary with layout information only.
/// This creates a minimal vocabulary with correct layout but no actual
/// embeddings. Sufficient for generating training data and entity mappings.
///
/// Note: Requires target-specific information from the first machine function
/// to determine the vocabulary layout (number of opcodes, register classes).
///
/// FIXME: Use --target option to get target info directly, avoiding the need
/// to parse machine functions for pre-training operations.
bool MIR2VecTool::initializeVocabularyForLayout(const Module &M) {
  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;

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

// ========================================================================
// Data structure methods
// ========================================================================

unsigned MIR2VecTool::generateTripletsForMF(
    const MachineFunction &MF, std::vector<Triplet> &Triplets) const {
  unsigned MaxRelation = MIRNextRelation;
  unsigned PrevOpcode = 0;
  bool HasPrevOpcode = false;

  if (!Vocab) {
    WithColor::error(errs(), ToolName)
        << "MIR Vocabulary must be initialized for triplet generation.\n";
    return MaxRelation;
  }

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      // Skip debug instructions
      if (MI.isDebugInstr())
        continue;

      // Get opcode entity ID
      unsigned OpcodeID = Vocab->getEntityIDForOpcode(MI.getOpcode());

      // Add "Next" relationship with previous instruction
      if (HasPrevOpcode) {
        Triplets.push_back({PrevOpcode, OpcodeID, MIRNextRelation});
        LLVM_DEBUG(dbgs() << Vocab->getStringKey(PrevOpcode) << '\t'
                          << Vocab->getStringKey(OpcodeID) << '\t'
                          << "Next\n");
      }

      // Add "Arg" relationships for operands
      unsigned ArgIndex = 0;
      for (const MachineOperand &MO : MI.operands()) {
        auto OperandID = Vocab->getEntityIDForMachineOperand(MO);
        unsigned RelationID = MIRArgRelation + ArgIndex;
        Triplets.push_back({OpcodeID, OperandID, RelationID});
        LLVM_DEBUG({
          std::string OperandStr = Vocab->getStringKey(OperandID);
          dbgs() << Vocab->getStringKey(OpcodeID) << '\t' << OperandStr << '\t'
                 << "Arg" << ArgIndex << '\n';
        });

        ++ArgIndex;
      }

      // Update MaxRelation if there were operands
      if (ArgIndex > 0)
        MaxRelation = std::max(MaxRelation, MIRArgRelation + ArgIndex - 1);

      PrevOpcode = OpcodeID;
      HasPrevOpcode = true;
    }
  }

  return MaxRelation;
}

TripletResult MIR2VecTool::getTriplets(const Module &M) const {
  TripletResult Result;
  Result.MaxRelation = MIRNextRelation;

  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;

    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF) {
      WithColor::warning(errs(), ToolName)
          << "No MachineFunction for " << F.getName() << "\n";
      continue;
    }

    unsigned FuncMaxRelation = generateTripletsForMF(*MF, Result.Triplets);
    Result.MaxRelation = std::max(Result.MaxRelation, FuncMaxRelation);
  }

  return Result;
}

TripletResult MIR2VecTool::getTriplets(const MachineFunction &MF) const {
  TripletResult Result;
  Result.MaxRelation = generateTripletsForMF(MF, Result.Triplets);
  return Result;
}

EntityMap MIR2VecTool::getEntityMappings() const {
  EntityMap Entities;

  if (!Vocab) {
    WithColor::error(errs(), ToolName)
        << "Vocabulary must be initialized for entity mappings.\n";
    return Entities;
  }

  const unsigned EntityCount = Vocab->getCanonicalSize();
  Entities.reserve(EntityCount);

  for (unsigned EntityID = 0; EntityID < EntityCount; ++EntityID)
    Entities.push_back(Vocab->getStringKey(EntityID));

  return Entities;
}

FuncVecMap MIR2VecTool::getFunctionEmbeddings(const Module &M) const {
  FuncVecMap FuncEmbeddings;

  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return FuncEmbeddings;
  }

  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;

    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF) {
      WithColor::warning(errs(), ToolName)
          << "No MachineFunction for " << F.getName() << "\n";
      continue;
    }

    auto Emb = MIREmbedder::create(MIR2VecKind::Symbolic, *MF, *Vocab);
    if (!Emb) {
      WithColor::error(errs(), ToolName)
          << "Failed to create embedder for " << MF->getName() << "\n";
      continue;
    }

    auto DemangledName = getDemagledName(&F);
    auto ActualName = getActualName(&F);
    FuncEmbeddings[DemangledName] = {ActualName, Emb->getMFunctionVector()};
  }

  return FuncEmbeddings;
}

std::pair<std::string, Embedding>
MIR2VecTool::getFunctionEmbedding(MachineFunction &MF) const {
  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return {"", Embedding()};
  }

  auto Emb = MIREmbedder::create(MIR2VecKind::Symbolic, MF, *Vocab);
  if (!Emb) {
    WithColor::error(errs(), ToolName)
        << "Failed to create embedder for " << MF.getName() << "\n";
    return {"", Embedding()};
  }

  return {MF.getName().str(), Emb->getMFunctionVector()};
}

BBVecList MIR2VecTool::getMBBEmbeddings(MachineFunction &MF) const {
  BBVecList BBEmbeddings;

  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return BBEmbeddings;
  }

  auto Emb = MIREmbedder::create(MIR2VecKind::Symbolic, MF, *Vocab);
  if (!Emb) {
    WithColor::error(errs(), ToolName)
        << "Failed to create embedder for " << MF.getName() << "\n";
    return BBEmbeddings;
  }

  for (const MachineBasicBlock &MBB : MF) {
    std::string BBName = MBB.getName().str();
    if (BBName.empty())
      BBName = "bb." + std::to_string(MBB.getNumber());
    BBEmbeddings.push_back({BBName, Emb->getMBBVector(MBB)});
  }

  return BBEmbeddings;
}

InstVecList MIR2VecTool::getMInstEmbeddings(MachineFunction &MF) const {
  InstVecList InstEmbeddings;

  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return InstEmbeddings;
  }

  auto Emb = MIREmbedder::create(MIR2VecKind::Symbolic, MF, *Vocab);
  if (!Emb) {
    WithColor::error(errs(), ToolName)
        << "Failed to create embedder for " << MF.getName() << "\n";
    return InstEmbeddings;
  }

  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      std::string InstStr;
      raw_string_ostream OS(InstStr);
      OS << MI;
      InstEmbeddings.push_back({OS.str(), Emb->getMInstVector(MI)});
    }
  }

  return InstEmbeddings;
}

// ========================================================================
// Stream output methods
// ========================================================================

void MIR2VecTool::generateTriplets(const Module &M, raw_ostream &OS) const {
  auto Result = getTriplets(M);

  // Write metadata header followed by relationships
  OS << "MAX_RELATION=" << Result.MaxRelation << '\n';

  for (const auto &Triplet : Result.Triplets) {
    OS << Triplet.Head << '\t' << Triplet.Tail << '\t' << Triplet.Relation
       << '\n';
  }
}

void MIR2VecTool::generateTriplets(const MachineFunction &MF,
                                   raw_ostream &OS) const {
  auto Result = getTriplets(MF);

  // Write metadata header followed by relationships
  OS << "MAX_RELATION=" << Result.MaxRelation << '\n';

  for (const auto &Triplet : Result.Triplets) {
    OS << Triplet.Head << '\t' << Triplet.Tail << '\t' << Triplet.Relation
       << '\n';
  }
}

void MIR2VecTool::generateEntityMappings(raw_ostream &OS) const {
  auto Entities = getEntityMappings();

  OS << Entities.size() << "\n";
  for (size_t EntityID = 0; EntityID < Entities.size(); ++EntityID) {
    OS << Entities[EntityID] << '\t' << EntityID << '\n';
  }
}

void MIR2VecTool::generateEmbeddings(const Module &M, raw_ostream &OS) const {
  if (!Vocab) {
    WithColor::error(errs(), ToolName) << "Vocabulary not initialized.\n";
    return;
  }

  for (const Function &F : M) {
    if (F.isDeclaration())
      continue;

    MachineFunction *MF = MMI.getMachineFunction(F);
    if (!MF) {
      WithColor::warning(errs(), ToolName)
          << "No MachineFunction for " << F.getName() << "\n";
      continue;
    }

    generateEmbeddings(*MF, OS);
  }
}

void MIR2VecTool::generateEmbeddings(MachineFunction &MF,
                                     raw_ostream &OS) const {
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
  case FunctionLevel: {
    OS << "Function vector: ";
    Emb->getMFunctionVector().print(OS);
    break;
  }
  case BasicBlockLevel: {
    OS << "Basic block vectors:\n";
    for (const MachineBasicBlock &MBB : MF) {
      OS << "MBB " << MBB.getName() << ": ";
      Emb->getMBBVector(MBB).print(OS);
    }
    break;
  }
  case InstructionLevel: {
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
}

} // namespace mir2vec
} // namespace llvm