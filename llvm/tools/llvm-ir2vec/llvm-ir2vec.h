//===- llvm-ir2vec.h - IR2Vec/MIR2Vec Tool Classes ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the IR2VecTool and MIR2VecTool class definitions and
/// implementations for the llvm-ir2vec embedding generation tool.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_IR2VEC_LLVM_IR2VEC_H
#define LLVM_TOOLS_LLVM_IR2VEC_LLVM_IR2VEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <memory>
#include <string>
#include <vector>

#define DEBUG_TYPE "ir2vec"

namespace llvm {

/// Tool name for error reporting
static const char *ToolName = "llvm-ir2vec";

/// Specifies the granularity at which embeddings are generated.
enum EmbeddingLevel {
  InstructionLevel, // Generate instruction-level embeddings
  BasicBlockLevel,  // Generate basic block-level embeddings
  FunctionLevel     // Generate function-level embeddings
};

/// Represents a single knowledge graph triplet (Head, Relation, Tail)
/// where indices reference entities in an EntityList
struct Triplet {
  unsigned Head = 0;     ///< Index of the head entity in the entity list
  unsigned Tail = 0;     ///< Index of the tail entity in the entity list
  unsigned Relation = 0; ///< Relation type (see RelationType enum)
};

/// Result structure containing all generated triplets and metadata
struct TripletResult {
  unsigned MaxRelation =
      0; ///< Highest relation index used (for ArgRelation + N)
  std::vector<Triplet> Triplets; ///< Collection of all generated triplets
};

/// Entity mappings: [entity_name]
using EntityList = std::vector<std::string>;

namespace ir2vec {

/// Relation types for triplet generation
enum RelationType {
  TypeRelation = 0, ///< Instruction to type relationship
  NextRelation = 1, ///< Sequential instruction relationship
  ArgRelation = 2   ///< Instruction to operand relationship (ArgRelation + N)
};

/// Helper class for collecting IR triplets and generating embeddings
class IR2VecTool {
private:
  Module &M;
  ModuleAnalysisManager MAM;
  const Vocabulary *Vocab = nullptr;

public:
  explicit IR2VecTool(Module &M) : M(M) {}

  /// Initialize the IR2Vec vocabulary analysis
  bool initializeVocabulary() {
    // Register and run the IR2Vec vocabulary analysis
    // The vocabulary file path is specified via --ir2vec-vocab-path global
    // option
    MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
    MAM.registerPass([&] { return IR2VecVocabAnalysis(); });
    // This will throw an error if vocab is not found or invalid
    Vocab = &MAM.getResult<IR2VecVocabAnalysis>(M);
    return Vocab->isValid();
  }

  /// Generate triplets for a single function
  /// Returns a TripletResult with:
  ///   - Triplets: vector of all (subject, object, relation) tuples
  ///   - MaxRelation: highest Arg relation ID used, or NextRelation if none
  TripletResult generateTriplets(const Function &F) const {
    if (F.isDeclaration())
      return {};

    TripletResult Result;
    Result.MaxRelation = 0;

    unsigned MaxRelation = NextRelation;
    unsigned PrevOpcode = 0;
    bool HasPrevOpcode = false;

    for (const BasicBlock &BB : F) {
      for (const auto &I : BB.instructionsWithoutDebug()) {
        unsigned Opcode = Vocabulary::getIndex(I.getOpcode());
        unsigned TypeID = Vocabulary::getIndex(I.getType()->getTypeID());

        // Add "Next" relationship with previous instruction
        if (HasPrevOpcode) {
          Result.Triplets.push_back({PrevOpcode, Opcode, NextRelation});
          LLVM_DEBUG(dbgs()
                     << Vocabulary::getVocabKeyForOpcode(PrevOpcode + 1) << '\t'
                     << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                     << "Next\n");
        }

        // Add "Type" relationship
        Result.Triplets.push_back({Opcode, TypeID, TypeRelation});
        LLVM_DEBUG(
            dbgs() << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                   << Vocabulary::getVocabKeyForTypeID(I.getType()->getTypeID())
                   << '\t' << "Type\n");

        // Add "Arg" relationships
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
        // Only update MaxRelation if there were operands
        if (ArgIndex > 0)
          MaxRelation = std::max(MaxRelation, ArgRelation + ArgIndex - 1);
        PrevOpcode = Opcode;
        HasPrevOpcode = true;
      }
    }

    Result.MaxRelation = MaxRelation;
    return Result;
  }

  /// Get triplets for the entire module
  TripletResult generateTriplets() const {
    TripletResult Result;
    Result.MaxRelation = NextRelation;

    for (const Function &F : M.getFunctionDefs()) {
      TripletResult FuncResult = generateTriplets(F);
      Result.MaxRelation = std::max(Result.MaxRelation, FuncResult.MaxRelation);
      Result.Triplets.insert(Result.Triplets.end(), FuncResult.Triplets.begin(),
                             FuncResult.Triplets.end());
    }

    return Result;
  }

  /// Collect triplets for the module and dump output to stream
  /// Output format: MAX_RELATION=N header followed by relationships
  void writeTripletsToStream(raw_ostream &OS) const {
    auto Result = generateTriplets();
    OS << "MAX_RELATION=" << Result.MaxRelation << '\n';
    for (const auto &T : Result.Triplets)
      OS << T.Head << '\t' << T.Tail << '\t' << T.Relation << '\n';
  }

  /// Generate entity mappings for the entire vocabulary
  /// Returns EntityList containing all entity strings
  static EntityList collectEntityMappings() {
    auto EntityLen = Vocabulary::getCanonicalSize();
    EntityList Result;
    for (unsigned EntityID = 0; EntityID < EntityLen; ++EntityID)
      Result.push_back(Vocabulary::getStringKey(EntityID).str());
    return Result;
  }

  /// Dump entity ID to string mappings
  static void writeEntitiesToStream(raw_ostream &OS) {
    auto Entities = collectEntityMappings();
    OS << Entities.size() << "\n";
    for (unsigned EntityID = 0; EntityID < Entities.size(); ++EntityID)
      OS << Entities[EntityID] << '\t' << EntityID << '\n';
  }

  /// Generate embeddings for the entire module
  void writeEmbeddingsToStream(raw_ostream &OS, EmbeddingLevel Level) const {
    if (!Vocab->isValid()) {
      WithColor::error(errs(), ToolName)
          << "Vocabulary is not valid. IR2VecTool not initialized.\n";
      return;
    }

    for (const Function &F : M.getFunctionDefs())
      writeEmbeddingsToStream(F, OS, Level);
  }

  /// Generate embeddings for a single function
  void writeEmbeddingsToStream(const Function &F, raw_ostream &OS,
                               EmbeddingLevel Level) const {
    if (!Vocab || !Vocab->isValid()) {
      WithColor::error(errs(), ToolName)
          << "Vocabulary is not valid. IR2VecTool not initialized.\n";
      return;
    }
    if (F.isDeclaration()) {
      OS << "Function " << F.getName() << " is a declaration, skipping.\n";
      return;
    }

    // Create embedder for this function
    auto Emb = Embedder::create(IR2VecEmbeddingKind, F, *Vocab);
    if (!Emb) {
      WithColor::error(errs(), ToolName)
          << "Failed to create embedder for function " << F.getName() << "\n";
      return;
    }

    OS << "Function: " << F.getName() << "\n";

    // Generate embeddings based on the specified level
    switch (Level) {
    case FunctionLevel:
      Emb->getFunctionVector().print(OS);
      break;
    case BasicBlockLevel:
      for (const BasicBlock &BB : F) {
        OS << BB.getName() << ":";
        Emb->getBBVector(BB).print(OS);
      }
      break;
    case InstructionLevel:
      for (const Instruction &I : instructions(F)) {
        OS << I;
        Emb->getInstVector(I).print(OS);
      }
      break;
    }
  }
};

} // namespace ir2vec

namespace mir2vec {

/// Relation types for MIR2Vec triplet generation
enum MIRRelationType {
  MIRNextRelation = 0, ///< Sequential instruction relationship
  MIRArgRelation = 1 ///< Instruction to operand relationship (ArgRelation + N)
};

/// Helper class for MIR2Vec embedding generation
class MIR2VecTool {
private:
  MachineModuleInfo &MMI;
  std::unique_ptr<MIRVocabulary> Vocab;

public:
  explicit MIR2VecTool(MachineModuleInfo &MMI) : MMI(MMI) {}

  /// Initialize MIR2Vec vocabulary from file (for embeddings generation)
  bool initializeVocabulary(const Module &M) {
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
  bool initializeVocabularyForLayout(const Module &M) {
    for (const Function &F : M.getFunctionDefs()) {
      MachineFunction *MF = MMI.getMachineFunction(F);
      if (!MF)
        continue;

      const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();
      const TargetRegisterInfo &TRI = *MF->getSubtarget().getRegisterInfo();
      const MachineRegisterInfo &MRI = MF->getRegInfo();

      auto VocabOrErr =
          MIRVocabulary::createDummyVocabForTest(TII, TRI, MRI, 1);
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

  /// Get triplets for a single machine function
  /// Returns TripletResult containing MaxRelation and vector of Triplets
  TripletResult generateTriplets(const MachineFunction &MF) const {
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
        // Skip debug instructions
        if (MI.isDebugInstr())
          continue;

        // Get opcode entity ID
        unsigned OpcodeID = Vocab->getEntityIDForOpcode(MI.getOpcode());

        // Add "Next" relationship with previous instruction
        if (HasPrevOpcode) {
          Result.Triplets.push_back({PrevOpcode, OpcodeID, MIRNextRelation});
          LLVM_DEBUG(dbgs()
                     << Vocab->getStringKey(PrevOpcode) << '\t'
                     << Vocab->getStringKey(OpcodeID) << '\t' << "Next\n");
        }

        // Add "Arg" relationships for operands
        unsigned ArgIndex = 0;
        for (const MachineOperand &MO : MI.operands()) {
          auto OperandID = Vocab->getEntityIDForMachineOperand(MO);
          unsigned RelationID = MIRArgRelation + ArgIndex;
          Result.Triplets.push_back({OpcodeID, OperandID, RelationID});
          LLVM_DEBUG({
            std::string OperandStr = Vocab->getStringKey(OperandID);
            dbgs() << Vocab->getStringKey(OpcodeID) << '\t' << OperandStr
                   << '\t' << "Arg" << ArgIndex << '\n';
          });

          ++ArgIndex;
        }

        // Update MaxRelation if there were operands
        if (ArgIndex > 0)
          Result.MaxRelation =
              std::max(Result.MaxRelation, MIRArgRelation + ArgIndex - 1);

        PrevOpcode = OpcodeID;
        HasPrevOpcode = true;
      }
    }

    return Result;
  }

  /// Get triplets for the entire module
  /// Returns TripletResult containing aggregated MaxRelation and all Triplets
  TripletResult generateTriplets(const Module &M) const {
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

  /// Collect triplets for the module and write to output stream
  /// Output format: MAX_RELATION=N header followed by relationships
  void writeTripletsToStream(const Module &M, raw_ostream &OS) const {
    auto Result = generateTriplets(M);
    OS << "MAX_RELATION=" << Result.MaxRelation << '\n';
    for (const auto &T : Result.Triplets)
      OS << T.Head << '\t' << T.Tail << '\t' << T.Relation << '\n';
  }

  /// Generate entity mappings for the entire vocabulary
  EntityList collectEntityMappings() const {
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

  /// Generate entity mappings and write to output stream
  void writeEntitiesToStream(raw_ostream &OS) const {
    auto Entities = collectEntityMappings();
    if (Entities.empty())
      return;

    OS << Entities.size() << "\n";
    for (unsigned EntityID = 0; EntityID < Entities.size(); ++EntityID)
      OS << Entities[EntityID] << '\t' << EntityID << '\n';
  }

  /// Generate embeddings for all machine functions in the module
  void writeEmbeddingsToStream(const Module &M, raw_ostream &OS,
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

  /// Generate embeddings for a specific machine function
  void writeEmbeddingsToStream(MachineFunction &MF, raw_ostream &OS,
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

    // Generate embeddings based on the specified level
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

  /// Get the MIR vocabulary instance
  const MIRVocabulary *getVocabulary() const { return Vocab.get(); }
};

/// Helper structure to hold MIR context
struct MIRContext {
  LLVMContext Context; // CRITICAL: Must be first for proper destruction order
  std::unique_ptr<Module> M;
  std::unique_ptr<MachineModuleInfo> MMI;
  std::unique_ptr<TargetMachine> TM;
};

} // namespace mir2vec

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_IR2VEC_LLVM_IR2VEC_H