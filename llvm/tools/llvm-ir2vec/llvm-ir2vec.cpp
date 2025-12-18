//===- llvm-ir2vec.cpp - IR2Vec/MIR2Vec Embedding Generation Tool --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the IR2Vec and MIR2Vec embedding generation tool.
///
/// This tool supports two modes:
/// - LLVM IR mode (-mode=llvm): Process LLVM IR
/// - Machine IR mode (-mode=mir): Process Machine IR
///
/// Available subcommands:
///
/// 1. Triplet Generation (triplets):
///    Generates numeric triplets (head, tail, relation) for vocabulary
///    training. Output format: MAX_RELATION=N header followed by
///    head\ttail\trelation lines. Relations: 0=Type, 1=Next, 2+=Arg0,Arg1,...
///
///    For LLVM IR:
///      llvm-ir2vec triplets input.bc -o train2id.txt
///
///    For Machine IR:
///      llvm-ir2vec triplets -mode=mir input.mir -o train2id.txt
///
/// 2. Entity Mappings (entities):
///    Generates entity mappings for vocabulary training.
///    Output format: <total_entities> header followed by entity\tid lines.
///
///    For LLVM IR:
///      llvm-ir2vec entities input.bc -o entity2id.txt
///
///    For Machine IR:
///      llvm-ir2vec entities -mode=mir input.mir -o entity2id.txt
///
/// 3. Embedding Generation (embeddings):
///    Generates IR2Vec/MIR2Vec embeddings using a trained vocabulary.
///
///    For LLVM IR:
///      llvm-ir2vec embeddings --ir2vec-vocab-path=vocab.json
///        --ir2vec-kind=<kind> --level=<level> input.bc -o embeddings.txt
///      Kind: --ir2vec-kind=symbolic (default), --ir2vec-kind=flow-aware
///
///    For Machine IR:
///      llvm-ir2vec embeddings -mode=mir --mir2vec-vocab-path=vocab.json
///        --level=<level> input.mir -o embeddings.txt
///
///    Levels: --level=inst (instructions), --level=bb (basic blocks),
///    --level=func (functions) (See IR2Vec.cpp/MIR2Vec.cpp for more embedding
///    generation options)
///
//===----------------------------------------------------------------------===//

#include "llvm-ir2vec.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/MIR2Vec.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#define DEBUG_TYPE "ir2vec"

namespace llvm {

// Common option category for options shared between IR2Vec and MIR2Vec
static cl::OptionCategory CommonCategory("Common Options",
                                         "Options applicable to both IR2Vec "
                                         "and MIR2Vec modes");

enum IRKind {
  LLVMIR = 0, ///< LLVM IR
  MIR         ///< Machine IR
};

static cl::opt<IRKind>
    IRMode("mode", cl::desc("Tool operation mode:"),
           cl::values(clEnumValN(LLVMIR, "llvm", "Process LLVM IR"),
                      clEnumValN(MIR, "mir", "Process Machine IR")),
           cl::init(LLVMIR), cl::cat(CommonCategory));

// Subcommands
static cl::SubCommand
    TripletsSubCmd("triplets", "Generate triplets for vocabulary training");
static cl::SubCommand
    EntitiesSubCmd("entities",
                   "Generate entity mappings for vocabulary training");
static cl::SubCommand
    EmbeddingsSubCmd("embeddings",
                     "Generate embeddings using trained vocabulary");

// Common options
static cl::opt<std::string> InputFilename(
    cl::Positional, cl::desc("<input bitcode/MIR file or '-' for stdin>"),
    cl::init("-"), cl::sub(TripletsSubCmd), cl::sub(EntitiesSubCmd),
    cl::sub(EmbeddingsSubCmd), cl::cat(CommonCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(CommonCategory));

// Embedding-specific options
static cl::opt<std::string>
    FunctionName("function", cl::desc("Process specific function only"),
                 cl::value_desc("name"), cl::Optional, cl::init(""),
                 cl::sub(EmbeddingsSubCmd), cl::cat(CommonCategory));

static cl::opt<EmbeddingLevel>
    Level("level", cl::desc("Embedding generation level:"),
          cl::values(clEnumValN(InstructionLevel, "inst",
                                "Generate instruction-level embeddings"),
                     clEnumValN(BasicBlockLevel, "bb",
                                "Generate basic block-level embeddings"),
                     clEnumValN(FunctionLevel, "func",
                                "Generate function-level embeddings")),
          cl::init(FunctionLevel), cl::sub(EmbeddingsSubCmd),
          cl::cat(CommonCategory));

namespace ir2vec {

IR2VecTool::IR2VecTool(Module &M) : M(M) {}

bool IR2VecTool::initializeVocabulary() {
  // Register and run the IR2Vec vocabulary analysis
  // The vocabulary file path is specified via --ir2vec-vocab-path global
  // option
  MAM.registerPass([&] { return PassInstrumentationAnalysis(); });
  MAM.registerPass([&] { return IR2VecVocabAnalysis(); });
  // This will throw an error if vocab is not found or invalid
  Vocab = &MAM.getResult<IR2VecVocabAnalysis>(M);
  return Vocab->isValid();
}

TripletResult IR2VecTool::generateTriplets(const Function &F) const {
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

TripletResult IR2VecTool::generateTriplets() const {
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

void IR2VecTool::writeTripletsToStream(raw_ostream &OS) const {
  auto Result = generateTriplets();
  OS << "MAX_RELATION=" << Result.MaxRelation << '\n';
  for (const auto &T : Result.Triplets)
    OS << T.Head << '\t' << T.Tail << '\t' << T.Relation << '\n';
}

EntityList IR2VecTool::collectEntityMappings() {
  auto EntityLen = Vocabulary::getCanonicalSize();
  EntityList Result;
  for (unsigned EntityID = 0; EntityID < EntityLen; ++EntityID)
    Result.push_back(Vocabulary::getStringKey(EntityID).str());
  return Result;
}

void IR2VecTool::writeEntitiesToStream(raw_ostream &OS) {
  auto Entities = collectEntityMappings();
  OS << Entities.size() << "\n";
  for (unsigned EntityID = 0; EntityID < Entities.size(); ++EntityID)
    OS << Entities[EntityID] << '\t' << EntityID << '\n';
}

void IR2VecTool::writeEmbeddingsToStream(raw_ostream &OS,
                                         EmbeddingLevel Level) const {
  if (!Vocab->isValid()) {
    WithColor::error(errs(), ToolName)
        << "Vocabulary is not valid. IR2VecTool not initialized.\n";
    return;
  }

  for (const Function &F : M.getFunctionDefs())
    writeEmbeddingsToStream(F, OS, Level);
}

void IR2VecTool::writeEmbeddingsToStream(const Function &F, raw_ostream &OS,
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

/// Process the module and generate output based on selected subcommand
Error processModule(Module &M, raw_ostream &OS) {
  IR2VecTool Tool(M);

  if (EmbeddingsSubCmd) {
    // Initialize vocabulary for embedding generation
    // Note: Requires --ir2vec-vocab-path option to be set
    auto VocabStatus = Tool.initializeVocabulary();
    assert(VocabStatus && "Failed to initialize IR2Vec vocabulary");
    (void)VocabStatus;

    if (!FunctionName.empty()) {
      // Process single function
      if (const Function *F = M.getFunction(FunctionName))
        Tool.writeEmbeddingsToStream(*F, OS, Level);
      else
        return createStringError(errc::invalid_argument,
                                 "Function '%s' not found",
                                 FunctionName.c_str());
    } else {
      // Process all functions
      Tool.writeEmbeddingsToStream(OS, Level);
    }
  } else {
    // Both triplets and entities use triplet generation
    Tool.writeTripletsToStream(OS);
  }
  return Error::success();
}
} // namespace ir2vec

namespace mir2vec {

MIR2VecTool::MIR2VecTool(MachineModuleInfo &MMI) : MMI(MMI) {}

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

const MIRVocabulary *MIR2VecTool::getVocabulary() const { return Vocab.get(); }

/// Setup MIR context from input file
Error setupMIRContext(const std::string &InputFile, MIRContext &Ctx) {
  SMDiagnostic Err;

  auto MIR = createMIRParserFromFile(InputFile, Err, Ctx.Context);
  if (!MIR) {
    Err.print(ToolName, errs());
    return createStringError(errc::invalid_argument,
                             "Failed to parse MIR file");
  }

  auto SetDataLayout = [&](StringRef DataLayoutTargetTriple,
                           StringRef OldDLStr) -> std::optional<std::string> {
    std::string IRTargetTriple = DataLayoutTargetTriple.str();
    Triple TheTriple = Triple(IRTargetTriple);
    if (TheTriple.getTriple().empty())
      TheTriple.setTriple(sys::getDefaultTargetTriple());

    auto TMOrErr = codegen::createTargetMachineForTriple(TheTriple.str());
    if (!TMOrErr) {
      Err.print(ToolName, errs());
      exit(1); // Match original behavior
    }
    Ctx.TM = std::move(*TMOrErr);
    return Ctx.TM->createDataLayout().getStringRepresentation();
  };

  Ctx.M = MIR->parseIRModule(SetDataLayout);
  if (!Ctx.M) {
    Err.print(ToolName, errs());
    return createStringError(errc::invalid_argument,
                             "Failed to parse IR module");
  }

  Ctx.MMI = std::make_unique<MachineModuleInfo>(Ctx.TM.get());
  if (!Ctx.MMI || MIR->parseMachineFunctions(*Ctx.M, *Ctx.MMI)) {
    Err.print(ToolName, errs());
    return createStringError(errc::invalid_argument,
                             "Failed to parse machine functions");
  }

  return Error::success();
}

/// Generic vocabulary initialization and processing
template <typename ProcessFunc>
Error processWithVocabulary(MIRContext &Ctx, raw_ostream &OS,
                            bool useLayoutVocab, ProcessFunc processFn) {
  MIR2VecTool Tool(*Ctx.MMI);

  // Initialize appropriate vocabulary type
  bool success = useLayoutVocab ? Tool.initializeVocabularyForLayout(*Ctx.M)
                                : Tool.initializeVocabulary(*Ctx.M);

  if (!success) {
    WithColor::error(errs(), ToolName)
        << "Failed to initialize MIR2Vec vocabulary"
        << (useLayoutVocab ? " for layout" : "") << ".\n";
    return createStringError(errc::invalid_argument,
                             "Vocabulary initialization failed");
  }

  assert(Tool.getVocabulary() &&
         "MIR2Vec vocabulary should be initialized at this point");

  LLVM_DEBUG(dbgs() << "MIR2Vec vocabulary loaded successfully.\n"
                    << "Vocabulary dimension: "
                    << Tool.getVocabulary()->getDimension() << "\n"
                    << "Vocabulary size: "
                    << Tool.getVocabulary()->getCanonicalSize() << "\n");

  // Execute the specific processing logic
  return processFn(Tool);
}

/// Process module for triplet generation
Error processModuleForTriplets(MIRContext &Ctx, raw_ostream &OS) {
  return processWithVocabulary(Ctx, OS, /*useLayoutVocab=*/true,
                               [&](MIR2VecTool &Tool) -> Error {
                                 Tool.writeTripletsToStream(*Ctx.M, OS);
                                 return Error::success();
                               });
}

/// Process module for entity generation
Error processModuleForEntities(MIRContext &Ctx, raw_ostream &OS) {
  return processWithVocabulary(Ctx, OS, /*useLayoutVocab=*/true,
                               [&](MIR2VecTool &Tool) -> Error {
                                 Tool.writeEntitiesToStream(OS);
                                 return Error::success();
                               });
}

/// Process module for embedding generation
Error processModuleForEmbeddings(MIRContext &Ctx, raw_ostream &OS) {
  return processWithVocabulary(
      Ctx, OS, /*useLayoutVocab=*/false, [&](MIR2VecTool &Tool) -> Error {
        if (!FunctionName.empty()) {
          // Process single function
          Function *F = Ctx.M->getFunction(FunctionName);
          if (!F) {
            WithColor::error(errs(), ToolName)
                << "Function '" << FunctionName << "' not found\n";
            return createStringError(errc::invalid_argument,
                                     "Function not found");
          }

          MachineFunction *MF = Ctx.MMI->getMachineFunction(*F);
          if (!MF) {
            WithColor::error(errs(), ToolName)
                << "No MachineFunction for " << FunctionName << "\n";
            return createStringError(errc::invalid_argument,
                                     "No MachineFunction");
          }

          Tool.writeEmbeddingsToStream(*MF, OS, Level);
        } else {
          // Process all functions
          Tool.writeEmbeddingsToStream(*Ctx.M, OS, Level);
        }
        return Error::success();
      });
}

/// Main entry point for MIR processing
Error processModule(const std::string &InputFile, raw_ostream &OS) {
  MIRContext Ctx;

  // Setup MIR context (parse file, setup target machine, etc.)
  if (auto Err = setupMIRContext(InputFile, Ctx))
    return Err;

  // Process based on subcommand
  if (TripletsSubCmd)
    return processModuleForTriplets(Ctx, OS);
  else if (EntitiesSubCmd)
    return processModuleForEntities(Ctx, OS);
  else if (EmbeddingsSubCmd)
    return processModuleForEmbeddings(Ctx, OS);
  else {
    WithColor::error(errs(), ToolName)
        << "Please specify a subcommand: triplets, entities, or embeddings\n";
    return createStringError(errc::invalid_argument, "No subcommand specified");
  }
}

} // namespace mir2vec

} // namespace llvm

int main(int argc, char **argv) {
  using namespace llvm;
  using namespace llvm::ir2vec;
  using namespace llvm::mir2vec;

  InitLLVM X(argc, argv);
  // Show Common, IR2Vec and MIR2Vec option categories
  cl::HideUnrelatedOptions(ArrayRef<const cl::OptionCategory *>{
      &CommonCategory, &ir2vec::IR2VecCategory, &mir2vec::MIR2VecCategory});
  cl::ParseCommandLineOptions(
      argc, argv,
      "IR2Vec/MIR2Vec - Embedding Generation Tool\n"
      "Generates embeddings for a given LLVM IR or MIR and "
      "supports triplet generation for vocabulary "
      "training and embedding generation.\n\n"
      "See https://llvm.org/docs/CommandGuide/llvm-ir2vec.html for more "
      "information.\n");

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename, EC);
  if (EC) {
    WithColor::error(errs(), ToolName)
        << "opening output file: " << EC.message() << "\n";
    return 1;
  }

  if (IRMode == IRKind::LLVMIR) {
    if (EntitiesSubCmd) {
      // Just dump entity mappings without processing any IR
      IR2VecTool::writeEntitiesToStream(OS);
      return 0;
    }

    // Parse the input LLVM IR file or stdin
    SMDiagnostic Err;
    LLVMContext Context;
    std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
    if (!M) {
      Err.print(ToolName, errs());
      return 1;
    }

    if (Error Err = processModule(*M, OS)) {
      handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EIB) {
        WithColor::error(errs(), ToolName) << EIB.message() << "\n";
      });
      return 1;
    }
    return 0;
  }
  if (IRMode == IRKind::MIR) {
    // Initialize targets for Machine IR processing
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllAsmPrinters();
    static codegen::RegisterCodeGenFlags CGF;

    if (Error Err = mir2vec::processModule(InputFilename, OS)) {
      handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EIB) {
        WithColor::error(errs(), ToolName) << EIB.message() << "\n";
      });
      return 1;
    }

    return 0;
  }

  return 0;
}
