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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
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

static const char *ToolName = "llvm-ir2vec";

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

enum EmbeddingLevel {
  InstructionLevel, // Generate instruction-level embeddings
  BasicBlockLevel,  // Generate basic block-level embeddings
  FunctionLevel     // Generate function-level embeddings
};

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

  /// Generate triplets for the module
  /// Output format: MAX_RELATION=N header followed by relationships
  void generateTriplets(raw_ostream &OS) const {
    unsigned MaxRelation = NextRelation; // Track maximum relation ID
    std::string Relationships;
    raw_string_ostream RelOS(Relationships);

    for (const Function &F : M) {
      unsigned FuncMaxRelation = generateTriplets(F, RelOS);
      MaxRelation = std::max(MaxRelation, FuncMaxRelation);
    }

    RelOS.flush();

    // Write metadata header followed by relationships
    OS << "MAX_RELATION=" << MaxRelation << '\n';
    OS << Relationships;
  }

  /// Generate triplets for a single function
  /// Returns the maximum relation ID used in this function
  unsigned generateTriplets(const Function &F, raw_ostream &OS) const {
    if (F.isDeclaration())
      return 0;

    unsigned MaxRelation = 1;
    unsigned PrevOpcode = 0;
    bool HasPrevOpcode = false;

    for (const BasicBlock &BB : F) {
      for (const auto &I : BB.instructionsWithoutDebug()) {
        unsigned Opcode = Vocabulary::getIndex(I.getOpcode());
        unsigned TypeID = Vocabulary::getIndex(I.getType()->getTypeID());

        // Add "Next" relationship with previous instruction
        if (HasPrevOpcode) {
          OS << PrevOpcode << '\t' << Opcode << '\t' << NextRelation << '\n';
          LLVM_DEBUG(dbgs()
                     << Vocabulary::getVocabKeyForOpcode(PrevOpcode + 1) << '\t'
                     << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                     << "Next\n");
        }

        // Add "Type" relationship
        OS << Opcode << '\t' << TypeID << '\t' << TypeRelation << '\n';
        LLVM_DEBUG(
            dbgs() << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                   << Vocabulary::getVocabKeyForTypeID(I.getType()->getTypeID())
                   << '\t' << "Type\n");

        // Add "Arg" relationships
        unsigned ArgIndex = 0;
        for (const Use &U : I.operands()) {
          unsigned OperandID = Vocabulary::getIndex(*U.get());
          unsigned RelationID = ArgRelation + ArgIndex;
          OS << Opcode << '\t' << OperandID << '\t' << RelationID << '\n';

          LLVM_DEBUG({
            StringRef OperandStr = Vocabulary::getVocabKeyForOperandKind(
                Vocabulary::getOperandKind(U.get()));
            dbgs() << Vocabulary::getVocabKeyForOpcode(Opcode + 1) << '\t'
                   << OperandStr << '\t' << "Arg" << ArgIndex << '\n';
          });

          ++ArgIndex;
        }
        // Only update MaxRelation if there were operands
        if (ArgIndex > 0) {
          MaxRelation = std::max(MaxRelation, ArgRelation + ArgIndex - 1);
        }
        PrevOpcode = Opcode;
        HasPrevOpcode = true;
      }
    }

    return MaxRelation;
  }

  /// Dump entity ID to string mappings
  static void generateEntityMappings(raw_ostream &OS) {
    auto EntityLen = Vocabulary::getCanonicalSize();
    OS << EntityLen << "\n";
    for (unsigned EntityID = 0; EntityID < EntityLen; ++EntityID)
      OS << Vocabulary::getStringKey(EntityID) << '\t' << EntityID << '\n';
  }

  /// Generate embeddings for the entire module
  void generateEmbeddings(raw_ostream &OS) const {
    if (!Vocab->isValid()) {
      WithColor::error(errs(), ToolName)
          << "Vocabulary is not valid. IR2VecTool not initialized.\n";
      return;
    }

    for (const Function &F : M)
      generateEmbeddings(F, OS);
  }

  /// Generate embeddings for a single function
  void generateEmbeddings(const Function &F, raw_ostream &OS) const {
    if (F.isDeclaration()) {
      OS << "Function " << F.getName() << " is a declaration, skipping.\n";
      return;
    }

    // Create embedder for this function
    assert(Vocab->isValid() && "Vocabulary is not valid");
    auto Emb = Embedder::create(IR2VecEmbeddingKind, F, *Vocab);
    if (!Emb) {
      WithColor::error(errs(), ToolName)
          << "Failed to create embedder for function " << F.getName() << "\n";
      return;
    }

    OS << "Function: " << F.getName() << "\n";

    // Generate embeddings based on the specified level
    switch (Level) {
    case FunctionLevel: {
      Emb->getFunctionVector().print(OS);
      break;
    }
    case BasicBlockLevel: {
      for (const BasicBlock &BB : F) {
        OS << BB.getName() << ":";
        Emb->getBBVector(BB).print(OS);
      }
      break;
    }
    case InstructionLevel: {
      for (const BasicBlock &BB : F) {
        for (const Instruction &I : BB) {
          I.print(OS);
          Emb->getInstVector(I).print(OS);
        }
      }
      break;
    }
    }
  }
};

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
        Tool.generateEmbeddings(*F, OS);
      else
        return createStringError(errc::invalid_argument,
                                 "Function '%s' not found",
                                 FunctionName.c_str());
    } else {
      // Process all functions
      Tool.generateEmbeddings(OS);
    }
  } else {
    // Both triplets and entities use triplet generation
    Tool.generateTriplets(OS);
  }
  return Error::success();
}
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
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;

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

  /// Generate triplets for the module
  /// Output format: MAX_RELATION=N header followed by relationships
  void generateTriplets(const Module &M, raw_ostream &OS) const {
    unsigned MaxRelation = MIRNextRelation; // Track maximum relation ID
    std::string Relationships;
    raw_string_ostream RelOS(Relationships);

    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;

      MachineFunction *MF = MMI.getMachineFunction(F);
      if (!MF) {
        WithColor::warning(errs(), ToolName)
            << "No MachineFunction for " << F.getName() << "\n";
        continue;
      }

      unsigned FuncMaxRelation = generateTriplets(*MF, RelOS);
      MaxRelation = std::max(MaxRelation, FuncMaxRelation);
    }

    RelOS.flush();

    // Write metadata header followed by relationships
    OS << "MAX_RELATION=" << MaxRelation << '\n';
    OS << Relationships;
  }

  /// Generate triplets for a single machine function
  /// Returns the maximum relation ID used in this function
  unsigned generateTriplets(const MachineFunction &MF, raw_ostream &OS) const {
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
          OS << PrevOpcode << '\t' << OpcodeID << '\t' << MIRNextRelation
             << '\n';
          LLVM_DEBUG(dbgs()
                     << Vocab->getStringKey(PrevOpcode) << '\t'
                     << Vocab->getStringKey(OpcodeID) << '\t' << "Next\n");
        }

        // Add "Arg" relationships for operands
        unsigned ArgIndex = 0;
        for (const MachineOperand &MO : MI.operands()) {
          auto OperandID = Vocab->getEntityIDForMachineOperand(MO);
          unsigned RelationID = MIRArgRelation + ArgIndex;
          OS << OpcodeID << '\t' << OperandID << '\t' << RelationID << '\n';
          LLVM_DEBUG({
            std::string OperandStr = Vocab->getStringKey(OperandID);
            dbgs() << Vocab->getStringKey(OpcodeID) << '\t' << OperandStr
                   << '\t' << "Arg" << ArgIndex << '\n';
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

  /// Generate entity mappings with vocabulary
  void generateEntityMappings(raw_ostream &OS) const {
    if (!Vocab) {
      WithColor::error(errs(), ToolName)
          << "Vocabulary must be initialized for entity mappings.\n";
      return;
    }

    const unsigned EntityCount = Vocab->getCanonicalSize();
    OS << EntityCount << "\n";
    for (unsigned EntityID = 0; EntityID < EntityCount; ++EntityID)
      OS << Vocab->getStringKey(EntityID) << '\t' << EntityID << '\n';
  }

  /// Generate embeddings for all machine functions in the module
  void generateEmbeddings(const Module &M, raw_ostream &OS) const {
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

  /// Generate embeddings for a specific machine function
  void generateEmbeddings(MachineFunction &MF, raw_ostream &OS) const {
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

  const MIRVocabulary *getVocabulary() const { return Vocab.get(); }
};

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
      IR2VecTool::generateEntityMappings(OS);
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

    // Parse MIR input file
    SMDiagnostic Err;
    LLVMContext Context;
    std::unique_ptr<TargetMachine> TM;

    auto MIR = createMIRParserFromFile(InputFilename, Err, Context);
    if (!MIR) {
      Err.print(ToolName, errs());
      return 1;
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
        exit(1);
      }
      TM = std::move(*TMOrErr);
      return TM->createDataLayout().getStringRepresentation();
    };

    std::unique_ptr<Module> M = MIR->parseIRModule(SetDataLayout);
    if (!M) {
      Err.print(ToolName, errs());
      return 1;
    }

    // Parse machine functions
    auto MMI = std::make_unique<MachineModuleInfo>(TM.get());
    if (!MMI || MIR->parseMachineFunctions(*M, *MMI)) {
      Err.print(ToolName, errs());
      return 1;
    }

    // Create MIR2Vec tool
    MIR2VecTool Tool(*MMI);

    // Initialize vocabulary. For triplet/entity generation, only layout is
    // needed For embedding generation, the full vocabulary is needed.
    //
    // Note: Unlike IR2Vec, MIR2Vec vocabulary initialization requires
    // target-specific information for generating the vocabulary layout. So, we
    // always initialize the vocabulary in this case.
    if (TripletsSubCmd || EntitiesSubCmd) {
      if (!Tool.initializeVocabularyForLayout(*M)) {
        WithColor::error(errs(), ToolName)
            << "Failed to initialize MIR2Vec vocabulary for layout.\n";
        return 1;
      }
    } else {
      if (!Tool.initializeVocabulary(*M)) {
        WithColor::error(errs(), ToolName)
            << "Failed to initialize MIR2Vec vocabulary.\n";
        return 1;
      }
    }
    assert(Tool.getVocabulary() &&
           "MIR2Vec vocabulary should be initialized at this point");
    LLVM_DEBUG(dbgs() << "MIR2Vec vocabulary loaded successfully.\n"
                      << "Vocabulary dimension: "
                      << Tool.getVocabulary()->getDimension() << "\n"
                      << "Vocabulary size: "
                      << Tool.getVocabulary()->getCanonicalSize() << "\n");

    // Handle subcommands
    if (TripletsSubCmd) {
      Tool.generateTriplets(*M, OS);
    } else if (EntitiesSubCmd) {
      Tool.generateEntityMappings(OS);
    } else if (EmbeddingsSubCmd) {
      if (!FunctionName.empty()) {
        // Process single function
        Function *F = M->getFunction(FunctionName);
        if (!F) {
          WithColor::error(errs(), ToolName)
              << "Function '" << FunctionName << "' not found\n";
          return 1;
        }

        MachineFunction *MF = MMI->getMachineFunction(*F);
        if (!MF) {
          WithColor::error(errs(), ToolName)
              << "No MachineFunction for " << FunctionName << "\n";
          return 1;
        }

        Tool.generateEmbeddings(*MF, OS);
      } else {
        // Process all functions
        Tool.generateEmbeddings(*M, OS);
      }
    } else {
      WithColor::error(errs(), ToolName)
          << "Please specify a subcommand: triplets, entities, or embeddings\n";
      return 1;
    }

    return 0;
  }

  return 0;
}
