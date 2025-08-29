//===- llvm-ir2vec.cpp - IR2Vec Embedding Generation Tool -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the IR2Vec embedding generation tool.
///
/// This tool provides three main subcommands:
///
/// 1. Triplet Generation (triplets):
///    Generates numeric triplets (head, tail, relation) for vocabulary
///    training. Output format: MAX_RELATION=N header followed by
///    head\ttail\trelation lines. Relations: 0=Type, 1=Next, 2+=Arg0,Arg1,...
///    Usage: llvm-ir2vec triplets input.bc -o train2id.txt
///
/// 2. Entity Mappings (entities):
///    Generates entity mappings for vocabulary training.
///    Output format: <total_entities> header followed by entity\tid lines.
///    Usage: llvm-ir2vec entities input.bc -o entity2id.txt
///
/// 3. Embedding Generation (embeddings):
///    Generates IR2Vec embeddings using a trained vocabulary.
///    Usage: llvm-ir2vec embeddings --ir2vec-vocab-path=vocab.json
///    --ir2vec-kind=<kind> --level=<level> input.bc -o embeddings.txt
///    Kind: --ir2vec-kind=symbolic (default), --ir2vec-kind=flow-aware
///    Levels: --level=inst (instructions), --level=bb (basic blocks),
///    --level=func (functions) (See IR2Vec.cpp for more embedding generation
///    options)
///
//===----------------------------------------------------------------------===//

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

#define DEBUG_TYPE "ir2vec"

namespace llvm {
namespace ir2vec {

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
static cl::opt<std::string>
    InputFilename(cl::Positional,
                  cl::desc("<input bitcode file or '-' for stdin>"),
                  cl::init("-"), cl::sub(TripletsSubCmd),
                  cl::sub(EmbeddingsSubCmd), cl::cat(ir2vec::IR2VecCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(ir2vec::IR2VecCategory));

// Embedding-specific options
static cl::opt<std::string>
    FunctionName("function", cl::desc("Process specific function only"),
                 cl::value_desc("name"), cl::Optional, cl::init(""),
                 cl::sub(EmbeddingsSubCmd), cl::cat(ir2vec::IR2VecCategory));

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
          cl::cat(ir2vec::IR2VecCategory));

namespace {

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
        unsigned Opcode = Vocabulary::getNumericID(I.getOpcode());
        unsigned TypeID = Vocabulary::getNumericID(I.getType()->getTypeID());

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
          unsigned OperandID = Vocabulary::getNumericID(U.get());
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
    // FIXME: Currently, the generated entity mappings are not one-to-one;
    // Multiple TypeIDs map to same string key (Like Half, BFloat, etc. map to
    // FloatTy). This would hinder learning good seed embeddings.
    // We should fix this in the future by ensuring unique string keys either by
    // post-processing here without changing the mapping in ir2vec::Vocabulary,
    // or by changing the Vocabulary generation logic to ensure unique keys.
    auto EntityLen = Vocabulary::expectedSize();
    OS << EntityLen << "\n";
    for (unsigned EntityID = 0; EntityID < EntityLen; ++EntityID)
      OS << Vocabulary::getStringKey(EntityID) << '\t' << EntityID << '\n';
  }

  /// Generate embeddings for the entire module
  void generateEmbeddings(raw_ostream &OS) const {
    if (!Vocab->isValid()) {
      OS << "Error: Vocabulary is not valid. IR2VecTool not initialized.\n";
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
      OS << "Error: Failed to create embedder for function " << F.getName()
         << "\n";
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
      const auto &BBVecMap = Emb->getBBVecMap();
      for (const BasicBlock &BB : F) {
        auto It = BBVecMap.find(&BB);
        if (It != BBVecMap.end()) {
          OS << BB.getName() << ":";
          It->second.print(OS);
        }
      }
      break;
    }
    case InstructionLevel: {
      const auto &InstMap = Emb->getInstVecMap();
      for (const BasicBlock &BB : F) {
        for (const Instruction &I : BB) {
          auto It = InstMap.find(&I);
          if (It != InstMap.end()) {
            I.print(OS);
            It->second.print(OS);
          }
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
} // namespace
} // namespace ir2vec
} // namespace llvm

int main(int argc, char **argv) {
  using namespace llvm;
  using namespace llvm::ir2vec;

  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(ir2vec::IR2VecCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "IR2Vec - Embedding Generation Tool\n"
      "Generates embeddings for a given LLVM IR and "
      "supports triplet generation for vocabulary "
      "training and embedding generation.\n\n"
      "See https://llvm.org/docs/CommandGuide/llvm-ir2vec.html for more "
      "information.\n");

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename, EC);
  if (EC) {
    errs() << "Error opening output file: " << EC.message() << "\n";
    return 1;
  }

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
    Err.print(argv[0], errs());
    return 1;
  }

  if (Error Err = processModule(*M, OS)) {
    handleAllErrors(std::move(Err), [&](const ErrorInfoBase &EIB) {
      errs() << "Error: " << EIB.message() << "\n";
    });
    return 1;
  }

  return 0;
}
