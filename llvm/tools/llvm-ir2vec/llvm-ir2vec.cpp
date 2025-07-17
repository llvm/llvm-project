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
/// This tool provides two main functionalities:
///
/// 1. Triplet Generation Mode (--mode=triplets):
///    Generates triplets (opcode, type, operands) for vocabulary training.
///    Usage: llvm-ir2vec --mode=triplets input.bc -o triplets.txt
///
/// 2. Embedding Generation Mode (--mode=embeddings):
///    Generates IR2Vec embeddings using a trained vocabulary.
///    Usage: llvm-ir2vec --mode=embeddings --ir2vec-vocab-path=vocab.json
///    --level=func input.bc -o embeddings.txt Levels: --level=inst
///    (instructions), --level=bb (basic blocks), --level=func (functions)
///    (See IR2Vec.cpp for more embedding generation options)
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

static cl::OptionCategory IR2VecToolCategory("IR2Vec Tool Options");

static cl::opt<std::string>
    InputFilename(cl::Positional,
                  cl::desc("<input bitcode file or '-' for stdin>"),
                  cl::init("-"), cl::cat(IR2VecToolCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(IR2VecToolCategory));

enum ToolMode {
  TripletMode,  // Generate triplets for vocabulary training
  EmbeddingMode // Generate embeddings using trained vocabulary
};

static cl::opt<ToolMode>
    Mode("mode", cl::desc("Tool operation mode:"),
         cl::values(clEnumValN(TripletMode, "triplets",
                               "Generate triplets for vocabulary training"),
                    clEnumValN(EmbeddingMode, "embeddings",
                               "Generate embeddings using trained vocabulary")),
         cl::init(EmbeddingMode), cl::cat(IR2VecToolCategory));

static cl::opt<std::string>
    FunctionName("function", cl::desc("Process specific function only"),
                 cl::value_desc("name"), cl::Optional, cl::init(""),
                 cl::cat(IR2VecToolCategory));

enum EmbeddingLevel {
  InstructionLevel, // Generate instruction-level embeddings
  BasicBlockLevel,  // Generate basic block-level embeddings
  FunctionLevel     // Generate function-level embeddings
};

static cl::opt<EmbeddingLevel>
    Level("level", cl::desc("Embedding generation level (for embedding mode):"),
          cl::values(clEnumValN(InstructionLevel, "inst",
                                "Generate instruction-level embeddings"),
                     clEnumValN(BasicBlockLevel, "bb",
                                "Generate basic block-level embeddings"),
                     clEnumValN(FunctionLevel, "func",
                                "Generate function-level embeddings")),
          cl::init(FunctionLevel), cl::cat(IR2VecToolCategory));

namespace {

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
    Vocab = &MAM.getResult<IR2VecVocabAnalysis>(M);
    return Vocab->isValid();
  }

  /// Generate triplets for the entire module
  void generateTriplets(raw_ostream &OS) const {
    for (const Function &F : M)
      generateTriplets(F, OS);
  }

  /// Generate triplets for a single function
  void generateTriplets(const Function &F, raw_ostream &OS) const {
    if (F.isDeclaration())
      return;

    std::string LocalOutput;
    raw_string_ostream LocalOS(LocalOutput);

    for (const BasicBlock &BB : F)
      traverseBasicBlock(BB, LocalOS);

    LocalOS.flush();
    OS << LocalOutput;
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
    auto Emb = Embedder::create(IR2VecKind::Symbolic, F, *Vocab);
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

private:
  /// Process a single basic block for triplet generation
  void traverseBasicBlock(const BasicBlock &BB, raw_string_ostream &OS) const {
    // Consider only non-debug and non-pseudo instructions
    for (const auto &I : BB.instructionsWithoutDebug()) {
      StringRef OpcStr = Vocabulary::getVocabKeyForOpcode(I.getOpcode());
      StringRef TypeStr =
          Vocabulary::getVocabKeyForTypeID(I.getType()->getTypeID());

      OS << '\n' << OpcStr << ' ' << TypeStr << ' ';

      LLVM_DEBUG({
        I.print(dbgs());
        dbgs() << "\n";
        I.getType()->print(dbgs());
        dbgs() << " Type\n";
      });

      for (const Use &U : I.operands())
        OS << Vocabulary::getVocabKeyForOperandKind(
                  Vocabulary::getOperandKind(U.get()))
           << ' ';
    }
  }
};

Error processModule(Module &M, raw_ostream &OS) {
  IR2VecTool Tool(M);

  if (Mode == EmbeddingMode) {
    // Initialize vocabulary for embedding generation
    // Note: Requires --ir2vec-vocab-path option to be set
    if (!Tool.initializeVocabulary())
      return createStringError(
          errc::invalid_argument,
          "Failed to initialize IR2Vec vocabulary. "
          "Make sure to specify --ir2vec-vocab-path for embedding mode.");

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
    // Triplet generation mode - no vocabulary needed
    if (!FunctionName.empty())
      // Process single function
      if (const Function *F = M.getFunction(FunctionName))
        Tool.generateTriplets(*F, OS);
      else
        return createStringError(errc::invalid_argument,
                                 "Function '%s' not found",
                                 FunctionName.c_str());
    else
      // Process all functions
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
  cl::HideUnrelatedOptions(IR2VecToolCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "IR2Vec - Embedding Generation Tool\n"
      "Generates embeddings for a given LLVM IR and "
      "supports triplet generation for vocabulary "
      "training and embedding generation.\n\n"
      "See https://llvm.org/docs/CommandGuide/llvm-ir2vec.html for more "
      "information.\n");

  // Validate command line options
  if (Mode == TripletMode && Level.getNumOccurrences() > 0)
    errs() << "Warning: --level option is ignored in triplet mode\n";

  // Parse the input LLVM IR file or stdin
  SMDiagnostic Err;
  LLVMContext Context;
  std::unique_ptr<Module> M = parseIRFile(InputFilename, Err, Context);
  if (!M) {
    Err.print(argv[0], errs());
    return 1;
  }

  std::error_code EC;
  raw_fd_ostream OS(OutputFilename, EC);
  if (EC) {
    errs() << "Error opening output file: " << EC.message() << "\n";
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
