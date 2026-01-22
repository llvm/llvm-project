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

#include "lib/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/IR2Vec.h"
#include "llvm/CodeGen/CommandFlags.h"
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
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
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

/// Process the module and generate output based on selected subcommand
static Error processModule(Module &M, raw_ostream &OS) {
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

/// Setup MIR context from input file
static Error setupMIRContext(const std::string &InputFile, MIRContext &Ctx) {
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
static Error processWithVocabulary(MIRContext &Ctx, raw_ostream &OS,
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
static Error processModuleForTriplets(MIRContext &Ctx, raw_ostream &OS) {
  return processWithVocabulary(Ctx, OS, /*useLayoutVocab=*/true,
                               [&](MIR2VecTool &Tool) -> Error {
                                 Tool.writeTripletsToStream(*Ctx.M, OS);
                                 return Error::success();
                               });
}

/// Process module for entity generation
static Error processModuleForEntities(MIRContext &Ctx, raw_ostream &OS) {
  return processWithVocabulary(Ctx, OS, /*useLayoutVocab=*/true,
                               [&](MIR2VecTool &Tool) -> Error {
                                 Tool.writeEntitiesToStream(OS);
                                 return Error::success();
                               });
}

/// Process module for embedding generation
static Error processModuleForEmbeddings(MIRContext &Ctx, raw_ostream &OS) {
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
static Error processModule(const std::string &InputFile, raw_ostream &OS) {
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
