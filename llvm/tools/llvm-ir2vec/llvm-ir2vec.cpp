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
/// Currently supports triplet generation for vocabulary training.
/// Future updates will support embedding generation using trained vocabulary.
///
/// Usage: llvm-ir2vec input.bc -o triplets.txt
///
/// TODO: Add embedding generation mode with vocabulary support
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IR2Vec.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace ir2vec;

#define DEBUG_TYPE "ir2vec"

static cl::OptionCategory IR2VecToolCategory("IR2Vec Tool Options");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input bitcode file>"),
                                          cl::Required,
                                          cl::cat(IR2VecToolCategory));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"),
                                           cl::cat(IR2VecToolCategory));

namespace {

/// Helper class for collecting IR information and generating triplets
class IR2VecTool {
private:
  Module &M;

public:
  explicit IR2VecTool(Module &M) : M(M) {}

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
  Tool.generateTriplets(OS);

  return Error::success();
}

} // anonymous namespace

int main(int argc, char **argv) {
  InitLLVM X(argc, argv);
  cl::HideUnrelatedOptions(IR2VecToolCategory);
  cl::ParseCommandLineOptions(
      argc, argv,
      "IR2Vec - Triplet Generation Tool\n"
      "Generates triplets for vocabulary training from LLVM IR.\n"
      "Future updates will support embedding generation.\n\n"
      "Usage:\n"
      "  llvm-ir2vec input.bc -o triplets.txt\n");

  // Parse the input LLVM IR file
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
