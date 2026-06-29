//===-- IRDocument.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_LSP_IRDOCUMENT_H
#define LLVM_TOOLS_LLVM_LSP_IRDOCUMENT_H

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/AsmParser/AsmParserContext.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/SourceMgr.h"

#include <memory>

namespace llvm {
// Tracks and Manages the Cache of all Artifacts for a given IR.
// LSP Server will use this class to query details about the IR file.
class IRDocument {
  LLVMContext C;
  std::unique_ptr<Module> ParsedModule;
  StringRef Filepath;

public:
  IRDocument(StringRef PathToIRFile) : Filepath(PathToIRFile) {
    ParsedModule = loadModuleFromIR(PathToIRFile, C);

    lsp::Logger::info("Finished setting up IR Document: {}",
                      PathToIRFile.str());
  }

  // ---------------- APIs that the Language Server can use  -----------------

  auto &getFunctions() { return ParsedModule->getFunctionList(); }

  Instruction *getInstructionAtLocation(unsigned Line, unsigned Col) {
    FileLoc FL(Line, Col);
    if (auto *MaybeI = ParserContext.getInstructionAtLocation(FL))
      return MaybeI;
    return nullptr;
  }

  AsmParserContext ParserContext;

private:
  std::unique_ptr<Module> loadModuleFromIR(StringRef Filepath, LLVMContext &C) {
    SMDiagnostic Err;
    // Try to parse as textual IR
    auto M = parseIRFile(Filepath, Err, C, {}, &ParserContext);
    if (!M) {
      // If parsing failed, print the error and crash
      lsp::Logger::error("Failed parsing IR file: {}", Err.getMessage().str());
      return nullptr;
    }
    return M;
  }
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_LSP_IRDOCUMENT_H
