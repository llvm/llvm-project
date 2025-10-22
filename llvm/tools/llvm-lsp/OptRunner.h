//===-- OptRunner.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_LSP_OPTRUNNER_H
#define LLVM_TOOLS_LLVM_LSP_OPTRUNNER_H

#include "Protocol.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <memory>
#include <string>

namespace llvm {

// FIXME: Maybe a better name?
class OptRunner {
  LLVMContext Context;
  const Module &InitialIR;

  SmallVector<std::unique_ptr<Module>, 256> IntermediateIRList;

public:
  OptRunner(Module &IIR) : InitialIR(IIR) {}

  llvm::Expected<SmallVector<std::pair<std::string, std::string>, 256>>
  getPassListAndDescription(const std::string PipelineText) {
    // First is Passname, Second is Pass Description.
    SmallVector<std::pair<std::string, std::string>, 256>
        PassListAndDescription;
    unsigned PassNumber = 0;
    // FIXME: Should we only consider passes that modify the IR?
    std::function<void(const StringRef, Any, const PreservedAnalyses)>
        RecordPassNamesAndDescription =
            [&PassListAndDescription, &PassNumber](
                const StringRef PassName, Any IR, const PreservedAnalyses &PA) {
              PassNumber++;
              std::string PassNameStr =
                  (std::to_string(PassNumber) + "-" + PassName.str());
              std::string PassDescStr = [&IR, &PassName]() -> std::string {
                if (auto *M = any_cast<const Module *>(&IR))
                  return "Module Pass on \"" + (**M).getName().str() + "\"";
                if (auto *F = any_cast<const Function *>(&IR))
                  return "Function Pass on \"" + (**F).getName().str() + "\"";
                if (auto *L = any_cast<const Loop *>(&IR)) {
                  Function *F = (*L)->getHeader()->getParent();
                  std::string Desc = "Loop Pass in Function \"" +
                                     F->getName().str() +
                                     "\" on loop with Header \"" +
                                     (*L)->getHeader()->getName().str() + "\"";
                  return Desc;
                }
                if (auto *SCC = any_cast<const LazyCallGraph::SCC *>(&IR)) {
                  Function &F = (**SCC).begin()->getFunction();
                  std::string Desc =
                      "CGSCC Pass on Function \"" + F.getName().str() + "\"";
                  return Desc;
                }
                lsp::Logger::error("Unknown Pass Type \"{}\"!", PassName.str());
                return "";
              }();

              PassListAndDescription.push_back({PassNameStr, PassDescStr});
            };

    auto RunOptResult = runOpt(PipelineText, RecordPassNamesAndDescription);
    if (!RunOptResult) {
      lsp::Logger::info("Handling error in getPassListAndDescription()");
      return RunOptResult.takeError();
    }
    return PassListAndDescription;
  }

  llvm::Expected<std::unique_ptr<Module>>
  runOpt(const std::string PipelineText,
         std::function<void(const StringRef, Any, const PreservedAnalyses)>
             &AfterPassCallback) {
    // Analysis Managers
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassInstrumentationCallbacks PIC;

    ModulePassManager MPM;
    PassBuilder PB;

    // Callback that redirects to a custom callback.
    PIC.registerAfterPassCallback(AfterPassCallback);

    PB = PassBuilder(nullptr, PipelineTuningOptions(), std::nullopt, &PIC);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Parse Pipeline text
    auto ParseError = PB.parsePassPipeline(MPM, PipelineText);
    if (ParseError) {
      lsp::Logger::info("Error parsing pipeline text!");
      return llvm::createStringError(toString(std::move(ParseError)).c_str());
    }

    // Run Opt on a copy of the original IR, so that we dont modify the original
    // IR.
    auto FinalIR = CloneModule(InitialIR);
    MPM.run(*FinalIR, MAM);
    return FinalIR;
  }

  // TODO: Check if N lies with in bounds for below methods. And to verify that
  // they are populated.
  // N is 1-Indexed
  llvm::Expected<std::unique_ptr<Module>>
  getModuleAfterPass(const std::string PipelineText, unsigned N) {
    unsigned PassNumber = 0;
    std::unique_ptr<Module> IntermediateIR = nullptr;
    std::function<void(const StringRef, Any, const PreservedAnalyses)>
        RecordIRAfterPass = [&PassNumber, &N,
                             &IntermediateIR](const StringRef PassName, Any IR,
                                              const PreservedAnalyses &PA) {
          PassNumber++;
          if (PassNumber == N) {
            IntermediateIR = [&IR, &PassName]() -> std::unique_ptr<Module> {
              if (auto *M = any_cast<const Module *>(&IR))
                return CloneModule(**M);
              if (auto *F = any_cast<const Function *>(&IR))
                return CloneModule(*(**F).getParent());
              if (auto *L = any_cast<const Loop *>(&IR))
                return CloneModule(
                    *((*L)->getHeader()->getParent())->getParent());
              if (auto *SCC = any_cast<const LazyCallGraph::SCC *>(&IR))
                return CloneModule(
                    *((**SCC).begin()->getFunction()).getParent());

              lsp::Logger::error("Unknown Pass Type \"{}\"!", PassName.str());
              return nullptr;
            }();
          }
        };

    auto RunOptResult = runOpt(PipelineText, RecordIRAfterPass);
    if (!RunOptResult) {
      return RunOptResult.takeError();
    }

    if (!IntermediateIR) {
      lsp::Logger::error("Unrecognized Pass Number {}!", std::to_string(N));
      return make_error<lsp::LSPError>(
          formatv("Unrecognized pass number {}!", N),
          lsp::ErrorCode::InvalidParams);
    }

    return IntermediateIR;
  }

  llvm::Expected<std::unique_ptr<Module>>
  getFinalModule(const std::string PipelineText) {
    std::function<void(const StringRef, Any, const PreservedAnalyses)>
        EmptyCallback = [](const StringRef, Any, const PreservedAnalyses &) {};
    return runOpt(PipelineText, EmptyCallback);
  }

  llvm::Expected<std::string> getPassName(std::string PipelineText,
                                          unsigned N) {
    unsigned PassNumber = 0;
    std::string IntermediatePassName = "";
    std::function<void(const StringRef, Any, const PreservedAnalyses)>
        RecordNameAfterPass =
            [&PassNumber, &N, &IntermediatePassName](
                const StringRef PassName, Any IR, const PreservedAnalyses &PA) {
              PassNumber++;
              if (PassNumber == N)
                IntermediatePassName = PassName.str();
            };

    auto RunOptResult = runOpt(PipelineText, RecordNameAfterPass);
    if (!RunOptResult) {
      return RunOptResult.takeError();
    }

    if (IntermediatePassName == "") {
      lsp::Logger::error("Unrecognized Pass Number {}!", std::to_string(N));
      return make_error<lsp::LSPError>(
          formatv("Unrecognized pass number {}!", N),
          lsp::ErrorCode::InvalidParams);
    }

    return IntermediatePassName;
  }
};

} // namespace llvm

#endif // LLVM_TOOLS_LLVM_LSP_OPTRUNNER_H
