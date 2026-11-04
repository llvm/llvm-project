//===--- mlir-opt-fuzzer.cpp - Entry point to the MLIR opt fuzzer ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tool to fuzz MLIR pass pipelines using libFuzzer.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/FuzzMutate/FuzzerCLI.h"
#include "llvm/Support/LibraryOptionsContext.h"

using namespace mlir;

static std::optional<llvm::StringRef>
getPipeline(llvm::StringRef slug) {
  return llvm::StringSwitch<std::optional<llvm::StringRef>>(slug)
#define MLIR_OPT_FUZZER_PIPELINE(slug, pipeline) .Case(slug, pipeline)
#include "MlirOptFuzzerPipelines.def"
#undef MLIR_OPT_FUZZER_PIPELINE
      .Default(std::nullopt);
}

static constexpr llvm::StringRef kModuleAnchor = "builtin.module";

static llvm::cl::opt<std::string> PipelineSlug(
    "pipeline",
    llvm::cl::desc("Pipeline slug from MlirOptFuzzerPipelines.def"),
    llvm::cl::init(""));

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerInitialize(int *argc,
                                                        char ***argv) {
  registerAllPasses();

  static llvm::cl::LibraryOptionsContext LibraryOptions;
  llvm::parseFuzzerCLOpts(LibraryOptions, *argc, *argv);

  if (PipelineSlug.empty()) {
    llvm::errs() << "mlir-opt-fuzzer: no --pipeline given (see "
                    "MlirOptFuzzerPipelines.def)\n";
    std::exit(1);
  }

  std::optional<llvm::StringRef> pipelineText = getPipeline(PipelineSlug);
  if (!pipelineText) {
    llvm::errs() << "mlir-opt-fuzzer: unknown pipeline slug '" << PipelineSlug
                 << "' (see MlirOptFuzzerPipelines.def)\n";
    std::exit(1);
  }
  if (pipelineText->starts_with("builtin.module(")) {
    llvm::errs() << "mlir-opt-fuzzer: pipeline '" << PipelineSlug
                 << "' re-anchors builtin.module; provide inner text only\n";
    std::exit(1);
  }

  // Validate the pipeline as it will run.
  MLIRContext context;
  PassManager pm(&context, kModuleAnchor, PassManager::Nesting::Implicit);
  if (failed(parsePassPipeline(*pipelineText, pm, llvm::nulls()))) {
    llvm::errs() << "mlir-opt-fuzzer: invalid pipeline '" << *pipelineText
                 << "'\n";
    std::exit(1);
  }
  return 0;
}

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerTestOneInput(const uint8_t *data,
                                                          size_t size) {
  // Skip empty inputs.
  if (size == 0)
    return 0;
  llvm::StringRef str(reinterpret_cast<const char *>(data), size);
  // Skip if bytecode.
  if (str.starts_with("ML\xefR"))
    return 0;

  // Validated at startup.
  llvm::StringRef pipelineText = *getPipeline(PipelineSlug);

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  MLIRContext context(registry);
  // Sequential pass execution for deterministic replay.
  context.disableMultithreading();
  // Register diagnostic handler to avoid triggering exit behavior.
  context.getDiagEngine().registerHandler(
      [](mlir::Diagnostic &diag) { return; });

  PassManager pm(&context, kModuleAnchor, PassManager::Nesting::Implicit);
  // Validated at startup; the parse cannot fail.
  (void)parsePassPipeline(pipelineText, pm, llvm::nulls());

  // An unparseable module is a rejected input, not a finding.
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(str, ParserConfig(&context));
  if (!module)
    return 0;

  // A pass signalling failure is a polite exit, not a finding.
  (void)pm.run(*module);

  return 0;
}

extern "C" LLVM_ATTRIBUTE_USED size_t LLVMFuzzerCustomMutator(uint8_t *Data, size_t Size,
                                                              size_t MaxSize, unsigned Seed) {
  return 0;
}

extern "C" LLVM_ATTRIBUTE_USED size_t LLVMFuzzerCustomCrossOver(const uint8_t *Data1, size_t Size1,
                                                                const uint8_t *Data2, size_t Size2, uint8_t *Out,
                                                                size_t MaxOutSize, unsigned Seed) {
  return 0;
}
