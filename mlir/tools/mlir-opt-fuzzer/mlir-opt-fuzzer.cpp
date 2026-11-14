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

#include "mlir/FuzzMutate/MLIRMutator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/FuzzMutate/FuzzerCLI.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LibraryOptionsContext.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static std::optional<llvm::StringRef> getPipeline(llvm::StringRef slug) {
  return llvm::StringSwitch<std::optional<llvm::StringRef>>(slug)
#define MLIR_OPT_FUZZER_PIPELINE(slug, pipeline) .Case(slug, pipeline)
#include "MlirOptFuzzerPipelines.def"
#undef MLIR_OPT_FUZZER_PIPELINE
      .Default(std::nullopt);
}

static constexpr llvm::StringRef kModuleAnchor = "builtin.module";

static llvm::cl::opt<std::string> PipelineSlug(
    "pipeline", llvm::cl::desc("Pipeline slug from MlirOptFuzzerPipelines.def"),
    llvm::cl::init(""));

static std::unique_ptr<MLIRMutator> Mutator;

static std::unique_ptr<MLIRMutator> createMutator() {
  std::vector<std::unique_ptr<MLIRGeneMutationStrategy>> Strategies;
  Strategies.push_back(std::make_unique<NonsenseMutationStrategy>());
  Strategies.push_back(std::make_unique<SynonymousMutationStrategy>());
  Strategies.push_back(std::make_unique<MissenseMutationStrategy>());
  return std::make_unique<MLIRMutator>(std::move(Strategies));
}

static void handleLLVMFatalError(void *, const char *Message, bool) {
  llvm::dbgs() << "LLVM ERROR: " << Message << "\n"
               << "Aborting to trigger fuzzer exit handling.\n";
  abort();
}

// Per-input context: quiet (hot-path diag noise is not a finding —
// triage re-runs the artifact for detail) and deterministic (serial).
static std::unique_ptr<MLIRContext> makeInputContext() {
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  auto context = std::make_unique<MLIRContext>(registry);
  context->disableMultithreading();
  context->getDiagEngine().registerHandler([](Diagnostic &) {});
  return context;
}

extern "C" LLVM_ATTRIBUTE_USED int LLVMFuzzerInitialize(int *argc,
                                                        char ***argv) {
  registerAllPasses();

  // Route LLVM/MLIR fatal errors through abort() so libFuzzer records them.
  llvm::install_fatal_error_handler(handleLLVMFatalError, nullptr);

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

  // Create mutator
  Mutator = createMutator();

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

  auto context = makeInputContext();
  PassManager pm(context.get(), kModuleAnchor, PassManager::Nesting::Implicit);
  // Validated at startup; the parse cannot fail.
  (void)parsePassPipeline(pipelineText, pm, llvm::nulls());

  // An unparseable module is a rejected input, not a finding.
  auto module = parseAndVerify(str, *context);
  if (!module)
    return 0;

  // A pass signalling failure is a polite exit, not a finding.
  (void)pm.run(*module);

  // Check that the pipeline resulted in a valid module.
  if (failed(verify(*module))) {
    llvm::errs() << "mlir-opt-fuzzer: post-pipeline verification failed\n";
    abort();
  }

  return 0;
}

extern "C" LLVM_ATTRIBUTE_USED size_t LLVMFuzzerCustomMutator(uint8_t *Data,
                                                              size_t Size,
                                                              size_t MaxSize,
                                                              unsigned Seed) {
  assert(Mutator &&
         "MLIR mutator should have been created during fuzzer initialization");

  auto context = makeInputContext();
  auto owned = parseAndVerify(
      llvm::StringRef(reinterpret_cast<const char *>(Data), Size), *context);
  if (!owned)
    // Dead input — nothing structural to mutate.
    return 0;

  ModuleOp module = *owned;
  Mutator->mutateModule(module, Seed, MaxSize);

  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  if (out.empty() || out.size() > MaxSize)
    // Mutation aborted — no corpus poisoning.
    return 0;

  // Corpus hygiene (llvm-opt-fuzzer discipline): only re-admit units
  // that survive the target's own front door — the alternative is a
  // corpus full of parser-error junk instead of pass behavior.
  if (!parseAndVerify(out, *context))
    return 0;
  std::memcpy(Data, out.data(), out.size());
  return out.size();
}

extern "C" LLVM_ATTRIBUTE_USED size_t LLVMFuzzerCustomCrossOver(
    const uint8_t *Data1, size_t Size1, const uint8_t *Data2, size_t Size2,
    uint8_t *Out, size_t MaxOutSize, unsigned Seed) {
  return 0;
}
