//===- SandboxVectorizer.cpp - Vectorizer based on Sandbox IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "../VectorizeOptions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/Support/Regex.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

using namespace llvm;

/// A magic string for the default pass pipeline.
static const char *DefaultPipelineMagicStr = "*";

static constexpr char AllowFilesDelim = ',';

SandboxVectorizerPass::SandboxVectorizerPass() : FPM("fpm") {}

void SandboxVectorizerPass::buildPipeline() {
  if (Opts->sbvec_passes == DefaultPipelineMagicStr) {
    // TODO: Add passes to the default pipeline. It currently contains:
    //       - Seed collection, which creates seed regions and runs the pipeline
    //         - Bundle Vectorizer pass that starts from a seed
    //         - Load-Store Vectorizer pass for load-store chains
    //         - Accept or revert IR state pass
    FPM.setPassPipeline(
        "seed-collection<tr-save,bundle-vec(bottom-up),load-store-vec,"
        "tr-accept-or-revert>",
        sandboxir::SandboxVectorizerPassBuilder::createFunctionPass);
  } else {
    // Create the user-defined pipeline.
    FPM.setPassPipeline(
        Opts->sbvec_passes,
        sandboxir::SandboxVectorizerPassBuilder::createFunctionPass);
  }
}

SandboxVectorizerPass::SandboxVectorizerPass(SandboxVectorizerPass &&) =
    default;

SandboxVectorizerPass::~SandboxVectorizerPass() = default;

PreservedAnalyses SandboxVectorizerPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  TTI = &AM.getResult<TargetIRAnalysis>(F);
  AA = &AM.getResult<AAManager>(F);
  SE = &AM.getResult<ScalarEvolutionAnalysis>(F);

  bool Changed = runImpl(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SandboxVectorizerPass::allowFile(const std::string &SrcFilePath) {
  // Iterate over all files in AllowFiles separated by `AllowFilesDelim`.
  size_t DelimPos = 0;
  do {
    size_t LastPos = DelimPos != 0 ? DelimPos + 1 : DelimPos;
    DelimPos = Opts->sbvec_allow_files.find(AllowFilesDelim, LastPos);
    auto FileNameToMatch =
        Opts->sbvec_allow_files.substr(LastPos, DelimPos - LastPos);
    if (FileNameToMatch.empty())
      return false;
    // Note: This only runs when debugging so its OK not to reuse the regex.
    Regex FileNameRegex((".*" + FileNameToMatch + "$").str());
    assert(FileNameRegex.isValid() && "Bad regex!");
    if (FileNameRegex.match(SrcFilePath))
      return true;
  } while (DelimPos != std::string::npos);
  return false;
}

bool SandboxVectorizerPass::runImpl(Function &LLVMF) {
  Opts = &LLVMF.getContext().getLibraryOptions().get<VectorizeOptions>();
  if (Ctx == nullptr) {
    Ctx = std::make_unique<sandboxir::Context>(LLVMF.getContext());
    buildPipeline();
  }

  if (Opts->sbvec_print_pass_pipeline) {
    FPM.printPipeline(outs());
    return false;
  }

  // This is used for debugging.
  if (LLVM_UNLIKELY(Opts->sbvec_allow_files != ".*")) {
    const auto &SrcFilePath = LLVMF.getParent()->getSourceFileName();
    if (!allowFile(SrcFilePath))
      return false;
  }

  // If the target claims to have no vector registers early return.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true))) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX
                      << "Target has no vector registers, return.\n");
    return false;
  }
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX << "Analyzing " << LLVMF.getName()
                    << ".\n");
  // Early return if the attribute NoImplicitFloat is used.
  if (LLVMF.hasFnAttribute(Attribute::NoImplicitFloat)) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX
                      << "NoImplicitFloat attribute, return.\n");
    return false;
  }

  // Create SandboxIR for LLVMF and run BundleVec on it.
  sandboxir::Function &F = *Ctx->createFunction(&LLVMF);
  sandboxir::Analyses A(*AA, *SE, *TTI);
  bool Change = FPM.runOnFunction(F, A);
  // Given that sandboxir::Context `Ctx` is defined at a pass-level scope, the
  // maps from LLVM IR to Sandbox IR may go stale as later passes remove LLVM IR
  // objects. To avoid issues caused by this clear the context's state.
  // NOTE: The alternative would be to define Ctx and FPM within runOnFunction()
  Ctx->clear();
  return Change;
}
