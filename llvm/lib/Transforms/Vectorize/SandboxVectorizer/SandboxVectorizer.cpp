//===- SandboxVectorizer.cpp - Vectorizer based on Sandbox IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Regex.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizerPassBuilder.h"

using namespace llvm;

static cl::opt<bool>
    PrintPassPipeline("sbvec-print-pass-pipeline", cl::init(false), cl::Hidden,
                      cl::desc("Prints the pass pipeline and returns."));

/// A magic string for the default pass pipeline.
static const char *DefaultPipelineMagicStr = "*";

static cl::opt<std::string> UserDefinedPassPipeline(
    "sbvec-passes", cl::init(DefaultPipelineMagicStr), cl::Hidden,
    cl::desc("Comma-separated list of vectorizer passes. If not set "
             "we run the predefined pipeline."));

// This option is useful for bisection debugging.
// For example you may use it to figure out which filename is the one causing a
// miscompile. You can specify a regex for the filename like: "/[a-m][^/]*"
// which will enable any file name starting with 'a' to 'm' and disable the
// rest. If the miscompile goes away, then we try "/[n-z][^/]*" for the other
// half of the range, from 'n' to 'z'. If we can reproduce the miscompile then
// we can keep looking in [n-r] and [s-z] and so on, in a binary-search fashion.
//
// Please note that we are using [^/]* and not .* to make sure that we are
// matching the actual filename and not some other directory in the path.
cl::opt<std::string> AllowFiles(
    "sbvec-allow-files", cl::init(".*"), cl::Hidden,
    cl::desc("Run the vectorizer only on file paths that match any in the "
             "list of comma-separated regex's."));
static constexpr const char AllowFilesDelim = ',';

SandboxVectorizerPass::SandboxVectorizerPass() : FPM("fpm") {
  if (UserDefinedPassPipeline == DefaultPipelineMagicStr) {
    // TODO: Add passes to the default pipeline. It currently contains:
    //       - Seed collection, which creates seed regions and runs the pipeline
    //         - Bottom-up Vectorizer pass that starts from a seed
    //         - Accept or revert IR state pass
    FPM.setPassPipeline(
        "seed-collection<tr-save,bottom-up-vec,tr-accept-or-revert>",
        sandboxir::SandboxVectorizerPassBuilder::createFunctionPass);
  } else {
    // Create the user-defined pipeline.
    FPM.setPassPipeline(
        UserDefinedPassPipeline,
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
    DelimPos = AllowFiles.find(AllowFilesDelim, LastPos);
    auto FileNameToMatch = AllowFiles.substr(LastPos, DelimPos - LastPos);
    if (FileNameToMatch.empty())
      return false;
    // Note: This only runs when debugging so its OK not to reuse the regex.
    Regex FileNameRegex(".*" + FileNameToMatch + "$");
    assert(FileNameRegex.isValid() && "Bad regex!");
    if (FileNameRegex.match(SrcFilePath))
      return true;
  } while (DelimPos != std::string::npos);
  return false;
}

bool SandboxVectorizerPass::runImpl(Function &LLVMF) {
  if (Ctx == nullptr)
    Ctx = std::make_unique<sandboxir::Context>(LLVMF.getContext());

  if (PrintPassPipeline) {
    FPM.printPipeline(outs());
    return false;
  }

  // This is used for debugging.
  if (LLVM_UNLIKELY(AllowFiles != ".*")) {
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

  // Create SandboxIR for LLVMF and run BottomUpVec on it.
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
