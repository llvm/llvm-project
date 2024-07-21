//=== UnpredictableProfileLoader.cpp - Unpredictable Profile Loader -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass reads a sample profile containing mispredict counts and a sample
// profile containing execution counts and computes branch mispredict ratios for
// each conditional instruction. If a sufficiently high mispredict ratio is
// found !unpredictable metadata is added.
//
// Note that this requires that the mispredict and frequency profiles have
// comparable magnitudes.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/UnpredictableProfileLoader.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

#define DEBUG_TYPE "unpredictable-profile-loader"

static cl::opt<std::string> UnpredictableHintsFile(
    "unpredictable-hints-file",
    cl::desc("Path to the unpredictability hints profile"), cl::Hidden);

// Typically this file will be provided via PGOOpt. This option is provided
// primarily for debugging and testing.
static cl::opt<std::string>
    FrequencyProfileOption("unpredictable-hints-frequency-profile",
                           cl::desc("Path to an execution frequency profile to "
                                    "use as a baseline for unpredictability"),
                           cl::Hidden);

// This determines the minimum apparent mispredict ratio which should earn a
// mispredict metadata annotation.
static cl::opt<double> MinimumRatio(
    "unpredictable-hints-min-ratio",
    cl::desc(
        "Absolute minimum branch miss ratio to apply MD_unpredictable from"),
    cl::init(0.2), cl::Hidden);

// This option is useful for dealing with two different sampling frequencies.
static cl::opt<double>
    RatioFactor("unpredictable-hints-factor",
                cl::desc("Multiply all ratios by this factor"), cl::init(1.0),
                cl::ReallyHidden);

// Lookup samples for an Instruction's corresponding location in a
// FunctionSamples profile. The count returned is directly from the profile
// representing the number of samples seen.
ErrorOr<double> UnpredictableProfileLoaderPass::getMispredictRatio(
    const FunctionSamples *FuncFreqSamples,
    const FunctionSamples *FuncMispSamples, const Instruction *I) {

  const auto &Loc = I->getDebugLoc();
  if (!Loc)
    return std::error_code();

  const FunctionSamples *FreqSamples =
      FuncFreqSamples->findFunctionSamples(Loc, FreqReader->getRemapper());
  if (!FreqSamples)
    return std::error_code();
  const ErrorOr<uint64_t> FreqCount = FreqSamples->findSamplesAt(
      FunctionSamples::getOffset(Loc), Loc->getBaseDiscriminator());
  if (!FreqCount)
    return std::error_code();

  const FunctionSamples *MispSamples =
      FuncMispSamples->findFunctionSamples(Loc, MispReader->getRemapper());
  if (!MispSamples)
    return std::error_code();
  const ErrorOr<uint64_t> MispCount = MispSamples->findSamplesAt(
      FunctionSamples::getOffset(Loc), Loc->getBaseDiscriminator());
  if (!MispCount)
    return std::error_code();

  const double Freq = FreqCount.get();
  if (!Freq)
    return std::error_code();

  const double Misp = MispCount.get();
  const double MissRatio = (Misp * RatioFactor) / Freq;

  LLVM_DEBUG(dbgs() << "Computing mispredict ratio of " << format("%0.2f", Misp)
                    << "/" << format("%0.2f", Freq) << " * "
                    << format("%0.2f", RatioFactor.getValue()) << " = "
                    << format("%0.2f", MissRatio) << " for instruction\n"
                    << *I << "\n");
  return MissRatio;
}

// Examine all Select and BranchInsts in a function, adding !unpredictable
// metadata if they appear in the mispredict profile with sufficient weight.
bool UnpredictableProfileLoaderPass::addUpredictableMetadata(Function &F) {

  const FunctionSamples *FreqSamples = FreqReader->getSamplesFor(F);
  if (!FreqSamples)
    return false;

  const FunctionSamples *MispSamples = MispReader->getSamplesFor(F);
  if (!MispSamples)
    return false;

  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (!isa<BranchInst>(&I) && !isa<SelectInst>(&I) && !isa<SwitchInst>(&I))
        continue;
      if (I.hasMetadata(LLVMContext::MD_unpredictable))
        continue;

      const ErrorOr<double> RatioOrError =
          getMispredictRatio(FreqSamples, MispSamples, &I);
      if (!RatioOrError)
        continue;
      const double MissRatio = RatioOrError.get();

      if (MissRatio < MinimumRatio) {
        LLVM_DEBUG(dbgs() << "\tRatio " << format("%0.2f", MissRatio)
                          << " is below threshold of "
                          << format("%0.2f", MinimumRatio.getValue())
                          << "; ignoring.\n");
        continue;
      }

      // In the future we probably want to attach more information here, such as
      // the mispredict count or ratio.
      MDNode *MD = MDNode::get(I.getContext(), std::nullopt);
      I.setMetadata(LLVMContext::MD_unpredictable, MD);
      MadeChange = true;
    }
  }

  return MadeChange;
}

bool UnpredictableProfileLoaderPass::addUpredictableMetadata(Module &M) {
  bool MadeChange = false;

  for (Function &F : M)
    MadeChange |= addUpredictableMetadata(F);

  // Return an indication of whether we changed anything or not.
  return MadeChange;
}

bool UnpredictableProfileLoaderPass::loadSampleProfile(Module &M) {
  if (MispReader && FreqReader)
    return true;

  assert(!MispReader && !FreqReader &&
         "Expected both or neither profile readers");

  LLVMContext &Ctx = M.getContext();
  auto FS = vfs::getRealFileSystem();

  auto ReadProfile = [&Ctx,
                      &FS](const std::string ProfileFile,
                           std::unique_ptr<SampleProfileReader> &ReaderPtr) {
    if (ProfileFile.empty())
      return false;

    ErrorOr<std::unique_ptr<SampleProfileReader>> ReaderOrErr =
        SampleProfileReader::create(ProfileFile, Ctx, *FS);
    if (std::error_code EC = ReaderOrErr.getError()) {
      std::string Msg = "Could not open profile: " + EC.message();
      Ctx.diagnose(DiagnosticInfoSampleProfile(ProfileFile, Msg,
                                               DiagnosticSeverity::DS_Warning));
      return false;
    }

    ReaderPtr = std::move(ReaderOrErr.get());
    if (std::error_code EC = ReaderPtr->read()) {
      std::string Msg = "Profile reading failed: " + EC.message();
      Ctx.diagnose(DiagnosticInfoSampleProfile(ProfileFile, Msg));
      return false;
    }

    return true;
  };

  if (!ReadProfile(UnpredictableHintsFile, MispReader))
    return false;

  if (!ReadProfile(FrequencyProfileFile, FreqReader))
    return false;

  return true;
}

UnpredictableProfileLoaderPass::UnpredictableProfileLoaderPass()
    : FrequencyProfileFile(FrequencyProfileOption) {}

UnpredictableProfileLoaderPass::UnpredictableProfileLoaderPass(
    StringRef PGOProfileFile)
    : FrequencyProfileFile(FrequencyProfileOption.empty()
                               ? PGOProfileFile
                               : FrequencyProfileOption) {}

PreservedAnalyses UnpredictableProfileLoaderPass::run(Module &M,
                                                      ModuleAnalysisManager &) {
  if (!loadSampleProfile(M))
    return PreservedAnalyses::all();

  if (addUpredictableMetadata(M)) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    return PA;
  }

  return PreservedAnalyses::all();
}
