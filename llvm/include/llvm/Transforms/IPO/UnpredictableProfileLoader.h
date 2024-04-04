//===-- UnpredictableProfileLoader.h - Unpredictable Profile Loader -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_UNPREDICTABLEPROFILELOADER_H
#define LLVM_TRANSFORMS_IPO_UNPREDICTABLEPROFILELOADER_H

#include "llvm/IR/PassManager.h"
#include "llvm/ProfileData/SampleProfReader.h"

namespace llvm {

class Module;

struct UnpredictableProfileLoaderPass
    : PassInfoMixin<UnpredictableProfileLoaderPass> {
  UnpredictableProfileLoaderPass(StringRef FrequencyProfileFile);
  UnpredictableProfileLoaderPass();
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &);
  std::unique_ptr<SampleProfileReader> FreqReader, MispReader;
  bool loadSampleProfile(Module &M);
  bool addUpredictableMetadata(Module &F);
  bool addUpredictableMetadata(Function &F);
  ErrorOr<double> getMispredictRatio(const FunctionSamples *FreqSamples,
                                     const FunctionSamples *MispSamples,
                                     const Instruction *I);
  const std::string FrequencyProfileFile;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_UNPREDICTABLEPROFILELOADER_H
