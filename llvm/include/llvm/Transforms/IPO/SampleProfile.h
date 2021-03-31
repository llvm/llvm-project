//===- SampleProfile.h - SamplePGO pass ---------- --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file provides the interface for the sampled PGO loader pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_IPO_SAMPLEPROFILE_H
#define LLVM_TRANSFORMS_IPO_SAMPLEPROFILE_H

#include "llvm/IR/PassManager.h"
#include <string>

namespace llvm {

class Module;

/// The sample profiler data loader pass.
class SampleProfileLoaderPass : public PassInfoMixin<SampleProfileLoaderPass> {
public:
  SampleProfileLoaderPass(
      std::string File = "", std::string RemappingFile = "",
      ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None)
      : ProfileFileName(File), ProfileRemappingFileName(RemappingFile),
        LTOPhase(LTOPhase) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::string ProfileFileName;
  std::string ProfileRemappingFileName;
  ThinOrFullLTOPhase LTOPhase;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_SAMPLEPROFILE_H
