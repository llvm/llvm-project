//===- Transforms/SampleProfile.h - SamplePGO pass--------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides the interface for the sampled PGO loader pass.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SAMPLEPROFILE_H
#define LLVM_TRANSFORMS_SAMPLEPROFILE_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// The sample profiler data loader pass.
class SampleProfileLoaderPass : public PassInfoMixin<SampleProfileLoaderPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  SampleProfileLoaderPass(std::string File = "", bool IsThinLTOPreLink = false)
      : ProfileFileName(File), IsThinLTOPreLink(IsThinLTOPreLink) {}

private:
  std::string ProfileFileName;
  bool IsThinLTOPreLink;
};

} // End llvm namespace
#endif
