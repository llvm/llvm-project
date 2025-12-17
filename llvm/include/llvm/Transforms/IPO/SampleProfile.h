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

#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include <string>

namespace llvm {

class Module;

LLVM_ABI extern cl::opt<int> SampleHotCallSiteThreshold;
LLVM_ABI extern cl::opt<int> SampleColdCallSiteThreshold;
LLVM_ABI extern cl::opt<int> ProfileInlineGrowthLimit;
LLVM_ABI extern cl::opt<int> ProfileInlineLimitMin;
LLVM_ABI extern cl::opt<int> ProfileInlineLimitMax;
LLVM_ABI extern cl::opt<bool> SortProfiledSCC;

namespace vfs {
class FileSystem;
} // namespace vfs

/// The sample profiler data loader pass.
class SampleProfileLoaderPass : public PassInfoMixin<SampleProfileLoaderPass> {
public:
  LLVM_ABI SampleProfileLoaderPass(
      std::string File = "", std::string RemappingFile = "",
      ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None,
      IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr,
      bool DisableSampleProfileInlining = false,
      bool UseFlattenedProfile = false);

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  std::string ProfileFileName;
  std::string ProfileRemappingFileName;
  const ThinOrFullLTOPhase LTOPhase;
  IntrusiveRefCntPtr<vfs::FileSystem> FS;
  bool DisableSampleProfileInlining;
  bool UseFlattenedProfile;
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_IPO_SAMPLEPROFILE_H
