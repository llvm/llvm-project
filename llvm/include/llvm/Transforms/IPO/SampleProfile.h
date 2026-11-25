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
#include "llvm/Support/Compiler.h"
#include "llvm/Support/VirtualFileSystemFwd.h"
#include <string>

namespace llvm {

class LLVMContext;
class Module;
namespace cl {
class LibraryOptionsContext;
}

/// The thresholds of the priority-based sample profile loader inliner. The
/// member initializers are the loader's defaults; llvm-profgen's preinliner
/// passes its own to getInlineOptions.
struct SampleProfileInlineOptions {
  int HotCallSiteThreshold = 3000;
  int ColdCallSiteThreshold = 45;
  unsigned GrowthLimit = 12;
  unsigned LimitMin = 100;
  unsigned LimitMax = 10000;
  bool SortProfiledSCC = true;
};

/// The sample profiler data loader pass.
class SampleProfileLoaderPass
    : public OptionalPassInfoMixin<SampleProfileLoaderPass> {
public:
  LLVM_ABI SampleProfileLoaderPass(
      std::string File = "", std::string RemappingFile = "",
      ThinOrFullLTOPhase LTOPhase = ThinOrFullLTOPhase::None,
      IntrusiveRefCntPtr<vfs::FileSystem> FS = nullptr,
      bool DisableSampleProfileInlining = false,
      bool UseFlattenedProfile = false);

  LLVM_ABI PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  /// Return -sample-profile-file.
  LLVM_ABI static StringRef getSampleProfileFile(const LLVMContext &C);

  /// Return \p Defaults overridden by the inliner options given in \p C.
  LLVM_ABI static SampleProfileInlineOptions
  getInlineOptions(const cl::LibraryOptionsContext &C,
                   SampleProfileInlineOptions Defaults);

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
