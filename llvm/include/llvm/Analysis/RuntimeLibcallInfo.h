//===-- RuntimeLibcallInfo.h - Runtime library information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_RUNTIMELIBCALLINFO_H
#define LLVM_ANALYSIS_RUNTIMELIBCALLINFO_H

#include "llvm/IR/RuntimeLibcalls.h"
#include "llvm/Pass.h"
#include <optional>
#include <string>

namespace llvm {

class LLVM_ABI RuntimeLibraryAnalysis
    : public AnalysisInfoMixin<RuntimeLibraryAnalysis> {
public:
  using Result = RTLIB::RuntimeLibcallsInfo;

  RuntimeLibraryAnalysis() = default;
  RuntimeLibraryAnalysis(ExceptionHandling ExceptionModel,
                         EABI EABIVersion = EABI::Default,
                         StringRef ABIName = "",
                         VectorLibrary VecLib = VectorLibrary::NoLibrary)
      : ExceptionModel(ExceptionModel), EABIVersion(EABIVersion),
        ABIName(ABIName.str()), VecLib(VecLib) {}

  RTLIB::RuntimeLibcallsInfo run(const Module &M, ModuleAnalysisManager &);

private:
  friend AnalysisInfoMixin<RuntimeLibraryAnalysis>;
  static AnalysisKey Key;

  // FIXME: These are TargetOptions values that are not yet represented in the
  // IR, copied here so run() can forward them to the RuntimeLibcallsInfo Module
  // constructor. Delete each one as they are migrated to module flags.
  ExceptionHandling ExceptionModel = ExceptionHandling::None;
  EABI EABIVersion = EABI::Default;
  std::string ABIName;
  VectorLibrary VecLib = VectorLibrary::NoLibrary;
};

class LLVM_ABI RuntimeLibraryInfoWrapper : public ImmutablePass {
  RuntimeLibraryAnalysis RTLA;
  std::optional<RTLIB::RuntimeLibcallsInfo> RTLCI;

public:
  static char ID;
  RuntimeLibraryInfoWrapper();
  RuntimeLibraryInfoWrapper(ExceptionHandling ExceptionModel,
                            EABI EABIVersion = EABI::Default,
                            StringRef ABIName = "",
                            VectorLibrary VecLib = VectorLibrary::NoLibrary);

  const RTLIB::RuntimeLibcallsInfo &getRTLCI(const Module &M) {
    if (!RTLCI) {
      ModuleAnalysisManager DummyMAM;
      RTLCI = RTLA.run(M, DummyMAM);
    }

    return *RTLCI;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

LLVM_ABI ModulePass *createRuntimeLibraryInfoWrapperPass();

} // namespace llvm

#endif
